"""R1c: where does a speculative step spend its time? One nsys capture of ~20 K=4 llama-server steps.

Not a stand-alone experiment: exps.r1 calls run_step(). The server (draft-simple K=4, draft backend sampling) runs under
`nsys profile --start-later`; after a warm-up request the capture covers one request, then the report is exported to sqlite,
parsed to JSON and the .nsys-rep / .sqlite are deleted. Secret safety: nsys and the server get an environment without
KEY / TOKEN / SECRET / PASSWORD / CREDENTIAL / RUNPOD_ variables (a report stores the environment), the files live in a
private temp dir outside the results dir, they are removed in a finally block, and *.nsys-rep / *.sqlite are in .gitignore
and on the results server deny list.

Timeline model (GPU activity only, from the CUDA trace; --cuda-graph-trace=node so graph nodes show as kernels):
  burst   = activity on any stream with gaps <= GAP_US between neighbours. Bursts shorter than TINY_US are noise.
  verify  = the long bursts (target forward over K + 1 tokens); drafts = the short ones. A step runs from one verify start
            to the next. Inside a step the first short burst is the draft re-eval of the accepted tokens, the others are draft
            token steps. Splitting by duration needs the draft pass to be well below the verify pass (about 1.6 ms vs 10 ms).
  memory ops are classed by direction and size: DtoD >= COPY_MIN is a recurrent-state snapshot copy, HtoD >= COPY_MIN a
            state / checkpoint restore, DtoH readback (logits, ids), the rest small copies.
  idle    = step time with no GPU activity, split by position: intra-burst gaps, after verify (CPU sampling, accept, draft
            bookkeeping), between draft bursts (draft sampling and launch), before verify (batch build and launch).
c1_removable = re-eval + readback + after-verify idle + between-draft idle (what backend sampling and no re-eval could take);
c1_floor = everything else that is not target verify or draft token steps. Heuristic, see the checks block of the result.
"""
from __future__ import annotations

import bisect
import json
import os
import pathlib
import shutil
import signal
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time

from exps import common

SESSION = "cr_r1"
GAP_US = 30.0
TINY_US = 400.0
COPY_MIN = 64 * 1024
SKIP_STEPS = 2       # prefill and the first step (draft state build) are not steady state
MAX_STEPS = 20
N_PREDICT = 56
COPY_KINDS = {1: "HtoD", 2: "DtoH", 3: "HtoA", 4: "AtoH", 5: "AtoA", 6: "AtoD", 7: "DtoA", 8: "DtoD", 9: "HtoH", 10: "PtoP"}
MS_KEYS = ("wall", "target_verify", "draft_gen", "draft_reeval", "snapshot_copy", "restore_copy", "readback", "small_copy",
           "other_kernel", "idle_intra_burst", "idle_after_verify", "idle_between_draft", "idle_before_verify", "idle_total",
           "cpu_sampling_est", "non_compute", "c1_removable", "c1_floor")


# ---------------------------------------------------------------- pure analysis

def classify_copy(kind: str | None, nbytes: int) -> str:
    if kind == "DtoD" and nbytes >= COPY_MIN:
        return "snapshot_copy"
    if kind == "HtoD" and nbytes >= COPY_MIN:
        return "restore_copy"
    if kind == "DtoH":
        return "readback"
    return "small_copy"


def merge(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def union_len(intervals: list[tuple[int, int]]) -> int:
    return sum(b - a for a, b in merge(intervals))


def overlap_len(merged: list[tuple[int, int]], lo: int, hi: int) -> int:
    """Length of [lo, hi) covered by the merged (sorted, disjoint) intervals."""
    if hi <= lo:
        return 0
    i = _first_end_after(merged, lo)
    tot = 0
    while i < len(merged) and merged[i][0] < hi:
        tot += max(min(merged[i][1], hi) - max(merged[i][0], lo), 0)
        i += 1
    return tot


def _first_end_after(merged: list[tuple[int, int]], lo: int) -> int:
    a, b = 0, len(merged)
    while a < b:
        m = (a + b) // 2
        if merged[m][1] <= lo:
            a = m + 1
        else:
            b = m
    return a


def make_bursts(acts: list[dict], gap_ns: float) -> list[dict]:
    """Cluster activity into bursts. Each burst: {t0, t1}."""
    bursts: list[dict] = []
    for a, b in merge([(x["t0"], x["t1"]) for x in acts]):
        if bursts and a - bursts[-1]["t1"] <= gap_ns:
            bursts[-1]["t1"] = b
        else:
            bursts.append({"t0": a, "t1": b})
    kern = merge([(x["t0"], x["t1"]) for x in acts if x["kind"] == "kernel"])
    for bu in bursts:  # a burst without kernels (a lone logits readback) is never a forward pass
        i = _first_end_after(kern, bu["t0"])
        bu["has_kernel"] = i < len(kern) and kern[i][0] < bu["t1"]
    return bursts


def is_tiny(b: dict, tiny_ns: float) -> bool:
    return b["t1"] - b["t0"] < tiny_ns or not b.get("has_kernel", True)


def split_long_short(bursts: list[dict], tiny_ns: float, min_ratio: float = 2.0) -> float | None:
    """Duration threshold between draft bursts and verify bursts: the widest jump in log duration above tiny_ns.
    None when the largest jump is below min_ratio (no two clusters)."""
    import math
    durs = sorted(math.log(b["t1"] - b["t0"]) for b in bursts if not is_tiny(b, tiny_ns))
    if len(durs) < 2:
        return None
    gaps = [(durs[i + 1] - durs[i], i) for i in range(len(durs) - 1)]
    g, i = max(gaps)
    if g < math.log(min_ratio):
        return None
    return math.exp(0.5 * (durs[i] + durs[i + 1]))


def _cat_of(x: dict, long_i: int, short_idx: dict, burst_of) -> str:
    if x["kind"] in ("memcpy", "memset"):
        return classify_copy(x.get("copy_kind"), x.get("bytes") or 0)
    bi = burst_of(x["t0"])
    if bi == long_i:
        return "target_verify"
    if bi in short_idx:
        return "draft_reeval" if short_idx[bi] == 0 else "draft_gen"
    return "other_kernel"


def analyze(acts: list[dict], k: int, gap_us: float = GAP_US, tiny_us: float = TINY_US, skip: int = SKIP_STEPS,
            max_steps: int = MAX_STEPS) -> dict:
    """Per-step time split from GPU activity. acts: [{t0, t1 (ns), kind: kernel|memcpy|memset, stream, name, bytes, copy_kind}]."""
    out: dict = {"ok": False, "error": None, "k": k, "n_acts": len(acts)}
    if not acts:
        out["error"] = "no GPU activity in the capture"
        return out
    acts = sorted(acts, key=lambda x: x["t0"])
    bursts = make_bursts(acts, gap_us * 1e3)
    thr = split_long_short(bursts, tiny_us * 1e3)
    out["n_bursts"] = len(bursts)
    out["streams"] = stream_summary(acts)
    if thr is None:
        out["error"] = "bursts do not split into verify and draft clusters (need verify >= 2x draft duration)"
        out["burst_ms_top"] = [round((b["t1"] - b["t0"]) / 1e6, 3) for b in sorted(bursts, key=lambda b: b["t0"] - b["t1"])[:12]]
        return out
    kind_of = ["tiny" if is_tiny(b, tiny_us * 1e3) else ("long" if b["t1"] - b["t0"] >= thr else "short") for b in bursts]
    longs = [i for i, kd in enumerate(kind_of) if kd == "long"]
    out.update(threshold_ms=thr / 1e6, n_long=len(longs), n_short=kind_of.count("short"), n_tiny=kind_of.count("tiny"))
    starts = [b["t0"] for b in bursts]
    merged_all = merge([(x["t0"], x["t1"]) for x in acts])

    def burst_of(t: int) -> int:
        return max(bisect.bisect_right(starts, t) - 1, 0)

    steps, shorts_per_step = [], []
    use = longs[skip:]
    for n in range(len(use) - 1):
        if len(steps) >= max_steps:
            break
        li, lj = use[n], use[n + 1]
        t_a, t_b = bursts[li]["t0"], bursts[lj]["t0"]
        short_idx = {bi: j for j, bi in enumerate(i for i in range(li + 1, lj) if kind_of[i] == "short")}
        shorts_per_step.append(len(short_idx))
        by_cat: dict[str, list[tuple[int, int]]] = {}
        for x in acts:
            if x["t0"] < t_a or x["t0"] >= t_b:
                continue
            by_cat.setdefault(_cat_of(x, li, short_idx, burst_of), []).append((x["t0"], min(x["t1"], t_b)))
        wall = t_b - t_a
        busy_all = union_len([iv for v in by_cat.values() for iv in v])
        idle_total = wall - busy_all
        # idle by position, over the burst sequence of this step; time covered by tiny bursts is not idle
        seq = [li] + [i for i in range(li + 1, lj) if kind_of[i] != "tiny"]
        idle = {"after_verify": 0, "between_draft": 0, "before_verify": 0, "intra_burst": 0}
        for j, bi in enumerate(seq):
            lo, hi = bursts[bi]["t1"], bursts[seq[j + 1]]["t0"] if j + 1 < len(seq) else t_b
            slot = "after_verify" if j == 0 else ("before_verify" if j == len(seq) - 1 else "between_draft")
            idle[slot] += max(hi - lo - overlap_len(merged_all, lo, hi), 0)
        idle["intra_burst"] = max(idle_total - sum(idle.values()), 0)
        ms = {c: union_len(by_cat.get(c, [])) / 1e6 for c in ("target_verify", "draft_gen", "draft_reeval", "snapshot_copy",
                                                             "restore_copy", "readback", "small_copy", "other_kernel")}
        ms.update(wall=wall / 1e6, idle_total=idle_total / 1e6, idle_after_verify=idle["after_verify"] / 1e6,
                  idle_between_draft=idle["between_draft"] / 1e6, idle_before_verify=idle["before_verify"] / 1e6,
                  idle_intra_burst=idle["intra_burst"] / 1e6)
        ms["cpu_sampling_est"] = ms["idle_after_verify"] + ms["readback"]
        ms["non_compute"] = ms["wall"] - ms["target_verify"] - ms["draft_gen"] - ms["draft_reeval"]
        ms["c1_removable"] = ms["draft_reeval"] + ms["readback"] + ms["idle_after_verify"] + ms["idle_between_draft"]
        ms["c1_floor"] = ms["wall"] - ms["target_verify"] - ms["draft_gen"] - ms["c1_removable"]
        steps.append(ms)
    if not steps:
        out["error"] = f"too few verify bursts for {skip} skipped + 1 steps ({len(longs)} found)"
        return out
    out["ok"] = True
    out["n_steps"] = len(steps)
    out["mean_ms"] = {key: statistics.fmean(s[key] for s in steps) for key in MS_KEYS}
    out["median_ms"] = {key: statistics.median(s[key] for s in steps) for key in MS_KEYS}
    out["shorts_per_step_median"] = statistics.median(shorts_per_step)
    out["checks"] = {"shorts_per_step_close_to_k": abs(out["shorts_per_step_median"] - k) <= 1,
                     "c1_floor_le_2ms": out["median_ms"]["c1_floor"] <= 2.0,
                     "note": "c1 is a heuristic: removable = re-eval + readback + idle after verify + idle between drafts"}
    out["steps_ms"] = [{key: round(s[key], 3) for key in MS_KEYS} for s in steps]
    return out


def stream_summary(acts: list[dict], top: int = 3) -> dict:
    """Busy ms and top kernel names per stream, to check by eye which stream is the target and which the draft."""
    per: dict = {}
    for x in acts:
        d = per.setdefault(str(x.get("stream")), {"busy_ns": 0, "n": 0, "names": {}})
        d["busy_ns"] += x["t1"] - x["t0"]
        d["n"] += 1
        if x["kind"] == "kernel":
            d["names"][x.get("name") or "?"] = d["names"].get(x.get("name") or "?", 0) + x["t1"] - x["t0"]
    return {s: {"busy_ms": d["busy_ns"] / 1e6, "n": d["n"],
                "top_kernels": [n[:100] for n, _ in sorted(d["names"].items(), key=lambda kv: -kv[1])[:top]]} for s, d in per.items()}


# ---------------------------------------------------------------- sqlite

def _cols(con, table: str) -> list[str]:
    return [r[1] for r in con.execute(f'PRAGMA table_info("{table}")')]


def read_sqlite(path) -> list[dict]:
    """Kernel, memcpy and memset activity of an nsys sqlite export. Raises on a file without the CUDA tables."""
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        tables = {r[0] for r in con.execute("select name from sqlite_master where type = 'table'")}
        strings = dict(con.execute("select id, value from StringIds")) if "StringIds" in tables else {}
        kinds = dict(COPY_KINDS)
        if "ENUM_CUDA_MEMCPY_OPER" in tables:
            kinds = {i: n.split("_")[-1] if n.startswith("CUDA_MEMCPY") else n for i, n in con.execute("select id, name from ENUM_CUDA_MEMCPY_OPER")}
            kinds = {i: {"HtoD": "HtoD", "DtoH": "DtoH", "DtoD": "DtoD"}.get(v, v) for i, v in kinds.items()} or dict(COPY_KINDS)
            if not any(v == "DtoD" for v in kinds.values()):
                kinds = dict(COPY_KINDS)
        acts: list[dict] = []
        if "CUPTI_ACTIVITY_KIND_KERNEL" in tables:
            c = _cols(con, "CUPTI_ACTIVITY_KIND_KERNEL")
            name_col = "demangledName" if "demangledName" in c else ("shortName" if "shortName" in c else None)
            q = f'select start, end, streamId, {name_col if name_col else "0"} from CUPTI_ACTIVITY_KIND_KERNEL'
            for t0, t1, st, nm in con.execute(q):
                acts.append({"t0": t0, "t1": t1, "kind": "kernel", "stream": st, "name": strings.get(nm, str(nm)) if isinstance(nm, int) else nm})
        if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
            for t0, t1, st, ck, nb in con.execute("select start, end, streamId, copyKind, bytes from CUPTI_ACTIVITY_KIND_MEMCPY"):
                acts.append({"t0": t0, "t1": t1, "kind": "memcpy", "stream": st, "bytes": nb, "copy_kind": kinds.get(ck, str(ck))})
        if "CUPTI_ACTIVITY_KIND_MEMSET" in tables:
            for t0, t1, st, nb in con.execute("select start, end, streamId, bytes from CUPTI_ACTIVITY_KIND_MEMSET"):
                acts.append({"t0": t0, "t1": t1, "kind": "memset", "stream": st, "bytes": nb})
        if not acts and not tables & {"CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMCPY"}:
            raise ValueError("no CUDA activity tables in %s (tables: %s)" % (path, sorted(tables)[:10]))
        return acts
    finally:
        con.close()


# ---------------------------------------------------------------- capture

def secret_env() -> dict:
    from exps import e4_bonsai
    return e4_bonsai.scrubbed_env()


def private_workdir() -> str:
    """Private temp dir outside the repo and the results dir (the results server must never see an nsys file)."""
    d = tempfile.mkdtemp(prefix="cr-r1-nsys-")
    os.chmod(d, 0o700)
    return d


def profile_cmd(nsys: str, prefix: str, argv: list[str]) -> list[str]:
    return [nsys, "profile", "-t", "cuda", "--sample=none", "--cpuctxsw=none", "--cuda-graph-trace=node", "--start-later=true",
            f"--session-new={SESSION}", "--export=sqlite", "-f", "true", "-o", prefix, *argv]


def control_cmd(nsys: str, verb: str) -> list[str]:
    return [nsys, verb, f"--session={SESSION}"]


def export_cmd(nsys: str, rep: str, sqlite: str) -> list[str]:
    return [nsys, "export", "--type", "sqlite", "--force-overwrite", "true", "-o", sqlite, rep]


def find_pids(needle_args: list[str], exclude_exe: str = "nsys") -> list[int]:
    """Pids whose cmdline contains every needle (from /proc), skipping processes whose exe name has exclude_exe."""
    out = []
    for p in pathlib.Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            args = (p / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        txt = [a.decode("utf-8", "replace") for a in args]
        if any(exclude_exe in os.path.basename(a) for a in txt[:2]):  # nsys itself, binary or script
            continue
        if all(n in txt for n in needle_args):
            out.append(int(p.name))
    return out


def _run(cmd: list[str], env: dict, timeout: float) -> tuple[int | None, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=max(1.0, timeout), env=env)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return None, "timeout"
    except OSError as e:
        return 127, str(e)


def cleanup(workdir: str) -> None:
    shutil.rmtree(workdir, ignore_errors=True)


def capture(*, nsys: str, server_argv: list[str], port: int, prompt: str, budget: common.Budget, workdir: str, log_path: str,
            k: int = 4, n_predict: int = N_PREDICT, notes: list | None = None) -> dict:
    """Run the nsys flow. Never raises; the report files are removed from workdir before return."""
    from exps import e5_spec
    notes = notes if notes is not None else []
    res: dict = {"ok": False, "error": None, "nsys": nsys, "server_cmd": server_argv, "log": log_path, "analysis": None}
    env = secret_env()
    prefix = os.path.join(workdir, "r1step")
    rep, sql = prefix + ".nsys-rep", prefix + ".sqlite"
    proc = None
    logf = open(log_path, "w")
    try:
        code, out = _run([nsys, "--version"], env, 20)
        res["nsys_version"] = out.strip()[:120] if code == 0 else None
        proc = subprocess.Popen(profile_cmd(nsys, prefix, server_argv), stdout=logf, stderr=subprocess.STDOUT, env=env,
                                start_new_session=True, preexec_fn=e5_spec._die_with_parent if sys.platform.startswith("linux") else None)
        t_end = time.monotonic() + budget.timeout(900.0, reserve_s=30)
        ready = False
        while time.monotonic() < t_end:
            if proc.poll() is not None:
                res["error"] = f"nsys / server exited early with code {proc.returncode}"
                return res
            if e5_spec.get_ok(f"http://127.0.0.1:{port}/health"):
                ready = True
                break
            time.sleep(1.0)
        if not ready:
            res["error"] = "timeout waiting for /health"
            return res
        base = f"http://127.0.0.1:{port}"
        text = e5_spec.post_json(base + "/apply-template", e5_spec.template_body(prompt, False))["prompt"]
        e5_spec.post_json(base + "/completion", e5_spec.completion_body(text, 24), budget.timeout(120.0, reserve_s=10))  # warm-up
        code, out = _run(control_cmd(nsys, "start"), env, 60)
        if code != 0:
            res["error"] = f"nsys start failed ({code}): {out[-300:].strip()}"
            return res
        time.sleep(0.5)
        resp = e5_spec.post_json(base + "/completion", e5_spec.completion_body(text, n_predict), budget.timeout(120.0, reserve_s=10))
        res["request"] = e5_spec.parse_completion(resp)
        code, out = _run(control_cmd(nsys, "stop"), env, budget.timeout(180.0, reserve_s=10) or 60)
        if code != 0:
            res["error"] = f"nsys stop failed ({code}): {out[-300:].strip()}"
            return res
        for pid in find_pids(["--port", str(port)]):  # the server, not nsys: its exit lets nsys finish the report
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
        try:
            proc.wait(timeout=budget.timeout(180.0, reserve_s=5))
        except subprocess.TimeoutExpired:
            notes.append("nsys did not exit after the server stopped; killing it")
        if not os.path.exists(rep):
            res["error"] = "no .nsys-rep after the capture"
            return res
        if not os.path.exists(sql):
            code, out = _run(export_cmd(nsys, rep, sql), env, budget.timeout(300.0, reserve_s=5))
            if code != 0 or not os.path.exists(sql):
                res["error"] = f"nsys export failed ({code}): {out[-300:].strip()}"
                return res
        acts = read_sqlite(sql)
        res["analysis"] = analyze(acts, k)
        res["ok"] = bool(res["analysis"]["ok"])
        res["error"] = res["analysis"].get("error")
        return res
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"
        return res
    finally:
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=20)
            except Exception:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    pass
        logf.close()
        for path in (rep, sql, sql + "-wal", sql + "-shm"):
            try:
                os.remove(path)
            except OSError:
                pass
        cleanup(workdir)


def main(argv=None) -> int:
    ap_help = "python -m exps.r1_step --analyze-sqlite FILE.sqlite [--k 4]  (parse an existing export; prints JSON, deletes nothing)"
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) >= 2 and argv[0] == "--analyze-sqlite":
        k = int(argv[argv.index("--k") + 1]) if "--k" in argv else 4
        print(json.dumps(analyze(read_sqlite(argv[1]), k), indent=1))
        return 0
    print(ap_help)
    return 2


if __name__ == "__main__":
    sys.exit(main())
