"""R2b multicol: the PQ2_0 multi-column mat-vec (mmvq-pq2_0.cuh) on sm_120, behind GGML_CUDA_PQ2_MULTICOL=1.

  python -m exps.fetch_models --models pq2 --timeout-min 6 && python -m exps.r2b_multicol [--budget-min 22]

ggml_cuda_pq2_multicol() (common.cuh) enables the #306 kernel only on Ada. With GGML_CUDA_PQ2_MULTICOL=1 (read once, host side) it
is also enabled on newer NVIDIA GPUs, for 3-8 columns, which is the verify width of a speculative step with K = 2..7. On sm_120 the
default is the generic mul_mat_vec_q<PQ2_0, ncols>. The kernel is launched through ggml_cuda_kernel_launch (PDL on sm_90+) and the
activation quantizer triggers programmatic launch completion at its start, so the kernel calls ggml_cuda_pdl_sync() before it reads vy.
Configs (environment of each process): off = variable unset (default), on = GGML_CUDA_PQ2_MULTICOL=1, on_pdl0 = on plus GGML_CUDA_PDL=0
(control: PDL off for every kernel, so the sync cannot matter).
t  correctness. test-backend-ops test -o MUL_MAT -b CUDA0 -p type_a=pq2_0, (every PQ2_0 case against the CPU backend; widths 1-32 at
   K 1024 and 5120, the Bonsai 2 shapes with odd row counts, the 27B projections). on, on_pdl0 and off, in that order (a short budget drops off first). A filter that matches nothing
   (an older test-backend-ops without -p) runs MUL_MAT whole.
k  kernel path. nsys CUDA trace (no capture range, private 0700 dir, report deleted after parsing, never in the results dir) of
   llama-bench -p 4 -n 0 -r 1: the kernel names for off and on. on must show mul_mat_vec_pq2_0_mc<(int)4> and no generic
   mul_mat_vec_q<(ggml_type)142, (int)4, ...>; off must show the generic one and no mc kernel.
b  llama-bench on Bonsai 2 27B PQ2_0, f16 KV, -fa on, -ub 2048, depth 0: -p 1,2,3,4,5,6,7,8,9 -n 0 -r 5 (one batch of W tokens is what a
   K = W - 1 verify step costs) and -p 0 -n 128 -r 5 (decode, one column). Two rounds, the second in reverse config order, so a
   clock drift cancels in the median. Reported: ms per batch per width for each config, the gain of on over off in percent
   (ms_off / ms_on - 1), and the decode tok/s of each config (width 1 and 2 do not use the kernel: they are the control).
f  op timing. test-backend-ops perf -o MUL_MAT -b CUDA0 -p type_a=pq2_0, (widths 1-32 at the 27B projection shapes), off and on:
   us/run per (m, n, k) and the gain of on over off for n = 3..8.
The kernel waits for the quantizer but never triggers launch completion for its successor (the generic kernel does), so the gain of
on over off is a slight underestimate; the run adds a note on this.
Parts run in the order t, k, b, f; a failed (or missing) test-backend-ops pass with the env set stops the run after t. Each has a share of --budget-min (weights 5 : 4 : 10 : 5, unused time rolls forward).
Result: results/r2b_mc/<gpu>/<stamp>.json with data.tbo, data.path, data.perf, data.bench, data.summary.
Time: pod failsafe 55 min = bootstrap with the cold build ~14 (test-backend-ops is built by bootstrap), download <= 6, run 22, reserve 5.
Needs CR_LLAMA_BIN with llama-bench and test-backend-ops (CR_BUILD_LLAMA=1); nsys is optional (part k is skipped without it).
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys

from exps import common, e4_bonsai, e4_pdl, e5_cost, hf_ids, r1, r1_step

EXP = "r2b_mc"
ENV_VAR = "GGML_CUDA_PQ2_MULTICOL"
CONFIGS = [("off", {}), ("on", {ENV_VAR: "1"}), ("on_pdl0", {ENV_VAR: "1", "GGML_CUDA_PDL": "0"})]
TBO_ORDER = [CONFIGS[1], CONFIGS[2], CONFIGS[0]]    # part t: the new path first, so a short budget drops off, not on
WIDTHS = list(range(1, 10))
MC_WIDTHS = range(3, 9)               # the kernel handles 3-8 columns
PQ2_FILTER = "type_a=pq2_0,"          # test-backend-ops -p: regex on the variant string of a case
REPS = 5
ROUNDS = 2
PARTS = "tkbf"
WEIGHTS = {"t": 5.0, "k": 4.0, "b": 10.0, "f": 5.0}
END_RESERVE_S = 25.0
PDL_LC_NOTE = ("the mc kernel waits for the quantizer (ggml_cuda_pdl_sync) but never calls ggml_cuda_pdl_lc, unlike mul_mat_vec_q: with 'on' "
               "the next kernel does not overlap its launch, so the gain of on over off is slightly understated; on_pdl0 vs on mixes this in")
MC_KERNEL_RE = re.compile(r"mul_mat_vec_pq2_0_mc<(?:\(int\))?(\d+)>")       # nsys prints template ints as (int)N
GEN_KERNEL_RE = re.compile(r"mul_mat_vec_q<\(ggml_type\)142,\s*(?:\(int\))?(\d+)\b")        # generic path for PQ2_0 (GGML_TYPE_PQ2_0 = 142)
PERF_RE = re.compile(r"MUL_MAT\((type_a=pq2_0,[^)]*)\):\s+(\d+) runs -\s+([0-9.]+) us/run")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
TBO_FAIL_RE = re.compile(r"\bFAIL\b")


# ---------------------------------------------------------------- pure helpers

def run_env(extra: dict) -> dict:
    """Our environment without the two knobs, plus extra."""
    env = {k: v for k, v in os.environ.items() if k not in (ENV_VAR, "GGML_CUDA_PDL")}
    env.update(extra)
    return env


def tbo_cmd(bin_dir, mode: str = "test", filt: str | None = PQ2_FILTER) -> list[str]:
    cmd = [str(bin_dir / "test-backend-ops"), mode, "-o", "MUL_MAT", "-b", "CUDA0"]
    return cmd + (["-p", filt] if filt else [])


def bench_pp_cmd(bin_dir, model: str, widths: list[int] = WIDTHS, reps: int = REPS) -> list[str]:
    """llama-bench -p W1,W2,... -n 0 -r reps at depth 0 (flags of E4/E5)."""
    return e5_cost.target_cmd(bin_dir, model, widths, 0, reps)


def bench_tg_cmd(bin_dir, model: str, reps: int = REPS) -> list[str]:
    return e5_cost.plain_cmd(bin_dir, model, 0, reps)


def parse_perf(text: str) -> list[dict]:
    """Console lines of test-backend-ops perf -> [{m, n, k, runs, us}] for the PQ2_0 cases that ran."""
    out = []
    for params, runs, us in PERF_RE.findall(text):
        kv = dict(p.split("=", 1) for p in re.split(r",(?=[a-z_]+=)", params) if "=" in p)
        try:
            out.append({"m": int(kv["m"]), "n": int(kv["n"]), "k": int(kv["k"]), "runs": int(runs), "us": float(us)})
        except (KeyError, ValueError):
            continue
    return out


def perf_gain(off: list[dict], on: list[dict]) -> list[dict]:
    """Per (m, k, n): us off, us on and gain_pct = us_off / us_on - 1 in percent, for the shapes both configs ran."""
    by_on = {(r["m"], r["k"], r["n"]): r["us"] for r in on}
    rows = []
    for r in off:
        key = (r["m"], r["k"], r["n"])
        if key in by_on and by_on[key] > 0:
            rows.append({"m": r["m"], "k": r["k"], "n": r["n"], "us_off": r["us"], "us_on": by_on[key],
                         "gain_pct": 100.0 * (r["us"] / by_on[key] - 1.0)})
    return sorted(rows, key=lambda x: (x["m"], x["k"], x["n"]))


def parse_tbo(text: str) -> dict:
    """e4_pdl.parse_tbo plus the number of FAIL lines (ANSI-colored in the console output)."""
    out = e4_pdl.parse_tbo(text)
    out["n_fail"] = sum(1 for ln in text.splitlines() if TBO_FAIL_RE.search(ANSI_RE.sub("", ln)))
    return out


def mc_kernels(kerns: list[dict]) -> dict:
    """From nsys kernel rows ({name, time_ns, count}): the PQ2_0 multi-column launches and the top kernels by time."""
    mc = [k for k in kerns if MC_KERNEL_RE.search(k["name"])]
    gen = [k for k in kerns if GEN_KERNEL_RE.search(k["name"])]
    return {"mc_kernel_instances": sum(k["count"] for k in mc),
            "mc_widths": sorted({int(MC_KERNEL_RE.search(k["name"]).group(1)) for k in mc}),
            "generic_pq2_instances": sum(k["count"] for k in gen),
            "generic_pq2_widths": sorted({int(GEN_KERNEL_RE.search(k["name"]).group(1)) for k in gen}),
            "top": [{"name": k["name"][:140], "count": k["count"], "time_ns": k["time_ns"]}
                    for k in sorted(kerns, key=lambda x: -x["time_ns"])[:6]]}


def summarize_bench(rows: list[dict], cfg_names: list[str]) -> dict:
    """rows: {config, round, kind (pp|tg), width, tok_s}. Per width: median ms per batch of each config over the rounds in which every
    config has the width, and gain_pct_<cfg> = ms_off / ms_<cfg> - 1 in percent. tg: median decode tok/s per config."""
    cells: dict = {}
    for r in rows:
        if r.get("tok_s"):
            cells.setdefault((r["kind"], r["width"]), {}).setdefault(r["config"], {}).setdefault(r["round"], []).append(r["tok_s"])
    out: dict = {"pp": {}, "tg": {}}
    for (kind, width), by_c in sorted(cells.items()):
        rounds = sorted(set.intersection(*[set(by_c.get(c, {})) for c in cfg_names])) if cfg_names else []
        if not rounds:
            continue
        tok = {c: common.median([x for rnd in rounds for x in by_c[c][rnd]]) for c in cfg_names}
        if kind == "tg":
            out["tg"][str(width)] = {"tok_s": tok, "rounds": rounds}
            continue
        ms = {c: 1e3 * width / tok[c] for c in cfg_names if tok[c]}
        ent = {"ms": ms, "tok_s": tok, "rounds": rounds, "uses_kernel": width in MC_WIDTHS}
        for c in cfg_names:
            if c != "off" and ms.get("off") and ms.get(c):
                ent[f"gain_pct_{c}"] = 100.0 * (ms["off"] / ms[c] - 1.0)
        out["pp"][str(width)] = ent
    base = out["tg"].get("1") or next(iter(out["tg"].values()), None)
    if base and base["tok_s"].get("off"):
        out["tg_gain_pct"] = {c: 100.0 * (base["tok_s"][c] / base["tok_s"]["off"] - 1.0) for c in cfg_names if base["tok_s"].get(c)}
    return out


def verdict_lines(summ: dict) -> list[str]:
    lines = []
    for w, e in summ.get("pp", {}).items():
        if "gain_pct_on" in e:
            lines.append("width %s: off %.2f ms, on %.2f ms, gain %+.1f%%%s" % (
                w, e["ms"]["off"], e["ms"]["on"], e["gain_pct_on"], "" if e["uses_kernel"] else " (not a multi-column width)"))
    return lines


# ---------------------------------------------------------------- running

def gpu_mem() -> str | None:
    """nvidia-smi memory used / total and processes, recorded before part t (a full GPU fails every test at allocation)."""
    try:
        p = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,name,driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as e:
        return f"nvidia-smi failed: {e}"
    return p.stdout.strip() or p.stderr.strip()


def run_tbo(cmd: list[str], env: dict, timeout: float) -> dict:
    row = {"cmd": cmd}
    try:
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=max(1.0, timeout))
    except subprocess.TimeoutExpired:
        row.update(ok=False, error="timeout")
        return row
    row.update(parse_tbo(p.stdout), returncode=p.returncode, ok=p.returncode == 0)
    if not row["ok"]:     # keep what failed: the console has no FAIL line for an allocation or setup error
        text = ANSI_RE.sub("", (p.stdout or "") + "\n" + (p.stderr or ""))
        bad = [ln.strip()[:300] for ln in text.splitlines() if ln.strip() and not ln.rstrip().endswith(("OK", "not supported [CPU]"))]
        row["problem_lines"] = r1.scrub(bad[:40], r1.secret_values())
        row["out_tail"] = r1.scrub(text[-3000:], r1.secret_values())
    return row


def tbo_part(bin_dir, budget: common.Budget, notes: list) -> list[dict]:
    out = []
    if not (bin_dir / "test-backend-ops").exists():
        notes.append(f"{bin_dir / 'test-backend-ops'} not found; part t and f skipped")
        return out
    for name, cenv in TBO_ORDER:
        if budget.expired(60):
            notes.append(f"t: time budget used up, config {name} skipped")
            continue
        row = run_tbo(tbo_cmd(bin_dir), run_env(cenv), budget.timeout(600, reserve_s=30))
        if not row.get("total") and row.get("ok"):          # the filter matched nothing: run MUL_MAT whole
            notes.append(f"t {name}: no test matched {PQ2_FILTER!r}; MUL_MAT run whole")
            row = dict(run_tbo(tbo_cmd(bin_dir, filt=None), run_env(cenv), budget.timeout(900, reserve_s=30)), filter=None)
        row.update(config=name, env=cenv)
        out.append(row)
        print("[r2b_mc] test-backend-ops %s ok=%s %s/%s" % (name, row.get("ok"), row.get("passed"), row.get("total")), flush=True)
    return out


def path_probe(nsys: str | None, bin_dir, model: str, budget: common.Budget, notes: list) -> dict:
    """Kernel names of llama-bench -p 4 -n 0 under nsys, per config (off, on). Never raises; the nsys files are deleted."""
    out: dict = {"cmd": bench_pp_cmd(bin_dir, "<model.gguf>", [4], 1), "configs": {}}
    if not nsys:
        out["error"] = "nsys not found (CR_NSYS, PATH, /usr/local/cuda*/bin, /opt/nvidia/nsight-systems)"
        notes.append("k: " + out["error"])
        return out
    for name, cenv in CONFIGS[:2]:
        if budget.expired(45):
            notes.append(f"k: time budget used up, config {name} skipped")
            continue
        wd = r1_step.private_workdir()
        ent: dict = {"env": cenv}
        try:
            env = {k: v for k, v in r1_step.nsys_env(wd).items() if k not in (ENV_VAR, "GGML_CUDA_PDL")}
            env.update(cenv)
            prefix = os.path.join(wd, "mc")
            cmd = [nsys, "profile", "-t", "cuda", "--sample=none", "--cpuctxsw=none", "--cuda-graph-trace=node", "-f", "true", "-o", prefix] + bench_pp_cmd(bin_dir, model, [4], 1)
            code, text = r1_step._run(cmd, env, budget.timeout(240, reserve_s=30))
            if code != 0 or not os.path.exists(prefix + ".nsys-rep"):
                ent["error"] = f"nsys profile exit {code}: {text[-300:].strip()}"
            else:
                code, text = r1_step._run(e4_bonsai.nsys_stats_cmd(nsys, prefix + ".nsys-rep"), env, budget.timeout(180, reserve_s=20))
                kerns = e4_bonsai.parse_kern_sum(text)
                if kerns:
                    ent.update(mc_kernels(kerns))
                else:
                    ent["error"] = f"nsys stats gave no kernels (exit {code}): {text[-300:].strip()}"
        finally:
            ent["files_deleted"] = sorted(os.listdir(wd)) if os.path.isdir(wd) else []
            shutil.rmtree(wd, ignore_errors=True)
        out["configs"][name] = ent
    on, off = out["configs"].get("on", {}), out["configs"].get("off", {})
    if "mc_kernel_instances" in on and "mc_kernel_instances" in off:
        out["on_uses_mc_kernel"] = on["mc_kernel_instances"] > 0
        out["off_uses_mc_kernel"] = off["mc_kernel_instances"] > 0
        out["off_uses_generic_w4"] = 4 in off.get("generic_pq2_widths", [])
        out["on_uses_generic_w4"] = 4 in on.get("generic_pq2_widths", [])
        if not out["on_uses_mc_kernel"] or out["off_uses_mc_kernel"] or not out["off_uses_generic_w4"] or out["on_uses_generic_w4"]:
            notes.append("k: the multi-column kernel is not on exactly in the 'on' config (off: generic mul_mat_vec_q at 4 columns, on: mul_mat_vec_pq2_0_mc<4>): read the timings of f and b with care")
    return out


def perf_part(bin_dir, budget: common.Budget, notes: list) -> dict:
    out: dict = {"cmd": tbo_cmd(bin_dir, "perf"), "configs": {}, "gain": []}
    if not (bin_dir / "test-backend-ops").exists():
        return out
    for name, cenv in CONFIGS[:2]:
        if budget.expired(60):
            notes.append(f"f: time budget used up, config {name} skipped")
            continue
        try:
            p = subprocess.run(out["cmd"], env=run_env(cenv), capture_output=True, text=True, timeout=budget.timeout(600, reserve_s=30))
        except subprocess.TimeoutExpired:
            out["configs"][name] = {"error": "timeout", "rows": []}
            continue
        rows = parse_perf(p.stdout)
        out["configs"][name] = {"returncode": p.returncode, "rows": rows}
        if not rows:
            notes.append(f"f {name}: no perf line parsed (exit {p.returncode}): {(p.stdout or p.stderr)[-200:].strip()}")
    if out["configs"].get("off", {}).get("rows") and out["configs"].get("on", {}).get("rows"):
        out["gain"] = perf_gain(out["configs"]["off"]["rows"], out["configs"]["on"]["rows"])
    return out


def bench_entries(cmd: list[str], env: dict, timeout: float) -> tuple[list[dict], str | None]:
    """All llama-bench JSON entries of one run in env, or ([], error)."""
    try:
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=max(1.0, timeout))
    except subprocess.TimeoutExpired:
        return [], "timeout"
    if p.returncode != 0:
        return [], f"exit {p.returncode}: {(p.stderr or '')[-300:].strip()}"
    got = e4_bonsai.parse_bench_json(p.stdout)
    return got, None if got else "no entries in llama-bench output"


def bench_part(bin_dir, model: str, budget: common.Budget, notes: list, rows: list, allow_power_cap: bool, save) -> None:
    unhealthy = False
    for rnd in range(ROUNDS):
        order = CONFIGS if rnd % 2 == 0 else list(reversed(CONFIGS))
        for name, cenv in order:
            for kind, cmd in (("pp", bench_pp_cmd(bin_dir, model)), ("tg", bench_tg_cmd(bin_dir, model))):
                if budget.expired(60):
                    notes.append(f"b: time budget used up, round {rnd} {name} {kind} and later skipped")
                    return
                with e4_pdl.SmiWatch() as watch:
                    got, err = bench_entries(cmd, run_env(cenv), budget.timeout(900, reserve_s=30))
                smi = watch.summary()
                bad = e4_pdl.health_problem(smi, check_power=not allow_power_cap)
                if bad and not unhealthy:
                    unhealthy = True
                    notes.append(f"host health: {bad} during round {rnd} {name} {kind}; timings after this may be invalid")
                if err:
                    notes.append(f"round {rnd} {name} {kind} bench failed: {err}")
                for r in e5_cost.rows_from(got, "pq2", 0):
                    rows.append({"config": name, "env": cenv, "round": rnd, "kind": "pp" if r["kind"] == "verify" else "tg",
                                 "width": r["width"], "tok_s": r["tok_s"], "stddev": r["stddev"], "ms_per_batch": r["ms_per_batch"],
                                 "samples_ts": r["samples_ts"], "smi": smi})
                save()
                print("[r2b_mc] r%d %s %s sm %s MHz" % (rnd, name, kind, smi.get("sm_mhz_med")), flush=True)


def part_budget_s(left_s: float, part: str, remaining: list[str]) -> float:
    tot = sum(WEIGHTS[p] for p in remaining) or 1.0
    return max(left_s - END_RESERVE_S, 0.0) * WEIGHTS[part] / tot


def run(args) -> int:
    notes: list[str] = []
    args.gpu = common.gpu_tag(args)
    bin_dir = e4_bonsai.llama_bin()
    if not (bin_dir / "llama-bench").exists():
        print(f"error: {bin_dir / 'llama-bench'} not found; build llama.cpp (CR_BUILD_LLAMA=1) or set CR_LLAMA_BIN", file=sys.stderr)
        return 2
    total = common.Budget(args.budget_min)
    config = {"parts": args.parts, "budget_min": args.budget_min, "configs": dict(CONFIGS), "widths": WIDTHS, "reps": REPS,
              "rounds": ROUNDS, "tbo_cmd": tbo_cmd(bin_dir), "perf_cmd": tbo_cmd(bin_dir, "perf"),
              "bench_pp_cmd": bench_pp_cmd(bin_dir, "<model.gguf>"), "bench_tg_cmd": bench_tg_cmd(bin_dir, "<model.gguf>"),
              "pq2_filter": PQ2_FILTER, "models_info": {}, "quick": args.quick}
    extra: dict = {"tbo": [], "path": None, "perf": None, "bench": [], "summary": {}}
    rows: list[dict] = extra["bench"]

    def save(final: bool = False) -> None:
        extra["summary"] = summarize_bench(rows, [c for c, _ in CONFIGS])
        (common.write_result if final else common.write_partial)(EXP, args, config, rows, notes, False, extra=extra)

    try:
        model, config["models_info"]["pq2"] = e4_pdl.model_path("pq2", notes)
    except Exception as e:
        notes.append(f"model download failed: {type(e).__name__}: {e}")
        save(final=True)
        print(f"error: {notes[-1]}", file=sys.stderr)
        return 3
    cut = {"part": None}
    parts = [p for p in PARTS if p in args.parts]
    rc = 0
    try:
        for i, p in enumerate(parts):
            sub = common.Budget(part_budget_s(total.left(), p, parts[i:]) / 60.0)
            if sub.left() < 60:
                notes.append(f"part {p} skipped: only {sub.left():.0f} s of budget left")
                continue
            cut["part"] = p
            print(f"[r2b_mc] part {p}: {sub.left() / 60:.1f} min of {total.left() / 60:.1f} min left", flush=True)
            try:
                if p == "t":
                    config["gpu_mem_before_t"] = gpu_mem()
                    extra["tbo"] = tbo_part(bin_dir, sub, notes)
                    if any(not t.get("ok") for t in extra["tbo"]):
                        rc = 1
                        notes.append("test-backend-ops reported failures: see data.tbo[].fail_lines before reading any timing")
                    bad_on = [t["config"] for t in extra["tbo"] if t.get("env") and (not t.get("ok") or t.get("n_fail"))]
                    if bad_on or not any(t.get("env") and t.get("ok") for t in extra["tbo"]):
                        notes.append("stop: test-backend-ops did not pass with %s set (%s); parts k, b, f not run" % (ENV_VAR, ", ".join(bad_on) or "no passing run"))
                        cut["part"] = None
                        save()
                        break
                elif p == "k":
                    extra["path"] = path_probe(e4_bonsai.find_nsys(), bin_dir, model, sub, notes)
                elif p == "f":
                    extra["perf"] = perf_part(bin_dir, sub, notes)
                else:
                    bench_part(bin_dir, model, sub, notes, rows, args.allow_power_cap, save)
            except Exception as e:  # a failed part must not lose the others
                notes.append(f"part {p} crashed: {type(e).__name__}: {e}")
            cut["part"] = None
            save()
    finally:
        if cut["part"]:
            notes.append(f"run cut during part {cut['part']}: the results of that part are partial")
        extra["summary"] = summarize_bench(rows, [c for c, _ in CONFIGS])
        notes.extend(verdict_lines(extra["summary"]))
        notes.append("widths 1 and 2 do not use the multi-column kernel: their gain is the noise floor of the comparison")
        notes.append(PDL_LC_NOTE)
        save(final=True)
    return rc


# ---------------------------------------------------------------- CLI

def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="exps.r2b_multicol", description=__doc__.split("\n")[0])
    common.add_common_args(ap)
    ap.add_argument("--parts", default=PARTS, help="t (test-backend-ops), k (kernel path), b (llama-bench), f (op timing)")
    ap.add_argument("--allow-power-cap", action="store_true", help="a power cap below default is a note, not a health problem")
    a = ap.parse_args(argv)
    bad = [p for p in a.parts if p not in PARTS]
    if bad:
        ap.error(f"unknown part(s) {bad}; parts: {PARTS}")
    a.budget_min = a.budget_min or (6.0 if a.quick else 22.0)
    return a


def plan_json(a) -> dict:
    bin_dir = e4_bonsai.llama_bin()
    return {"exp": EXP, "parts": a.parts, "budget_min": a.budget_min, "weights": WEIGHTS, "configs": dict(CONFIGS),
            "model": hf_ids.resolve("bonsai2", "PQ2_0", offline=True), "tbo_cmd": tbo_cmd(bin_dir), "perf_cmd": tbo_cmd(bin_dir, "perf"),
            "bench_pp_cmd": bench_pp_cmd(bin_dir, "<model.gguf>"), "bench_tg_cmd": bench_tg_cmd(bin_dir, "<model.gguf>"),
            "rounds": ROUNDS, "order": [[c for c, _ in (CONFIGS if r % 2 == 0 else list(reversed(CONFIGS)))] for r in range(ROUNDS)],
            "path_probe": "nsys profile -t cuda (private dir, deleted) of llama-bench -p 4 -n 0 -r 1, off and on", "quick": a.quick}


def main(argv=None) -> int:
    a = parse_args(argv)
    if a.dry_run:
        common.print_plan(plan_json(a))
        return 0
    with common.term_exits():
        return run(a)


if __name__ == "__main__":
    sys.exit(main())
