"""E4: Bonsai 2 27B (ternary, PrismML fork) decode sweep with llama-bench plus an ncu kernel split.

  python -m exps.e4_bonsai --quant PQ2_0|PTQ1_0 [--depths 0,32768,131072,262144] [--kv f16,q8_0]
                           [--ncu-depths 0,32768] [--ncu-kv f16,q8_0]

Throughput: llama-bench tg128 at each depth (-d) and KV type (-ctk/-ctv). Kernel split: ncu on one decode
window (4 tokens) of exps/csrc/cr_decode_prof.cpp, a small libllama program that fills the context to the
depth and calls cudaProfilerStart/Stop around the decode, so no launch counting is needed at depth > 0.
If the tool cannot be built, depth 0 falls back to ncu on llama-bench (-p 0 -n 4 -r 1 --no-warmup) and
deeper depths get no kernel split. llama-bench never calls cudaProfilerStart, so that fallback profiles from
the start of the process (no --profile-from-start off). PQ2_0/PTQ1_0 unpack is fused inside mul_mat_vec.
When ncu has no counters (RunPod), the split comes from nsys instead: CUDA trace of cr_decode_prof with
--capture-range=cudaProfilerApi and --cuda-graph-trace=node, kernel times summed per bucket (no DRAM bytes).
All llama-bench rows run first, then the ncu/nsys passes use the leftover time budget. Rows are written to
raw/partial.json as they complete, and every subprocess timeout is bounded by the time budget.
Needs CR_LLAMA_BIN (default <repo>/build-cuda/bin) with llama-bench built (CR_BUILD_LLAMA=1).
"""
from __future__ import annotations

import argparse
import csv
import glob
import io
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

from exps import common, hf_ids

BUCKETS = {"hadamard": "fwht|hadamard", "act_quant": "quantize_q8_1|quantize_mmq", "mul_mat_vec": "mul_mat_vec",
           "flash_attn": "flash_attn", "gated_delta_net": "gated_delta|gdn"}
EXP_NAMES = {"PQ2_0": "e4_pq2", "PTQ1_0": "e4_ptq1"}  # result dir must equal the manifest key
DEFAULT_DEPTHS = [0, 32768, 131072, 262144]
UB = 2048
N_GEN = 128
NCU_TOKENS = 4
NSYS_TOKENS = 16
NSYS_GLOBS = ("/usr/local/cuda*/bin/nsys", "/opt/nvidia/nsight-systems/*/bin/nsys", "/opt/nvidia/nsight-systems/*/target-linux-x64/nsys")
REPO_ROOT = common.EXP_DIR.parent.parent
PROF_SRC = common.EXP_DIR / "exps" / "csrc" / "cr_decode_prof.cpp"


# ---------------------------------------------------------------- pure helpers

def llama_bin() -> pathlib.Path:
    return pathlib.Path(os.environ.get("CR_LLAMA_BIN") or (REPO_ROOT / "build-cuda" / "bin"))


def reps_for(depth: int) -> int:
    return 3 if depth <= 32768 else 1


def bench_cmd(bin_dir, model: str, depth: int, kv: str, reps: int | None = None, n_gen: int = N_GEN) -> list[str]:
    """llama-bench tg run at a depth. Flags as in tools/llama-bench."""
    return [str(pathlib.Path(bin_dir) / "llama-bench"), "-m", model, "-ngl", "99", "-fa", "on", "-p", "0",
            "-n", str(n_gen), "-d", str(depth), "-ctk", kv, "-ctv", kv, "-ub", str(UB),
            "-r", str(reps if reps is not None else reps_for(depth)), "-o", "json"]


def bench_ncu_cmd(bin_dir, model: str, kv: str) -> list[str]:
    """Fallback depth-0 ncu target: 4 tokens, no warmup, one repetition."""
    return [str(pathlib.Path(bin_dir) / "llama-bench"), "-m", model, "-ngl", "99", "-fa", "on", "-p", "0",
            "-n", str(NCU_TOKENS), "-d", "0", "-ctk", kv, "-ctv", kv, "-ub", str(UB), "-r", "1", "--no-warmup", "-o", "json"]


def prof_cmd(tool, model: str, depth: int, kv: str, n: int = NCU_TOKENS) -> list[str]:
    return [str(tool), "-m", model, "-ngl", "99", "-fa", "on", "-d", str(depth), "-n", str(n), "-ctk", kv, "-ctv", kv,
            "-ub", str(UB)]


def parse_bench_json(text: str) -> list[dict]:
    """Parse llama-bench -o json output (a JSON array; log lines before/after are ignored)."""
    a, b = text.find("["), text.rfind("]")
    if a < 0 or b < a:
        return []
    try:
        data = json.loads(text[a:b + 1])
    except ValueError:
        return []
    return [d for d in data if isinstance(d, dict)]


def tg_result(entries: list[dict]) -> dict | None:
    """First generation entry: {tok_s, stddev, n_gen, n_depth}."""
    for e in entries:
        if int(e.get("n_gen", 0) or 0) > 0:
            return {"tok_s": float(e["avg_ts"]), "stddev": float(e.get("stddev_ts", 0.0) or 0.0),
                    "n_gen": int(e["n_gen"]), "n_depth": int(e.get("n_depth", 0) or 0),
                    "model_type": e.get("model_type"), "type_k": e.get("type_k"), "type_v": e.get("type_v")}
    return None


def bucket_summary(rows: list[dict], tokens: int) -> dict:
    """Per-token time/bytes per kernel bucket plus time shares."""
    from harness import counters
    b = counters.bucket(rows, BUCKETS)
    tot_ns = sum(v["time_ns"] for v in b.values()) or 1.0
    out = {}
    for name, v in b.items():
        out[name] = {"us_per_token": v["time_ns"] / 1000.0 / tokens, "dram_bytes_per_token": v["dram_bytes"] / tokens,
                     "hit_rate": v["hit_rate"], "count": v["count"], "time_share": v["time_ns"] / tot_ns}
    return out


def bw_bound_ratio(dram_bytes_per_token, us_per_token, peak_gbps) -> float | None:
    if not dram_bytes_per_token or not us_per_token or not peak_gbps:
        return None
    return dram_bytes_per_token / (us_per_token * 1e3) / peak_gbps


def ncu_target(tool, bin_dir, model: str, depth: int, kv: str):
    """(argv, from_start_off) for the ncu pass, or None. Only cr_decode_prof calls cudaProfilerStart/Stop."""
    if tool is not None:
        return prof_cmd(tool, model, depth, kv), True
    if depth == 0:
        return bench_ncu_cmd(bin_dir, model, kv), False
    return None


def make_row(quant: str, depth: int, kv: str, tg: dict | None, ncu: dict | None, peak: float | None) -> dict:
    us = 1e6 / tg["tok_s"] if tg and tg["tok_s"] else None
    dram = ncu["summary"]["dram_bytes_per_token"] if ncu and ncu.get("summary") else None
    return {"quant": quant, "depth": depth, "kv": kv, "tg_tok_s": tg["tok_s"] if tg else None,
            "tg_stddev": tg["stddev"] if tg else None, "us_per_token": us,
            "buckets": ncu.get("buckets") if ncu else None, "dram_bytes_per_token": dram,
            "ncu_us_per_token": ncu["summary"]["us_per_token"] if ncu and ncu.get("summary") else None,
            "l2_hit_rate": ncu["summary"]["hit_rate"] if ncu and ncu.get("summary") else None,
            "ncu_replay_mode": ncu.get("replay_mode") if ncu else None, "ncu_passes": ncu.get("passes") if ncu else None,
            "ncu_l2_state_ok": ncu.get("l2_state_ok") if ncu else None,
            "bw_bound_ratio": bw_bound_ratio(dram, us, peak)}


def find_nsys() -> str | None:
    """CR_NSYS, else nsys on PATH, else a CUDA / Nsight Systems install dir."""
    env = os.environ.get("CR_NSYS")
    if env:
        return env if os.path.exists(env) else None
    hit = shutil.which("nsys")
    if hit:
        return hit
    for g in NSYS_GLOBS:
        found = sorted(glob.glob(g))
        if found:
            return found[-1]
    return None


def nsys_profile_cmd(nsys: str, argv: list[str], out_prefix: str) -> list[str]:
    return [nsys, "profile", "-t", "cuda", "--sample=none", "--cpuctxsw=none", "--cuda-graph-trace=node",
            "--capture-range=cudaProfilerApi", "--capture-range-end=stop", "-f", "true", "-o", out_prefix, *argv]


def nsys_stats_cmd(nsys: str, rep_path: str) -> list[str]:
    return [nsys, "stats", "-r", "cuda_gpu_kern_sum", "-f", "csv", "--force-export=true", rep_path]


def parse_kern_sum(text: str) -> list[dict]:
    """cuda_gpu_kern_sum CSV (header row has 'Total Time (ns)' and 'Name'; other lines are ignored) -> [{name, time_ns, count}]."""
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if "Total Time (ns)" in ln and "Name" in ln), None)
    if start is None:
        return []
    out = []
    for r in csv.DictReader(io.StringIO("\n".join(lines[start:]))):
        try:
            out.append({"name": r["Name"], "time_ns": float(r["Total Time (ns)"]), "count": int(float(r["Instances"]))})
        except (KeyError, TypeError, ValueError):
            continue
    return out


def nsys_buckets(kerns: list[dict], tokens: int) -> dict:
    """Same buckets as ncu (first match wins, rest in other), time only."""
    pats = [(n, re.compile(rx, re.IGNORECASE)) for n, rx in BUCKETS.items()]
    acc = {n: [0.0, 0] for n, _ in pats}
    acc["other"] = [0.0, 0]
    for k in kerns:
        name = next((n for n, p in pats if p.search(k["name"])), "other")
        acc[name][0] += k["time_ns"]
        acc[name][1] += k["count"]
    tot = sum(v[0] for v in acc.values()) or 1.0
    return {n: {"us_per_token": v[0] / 1000.0 / tokens, "count": v[1], "time_share": v[0] / tot} for n, v in acc.items()}


def verdict(rows: list[dict]) -> str | None:
    """One line on bandwidth-bound vs unpack-bound from the depth-0 f16 row."""
    r = next((x for x in rows if x["depth"] == 0 and (x.get("buckets") or x.get("nsys_buckets"))), None)
    if not r:
        return None
    b = r.get("buckets") or r["nsys_buckets"]
    unpack = b.get("hadamard", {}).get("time_share", 0) + b.get("act_quant", {}).get("time_share", 0)
    mmv = b.get("mul_mat_vec", {}).get("time_share", 0)
    ratio = r.get("bw_bound_ratio")
    return ("depth 0 kv=%s: mul_mat_vec %.0f%% of kernel time, hadamard+act_quant %.0f%%, bw_bound_ratio %s"
            % (r["kv"], 100 * mmv, 100 * unpack, "n/a" if ratio is None else "%.2f" % ratio))


# ---------------------------------------------------------------- running

def build_prof_tool(bin_dir: pathlib.Path, notes: list) -> pathlib.Path | None:
    """Compile cr_decode_prof against libllama in bin_dir. Returns the binary or None."""
    out = bin_dir / "cr_decode_prof"
    if out.exists() and out.stat().st_mtime >= PROF_SRC.stat().st_mtime:
        return out
    cuda = os.environ.get("CUDA_HOME") or "/usr/local/cuda"
    cmd = ["g++", "-O2", "-std=c++17", "-I", str(REPO_ROOT / "include"), "-I", str(REPO_ROOT / "ggml" / "include"),
           str(PROF_SRC), "-o", str(out), f"-L{bin_dir}", "-lllama", "-lggml", "-lggml-base", f"-Wl,-rpath,{bin_dir}",
           f"-L{cuda}/lib64", f"-Wl,-rpath,{cuda}/lib64", "-lcudart"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except Exception as e:
        notes.append(f"cr_decode_prof build failed to run: {e}")
        return None
    if r.returncode != 0:
        notes.append("cr_decode_prof build failed: " + (r.stderr or "")[-400:])
        return None
    return out


def run_bench(cmd: list[str], timeout: float) -> tuple[dict | None, str | None]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=max(1.0, timeout))
    except subprocess.TimeoutExpired:
        return None, "timeout"
    if p.returncode != 0:
        return None, f"exit {p.returncode}: {(p.stderr or '')[-300:].strip()}"
    tg = tg_result(parse_bench_json(p.stdout))
    return tg, None if tg else "no generation entry in llama-bench output"


def ncu_split(argv, tokens: int, timeout: float, csv_out: str | None, from_start_off: bool = True) -> dict:
    res = common.ncu_run(argv, tokens, profile_from_start_off=from_start_off, app_replay=False, timeout=max(1.0, timeout),
                         csv_out=csv_out)
    info = common.replay_info(res)
    out = {"ok": res["ok"], "error": res["error"], "summary": res.get("summary"), "replay_mode": info["ncu_replay_mode"],
           "passes": info["ncu_passes"], "l2_state_ok": info["ncu_l2_state_ok"]}
    if res.get("rows"):
        out["buckets"] = bucket_summary(res["rows"], tokens)
    return out


def nsys_split(nsys: str, argv: list[str], tokens: int, timeout: float, out_prefix: str) -> dict:
    """Kernel time per bucket from an nsys CUDA trace of the profiler range. Never raises."""
    out = {"ok": False, "error": None, "kernels_top": None, "buckets": None, "kernel_us_per_token": None}
    try:
        p = subprocess.run(nsys_profile_cmd(nsys, argv, out_prefix), capture_output=True, text=True, timeout=max(1.0, timeout))
        rep_path = out_prefix + ".nsys-rep"
        if p.returncode != 0 or not os.path.exists(rep_path):
            out["error"] = f"nsys profile exit {p.returncode}: {((p.stderr or '') + (p.stdout or ''))[-300:].strip()}"
            return out
        st = subprocess.run(nsys_stats_cmd(nsys, rep_path), capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        out["error"] = "timeout"
        return out
    except OSError as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out
    kerns = parse_kern_sum(st.stdout)
    if not kerns:
        out["error"] = f"nsys stats gave no kernels (exit {st.returncode}): {(st.stderr or '')[-300:].strip()}"
        return out
    kerns.sort(key=lambda k: -k["time_ns"])
    out.update(ok=True, buckets=nsys_buckets(kerns, tokens), kernel_us_per_token=sum(k["time_ns"] for k in kerns) / 1000.0 / tokens,
               kernels_top=[dict(k, name=k["name"][:160], us_per_token=k["time_ns"] / 1000.0 / tokens) for k in kerns[:15]])
    return out


def raw_path(exp: str, name: str, gpu: str) -> str:
    from infra.results import artifact_path
    return str(artifact_path(exp, name, gpu))


def download(quant: str, notes: list) -> tuple[str, dict]:
    info = hf_ids.resolve("bonsai2", quant)
    if info["exists"] is False:
        notes.append(f"{info['repo']}/{info['file']} not found; alternatives {info['alternatives']}")
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(info["repo"], info["file"])
    return path, info


def run(args) -> int:
    notes: list[str] = []
    args.gpu = common.gpu_tag(args)
    bin_dir = llama_bin()
    bench = bin_dir / "llama-bench"
    if not bench.exists():
        print(f"error: {bench} not found; build llama.cpp (CR_BUILD_LLAMA=1) or set CR_LLAMA_BIN", file=sys.stderr)
        return 2
    from infra.gpus import peak_dram_gbps
    peak = peak_dram_gbps(args.gpu)
    ncu_ok, ncu_note = common.counters_available(args)
    if ncu_note:
        notes.append(ncu_note)
    budget = common.Budget(args.budget_min or (8 if args.quick else 31))
    exp = EXP_NAMES[args.quant]
    try:
        model, hf_info = download(args.quant, notes)
    except Exception as e:
        notes.append(f"model download failed: {type(e).__name__}: {e}")
        common.write_result(exp, args, {"quant": args.quant, "hf": hf_ids.resolve("bonsai2", args.quant)}, [], notes, False)
        print(f"error: {notes[-1]}", file=sys.stderr)
        return 3
    nsys = None if ncu_ok or args.no_nsys else find_nsys()
    if not ncu_ok and not args.no_nsys:
        notes.append(f"kernel split via nsys {nsys}" if nsys else "no ncu counters and no nsys: no kernel split")
    tool = build_prof_tool(bin_dir, notes) if (ncu_ok or nsys) else None
    if nsys and tool is None:
        notes.append("nsys found but cr_decode_prof did not build: no kernel split")
        nsys = None
    config = {"quant": args.quant, "depths": args.depths, "kv": args.kv, "ncu_depths": args.ncu_depths,
              "ncu_kv": args.ncu_kv, "nsys": nsys, "nsys_tokens": NSYS_TOKENS, "ub": UB, "n_gen": N_GEN, "bench_cmd_example": bench_cmd(bin_dir, model, 0, "f16"),
              "model_path": model, "hf": hf_info, "buckets": BUCKETS, "peak_gbps": peak, "prof_tool": str(tool) if tool else None,
              "quick": args.quick}
    rows: list[dict] = []
    benched: list[tuple] = []  # (row index, tg) so the ncu pass can rebuild the row
    counters_all = ncu_ok
    for depth in sorted(args.depths):  # all throughput rows first: the deep rows must not wait behind ncu passes
        for kv in args.kv:
            if budget.expired(60):
                notes.append(f"time budget used up; depth {depth} kv {kv} and later skipped")
                continue
            tg, err = run_bench(bench_cmd(bin_dir, model, depth, kv, 1 if args.quick else None,
                                          32 if args.quick else N_GEN), budget.timeout(86400, reserve_s=30))
            if err:
                notes.append(f"depth {depth} kv {kv} bench failed: {err}")
            row = make_row(args.quant, depth, kv, tg, None, peak)
            if err:
                row["error"] = err
            benched.append((len(rows), tg))
            rows.append(row)
            common.write_partial(exp, args, config, rows, notes, False)
            print("[e4] %s depth=%d kv=%s tg=%s tok/s" % (args.quant, depth, kv, row["tg_tok_s"]), flush=True)
    for i, tg in benched:  # then spend the leftover budget on kernel splits
        depth, kv = rows[i]["depth"], rows[i]["kv"]
        if nsys and depth in args.ncu_depths and kv in args.ncu_kv:
            if budget.expired(180):
                rows[i]["nsys_error"] = "skipped: time budget"
                continue
            res = nsys_split(nsys, prof_cmd(tool, model, depth, kv, NSYS_TOKENS), NSYS_TOKENS, budget.timeout(1200, reserve_s=60),
                             raw_path(exp, f"nsys-d{depth}-{kv}", args.gpu))
            rows[i].update(nsys_buckets=res["buckets"], nsys_kernel_us_per_token=res["kernel_us_per_token"],
                           nsys_kernels_top=res["kernels_top"], nsys_error=res["error"])
            if not res["ok"]:
                notes.append(f"depth {depth} kv {kv} nsys failed: {res['error']}")
            common.write_partial(exp, args, config, rows, notes, False)
            print("[e4] nsys %s depth=%d kv=%s kernel us/tok=%s" % (args.quant, depth, kv, res["kernel_us_per_token"]), flush=True)
            continue
        if not (ncu_ok and depth in args.ncu_depths and kv in args.ncu_kv):
            continue
        if budget.expired(240):
            rows[i]["ncu_error"] = "skipped: time budget"
            counters_all = False
            continue
        target = ncu_target(tool, bin_dir, model, depth, kv)
        if target is None:
            notes.append(f"depth {depth} kv {kv}: no kernel split (cr_decode_prof unavailable)")
            continue
        argv, from_start_off = target
        ncu = ncu_split(argv, NCU_TOKENS, budget.timeout(1800, reserve_s=60),
                        raw_path(exp, f"ncu-d{depth}-{kv}.txt", args.gpu), from_start_off)
        counters_all = counters_all and ncu["ok"]
        if not ncu["ok"]:
            notes.append(f"depth {depth} kv {kv} ncu failed: {ncu['error']}")
        err = rows[i].get("error")
        rows[i] = make_row(args.quant, depth, kv, tg, ncu, peak)
        if err:
            rows[i]["error"] = err
        common.write_partial(exp, args, config, rows, notes, False)
        print("[e4] ncu %s depth=%d kv=%s dram/tok=%s" % (args.quant, depth, kv, rows[i]["dram_bytes_per_token"]), flush=True)
    v = verdict(rows)
    if v:
        notes.append(v)
    notes.append("unpack (Hadamard + ternary decode) is fused in mul_mat_vec for PQ2_0/PTQ1_0; ncu is kernel-replay, "
                 "dram_bytes counts reads only; rows with ncu_l2_state_ok false ran more than one pass on a warmed L2 "
                 "(--cache-control none), so dram bytes may be under-reported")
    common.write_result(exp, args, config, rows, notes, counters_all)
    return 0


# ---------------------------------------------------------------- CLI

def _ints(text):
    return [int(t) for t in text.split(",") if t.strip() != ""]


def _strs(text):
    return [t.strip() for t in text.split(",") if t.strip()]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="exps.e4_bonsai", description=__doc__.split("\n")[0])
    common.add_common_args(ap)
    ap.add_argument("--quant", choices=sorted(hf_ids.BONSAI2_GGUF[1]), default="PQ2_0")
    ap.add_argument("--depths", type=_ints, default=list(DEFAULT_DEPTHS))
    ap.add_argument("--kv", type=_strs, default=["f16", "q8_0"])
    ap.add_argument("--ncu-depths", type=_ints, default=[0, 32768])
    ap.add_argument("--ncu-kv", type=_strs, default=["f16", "q8_0"])
    ap.add_argument("--no-nsys", action="store_true", help="no nsys kernel split when ncu has no counters")
    a = ap.parse_args(argv)
    if a.quick:
        a.depths, a.ncu_depths, a.kv, a.ncu_kv = [0], [0], ["f16"], ["f16"]
    for k in a.kv + a.ncu_kv:
        if k not in ("f16", "q8_0"):
            ap.error(f"unsupported kv type {k} (cr_decode_prof supports f16, q8_0)")
    return a


def plan_json(a) -> dict:
    from harness import counters
    bin_dir = llama_bin()
    model = "<model.gguf>"
    return {"exp": EXP_NAMES[a.quant], "quant": a.quant, "hf": hf_ids.resolve("bonsai2", a.quant, offline=True),
            "llama_bin": str(bin_dir), "runs": [{"depth": d, "kv": k, "cmd": bench_cmd(bin_dir, model, d, k)}
                                                  for d in sorted(a.depths) for k in a.kv],
            "ncu": {"depths": a.ncu_depths, "kv": a.ncu_kv, "tool_cmd": prof_cmd(bin_dir / "cr_decode_prof", model, 0, "f16"),
                    "fallback_bench_cmd": bench_ncu_cmd(bin_dir, model, "f16"),
                    "ncu_cmd": counters.build_ncu_cmd(prof_cmd(bin_dir / "cr_decode_prof", model, 0, "f16"),
                                                      extra=("--profile-from-start", "off"))},
            "buckets": BUCKETS, "no_ncu": a.no_ncu, "quick": a.quick}


def main(argv=None) -> int:
    a = parse_args(argv)
    if a.dry_run:
        common.print_plan(plan_json(a))
        return 0
    with common.term_exits():
        return run(a)


if __name__ == "__main__":
    sys.exit(main())
