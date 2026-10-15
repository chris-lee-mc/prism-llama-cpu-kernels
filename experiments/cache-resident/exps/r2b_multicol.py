"""R2b multicol: the PQ2_0 multi-column mat-vec (mmvq-pq2_0.cuh) on sm_120, behind GGML_CUDA_PQ2_MULTICOL=1.

  python -m exps.fetch_models --models pq2 --timeout-min 6 && python -m exps.r2b_multicol [--budget-min 22]

ggml_cuda_pq2_multicol() (common.cuh) enables the #306 kernel only on Ada. With GGML_CUDA_PQ2_MULTICOL=1 (read once, host side) it
is also enabled on newer NVIDIA GPUs, for 3-8 columns, which is the verify width of a speculative step with K = 2..7. On sm_120 the
default is the generic mul_mat_vec_q<PQ2_0, ncols>. The kernel is launched through ggml_cuda_kernel_launch (PDL on sm_90+) and the
activation quantizer triggers programmatic launch completion at its start, so the kernel calls ggml_cuda_pdl_sync() before it reads vy.
Configs (environment of each process): off = variable unset (default), on = GGML_CUDA_PQ2_MULTICOL=1, on_pdl0 = on plus GGML_CUDA_PDL=0
(control: PDL off for every kernel, so the sync cannot matter).
t  correctness. test-backend-ops test -o MUL_MAT -b CUDA0 -p type_a=pq2_0, (every PQ2_0 case against the CPU backend; widths 1-32 at
   K 1024 and 5120, the Bonsai 2 shapes with odd row counts, the 27B projections). on, on_pdl0 and off, in that order (a short budget drops off first). Every run streams its output to results/<exp>/<gpu>/raw/<stamp>-tbo-<config>.{out,err}.log, is sampled with nvidia-smi, and on a timeout keeps the partial output (last case, cases done, hang or slow diagnosis) and kills the process group. A filter that matches nothing
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
import glob
import os
import pathlib
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

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
GPU600_TAGS = ("pro6000-ws", "pro6000-server")     # the 600 W editions; a Max-Q (300 W) gives other timings
GPU600_MIN_DEFAULT_W = 550.0
POWER_TOL = 0.01                                    # PR evidence: power.limit must equal power.default_limit within this fraction
TBO_TIMEOUT_S = 600.0                               # default limit of one test-backend-ops run (wrap_a passes 300)
TBO_SMI_FIELDS = ("clocks.sm", "clocks.mem", "power.draw", "utilization.gpu", "utilization.memory", "memory.used")
TBO_SMI_PERIOD_S = 5.0
TBO_DIAG_WINDOW_S = 60.0                            # no progress over this long: GPU idle or pegged means a hang, else slow
IDLE_UTIL, BUSY_UTIL = 5.0, 90.0                    # utilization.gpu in percent
CASE_RE = re.compile(r"^\s*([A-Z][A-Z_0-9]*)\((.*?)\):\s*(.*)$")
RACECHECK_RE = re.compile(r"RACECHECK SUMMARY:\s*(\d+) hazards? displayed \((\d+) errors?, (\d+) warnings?\)")
SAN_RUN_S, CLB_RUN_S, LOCALIZE_MIN_S = 200.0, 300.0, 60.0    # limits of the racecheck run and of the launch-blocking run, time left that a run needs
SAN_FILTER = r"type_a=pq2_0,type_b=f32,m=97,n=(3|4|8),k=(1024|5120),"       # 6 multi-column cases for compute-sanitizer
BOOT_POWER_RE = re.compile(r"host check: power limit,default ([0-9.]+),([0-9.]+) W ok")


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


def tbo_passed(row: dict) -> bool:
    """Exit 0, at least one case ran on CUDA0 and every case passed (0/0 means no CUDA backend, e.g. a CPU-only fallback)."""
    return bool(row.get("ok")) and row.get("total", 0) > 0 and row.get("passed") == row.get("total") and not row.get("n_fail")


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


def gpu600_problem(tag: str, smi: dict | None) -> str | None:
    """Why this host is not a 600 W RTX PRO 6000 (Workstation or Server Edition), else None. smi: one e4_pdl.smi_sample()."""
    if tag not in GPU600_TAGS:
        return f"GPU tag {tag} is not a 600 W RTX PRO 6000 ({' or '.join(GPU600_TAGS)})"
    dflt = (smi or {}).get("power.default_limit")
    if dflt is not None and dflt < GPU600_MIN_DEFAULT_W:
        return f"default power limit {dflt:.0f} W < {GPU600_MIN_DEFAULT_W:.0f} W"
    return None


def power_problem(limit, default, tol: float = POWER_TOL) -> str | None:
    """Why the power limit is not the default (within tol), else None. An unreadable limit counts: the host cannot be shown healthy."""
    if not limit or not default:
        return "power limit or default limit not readable"
    if abs(limit - default) > tol * default:
        return "power limit %.0f W is not the default %.0f W (tolerance %.0f%%)" % (limit, default, 100 * tol)
    return None


def bootstrap_power(path=None) -> dict | None:
    """The limit and default the bootstrap host check printed (the last such line of bootstrap.log), else None."""
    from infra.results import results_root
    p = pathlib.Path(path) if path else results_root() / "_status" / "bootstrap.log"
    try:
        found = BOOT_POWER_RE.findall(p.read_text(errors="replace"))
    except OSError:
        return None
    return {"limit": float(found[-1][0]), "default": float(found[-1][1])} if found else None


def enforced_limit() -> float | None:
    """nvidia-smi enforced.power.limit (the lowest of all power limiters), recorded only; None when unreadable."""
    try:
        p = subprocess.run(["nvidia-smi", "--query-gpu=enforced.power.limit", "--format=csv,noheader,nounits", "-i", "0"],
                           capture_output=True, text=True, timeout=10)
        return float(p.stdout.split()[0]) if p.returncode == 0 else None
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


class PowerGuard:
    """PR evidence must come from a card at its default power limit. check() reads nvidia-smi, records both values and returns the problem
    (None when the limit is the default within tol, or when enforce is off: the values are recorded either way). doc goes into the result."""

    def __init__(self, enforce: bool = True, tol: float = POWER_TOL, bootstrap: dict | None = None):
        self.enforce = enforce
        self.doc: dict = {"tol": tol, "enforced": enforce, "bootstrap": bootstrap, "checks": [], "first_problem": None}

    def check(self, label: str) -> str | None:
        smi = e4_pdl.smi_sample() or {}
        if not smi.get("power.limit") or not smi.get("power.default_limit"):      # one retry: a slow nvidia-smi must not skip a part
            time.sleep(2)
            smi = e4_pdl.smi_sample() or {}
        limit, default = smi.get("power.limit"), smi.get("power.default_limit")
        bad = power_problem(limit, default, self.doc["tol"])
        self.doc["checks"].append({"at": label, "limit": limit, "default": default, "enforced": enforced_limit(), "problem": bad, "t": time.time()})
        if bad and self.doc["first_problem"] is None:
            self.doc["first_problem"] = {"at": label, "problem": bad}
        return bad if self.enforce else None


def power_line(power: dict) -> str:
    """One notes line with every recorded limit/default pair."""
    pairs = ", ".join("%s %s/%s W%s%s" % (c["at"], c["limit"], c["default"], " (enforced %s W)" % c["enforced"] if c.get("enforced") else "",
                                         " PROBLEM" if c.get("problem") else "") for c in power.get("checks", []))
    boot = power.get("bootstrap")
    return "power limit/default: bootstrap %s, %s" % ("%s/%s W" % (boot["limit"], boot["default"]) if boot else "not seen", pairs or "no check")


def round_configs(configs: list, rnd: int) -> list:
    """Even rounds in the given order, odd rounds reversed: with an even round count every config runs first equally often."""
    return list(configs) if rnd % 2 == 0 else list(reversed(configs))


def paired_rounds(rows: list[dict], a: str = "off", b: str = "on", before_round: int | None = None) -> dict:
    """Gain of b over a per width and kind, round by round (ms_a / ms_b - 1 in percent, each from the median of the round's reps),
    and over the rounds: median, min, max and whether every round has the same sign. A round counts when both configs ran it.
    before_round: only rounds below it count (the first unhealthy round and later ones are left out)."""
    cell: dict = {}
    for r in rows:
        if r.get("tok_s") and (before_round is None or r["round"] < before_round):
            cell.setdefault((r["kind"], r["width"]), {}).setdefault(r["round"], {})[r["config"]] = r["tok_s"]
    out: dict = {"a": a, "b": b, "pp": {}, "tg": {}}
    for (kind, width), by_r in sorted(cell.items()):
        rnds = [rnd for rnd in sorted(by_r) if a in by_r[rnd] and b in by_r[rnd]]
        if not rnds:
            continue
        g = [100.0 * (by_r[rnd][b] / by_r[rnd][a] - 1.0) for rnd in rnds]
        out[kind][str(width)] = {"rounds": rnds, "gain_pct": g, "median": common.median(g), "min": min(g), "max": max(g), "n": len(g),
                                 "same_sign": all(x > 0 for x in g) or all(x < 0 for x in g)}
    return out


def summarize_bench(rows: list[dict], cfg_names: list[str], ref: str = "off", before_round: int | None = None) -> dict:
    """rows: {config, round, kind (pp|tg), width, tok_s}. Per width: median ms per batch of each config over the rounds in which every
    config has the width, and gain_pct_<cfg> = ms_<ref> / ms_<cfg> - 1 in percent. tg: median decode tok/s per config."""
    cells: dict = {}
    for r in rows:
        if r.get("tok_s") and (before_round is None or r["round"] < before_round):
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
            if c != ref and ms.get(ref) and ms.get(c):
                ent[f"gain_pct_{c}"] = 100.0 * (ms[ref] / ms[c] - 1.0)
        out["pp"][str(width)] = ent
    base = out["tg"].get("1") or next(iter(out["tg"].values()), None)
    if base and base["tok_s"].get(ref):
        out["tg_gain_pct"] = {c: 100.0 * (base["tok_s"][c] / base["tok_s"][ref] - 1.0) for c in cfg_names if base["tok_s"].get(c)}
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


def tbo_progress(text: str) -> dict:
    """Cases in the console output of test-backend-ops so far: the last case line, cases with a result, passed, failed."""
    last, done, passed, failed, has_result = None, 0, 0, 0, True
    for ln in ANSI_RE.sub("", text).splitlines():
        m = CASE_RE.match(ln)
        if not m:
            continue
        res = m.group(3).strip()
        last, has_result = "%s(%s)" % (m.group(1), m.group(2)), bool(res)
        if res.startswith("OK"):
            done, passed = done + 1, passed + 1
        elif res.startswith("FAIL"):
            done, failed = done + 1, failed + 1
        elif res:
            done += 1
    return {"last_case": last[:300] if last else None, "last_case_has_result": has_result, "cases_done": done, "passed_so_far": passed,
            "fail_so_far": failed}


def smi_tbo_sample() -> dict | None:
    """One nvidia-smi sample (clocks, power, utilization, memory) of GPU 0; None when nvidia-smi fails or hangs."""
    cmd = ["nvidia-smi", "--query-gpu=" + ",".join(TBO_SMI_FIELDS), "--format=csv,noheader,nounits", "-i", "0"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    vals = [v.strip() for v in p.stdout.splitlines()[0].split(",")] if p.returncode == 0 and p.stdout.strip() else []
    if len(vals) != len(TBO_SMI_FIELDS):
        return None
    out = {}
    for k, v in zip(("sm_mhz", "mem_mhz", "power_w", "util", "mem_util", "mem_mib"), vals):
        try:
            out[k] = float(v)
        except ValueError:
            out[k] = None
    return out


def tbo_diagnosis(samples: list[dict], window_s: float | None = None) -> dict:
    """Hang or slow, from the samples of a run that timed out (each has t, util, power_w, done = cases with a result).
    slow_progress: cases finished in the last window. No case finished: idle_no_progress (GPU idle), pegged_no_progress (GPU busy
    at >= 90%), partial_no_progress (between)."""
    window_s = TBO_DIAG_WINDOW_S if window_s is None else window_s
    ss = [s for s in samples if s.get("util") is not None]
    if len(ss) < 3:
        return {"verdict": "no_data", "n": len(ss)}
    win = [s for s in ss if s["t"] >= ss[-1]["t"] - window_s]
    win = win if len(win) >= 3 else ss[-3:]
    prog = win[-1]["done"] - win[0]["done"]
    util, power = common.median([s["util"] for s in win]), common.median([s.get("power_w") for s in win])
    verdict = ("slow_progress" if prog > 0 else "idle_no_progress" if util <= IDLE_UTIL else
               "pegged_no_progress" if util >= BUSY_UTIL else "partial_no_progress")
    return {"verdict": verdict, "window_s": round(win[-1]["t"] - win[0]["t"], 1), "n": len(win), "cases_in_window": prog,
            "util_med": util, "power_w_med": power}


def smi_run_summary(samples: list[dict]) -> dict:
    def col(k, f):
        xs = [s[k] for s in samples if s.get(k) is not None]
        return f(xs) if xs else None
    return {"n": len(samples), "period_s": TBO_SMI_PERIOD_S, "util_med": col("util", common.median), "util_min": col("util", min),
            "util_max": col("util", max), "power_w_med": col("power_w", common.median), "power_w_max": col("power_w", max),
            "sm_mhz_med": col("sm_mhz", common.median), "sm_mhz_min": col("sm_mhz", min), "mem_mhz_med": col("mem_mhz", common.median),
            "mem_mib_max": col("mem_mib", max)}


def kill_group(proc: subprocess.Popen, grace_s: float = 10.0) -> str:
    """SIGTERM to the process group, SIGKILL after grace_s; returns sigterm, sigkill or unkillable (a process stuck in the driver)."""
    for sig, wait in ((signal.SIGTERM, grace_s), (signal.SIGKILL, 15.0)):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.wait(timeout=wait)
            if sig == signal.SIGTERM:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)      # children that ignored the TERM
                except (ProcessLookupError, PermissionError):
                    pass
            return "sigterm" if sig == signal.SIGTERM else "sigkill"
        except subprocess.TimeoutExpired:
            continue
    return "unkillable"


def _read(path: pathlib.Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def run_tbo(cmd: list[str], env: dict, timeout: float, log_base: pathlib.Path | None = None) -> dict:
    """Run one test-backend-ops (or sanitizer) command. stdout and stderr stream to <log_base>.out.log and .err.log (a temp dir when
    log_base is None) and are kept on a timeout. nvidia-smi is sampled during the run. On a timeout the process group is killed and the
    row has last_case, cases_done, elapsed_s and a hang-or-slow diagnosis."""
    row: dict = {"cmd": cmd}
    tmp = None if log_base else tempfile.mkdtemp(prefix="cr-tbo-")
    base = log_base or pathlib.Path(tmp) / "tbo"
    out_p, err_p = (base.with_name(base.name + ".%s.log" % k) for k in ("out", "err"))
    row.update(log_out=out_p.name, log_err=err_p.name)
    timeout = max(1.0, timeout)
    samples: list[dict] = []
    stop = threading.Event()
    t0 = time.monotonic()
    wrap = ["stdbuf", "-oL", "-eL"] if shutil.which("stdbuf") else []          # a hung run must not hold finished cases in a stdio buffer

    def sampler():
        while True:
            s = smi_tbo_sample()
            if s:
                s.update(t=round(time.monotonic() - t0, 1), done=tbo_progress(_read(out_p))["cases_done"])
                samples.append(s)
            if stop.wait(TBO_SMI_PERIOD_S):
                return
    timed_out, killed, rc = False, None, None
    try:
        with open(out_p, "wb") as fo, open(err_p, "wb") as fe:
            proc = subprocess.Popen(wrap + cmd, env=env, stdin=subprocess.DEVNULL, stdout=fo, stderr=fe, start_new_session=True)
            th = threading.Thread(target=sampler, daemon=True)
            th.start()
            try:
                rc = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                row["elapsed_s"] = round(time.monotonic() - t0, 1)
                timed_out, killed = True, kill_group(proc)
            finally:
                stop.set()
                th.join(timeout=12)
                if proc.poll() is None:
                    kill_group(proc)
    except OSError as e:
        row.update(ok=False, error="%s: %s" % (type(e).__name__, e))
        return row
    finally:
        stop.set()
    row.setdefault("elapsed_s", round(time.monotonic() - t0, 1))
    secrets = r1.secret_values()
    out_t, err_t = _read(out_p), _read(err_p)
    for path, text in ((out_p, out_t), (err_p, err_t)):
        if r1.scrub(text, secrets) != text:                    # the logs ship with the results
            path.write_text(r1.scrub(text, secrets), encoding="utf-8")
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    row.update(tbo_progress(out_t), smi=smi_run_summary(samples), smi_samples=samples)
    m = RACECHECK_RE.search(out_t + "\n" + err_t)
    if m:
        row["racecheck"] = {"hazards": int(m.group(1)), "errors": int(m.group(2)), "warnings": int(m.group(3))}
    if timed_out:
        text = ANSI_RE.sub("", out_t + "\n" + err_t)
        row.update(ok=False, error="timeout", timeout_s=round(timeout, 1), killed=killed, diagnosis=tbo_diagnosis(samples),
                   gpu_mem_after_kill=gpu_mem(), out_tail=r1.scrub(text[-3000:], secrets))
        return row
    parsed = parse_tbo(out_t)
    row.update(parsed, returncode=rc, ok=rc == 0 and parsed["total"] > 0)  # 0/0 tests (no CUDA) is not ok
    if not row["ok"]:     # keep what failed: the console has no FAIL line for an allocation or setup error
        text = ANSI_RE.sub("", (out_t or "") + "\n" + (err_t or ""))
        bad = [ln.strip()[:300] for ln in text.splitlines() if ln.strip() and not ln.rstrip().endswith(("OK", "not supported [CPU]"))]
        row["problem_lines"] = r1.scrub(bad[:40], secrets)
        row["out_tail"] = r1.scrub(text[-3000:], secrets)
    return row


def tbo_log_base(exp: str, gpu: str | None, name: str) -> pathlib.Path | None:
    """results/<exp>/<gpu>/raw/<stamp>-tbo-<name>; None when the results dir is not writable (the logs then go to a temp dir)."""
    try:
        from infra.results import artifact_path
        return artifact_path(exp, "tbo-" + name, gpu)
    except (OSError, ValueError):
        return None


def tbo_line(name: str, row: dict, exp: str = EXP) -> str:
    if row.get("error") == "timeout":
        d = row.get("diagnosis") or {}
        return "[%s] test-backend-ops %s TIMEOUT after %.0f s: %s cases done, last %s, %s" % (
            exp, name, row.get("elapsed_s", 0), row.get("cases_done"), row.get("last_case"), d.get("verdict"))
    return "[%s] test-backend-ops %s ok=%s %s/%s in %s s" % (exp, name, row.get("ok"), row.get("passed"), row.get("total"), row.get("elapsed_s"))


def tbo_part(bin_dir, budget: common.Budget, notes: list, order: list | None = None, per_run_s: float = TBO_TIMEOUT_S,
             exp: str = EXP, gpu: str | None = None) -> list[dict]:
    """One run per config in order, each with its own timeout (per_run_s) and logs; a timeout or failure never skips the next run."""
    out = []
    if not (bin_dir / "test-backend-ops").exists():
        notes.append(f"{bin_dir / 'test-backend-ops'} not found; part t and f skipped")
        return out
    for name, cenv in (TBO_ORDER if order is None else order):
        if budget.expired(60):
            notes.append(f"t: time budget used up, config {name} skipped")
            continue
        row = run_tbo(tbo_cmd(bin_dir), run_env(cenv), budget.timeout(per_run_s, reserve_s=30), tbo_log_base(exp, gpu, name))
        if not row.get("total") and row.get("returncode") == 0:        # the filter matched nothing: run MUL_MAT whole
            notes.append(f"t {name}: no test matched {PQ2_FILTER!r}; MUL_MAT run whole")
            row = dict(run_tbo(tbo_cmd(bin_dir, filt=None), run_env(cenv), budget.timeout(per_run_s * 1.5, reserve_s=30),
                               tbo_log_base(exp, gpu, name + "-whole")), filter=None)
        row.update(config=name, env=cenv)
        if row.get("error") == "timeout":
            notes.append("t %s: timeout after %.0f s, %s cases done, last case %s; diagnosis %s (logs in raw/)" % (
                name, row.get("elapsed_s", 0), row.get("cases_done"), row.get("last_case"), (row.get("diagnosis") or {}).get("verdict")))
        out.append(row)
        print(tbo_line(name, row, exp), flush=True)
    return out


def nvcc_release() -> str | None:
    """CUDA toolkit release of nvcc (the build), e.g. 12.8; None when nvcc is not found."""
    nvcc = shutil.which("nvcc") or next(iter(sorted(glob.glob("/usr/local/cuda*/bin/nvcc"))[-1:]), None)
    try:
        p = subprocess.run([nvcc, "--version"], capture_output=True, text=True, timeout=15) if nvcc else None
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"release ([0-9.]+), (V[0-9.]+)", p.stdout) if p else None
    return "%s (%s)" % m.groups() if m else None


def host_info() -> dict:
    """CPU model, nproc, driver, CUDA versions (nvidia-smi driver ceiling, nvcc of the build) and git sha; never raises."""
    from infra import results
    info: dict = {k: None for k in ("cpu_model", "nproc", "nproc_all", "driver", "cuda_driver", "nvcc", "git_sha", "git_dirty", "compute_sanitizer")}
    try:
        for ln in pathlib.Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
            if ln.lower().startswith("model name"):
                info["cpu_model"] = ln.split(":", 1)[1].strip()
                break
        info["nproc_all"] = os.cpu_count()
        info["nproc"] = len(os.sched_getaffinity(0))
    except (OSError, AttributeError):
        pass
    try:
        info["driver"], info["cuda_driver"] = results._smi_driver()
        info["nvcc"] = nvcc_release()
        info["git_sha"] = os.environ.get("CR_GIT_SHA") or results._git("rev-parse", "HEAD")
        st = results._git("status", "--porcelain", "--untracked-files=no")
        info["git_dirty"] = None if st is None else bool(st.strip())
        info["compute_sanitizer"] = find_sanitizer()
    except Exception:
        pass
    return info


def find_sanitizer() -> str | None:
    p = shutil.which("compute-sanitizer")
    if p:
        return p
    found = sorted(glob.glob("/usr/local/cuda*/bin/compute-sanitizer") + glob.glob("/opt/nvidia/*/bin/compute-sanitizer"))
    return found[-1] if found else None


def tbo_localize(bin_dir, cenv: dict, budget: common.Budget, notes: list, exp: str = EXP, gpu: str | None = None) -> dict:
    """Part t after a timeout with the switch on while off passed. Inside budget: racecheck on SAN_FILTER (few pq2_0 cases) when
    compute-sanitizer exists (recorded when not), then the whole pq2_0 filter with CUDA_LAUNCH_BLOCKING=1: with blocking launches the
    last case line names the case that hangs. Racecheck sees shared memory hazards only, not a global memory race with PDL."""
    out: dict = {"sanitizer": find_sanitizer(), "runs": []}
    san = out["sanitizer"]
    if san:
        row = run_tbo([san, "--tool", "racecheck", "--target-processes", "all"] + tbo_cmd(bin_dir, filt=SAN_FILTER), run_env(cenv), budget.timeout(SAN_RUN_S),
                      tbo_log_base(exp, gpu, "on-racecheck"))
        out["runs"].append(dict(row, config="on-racecheck", env=cenv))
        print(tbo_line("on-racecheck", row, exp), flush=True)
    else:
        notes.append("t: compute-sanitizer not found on this image; localizing with CUDA_LAUNCH_BLOCKING=1 only")
    if budget.left() >= LOCALIZE_MIN_S:
        env = dict(cenv, CUDA_LAUNCH_BLOCKING="1")
        row = run_tbo(tbo_cmd(bin_dir), run_env(env), budget.timeout(CLB_RUN_S, reserve_s=5), tbo_log_base(exp, gpu, "on-launchblocking"))
        out["runs"].append(dict(row, config="on-launchblocking", env=env))
        print(tbo_line("on-launchblocking", row, exp), flush=True)
    else:
        notes.append("t: no time left for the CUDA_LAUNCH_BLOCKING=1 run")
    return out


def path_probe(nsys: str | None, bin_dir, model: str, budget: common.Budget, notes: list, configs: list | None = None) -> dict:
    """Kernel names of llama-bench -p 4 -n 0 under nsys, per config (default off, on; configs: [(name, env)] in the order to run).
    Never raises; the nsys files are deleted."""
    out: dict = {"cmd": bench_pp_cmd(bin_dir, "<model.gguf>", [4], 1), "configs": {}}
    if not nsys:
        out["error"] = "nsys not found (CR_NSYS, PATH, /usr/local/cuda*/bin, /opt/nvidia/nsight-systems)"
        notes.append("k: " + out["error"])
        return out
    for name, cenv in (CONFIGS[:2] if configs is None else configs):
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


def bench_part(bin_dir, model: str, budget: common.Budget, notes: list, rows: list, allow_power_cap: bool, save,
               configs: list | None = None, rounds: int = ROUNDS, bins: dict | None = None, tag: str = "r2b_mc",
               health: dict | None = None, power_tol: float | None = None) -> None:
    """configs: [(name, env)] (default CONFIGS); bins: {name: bin_dir} when the configs are different builds.
    health: filled with first_bad_round and problem at the first run that fails the health check; rows from that round on are flagged
    unhealthy (the caller leaves them out with before_round). power_tol: the power cap seen during a run must also equal the default
    within that fraction (PR evidence), else the run counts as unhealthy."""
    unhealthy = False
    configs = CONFIGS if configs is None else configs
    for rnd in range(rounds):
        for name, cenv in round_configs(configs, rnd):
            bd = (bins or {}).get(name, bin_dir)
            for kind, cmd in (("pp", bench_pp_cmd(bd, model)), ("tg", bench_tg_cmd(bd, model))):
                if budget.expired(60):
                    notes.append(f"b: time budget used up, round {rnd} {name} {kind} and later skipped")
                    return
                with e4_pdl.SmiWatch() as watch:
                    got, err = bench_entries(cmd, run_env(cenv), budget.timeout(900, reserve_s=30))
                smi = watch.summary()
                bad = e4_pdl.health_problem(smi, check_power=not allow_power_cap)
                if not bad and power_tol and not allow_power_cap:
                    bad = power_problem(smi.get("power_cap_w_min", smi.get("power_cap_w")), smi.get("power_default_w"), power_tol)      # lowest cap of all samples
                if bad and not unhealthy:
                    unhealthy = True
                    notes.append(f"host health: {bad} during round {rnd} {name} {kind}; timings after this may be invalid")
                    if health is not None:
                        health.update(first_bad_round=rnd, problem=bad, where=f"{name} {kind}")
                if err:
                    notes.append(f"round {rnd} {name} {kind} bench failed: {err}")
                for r in e5_cost.rows_from(got, "pq2", 0):
                    rows.append({"config": name, "env": cenv, "round": rnd, "kind": "pp" if r["kind"] == "verify" else "tg",
                                 "width": r["width"], "tok_s": r["tok_s"], "stddev": r["stddev"], "ms_per_batch": r["ms_per_batch"],
                                 "samples_ts": r["samples_ts"], "smi": smi, "unhealthy": unhealthy})
                save()
                print("[%s] r%d %s %s sm %s MHz" % (tag, rnd, name, kind, smi.get("sm_mhz_med")), flush=True)


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
                    extra["tbo"] = tbo_part(bin_dir, sub, notes, gpu=args.gpu)
                    if any(not t.get("ok") for t in extra["tbo"]):
                        rc = 1
                        notes.append("test-backend-ops reported failures: see data.tbo[].fail_lines before reading any timing")
                    bad_on = [t["config"] for t in extra["tbo"] if t.get("env") and not tbo_passed(t)]
                    if bad_on or not any(t.get("env") and tbo_passed(t) for t in extra["tbo"]):
                        notes.append("stop: test-backend-ops did not pass with %s set (%s); parts k, b, f not run" % (ENV_VAR, ", ".join(bad_on) or "no passing run"))
                        rc = 1
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
