"""Phase 0 exit check (PLAN section 3): does persisting L2 raise the hit rate on a re-read buffer?

python -m harness.p0_l2_check [--mb 64] [--reps 100] [--hit-ratio 1.0] [--no-ncu] [--inner off|on]
                              [--method all|eager|graph-nodes|graph-stream] [--pollute-mb N]
                              [--ncu-timeout S] [--ncu-budget S]

A buffer of --mb MiB fp32 is re-read --reps times by kernel cr_reread on one stream, window off then on.
Deviation from a plain re-read loop: after every rep a streaming kernel (cr_pollute, 2x L2 by default)
reads a large other buffer. Without it a 64 MiB buffer simply stays in a 96-128 MiB L2 and the window
shows nothing. The pollute kernel is outside the cr_reread ncu filter and outside the timed region.

--method says how the window reaches the kernel (default all = every method, one verdict each):
  eager         stream window, kernels launched directly
  graph-nodes   window set on the graph kernel nodes (cudaGraphKernelNodeSetAttribute), stream has none
  graph-stream  stream window set before capture; cleared after capture, so only the window copied into
                the captured nodes can explain a change (this is what E1 --window-method stream relies on)
Graph methods record the window read back from a kernel node (node_window_readback).
methods_honored lists the methods whose verdict passed on a non-control run.

Outer mode times each condition with CUDA events and re-runs itself under ncu (--inner off|on,
kernel regex cr_reread, launch skip 5, application replay). A timing-only result file is written before
ncu starts (partial: true) and replaced by the final one. Each ncu call has --ncu-timeout seconds, all
ncu calls together --ncu-budget seconds; a timed-out call kills the whole ncu process group.
Pass rule:
  window fits (persisting_max >= 0.75 * buffer): hit_on - hit_off >= 0.20 and dram_on <= 0.5 * dram_off
  otherwise (control, e.g. RTX 3090): no gain beyond the persisting capacity, |hit_on - hit_off| < 0.05 + persisting_max / buffer
    (a working window may hold persisting_max bytes, so up to that share of re-reads hits legitimately)
Without counters the decision uses timing only (on >= 1.3x faster; control: < 1.15x) and counters_ok is false.
Counters from kernel replay (application replay failed) are flagged: counters_basis kernel_replay, counters_suspect.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import statistics
import sys
import time

SKIP = 5
METHODS = ("eager", "graph-nodes", "graph-stream")
NCU_CALL_TIMEOUT = 300.0
NCU_BUDGET = 900.0
NCU_MIN_CALL = 30.0
KERNEL_REGEX = "cr_reread"
INNER_TAG = "CR_L2_INNER "
EXP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLOCK = 256
MiB = 1024 * 1024

# own 16 byte struct: NVRTC (cupy) needs no vector_types.h include
_PRELUDE = "struct alignas(16) cr_f4 { float x, y, z, w; };\n"

_KERNEL = r'''
extern "C" __global__ void %(name)s(const cr_f4* __restrict__ p, float* __restrict__ out, long long n4)
{
    float acc = 0.f;
    long long stride = (long long) gridDim.x * blockDim.x;
    #pragma unroll 4
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
        cr_f4 v = p[i];
        acc += (v.x + v.y) + (v.z + v.w);
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    __shared__ float part[32];
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) part[wid] = acc;
    __syncthreads();
    if (wid == 0) {
        acc = lane < (int) ((blockDim.x + 31) >> 5) ? part[lane] : 0.f;
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
        if (lane == 0) out[blockIdx.x] = acc;
    }
}
'''

_LAUNCHERS = r'''
int64_t cr_launch_reread(int64_t p, int64_t out, int64_t n4, int64_t grid, int64_t stream) {
    cr_reread<<<(unsigned) grid, %(block)d, 0, (cudaStream_t) stream>>>((const cr_f4 *) p, (float *) out, n4);
    return (int64_t) cudaGetLastError();
}
int64_t cr_launch_pollute(int64_t p, int64_t out, int64_t n4, int64_t grid, int64_t stream) {
    cr_pollute<<<(unsigned) grid, %(block)d, 0, (cudaStream_t) stream>>>((const cr_f4 *) p, (float *) out, n4);
    return (int64_t) cudaGetLastError();
}
'''


def cuda_source(with_launchers: bool = False) -> str:
    """Kernel source: cr_reread (the measured kernel) and cr_pollute (same body, other name)."""
    src = _PRELUDE + _KERNEL % {"name": "cr_reread"} + _KERNEL % {"name": "cr_pollute"}
    if with_launchers:
        src = "#include <cstdint>\n#include <cuda_runtime.h>\n" + src + _LAUNCHERS % {"block": BLOCK}
    return src


# ---------------------------------------------------------------- pure logic

def make_plan(mb: float, hit_ratio: float, info: dict, pollute_mb: float | None = None) -> dict:
    """Sizes and control flag from device-reported numbers."""
    buf = int(mb * MiB) // 16 * 16
    if buf <= 0:
        raise ValueError("--mb too small")
    if pollute_mb is None:
        pollute = max(2 * int(info["l2_bytes"]), 64 * MiB)
    else:
        pollute = int(pollute_mb * MiB) // 16 * 16
    pmax = int(info["persisting_max_bytes"])
    window = min(buf, int(info["window_max_bytes"]) or buf, pmax)
    return {"buffer_bytes": buf, "pollute_bytes": pollute, "window_bytes": window // 16 * 16,
            "window_clamped": window < buf, "hit_ratio": hit_ratio, "persisting_max_bytes": pmax,
            "control": pmax < 0.75 * buf}


def decide(off: dict, on: dict, plan: dict) -> dict:
    """Pass/fail from two condition dicts (keys hit_rate, dram_bytes_per_rep, us_per_rep, counters_ok)."""
    control = plan["control"]
    counters_ok = bool(off.get("counters_ok") and on.get("counters_ok")
                       and off.get("hit_rate") is not None and on.get("hit_rate") is not None)
    speedup = off["us_per_rep"] / on["us_per_rep"] if on["us_per_rep"] > 0 else None
    out = {"control": control, "counters_ok": counters_ok, "speedup": speedup}
    if counters_ok:
        out["counters_suspect"] = "kernel" in (off.get("replay_mode"), on.get("replay_mode"))
        d_hit = on["hit_rate"] - off["hit_rate"]
        out["hit_delta"] = d_hit
        out["dram_ratio"] = (on["dram_bytes_per_rep"] / off["dram_bytes_per_rep"]
                             if off.get("dram_bytes_per_rep") else None)
        if control:
            tol = 0.05 + plan["persisting_max_bytes"] / float(plan["buffer_bytes"])
            out["tolerance"] = tol
            out["pass"] = abs(d_hit) < tol
        else:
            dram_ok = (on.get("dram_bytes_per_rep") or 0.0) <= 0.5 * (off.get("dram_bytes_per_rep") or 0.0)
            out["pass"] = d_hit >= 0.20 and dram_ok
        out["basis"] = "counters"
    else:
        out["basis"] = "timing"
        if speedup is None:
            out["pass"] = False
        elif control:
            out["pass"] = speedup < 1.15
        else:
            out["pass"] = speedup >= 1.3
    return out


def trimmed_stats(times_us: list[float], skip: int = SKIP) -> dict:
    """Median and mean of per-rep times, dropping the first reps (cold start)."""
    use = times_us[skip:] if len(times_us) > skip + 1 else times_us
    return {"us_per_rep": statistics.median(use), "us_per_rep_mean": statistics.fmean(use), "n_timed": len(use)}


# ---------------------------------------------------------------- GPU side

class _Kernels:
    """Launches cr_reread / cr_pollute. cupy RawModule first, torch load_inline (nvcc) as fallback."""

    def __init__(self, buf, pol, grid):
        import torch

        self.buf, self.pol, self.grid = buf, pol, grid
        self.out = torch.zeros(grid, dtype=torch.float32, device="cuda")
        self.n4 = buf.numel() // 4
        self.pn4 = pol.numel() // 4 if pol is not None else 0
        self.how = None
        try:
            import cupy as cp

            mod = cp.RawModule(code=cuda_source())
            self._cp = cp
            self._re, self._po = mod.get_function("cr_reread"), mod.get_function("cr_pollute")
            self._cbuf = cp.asarray(buf)
            self._cpol = cp.asarray(pol) if pol is not None else None
            self._cout = cp.asarray(self.out)
            self._es = {}
            self.how = "cupy"
        except Exception as e:
            self.cupy_error = repr(e)
            from torch.utils.cpp_extension import load_inline

            ext = load_inline(name="cr_p0_reread", cpp_sources=["int64_t cr_launch_reread(int64_t,int64_t,int64_t,int64_t,int64_t); "
                                                                "int64_t cr_launch_pollute(int64_t,int64_t,int64_t,int64_t,int64_t);"],
                              cuda_sources=[cuda_source(with_launchers=True)],
                              functions=["cr_launch_reread", "cr_launch_pollute"], with_cuda=True, verbose=False)
            self._ext = ext
            self.how = "load_inline"

    @staticmethod
    def _check(err):
        if err != 0:
            raise RuntimeError("kernel launch failed, cudaError %d" % err)

    def _stream_ctx(self, handle):
        es = self._es.get(handle)
        if es is None:
            es = self._es[handle] = self._cp.cuda.ExternalStream(handle)
        return es

    def reread(self, handle):
        if self.how == "cupy":
            import numpy as np

            with self._stream_ctx(handle):
                self._re((self.grid,), (BLOCK,), (self._cbuf, self._cout, np.int64(self.n4)))
        else:
            self._check(self._ext.cr_launch_reread(self.buf.data_ptr(), self.out.data_ptr(), self.n4, self.grid, handle))

    def pollute(self, handle):
        if self.pol is None:
            return
        if self.how == "cupy":
            import numpy as np

            with self._stream_ctx(handle):
                self._po((self.grid,), (BLOCK,), (self._cpol, self._cout, np.int64(self.pn4)))
        else:
            self._check(self._ext.cr_launch_pollute(self.pol.data_ptr(), self.out.data_ptr(), self.pn4, self.grid, handle))


def _node_window(graph) -> dict:
    """Window of the first kernel node of a keep_graph graph, or {"error": ...}."""
    from harness import l2persist

    try:
        w = l2persist.get_graph_window(graph)
    except Exception as e:
        return {"error": repr(e)}
    return w if w is not None else {"error": "no kernel node"}


def run_condition(cond: str, args, plan: dict, timed: bool) -> dict:
    """Run --reps reread kernels with the window off or on. Returns timings and driver read-back."""
    import contextlib

    import torch

    from harness import l2persist, timing

    if args.method not in METHODS:
        raise ValueError("run_condition needs one method, got %r" % args.method)
    dev = torch.cuda.current_device()
    sm = torch.cuda.get_device_properties(dev).multi_processor_count
    buf = torch.rand(plan["buffer_bytes"] // 4, device="cuda")
    pol = torch.zeros(plan["pollute_bytes"] // 4, device="cuda") if plan["pollute_bytes"] else None
    k = _Kernels(buf, pol, sm * 8)
    s = torch.cuda.Stream()
    h = s.cuda_stream
    info = {"kernels": k.how, "method": args.method}
    if k.how != "cupy":
        info["cupy_error"] = k.cupy_error
    graph = None

    with contextlib.ExitStack() as stack:
        if cond == "on":
            l2persist.set_limit(None)

            def _cleanup():
                try:
                    l2persist.clear_stream_window(s)
                finally:
                    try:
                        l2persist.reset()
                    finally:
                        l2persist.set_limit(0)

            stack.callback(_cleanup)
            if args.method in ("eager", "graph-stream"):
                info["window"] = l2persist.set_stream_window(buf, plan["window_bytes"], plan["hit_ratio"], stream=s)
                info["window_readback"] = l2persist.get_stream_window(s)
        if args.method != "eager":
            k.reread(h)
            k.pollute(h)
            s.synchronize()
            keep = cond == "on"
            graph = timing.capture_graph(lambda: k.reread(h), warmup=1, stream=s, keep_graph=keep)
            if keep:
                if args.method == "graph-nodes":
                    info["graph_kernel_nodes"] = l2persist.apply_window_to_graph(graph, buf, plan["window_bytes"], plan["hit_ratio"])
                    info["window"] = {"num_bytes": plan["window_bytes"], "hit_ratio": plan["hit_ratio"]}
                info["node_window_readback"] = _node_window(graph)
                if args.method == "graph-stream":
                    l2persist.clear_stream_window(s)
                    info["window_readback_after_clear"] = l2persist.get_stream_window(s)
                graph.instantiate()
        run = (lambda: graph.replay()) if graph is not None else (lambda: k.reread(h))
        evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(args.reps)] if timed else []
        with torch.cuda.stream(s):
            for i in range(args.reps):
                if timed:
                    evs[i][0].record(s)
                run()
                if timed:
                    evs[i][1].record(s)
                k.pollute(h)
        s.synchronize()
        torch.cuda.synchronize()
    del k, buf, pol
    torch.cuda.empty_cache()
    if timed:
        info["times_us"] = [a.elapsed_time(b) * 1000.0 for a, b in evs]
    return info


def _inner_argv(cond: str, args) -> list[str]:
    if args.method not in METHODS:
        raise ValueError("inner run needs one method, got %r" % args.method)
    return [sys.executable, "-m", "harness.p0_l2_check", "--inner", cond, "--mb", repr(args.mb), "--reps", str(args.reps),
            "--hit-ratio", repr(args.hit_ratio), "--method", args.method, "--no-ncu"] + (
        ["--pollute-mb", repr(args.pollute_mb)] if args.pollute_mb is not None else [])


def ncu_condition(cond: str, args, timeout: float = NCU_CALL_TIMEOUT) -> dict:
    """Profile one condition under ncu (args.method must be one method).

    Returns hit_rate / dram_bytes_per_rep / ncu_us_per_rep, counters_ok, replay_mode, counters_basis, ncu_cmd.
    timeout covers the application replay run and the kernel replay retry together.
    """
    from harness import counters

    argv = _inner_argv(cond, args)
    n = max(1, args.reps - SKIP)
    t_end = time.monotonic() + timeout
    kw = dict(kernel_regex=KERNEL_REGEX, launch_skip=SKIP, launch_count=n, cwd=EXP_DIR)
    mode = "application"
    res = counters.run_ncu(argv, 1, extra=("--replay-mode", "application"), timeout=timeout, **kw)
    if not res["ok"] and res["error"] != counters.ERR_PERM and not res.get("timed_out") and counters.ncu_path():
        left = t_end - time.monotonic()
        if left >= 10.0:
            mode = "kernel"
            res = counters.run_ncu(argv, 1, timeout=left, **kw)
    out = {"counters_ok": bool(res["ok"]), "ncu_error": res["error"], "hit_rate": None, "dram_bytes_per_rep": None,
           "ncu_us_per_rep": None, "ncu_kernels": len(res["rows"]), "replay_mode": mode,
           "counters_basis": mode + "_replay", "ncu_cmd": res.get("cmd"), "ncu_timed_out": bool(res.get("timed_out"))}
    if res["rows"]:
        s = counters.summarize(res["rows"], len(res["rows"]))
        out.update(hit_rate=s["hit_rate"], dram_bytes_per_rep=s["dram_bytes_per_token"], ncu_us_per_rep=s["us_per_token"])
    return out


def write_result(payload: dict):
    """infra.results.write_result when available, else a minimal local writer with the same layout."""
    try:
        from infra.results import write_result as wr
    except ImportError:
        wr = None
    if wr is not None:
        return wr("p0_l2", None, payload)
    import torch

    gpu = os.environ.get("CR_GPU_TAG") or re.sub(r"[^a-z0-9]+", "-", torch.cuda.get_device_name(0).lower()).strip("-")
    root = os.environ.get("CR_RESULTS_DIR") or os.path.join(EXP_DIR, "results")
    d = os.path.join(root, "p0_l2", gpu)
    os.makedirs(d, exist_ok=True)
    now = datetime.datetime.now(datetime.timezone.utc)
    path = os.path.join(d, now.strftime("%Y%m%dT%H%M%SZ") + ".json")
    with open(path, "w") as f:
        json.dump({"schema": 1, "exp": "p0_l2", "gpu": gpu, "utc": now.isoformat(), "env": {}, "data": payload}, f, indent=1, default=str)
    return path


# ---------------------------------------------------------------- CLI

def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="harness.p0_l2_check", description="Phase 0 persisting-L2 exit check")
    ap.add_argument("--mb", type=float, default=64.0, help="buffer size in MiB")
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--hit-ratio", type=float, default=1.0)
    ap.add_argument("--no-ncu", action="store_true", help="timing only")
    ap.add_argument("--inner", choices=["off", "on"], help="single condition for ncu (internal)")
    ap.add_argument("--method", choices=["all"] + list(METHODS), default="all",
                    help="how the window reaches the kernel: stream window, graph kernel nodes, stream window before capture; all = each")
    ap.add_argument("--pollute-mb", type=float, default=None, help="streaming buffer read after each rep (default 2x L2, 0 = none)")
    ap.add_argument("--ncu-timeout", type=float, default=NCU_CALL_TIMEOUT, help="seconds per ncu call")
    ap.add_argument("--ncu-budget", type=float, default=NCU_BUDGET, help="seconds for all ncu calls together")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and the ncu command, no CUDA")
    return ap.parse_args(argv)


def _for_method(args, method: str):
    return argparse.Namespace(**dict(vars(args), method=method))


def time_method(method: str, args, plan: dict) -> dict:
    """Timed off/on conditions of one method. An exception is recorded as {"error": ...}."""
    conds = {}
    try:
        for cond in ("off", "on"):
            r = run_condition(cond, _for_method(args, method), plan, timed=True)
            st = trimmed_stats(r.pop("times_us"))
            c = dict(st, cond=cond, method=method, counters_ok=False, hit_rate=None, dram_bytes_per_rep=None,
                     ncu_error=None, replay_mode=None, counters_basis=None, ncu_cmd=None)
            c.update(r)
            conds[cond] = c
            print("[p0_l2] %-12s %-3s %8.1f us/rep (median of %d)" % (method, cond, st["us_per_rep"], st["n_timed"]), flush=True)
    except Exception as e:
        print("[p0_l2] %-12s failed: %r" % (method, e), flush=True)
        return {"conditions": conds, "error": repr(e)}
    return {"conditions": conds}


def _node_window_ok(method: str, conds: dict, plan: dict):
    """True when the 'on' graph kernel node carries the window; None for eager."""
    if method == "eager":
        return None
    rb = conds["on"].get("node_window_readback") or {}
    return bool(rb.get("num_bytes") == plan["window_bytes"] and rb.get("hit_prop") == "persisting")


def build_payload(args, plan: dict, info: dict, entries: dict, notes: list, partial: bool) -> dict:
    """One verdict per method plus the top-level fields of the primary (first) method."""
    methods, notes = {}, list(notes)
    for m, e in entries.items():
        conds = e["conditions"]
        if "error" in e or "off" not in conds or "on" not in conds:
            methods[m] = {"conditions": conds, "error": e.get("error", "missing condition"), "pass": False,
                          "verdict": {"pass": False, "basis": "error", "control": plan["control"], "counters_ok": False, "speedup": None}}
            notes.append("%s: failed: %s" % (m, e.get("error", "missing condition")))
            continue
        v = decide(conds["off"], conds["on"], plan)
        methods[m] = {"conditions": conds, "verdict": v, "pass": v["pass"], "node_window_ok": _node_window_ok(m, conds, plan)}
        if methods[m]["node_window_ok"] is False:
            notes.append("%s: kernel node window read back differs from the requested window: %s" % (m, conds["on"].get("node_window_readback")))
        if v.get("counters_suspect"):
            notes.append("%s: counters from kernel replay (L2 state may carry over between passes)" % m)
    primary = next(iter(methods))
    pv = methods[primary]["verdict"]
    all_counters = all(x["verdict"].get("counters_ok") for x in methods.values())
    if not partial and not all_counters:
        from harness import counters

        notes.append(counters.NO_COUNTERS_NOTE)
    if plan["window_clamped"]:
        notes.append("window clamped to %d bytes (buffer %d, persisting max %d, window max %d)" % (
            plan["window_bytes"], plan["buffer_bytes"], info["persisting_max_bytes"], info["window_max_bytes"]))
    if plan["control"]:
        notes.append("control run: persisting max < 0.75 * buffer, no cache win expected")
    pconds = methods[primary]["conditions"]
    return {"config": dict(vars(args), **plan), "device": info, "methods": methods, "primary_method": primary,
            "methods_passed": [m for m, x in methods.items() if x["pass"]],
            "methods_honored": [] if plan["control"] else [m for m, x in methods.items() if x["pass"]],
            "conditions": pconds, "rows": [c for x in methods.values() for c in x["conditions"].values()],
            "verdict": pv, "pass": all(x["pass"] for x in methods.values()), "control": plan["control"],
            "counters_ok": all_counters, "partial": partial, "notes": notes}


def run_ncu_phase(entries: dict, args) -> None:
    """ncu every condition of every method inside the shared budget; results are merged into the conditions."""
    deadline = time.monotonic() + args.ncu_budget
    for m, e in entries.items():
        if "error" in e:
            continue
        margs = _for_method(args, m)
        for cond in ("off", "on"):
            left = deadline - time.monotonic()
            if left < NCU_MIN_CALL:
                upd = {"counters_ok": False, "ncu_error": "ncu budget exhausted (%.0f s)" % args.ncu_budget}
            else:
                upd = ncu_condition(cond, margs, timeout=min(args.ncu_timeout, left))
            e["conditions"][cond].update(upd)
            c = e["conditions"][cond]
            print("[p0_l2] %-12s %-3s hit=%s dram/rep=%s ncu_ok=%s replay=%s %s" % (
                m, cond, c["hit_rate"], c["dram_bytes_per_rep"], c["counters_ok"], c["replay_mode"], c["ncu_error"] or ""), flush=True)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.reps <= SKIP:
        print("--reps must be > %d" % SKIP, file=sys.stderr)
        return 2
    if args.inner and args.method == "all":
        print("--inner needs one --method", file=sys.stderr)
        return 2
    from harness import counters, l2persist

    methods = METHODS if args.method == "all" else (args.method,)
    if args.dry_run:
        cmds = {m: counters.build_ncu_cmd(_inner_argv("on", _for_method(args, m)), kernel_regex=KERNEL_REGEX, launch_skip=SKIP,
                                          launch_count=args.reps - SKIP, extra=("--replay-mode", "application")) for m in methods}
        print(json.dumps({"config": vars(args), "methods": list(methods), "ncu_cmd": cmds[methods[0]], "ncu_cmds": cmds,
                          "kernels": ["cr_reread", "cr_pollute"]}, indent=1))
        return 0
    try:
        info = l2persist.log_device(0)
    except RuntimeError as e:
        print(json.dumps({"ok": False, "error": str(e)}))
        return 2
    plan = make_plan(args.mb, args.hit_ratio, info, args.pollute_mb)

    if args.inner:
        r = run_condition(args.inner, args, plan, timed=False)
        r.pop("times_us", None)
        print(INNER_TAG + json.dumps(r, default=str), flush=True)
        return 0

    notes = []
    entries = {m: time_method(m, args, plan) for m in methods}
    use_ncu = not args.no_ncu
    if args.no_ncu:
        notes.append("--no-ncu: timing-only")
    elif counters.ncu_path() is None:
        use_ncu = False
        notes.append("ncu not found: timing-only")
        for e in entries.values():
            for c in e["conditions"].values():
                c["ncu_error"] = "ncu not found"
    partial_path = None
    if use_ncu:
        partial_path = write_result(build_payload(args, plan, info, entries, notes + ["partial: timing only, ncu not finished"], True))
        print("[p0_l2] timing-only result written: %s" % partial_path, flush=True)
        run_ncu_phase(entries, args)
    payload = build_payload(args, plan, info, entries, notes, False)
    path = write_result(payload)
    if partial_path is not None and str(partial_path) != str(path):
        try:
            os.remove(partial_path)
        except OSError:
            pass
    print(json.dumps({"pass": payload["pass"], "verdict": payload["verdict"],
                      "methods": {m: x["pass"] for m, x in payload["methods"].items()},
                      "methods_honored": payload["methods_honored"], "result": str(path)}, indent=1, default=str))
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
