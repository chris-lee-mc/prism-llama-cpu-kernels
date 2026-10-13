"""Shared helpers for the experiment drivers (flags, result writing, ncu wrapper, time budget)."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import pathlib
import re
import signal
import statistics
import sys
import tempfile
import threading
import time

EXP_DIR = pathlib.Path(__file__).resolve().parent.parent
MIB = 1 << 20


def add_common_args(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    ap.add_argument("--dry-run", action="store_true", help="print the plan as JSON, no CUDA, exit 0")
    ap.add_argument("--quick", action="store_true", help="tiny sizes for a smoke run")
    ap.add_argument("--no-ncu", action="store_true", help="timing only, skip Nsight Compute passes")
    ap.add_argument("--gpu", default=None, help="override the GPU tag used for the result path")
    ap.add_argument("--budget-min", type=float, default=None,
                    help="stop starting new work after this many minutes (default: per experiment)")
    return ap


def have_cuda() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def gpu_tag(args=None) -> str:
    if args is not None and getattr(args, "gpu", None):
        return args.gpu
    from infra.results import gpu_tag as _tag
    return _tag()


def print_plan(plan: dict) -> None:
    print(json.dumps(plan, indent=1, default=str))


def _doc_data(config, rows, notes, counters_ok, extra):
    data = {"config": config, "rows": rows, "notes": notes, "counters_ok": bool(counters_ok)}
    if extra:
        data.update(extra)
    return data


def partial_path(exp: str, args) -> pathlib.Path:
    from infra.results import run_dir
    return run_dir(exp, getattr(args, "gpu", None)) / "raw" / "partial.json"


def write_partial(exp: str, args, config: dict, rows: list, notes: list, counters_ok: bool, extra: dict | None = None):
    """Atomic snapshot of the run so far (raw/partial.json). A run killed at the deadline still ships its rows."""
    try:
        import datetime as dt
        from infra.results import _clean
        p = partial_path(exp, args)
        p.parent.mkdir(parents=True, exist_ok=True)
        doc = _clean({"schema": 1, "exp": exp, "gpu": getattr(args, "gpu", None), "partial": True,
                      "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                      "data": _doc_data(config, rows, notes, counters_ok, extra)})
        tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(doc, indent=1, default=str, allow_nan=False) + "\n", encoding="ascii")
        os.replace(tmp, p)
    except Exception as e:
        print(f"[{exp}] partial write failed: {type(e).__name__}: {e}", file=sys.stderr, flush=True)


def write_result(exp: str, args, config: dict, rows: list, notes: list, counters_ok: bool, extra: dict | None = None):
    """Write one result file: data = {config, rows, notes, counters_ok} (+ extra keys). Drops the partial snapshot."""
    from infra.results import write_result as _wr
    path = _wr(exp, getattr(args, "gpu", None), _doc_data(config, rows, notes, counters_ok, extra))
    print(f"[{exp}] wrote {path}", flush=True)
    try:
        partial_path(exp, args).unlink()
    except (OSError, ValueError):
        pass
    return path


@contextlib.contextmanager
def term_exits():
    """Main thread only: SIGTERM raises SystemExit so finally blocks (final result write) run before the SIGKILL."""
    old = None
    if threading.current_thread() is threading.main_thread():
        def _handler(signum, frame):
            raise SystemExit(128 + signum)
        old = signal.signal(signal.SIGTERM, _handler)
    try:
        yield
    finally:
        if old is not None:
            signal.signal(signal.SIGTERM, old)


FAILSAFE_RESERVE_S = 300.0  # the runner kills the child at deadline - 180 s; stay clear of that


def deadline_left_s() -> float | None:
    """Seconds until CR_START_EPOCH + CR_FAILSAFE_MIN * 60 - reserve (the pod clock includes clone and build); None off-pod."""
    try:
        start = float(os.environ["CR_START_EPOCH"])
    except (KeyError, ValueError):
        return None
    try:
        failsafe_min = float(os.environ.get("CR_FAILSAFE_MIN") or 40)
    except ValueError:
        failsafe_min = 40.0
    return start + failsafe_min * 60.0 - FAILSAFE_RESERVE_S - time.time()


class Budget:
    """Wall-clock budget; left() is seconds remaining, expired() true once it is used up.

    On a pod the limit is also capped by the pod deadline (deadline_left_s), so a run never outlives the runner.
    """

    def __init__(self, minutes: float | None):
        self.t0 = time.monotonic()
        self.limit = None if minutes is None else float(minutes) * 60.0
        dl = deadline_left_s()
        if dl is not None:
            self.limit = dl if self.limit is None else min(self.limit, dl)

    def elapsed(self) -> float:
        return time.monotonic() - self.t0

    def left(self) -> float:
        return float("inf") if self.limit is None else self.limit - self.elapsed()

    def expired(self, reserve_s: float = 0.0) -> bool:
        return self.left() <= reserve_s

    def timeout(self, cap: float | None = None, reserve_s: float = 0.0) -> float | None:
        """Subprocess timeout: min(cap, left - reserve), at least 1 s; cap (maybe None) when the budget is unbounded."""
        t = self.left() - reserve_s
        if t == float("inf"):
            return cap
        return max(1.0, t if cap is None else min(cap, t))


def median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def counters_available(args) -> tuple[bool, str | None]:
    """(ok, note). Probes ncu once; --no-ncu and a failing probe both give ok False with a note."""
    if getattr(args, "no_ncu", False):
        return False, "ncu skipped (--no-ncu): timing only, analytic bytes"
    try:
        from harness import counters
        r = counters.probe()
    except Exception as e:
        return False, f"ncu probe failed to run: {e}"
    if r.get("ok"):
        return True, None
    from harness.counters import NO_COUNTERS_NOTE
    return False, f"{NO_COUNTERS_NOTE} [{r.get('error')}]"


_PASSES_RE = re.compile(r"(\d+)\s+passes?\b")


def parse_passes(text: str) -> int | None:
    """Largest pass count in the '==PROF== Profiling ... N passes' lines of ncu output; None if absent."""
    ns = [int(m.group(1)) for ln in text.splitlines() if ln.startswith("==PROF==") for m in [_PASSES_RE.search(ln)] if m]
    return max(ns) if ns else None


def ncu_run(argv, tokens: int, *, kernel_regex=None, launch_skip=0, launch_count=None, profile_from_start_off=False,
            app_replay=True, kernel_fallback=True, timeout=1800, cwd=None, env=None, csv_out=None) -> dict:
    """Run argv under ncu via harness.counters.run_ncu. The result has replay_mode ("application"/"kernel") and passes.

    profile_from_start_off: the target calls cudaProfilerStart/Stop around the measured region, so warmup
    and setup kernels are never profiled (no launch counting needed). Only set it if the target does.
    app_replay: re-run the whole process per pass so L2 state is not disturbed by kernel replay.
    kernel_fallback: after a failed application replay (not a counter permission error) retry in kernel replay.
    Turn it off for L2-persistence runs: with --cache-control none a later kernel-replay pass sees L2 warmed by the
    earlier ones, so hit rate and DRAM bytes are not trustworthy.
    """
    from harness import counters
    extra = ["--profile-from-start", "off"] if profile_from_start_off else []
    tmp = None
    out = csv_out
    if out is None:
        fd, tmp = tempfile.mkstemp(prefix="cr-ncu-", suffix=".txt")
        os.close(fd)
        out = tmp
    kw = dict(kernel_regex=kernel_regex, launch_skip=launch_skip, launch_count=launch_count, cwd=str(cwd or EXP_DIR),
              timeout=timeout, env=env, csv_out=out)

    def once(mode_extra: tuple, mode: str) -> dict:
        res = counters.run_ncu(argv, tokens, extra=tuple(extra) + mode_extra, **kw)
        res["replay_mode"] = mode
        try:
            res["passes"] = parse_passes(pathlib.Path(out).read_text(errors="replace"))
        except OSError:
            res["passes"] = None
        return res

    try:
        if app_replay:
            res = once(("--replay-mode", "application"), "application")
            if res["ok"] or res["error"] == counters.ERR_PERM or counters.ncu_path() is None or not kernel_fallback:
                return res
            first_error = res["error"]
            res = once((), "kernel")
            res["app_replay_error"] = first_error
            return res
        return once((), "kernel")
    finally:
        if tmp:
            with contextlib.suppress(OSError):
                os.unlink(tmp)


def replay_info(res: dict | None) -> dict:
    """ncu_replay_mode, ncu_passes and ncu_l2_state_ok for a row.

    Kernel replay with more than one pass runs later passes on an L2 warmed by the first (--cache-control none),
    so DRAM bytes can be under-reported and the hit rate inflated: ncu_l2_state_ok is False then.
    """
    mode = res.get("replay_mode") if res else None
    passes = res.get("passes") if res else None
    ok = None if mode is None else (mode == "application" or passes == 1)
    out = {"ncu_replay_mode": mode, "ncu_passes": passes, "ncu_l2_state_ok": ok}
    if res and res.get("app_replay_error"):
        out["ncu_app_replay_error"] = res["app_replay_error"]
    return out


def ncu_stats(res: dict | None) -> dict:
    """Flatten a run_ncu result: counters_ok, error, l2_hit_rate, dram_bytes_per_token, ncu_us_per_token, n_kernels, replay info."""
    out = {"counters_ok": False, "ncu_error": None, "l2_hit_rate": None, "dram_bytes_per_token": None,
           "ncu_us_per_token": None, "n_kernels": 0, **replay_info(res)}
    if not res:
        return out
    out["counters_ok"] = bool(res.get("ok"))
    out["ncu_error"] = res.get("error")
    s = res.get("summary")
    if s:
        out.update(l2_hit_rate=s["hit_rate"], dram_bytes_per_token=s["dram_bytes_per_token"],
                   ncu_us_per_token=s["us_per_token"], n_kernels=s["n_kernels"])
    return out


def inner_env() -> dict:
    """Env for ncu child processes: same results dir, no extra output buffering."""
    return {"PYTHONUNBUFFERED": "1"}


def py_module_argv(module: str, *args: str) -> list[str]:
    return [sys.executable, "-m", module, *args]


def clear_capture_windows(*streams) -> None:
    """Clear the access window on streams and on torch's default graph capture stream.

    torch.cuda.Stream() comes from a round-robin pool, so a stream with an old window can come back later.
    """
    import torch
    from harness import l2persist
    for s in (*streams, torch.cuda.graph.default_capture_stream):
        if s is not None:
            l2persist.clear_stream_window(s)


def window_readback(obj, graph: bool = False) -> dict | None:
    """Window the driver reports for a stream or a keep_graph template (first kernel node); None on any error."""
    try:
        from harness import l2persist
        return l2persist.get_graph_window(obj) if graph else l2persist.get_stream_window(obj)
    except Exception:
        return None


def env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    return default if v is None else v not in ("", "0", "false", "False")
