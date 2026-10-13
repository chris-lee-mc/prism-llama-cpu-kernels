"""Download the GGUFs of several drivers in parallel before they run, so the download does not eat their time budget.

  python -m exps.fetch_models [--models pq2,ptq1,draft_q8_0,draft_bf16] [--timeout-min 12]

A stalled download ends the command with exit 1 after --timeout-min, so a chained driver does not start and the runner timeout
is not what stops the pod.
"""
from __future__ import annotations

import argparse
import sys
import threading
import time

from exps import e4_pdl

DEFAULT_TIMEOUT_MIN = 12.0


def fetch_all(names: list[str], timeout_s: float, fetch=None) -> dict:
    """{name: (path or None, notes)}. A model still downloading at the deadline gets (None, ["timeout ..."]); its thread is a
    daemon thread, so it cannot hold the process open."""
    fetch = fetch or (lambda m, notes: e4_pdl.model_path(m, notes)[0])
    res: dict = {}

    def one(m: str) -> None:
        notes: list = []
        try:
            res[m] = (fetch(m, notes), notes)
        except Exception as e:
            res[m] = (None, notes + [f"{type(e).__name__}: {e}"])
    threads = {m: threading.Thread(target=one, args=(m,), daemon=True) for m in names}
    for t in threads.values():
        t.start()
    t_end = time.monotonic() + timeout_s
    for m, t in threads.items():
        t.join(max(t_end - time.monotonic(), 0.0))
        if t.is_alive():
            res[m] = (None, [f"timeout: not finished after {timeout_s / 60:.1f} min"])
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="exps.fetch_models", description=__doc__.split("\n")[0])
    ap.add_argument("--models", default=",".join(e4_pdl.MODELS))
    ap.add_argument("--timeout-min", type=float, default=DEFAULT_TIMEOUT_MIN)
    a = ap.parse_args(argv)
    names = [m for m in a.models.split(",") if m]
    t0 = time.monotonic()
    res = fetch_all(names, a.timeout_min * 60.0)
    for m in names:
        path, notes = res[m]
        print(f"[fetch_models] {m}: {path} {notes if notes else ''}", flush=True)
    print(f"[fetch_models] {time.monotonic() - t0:.0f} s", flush=True)
    return 0 if all(res[m][0] for m in names) else 1


if __name__ == "__main__":
    sys.exit(main())
