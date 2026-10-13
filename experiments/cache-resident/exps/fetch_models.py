"""Download the GGUFs of several drivers in parallel before they run, so the download does not eat their time budget.

  python -m exps.fetch_models [--models pq2,ptq1,draft_q8_0,draft_bf16]
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from exps import e4_pdl


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="exps.fetch_models", description=__doc__.split("\n")[0])
    ap.add_argument("--models", default=",".join(e4_pdl.MODELS))
    a = ap.parse_args(argv)
    names = [m for m in a.models.split(",") if m]
    t0 = time.monotonic()

    def one(m):
        notes: list = []
        try:
            return m, e4_pdl.model_path(m, notes)[0], notes
        except Exception as e:
            return m, None, notes + [f"{type(e).__name__}: {e}"]
    with ThreadPoolExecutor(len(names) or 1) as ex:
        res = list(ex.map(one, names))
    for m, path, notes in res:
        print(f"[fetch_models] {m}: {path} {notes if notes else ''}", flush=True)
    print(f"[fetch_models] {time.monotonic() - t0:.0f} s", flush=True)
    return 0 if all(p for _, p, _ in res) else 1


if __name__ == "__main__":
    sys.exit(main())
