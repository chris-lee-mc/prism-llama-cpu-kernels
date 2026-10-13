"""Plots for E3d (resident ternary draft): draft step time, on-vs-off gain, refill after verify, projected speculative speedup.

  python -m exps.plot_e3d [--results results/e3d] [--out-dir results/e3d] [--K 4] [--pdl 1]

Reads the newest e3d result JSON per GPU directory. One PNG per GPU and PDL setting: <out-dir>/e3d_<gpu>_pdl<p>_K<k>.png.
Panels: (1) draft us/step vs D (S1 steady, S2 later steps, S2 first step; off grey, on blue),
(2) gain of on vs off (%) with the paired-delta noise as error bars, (3) refill penalty first - later step (us),
(4) projected speculative speedup (S2 source) for a = 0.6, 0.7, 0.8; off dashed, on solid.
Vertical dotted line: device persisting max.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

from exps import common

MIB = 1 << 20
OFF, ON = "#52514e", "#2a78d6"
ACC_COLORS = ["#eb6834", "#1baf7a", "#6250d6"]


def load_latest(results_dir) -> dict[str, dict]:
    """{gpu_dir_name: parsed json} of the newest e3d result per GPU directory."""
    root = pathlib.Path(results_dir)
    out = {}
    if not root.is_dir():
        return out
    for d in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        for f in reversed(sorted(f for f in d.glob("*.json") if f.is_file())):
            try:
                doc = json.loads(f.read_text())
            except (OSError, ValueError):
                continue
            rows = doc.get("data", {}).get("rows") if isinstance(doc, dict) else None
            if rows and all("scenario" in r for r in rows):
                out[d.name] = doc
                break
    return out


def pdl_settings(doc: dict) -> list[int]:
    return sorted({r["pdl"] for r in doc["data"]["rows"] if r.get("pdl") is not None})


def series(doc: dict, pdl: int, scn: str, k: int, mode: str, key: str) -> list[tuple[float, float, float, float]]:
    """[(D_mib, value, lo, hi)] sorted by D; lo/hi from the per-round values."""
    out = []
    for r in doc["data"]["rows"]:
        if (r.get("pdl"), r["scenario"], r["K"], r["mode"]) == (pdl, scn, k, mode) and r.get(key) is not None:
            a = r.get(key + "_all") or [r[key]]
            out.append((r["D_mib"], r[key], r[key] - min(a), max(a) - r[key]))
    return sorted(out)


def gain_series(doc: dict, pdl: int, scn: str, k: int, key: str) -> list[tuple[float, float, float]]:
    """[(D_mib, gain %, noise %)] of mode on vs off."""
    out = []
    for r in doc["data"]["rows"]:
        if (r.get("pdl"), r["scenario"], r["K"], r["mode"]) == (pdl, scn, k, "on") and r.get("gain_" + key) is not None:
            base = r["delta_" + key] / r["gain_" + key] if r["gain_" + key] else None
            noise = 100 * (r.get("noise_" + key) or 0.0) / base if base else 0.0
            out.append((r["D_mib"], 100 * r["gain_" + key], noise))
    return sorted(out)


def refill_series(doc: dict, pdl: int, scn: str, k: int, mode: str) -> list[tuple[float, float]]:
    return [(d, v) for d, v, _, _ in series(doc, pdl, scn, k, mode, "refill_us")]


def proj_series(doc: dict, pdl: int, k: int, acc: float, key: str, src: str = "S2") -> list[tuple[float, float]]:
    return sorted((p["D_mib"], p[key]) for p in doc["data"].get("projection", [])
                  if (p["pdl"], p["K"], p["source"]) == (pdl, k, src) and abs(p["acc"] - acc) < 1e-9)


def plot(doc: dict, out, pdl: int, k: int) -> pathlib.Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(12.5, 9.0))
    pm = (doc.get("env") or {}).get("persisting_l2_max_bytes")
    gpu = doc.get("gpu") or ""
    # 1: step time
    a = ax[0][0]
    for scn, key, ls, name in (("S1", "step_us", "-", "S1 steady"), ("S2", "later_us", "--", "S2 later steps"),
                               ("S2", "first_us", ":", "S2 first step")):
        for mode, col in (("off", OFF), ("on", ON)):
            s = series(doc, pdl, scn, k, mode, key)
            if s:
                a.errorbar([x[0] for x in s], [x[1] for x in s], yerr=[[x[2] for x in s], [x[3] for x in s]], color=col, ls=ls,
                           marker="o", ms=4, lw=1.5, capsize=2, label=f"{name} {mode}")
    a.set_title(f"{gpu} PDL={pdl} K={k}: draft us / step")
    a.set_ylabel("us / step")
    # 2: gain
    a = ax[0][1]
    for (scn, key, name), col in zip((("S1", "step_us", "S1 steady"), ("S2", "first_us", "S2 first"), ("S2", "later_us", "S2 later"),
                                      ("S3", "first_us", "S3 first")), ("#2a78d6", "#eb6834", "#1baf7a", "#6250d6")):
        g = gain_series(doc, pdl, scn, k, key)
        if g:
            a.errorbar([x[0] for x in g], [x[1] for x in g], yerr=[x[2] for x in g], color=col, marker="o", ms=4, lw=1.5,
                       capsize=2, label=name)
    a.axhline(0, color="0.7", lw=0.8)
    a.set_title("gain of persisting on vs off (%)")
    a.set_ylabel("gain %")
    # 3: refill
    a = ax[1][0]
    for scn, ls in (("S2", "-"), ("S3", "--")):
        for mode, col in (("off", OFF), ("on", ON)):
            s = refill_series(doc, pdl, scn, k, mode)
            if s:
                a.plot([x[0] for x in s], [x[1] for x in s], color=col, ls=ls, marker="o", ms=4, lw=1.5, label=f"{scn} {mode}")
    a.axhline(0, color="0.7", lw=0.8)
    a.set_title("first step after verify minus later steps (us)")
    a.set_ylabel("us")
    # 4: projection
    a = ax[1][1]
    for acc, col in zip((0.6, 0.7, 0.8), ACC_COLORS):
        for key, ls in (("speedup_off", "--"), ("speedup_on", "-")):
            s = proj_series(doc, pdl, k, acc, key)
            if s:
                a.plot([x[0] for x in s], [x[1] for x in s], color=col, ls=ls, marker="o", ms=4, lw=1.5,
                       label=f"a={acc} {key.split('_')[1]}")
    a.axhline(1, color="0.7", lw=0.8)
    a.set_title("projected speculative speedup vs target-only (S2 timings)")
    a.set_ylabel("speedup x")
    for a in ax.flat:
        a.set_xlabel("draft working set D (MiB)")
        a.set_xscale("log", base=2)
        a.grid(True, alpha=0.25)
        if pm:
            a.axvline(pm / MIB, color="#e34948", ls=":", lw=1.2)
        if a.get_legend_handles_labels()[0]:
            a.legend(fontsize=7)
    fig.tight_layout()
    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="exps.plot_e3d", description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=str(common.EXP_DIR / "results" / "e3d"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--pdl", type=int, default=None, help="one PDL setting (default: all present)")
    a = ap.parse_args(argv)
    docs = load_latest(a.results)
    if not docs:
        print(f"error: no usable e3d results under {a.results}", file=sys.stderr)
        return 1
    out_dir = pathlib.Path(a.out_dir or a.results)
    for gpu, doc in docs.items():
        for p in ([a.pdl] if a.pdl is not None else pdl_settings(doc)):
            print(plot(doc, out_dir / f"e3d_{gpu}_pdl{p}_K{a.K}.png", p, a.K))
    return 0


if __name__ == "__main__":
    sys.exit(main())
