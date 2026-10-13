"""Knee plot for E1: us/token vs working set W, plus DRAM bytes ratio, with the rtx3090 control overlay.

  python -m exps.plot_e1 [--results results/e1] [--out results/e1/e1_knee.png]

Reads the newest result JSON per GPU directory under --results. One column per non-control GPU;
the rtx3090 control (if present) is drawn as dashed grey lines in every column.
Row 1: us/token vs W (log2 x), one line per mode, vertical line at the GPU persisting max.
Row 2: dram_bytes_per_token / bytes_per_token.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

from exps import common

CONTROL = "rtx3090"
MIB = 1 << 20


def load_latest(results_dir) -> dict[str, dict]:
    """{gpu_dir_name: parsed json} using the newest *.json (by name, which sorts by UTC stamp) per GPU."""
    root = pathlib.Path(results_dir)
    out = {}
    if not root.is_dir():
        return out
    for d in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        files = sorted(f for f in d.glob("*.json") if f.is_file())
        for f in reversed(files):
            try:
                doc = json.loads(f.read_text())
            except (OSError, ValueError):
                continue
            if isinstance(doc, dict) and doc.get("data", {}).get("rows"):
                out[d.name] = doc
                break
    return out


def series(doc: dict) -> dict[str, list[dict]]:
    """{mode: rows sorted by W} for rows with a timing."""
    by: dict[str, list[dict]] = {}
    for r in doc["data"]["rows"]:
        if r.get("us_per_token") is None:
            continue
        by.setdefault(str(r["mode"]), []).append(r)
    for rows in by.values():
        rows.sort(key=lambda r: r["W_mib"])
    return by


def mode_order(modes) -> list[str]:
    return sorted(modes, key=lambda m: -1.0 if m == "off" else float(m))


def persist_max_mib(doc: dict) -> float | None:
    v = (doc.get("env") or {}).get("persisting_l2_max_bytes")
    return v / MIB if v else None


def plot(docs: dict[str, dict], out: str | pathlib.Path) -> pathlib.Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    main = [g for g in docs if g != CONTROL] or list(docs)
    ctl = docs.get(CONTROL) if CONTROL in docs and CONTROL not in main else None
    fig, axes = plt.subplots(2, len(main), figsize=(6.2 * len(main), 8.0), squeeze=False)
    cmap = plt.get_cmap("viridis")
    for col, gpu in enumerate(main):
        doc = docs[gpu]
        ser = series(doc)
        modes = mode_order(ser)
        colors = {m: cmap(i / max(1, len(modes) - 1) * 0.85) for i, m in enumerate(modes)}
        ax1, ax2 = axes[0][col], axes[1][col]
        for m in modes:
            xs = [r["W_mib"] for r in ser[m]]
            ax1.plot(xs, [r["us_per_token"] for r in ser[m]], marker="o", ms=4, color=colors[m],
                     label="off" if m == "off" else f"hitRatio {m}")
            pts = [(r["W_mib"], r["dram_bytes_per_token"] / r["bytes_per_token"]) for r in ser[m]
                   if r.get("dram_bytes_per_token") is not None and r.get("bytes_per_token")]
            if pts:
                ax2.plot(*zip(*pts), marker="o", ms=4, color=colors[m])
        if ctl is not None:
            cser = series(ctl)
            for m in mode_order(cser):
                if m in ("off", "1.0"):
                    ax1.plot([r["W_mib"] for r in cser[m]], [r["us_per_token"] for r in cser[m]], ls="--",
                             color="0.55", marker="x", ms=4, label=f"{CONTROL} {m}")
                    pts = [(r["W_mib"], r["dram_bytes_per_token"] / r["bytes_per_token"]) for r in cser[m]
                           if r.get("dram_bytes_per_token") is not None and r.get("bytes_per_token")]
                    if pts:
                        ax2.plot(*zip(*pts), ls="--", color="0.55", marker="x", ms=4)
        pm = persist_max_mib(doc)
        for ax in (ax1, ax2):
            ax.set_xscale("log", base=2)
            ax.grid(True, alpha=0.3)
            if pm:
                ax.axvline(pm, color="tab:red", ls=":", lw=1.2)
        if pm:
            ax1.text(pm, ax1.get_ylim()[1], f" persisting max {pm:.0f} MiB", color="tab:red", va="top", fontsize=8)
        ax1.set_title(f"{gpu}: us/token vs working set")
        ax1.set_ylabel("us / token")
        ax1.legend(fontsize=8)
        ax2.set_xlabel("W (MiB of ternary weights per token)")
        ax2.set_ylabel("DRAM bytes read / bytes per token")
        ax2.set_title(f"{gpu}: DRAM traffic ratio")
    fig.tight_layout()
    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="exps.plot_e1", description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=str(common.EXP_DIR / "results" / "e1"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    docs = load_latest(a.results)
    if not docs:
        print(f"error: no usable e1 results under {a.results}", file=sys.stderr)
        return 1
    out = a.out or str(pathlib.Path(a.results) / "e1_knee.png")
    print(plot(docs, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
