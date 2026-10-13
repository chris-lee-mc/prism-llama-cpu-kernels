"""Knee plot for E1 (hot/cold design): hot us/token vs hot set W, off vs on, with the rtx3090 control overlay.

  python -m exps.plot_e1 [--results results/e1] [--out results/e1/e1_knee.png]

Reads the newest hot/cold result JSON per GPU directory (rows with C_mib). One column per non-control GPU;
the rtx3090 control (if present) is drawn dashed grey (off and 1.0) in every column.
Row 1: hot us/token with the cold set streamed (split timing), one line per mode, round min-max as error bars.
Row 2: hot us/token with C=0 (hot only, natural L2 residency).
Row 3: hot gain vs off (%), C > 0, with the paired-delta noise as error bars; dotted line at the 20% stop rule.
Vertical dotted line: device persisting max.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

from exps import common

CONTROL = "rtx3090"
MIB = 1 << 20
COLORS = ["#52514e", "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6", "#e34948"]  # off first


def load_latest(results_dir) -> dict[str, dict]:
    """{gpu_dir_name: parsed json} using the newest hot/cold *.json (by name, which sorts by UTC stamp) per GPU."""
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
            if rows and all("C_mib" in r for r in rows):
                out[d.name] = doc
                break
    return out


def cold_c(doc: dict) -> int:
    return max((r["C_mib"] for r in doc["data"]["rows"]), default=0)


def series(doc: dict, c: int, key: str = "hot_us") -> dict[str, list[dict]]:
    """{mode: rows at C=c sorted by W} for rows with `key`."""
    by: dict[str, list[dict]] = {}
    for r in doc["data"]["rows"]:
        if r["C_mib"] == c and r.get(key) is not None:
            by.setdefault(str(r["mode"]), []).append(r)
    for rows in by.values():
        rows.sort(key=lambda r: r["W_mib"])
    return by


def mode_order(modes) -> list[str]:
    def key(m):
        return (-1.0, 0) if m == "off" else (float(m.rstrip("L")), m.endswith("L"))
    return sorted(modes, key=key)


def mode_label(m: str) -> str:
    if m == "off":
        return "off"
    return f"hitRatio {m.rstrip('L')}, limit min(W, max)" if m.endswith("L") else f"hitRatio {m}"


def persist_max_mib(doc: dict) -> float | None:
    v = (doc.get("env") or {}).get("persisting_l2_max_bytes") or ((doc["data"].get("config") or {}).get("device") or {}).get("persisting_max_bytes")
    return v / MIB if v else None


def _err(rows, key):
    lo, hi = [], []
    for r in rows:
        a = r.get(key + "_all") or [r[key]]
        lo.append(r[key] - min(a))
        hi.append(max(a) - r[key])
    return [lo, hi]


def _gain_pts(rows):
    pts = [(r["W_mib"], 100 * r["hot_gain_vs_off"], 100 * (r.get("hot_delta_noise_us") or 0.0) / (r["hot_us"] + r["hot_delta_vs_off_us"]))
           for r in rows if r.get("hot_gain_vs_off") is not None]
    return list(zip(*pts)) if pts else None


def plot(docs: dict[str, dict], out: str | pathlib.Path) -> pathlib.Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    main = [g for g in docs if g != CONTROL] or list(docs)
    ctl = docs.get(CONTROL) if CONTROL in docs and CONTROL not in main else None
    fig, axes = plt.subplots(3, len(main), figsize=(6.4 * len(main), 11.0), squeeze=False)
    for col, gpu in enumerate(main):
        doc = docs[gpu]
        c = cold_c(doc)
        panels = [(axes[0][col], c, f"{gpu}: hot us/token, cold {c} MiB streamed"),
                  (axes[1][col], 0, f"{gpu}: hot us/token, C=0 (hot only)")]
        for ax, cc, title in panels:
            ser = series(doc, cc)
            for i, m in enumerate(mode_order(ser)):
                rs = ser[m]
                ax.errorbar([r["W_mib"] for r in rs], [r["hot_us"] for r in rs], yerr=_err(rs, "hot_us"), marker="o", ms=4,
                            lw=1.6, capsize=2, color=COLORS[i % len(COLORS)], label=mode_label(m))
            if ctl is not None:
                cser = series(ctl, cold_c(ctl) if cc else 0)
                for m, ls in (("off", "--"), ("1.0", ":")):
                    if m in cser:
                        ax.plot([r["W_mib"] for r in cser[m]], [r["hot_us"] for r in cser[m]], ls=ls, color="0.6",
                                marker="x", ms=4, label=f"{CONTROL} {m}")
            ax.set_title(title, fontsize=10)
            ax.set_ylabel("us / token (hot part)")
            ax.set_yscale("log")
        ax3 = axes[2][col]
        ser = series(doc, c)
        for i, m in enumerate(mode_order(ser)):
            if m == "off":
                continue
            g = _gain_pts(ser[m])
            if g:
                ax3.errorbar(g[0], g[1], yerr=g[2], marker="o", ms=4, lw=1.6, capsize=2, color=COLORS[i % len(COLORS)],
                             label=mode_label(m))
        if ctl is not None:
            g = _gain_pts(series(ctl, cold_c(ctl)).get("1.0", []))
            if g:
                ax3.plot(g[0], g[1], ls=":", color="0.6", marker="x", ms=4, label=f"{CONTROL} 1.0")
        ax3.axhline(20, color="#e34948", ls=":", lw=1.0)
        ax3.axhline(0, color="0.7", lw=0.8)
        ax3.set_title(f"{gpu}: hot gain vs off (cold {c} MiB)", fontsize=10)
        ax3.set_ylabel("gain %")
        ax3.set_xlabel("W (MiB of ternary weights in the hot set)")
        pm = persist_max_mib(doc)
        for ax in (axes[0][col], axes[1][col], ax3):
            ax.set_xscale("log", base=2)
            ax.grid(True, alpha=0.25)
            if pm:
                ax.axvline(pm, color="#e34948", ls=":", lw=1.2)
            ax.legend(fontsize=7)
        if pm:
            axes[0][col].text(pm, axes[0][col].get_ylim()[1], f" persisting max {pm:.0f} MiB", color="#e34948", va="top", fontsize=8)
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
