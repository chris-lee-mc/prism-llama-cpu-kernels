"""Re-implementation of the proposal's projection; original proj.py/proj_hybrid.py not in repo.

Small roofline for batch-1 decode of a hybrid (GDN + attention) model, with optional persisting-L2 residency:

  l2_bytes   = min(persisted_bytes, persist_max_bytes) * hit
  dram_bytes = weight_bytes + head_bytes + 2 * state_bytes + kv_bytes_per_ctx_token * ctx - l2_bytes
  us/token   = n_layers * floor_us_per_layer + dram_bytes / bw_dram + l2_bytes / bw_l2

Speculative decoding: E[tokens per verify] = (1 - a^(k+1)) / (1 - a); tok/s = E / (k * t_draft + t_verify(k+1)).
t_verify is taken as one target decode step (weight bound: a batch of k+1 reads the weights once).

Inputs are measured numbers: floor us/layer and effective DRAM bandwidth from E2 (fit) or P0.5 (toy hybrid: floor_us_per_layer + eff_gbps_net),
L2 bandwidth from E1 (not measured yet: default is an assumed multiple of DRAM bandwidth, flagged in output).

  python -m exps.proj_hybrid --floor-us F --bw-gbps B [--l2-gbps X] [--persist-mb 72] [--hit 1.0]
                             [--from-results DIR] [--models a,b] [--ctx 0,32768] [--write-result]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import warnings
from dataclasses import dataclass, asdict

from exps import common, modelspec

TERNARY_BITS = 2.125  # 2-bit trits + one fp16 scale per 128 weights
L2_BW_RATIO = 3.0     # assumed L2 / DRAM bandwidth when E1 has not measured it


@dataclass
class ModelSpec:
    name: str
    n_layers: int
    weight_bytes: int  # body weights read per token (no LM head)
    head_bytes: int
    state_bytes: int   # GDN recurrent state, read and written each token
    kv_bytes_per_ctx_token: int


@dataclass
class HwSpec:
    name: str
    bw_dram_gbps: float
    bw_l2_gbps: float
    floor_us_per_layer: float
    persist_max_bytes: int


def _from_facts(name: str, key: str, bits_per_weight: float | None = None) -> ModelSpec:
    f = modelspec.facts(modelspec.TEXT_CFG[key])
    if bits_per_weight is None:
        body, head = f["body_weight_bytes"], f["head_bytes"]
    else:
        scale = bits_per_weight / 16.0
        body, head = int(f["body_weight_bytes"] * scale), int(f["head_bytes"] * scale)
    return ModelSpec(name, f["n_layers"], body, head, f["state_bytes"], f["kv_bytes_per_token"])


def _models() -> dict[str, ModelSpec]:
    b27 = modelspec.facts(modelspec.TEXT_CFG["27B"])
    return {
        "qwen3.5-0.8b": _from_facts("qwen3.5-0.8b", "0.8B"),
        "qwen3.5-2b": _from_facts("qwen3.5-2b", "2B"),
        "ternary-0.8b-g128": _from_facts("ternary-0.8b-g128", "0.8B", TERNARY_BITS),
        # Bonsai 2 27B PQ2_0: whole 7.21 GB file counted as body (head included), f16 KV.
        "bonsai2-27b-pq2_0": ModelSpec("bonsai2-27b-pq2_0", b27["n_layers"], int(7.21e9), 0, b27["state_bytes"],
                                       b27["kv_bytes_per_token"]),
    }


MODELS: dict[str, ModelSpec] = _models()


def decode_us(m: ModelSpec, hw: HwSpec, ctx: int = 0, persisted_bytes: int = 0, hit: float = 1.0) -> dict:
    """Projected decode time per token. Returns {us, dram_bytes, l2_bytes, floor_us, dram_us, l2_us}."""
    total = m.weight_bytes + m.head_bytes + 2 * m.state_bytes + m.kv_bytes_per_ctx_token * ctx
    l2 = min(max(persisted_bytes, 0), hw.persist_max_bytes) * hit
    l2 = min(l2, total)
    dram = total - l2
    floor = m.n_layers * hw.floor_us_per_layer
    dram_us = dram / (hw.bw_dram_gbps * 1e3)
    l2_us = l2 / (hw.bw_l2_gbps * 1e3)
    return {"us": floor + dram_us + l2_us, "dram_bytes": dram, "l2_bytes": l2, "floor_us": floor,
            "dram_us": dram_us, "l2_us": l2_us}


def spec_tokens_per_verify(a: float, k: int) -> float:
    if abs(1.0 - a) < 1e-12:
        return float(k + 1)
    return (1.0 - a ** (k + 1)) / (1.0 - a)


def spec_tok_s(a: float, k: int, t_draft_us: float, t_verify_us: float) -> float:
    """Tokens per second with acceptance a, k drafted tokens, per-token draft time and one verify step (us)."""
    return spec_tokens_per_verify(a, k) / (k * t_draft_us + t_verify_us) * 1e6


def hw_from_results(path, l2_ratio: float = L2_BW_RATIO, notes: list | None = None) -> HwSpec:
    """HwSpec from an E2 result (data.fit) or a P0.5 toy result. path: file or dir.

    Toy rows: floor_us_per_layer + eff_gbps_net (floor removed from the bandwidth) are preferred over
    us_per_layer_graph + eff_gbps, which count the floor twice. Warnings go to notes when given.
    """
    p = pathlib.Path(path)
    if p.is_dir():
        skip = {"_status", "raw", "logs"}  # raw/partial.json sorts after the final <stamp>.json: never pick a snapshot
        files = sorted(f for f in p.rglob("*.json")
                       if not skip & set(f.relative_to(p).parts[:-1]) and not f.name.startswith("partial"))
        if not files:
            raise FileNotFoundError(f"no final result json under {p}")
        p = files[-1]
    doc = json.loads(p.read_text())
    data = doc.get("data", doc)
    env = doc.get("env") or {}
    fit = data.get("fit") or {}
    floor = fit.get("floor_us_per_layer")
    bw = fit.get("eff_bw_gbps")
    warns: list[str] = []
    impl = fit.get("gdn_impl") or ((data.get("config") or {}).get("load") or {}).get("gdn_impl")
    if floor is not None and bw is not None and impl not in (None, "fla"):
        warns.append(f"fit comes from gdn_impl={impl} (HF torch ops, not a fused GDN kernel): floor and bandwidth are not representative")
    if floor is None or bw is None:
        rows = data.get("rows") or []
        src = [data] + rows[:1]
        pick = lambda key: next((s[key] for s in src if s.get(key) is not None), None)
        floor, bw = pick("floor_us_per_layer"), pick("eff_gbps_net")
        if floor is None or bw is None:
            floor, bw = pick("us_per_layer_graph"), pick("eff_gbps")
            if floor is not None and bw is not None:
                warns.append("toy result has no floor_us_per_layer/eff_gbps_net: us_per_layer_graph and eff_gbps count the floor twice")
        if any(s.get("l2_assisted") for s in src):
            warns.append("toy result is L2-assisted: bandwidth is not a DRAM number")
    if floor is None or bw is None:
        raise ValueError(f"{p}: no fit.floor_us_per_layer/eff_bw_gbps or floor_us_per_layer/eff_gbps_net or us_per_layer_graph/eff_gbps")
    for w in warns:
        warnings.warn(w)
        if notes is not None:
            notes.append(w)
    return HwSpec(env.get("gpu_tag") or doc.get("gpu") or "measured", float(bw), float(bw) * l2_ratio, float(floor),
                  int(env.get("persisting_l2_max_bytes") or 0))


def project(models: list[str], hw: HwSpec, ctxs: list[int], hit: float) -> list[dict]:
    """For each model and ctx: streamed vs persisting state+head (capped at persist max)."""
    rows = []
    for name in models:
        m = MODELS[name]
        for ctx in ctxs:
            base = decode_us(m, hw, ctx)
            want = m.head_bytes + 2 * m.state_bytes if m.head_bytes else 2 * m.state_bytes
            res = decode_us(m, hw, ctx, want, hit)
            rows.append({"model": name, "ctx": ctx, "streamed_us": base["us"], "streamed_tok_s": 1e6 / base["us"],
                         "persisted_want_bytes": want, "persisted_us": res["us"], "persisted_tok_s": 1e6 / res["us"],
                         "dram_bytes_streamed": base["dram_bytes"], "dram_bytes_persisted": res["dram_bytes"],
                         "floor_us": base["floor_us"], "gain_pct": 100.0 * (base["us"] - res["us"]) / base["us"]})
    return rows


def spec_table(target: str, draft: str, hw: HwSpec, ctx: int, ks=(4, 6, 8), accs=(0.5, 0.7, 0.8, 0.9)) -> list[dict]:
    t_verify = decode_us(MODELS[target], hw, ctx)["us"]
    t_draft = decode_us(MODELS[draft], hw, 0)["us"]
    return [{"a": a, "k": k, "tok_s": spec_tok_s(a, k, t_draft, t_verify), "t_draft_us": t_draft, "t_verify_us": t_verify}
            for k in ks for a in accs]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="exps.proj_hybrid", description=__doc__.split("\n")[0])
    ap.add_argument("--floor-us", type=float, help="measured floor, us per layer")
    ap.add_argument("--bw-gbps", type=float, help="measured effective DRAM bandwidth, GB/s")
    ap.add_argument("--l2-gbps", type=float, default=None, help="L2 bandwidth; default %.1fx DRAM (assumption)" % L2_BW_RATIO)
    ap.add_argument("--persist-mb", type=float, default=None, help="persisting L2 max in MiB")
    ap.add_argument("--hit", type=float, default=1.0)
    ap.add_argument("--from-results", default=None, help="E2 or P0.5 result file or directory")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--ctx", default="0,32768")
    ap.add_argument("--write-result", action="store_true")
    a = ap.parse_args(argv)
    notes = []
    if a.from_results:
        hw = hw_from_results(a.from_results, notes=notes)
        if a.floor_us is not None:
            hw.floor_us_per_layer = a.floor_us
        if a.bw_gbps is not None:
            hw.bw_dram_gbps = a.bw_gbps
    elif a.floor_us is not None and a.bw_gbps is not None:
        hw = HwSpec("cli", a.bw_gbps, 0.0, a.floor_us, 0)
    else:
        ap.error("give --floor-us and --bw-gbps, or --from-results")
    if a.l2_gbps:
        hw.bw_l2_gbps = a.l2_gbps
    else:
        hw.bw_l2_gbps = hw.bw_dram_gbps * L2_BW_RATIO
        notes.append("L2 bandwidth is an assumption (%.1fx DRAM); replace with the E1 measurement" % L2_BW_RATIO)
    if a.persist_mb is not None:
        hw.persist_max_bytes = int(a.persist_mb * common.MIB)
    elif not hw.persist_max_bytes:
        hw.persist_max_bytes = 72 * common.MIB
        notes.append("persisting L2 max assumed 72 MiB; pass --persist-mb with the measured value")
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    ctxs = [int(c) for c in a.ctx.split(",") if c.strip() != ""]
    rows = project(models, hw, ctxs, a.hit)
    spec = spec_table("bonsai2-27b-pq2_0", "qwen3.5-0.8b", hw, 0) if "bonsai2-27b-pq2_0" in MODELS else []
    print("hw:", json.dumps(asdict(hw)))
    print("%-20s %8s %12s %10s %12s %10s %8s" % ("model", "ctx", "streamed_us", "tok/s", "persisted_us", "tok/s", "gain%"))
    for r in rows:
        print("%-20s %8d %12.1f %10.1f %12.1f %10.1f %8.1f" % (r["model"], r["ctx"], r["streamed_us"], r["streamed_tok_s"],
                                                               r["persisted_us"], r["persisted_tok_s"], r["gain_pct"]))
    print("speculation (target bonsai2-27b-pq2_0, draft qwen3.5-0.8b, ctx 0):")
    for s in spec:
        print("  k=%d a=%.2f -> %.1f tok/s" % (s["k"], s["a"], s["tok_s"]))
    for n in notes:
        print("note:", n)
    if a.write_result:
        from infra.results import write_result
        payload = {"config": {"hw": asdict(hw), "models": models, "ctx": ctxs, "hit": a.hit, "reimplementation": True},
                   "rows": rows, "spec": spec, "notes": notes + ["re-implementation; original proj_hybrid.py not in repo"],
                   "counters_ok": False}
        print(write_result("proj_hybrid", None, payload))
    return 0


if __name__ == "__main__":
    sys.exit(main())
