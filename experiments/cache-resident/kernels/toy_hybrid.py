"""P0.5 toy hybrid decode step (GDN + attention + ternary MLP) in a CUDA graph.

python -m kernels.toy_hybrid [--layers 24] [--d 1024] [--attn-every 4] [--H 16] [--dk 128] [--dv 128]
    [--Hq 8] [--Hkv 2] [--D 256] [--F 3584] [--ctx 4096] [--tokens 200] [--variant lut|fp4] [--result-exp p0_toy]

Eager mode replays a precomputed launch list from Python (best-case eager). The graph captures the same list.
Weights are random; the residual stream is not meant to be meaningful, only the traffic and launch pattern.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from dataclasses import dataclass

import torch

from kernels.ternary import QK, bytes_moved, random_ternary

ARENA_ALIGN = 256
EPS = 1e-6


@dataclass
class ToyConfig:
    layers: int = 24
    d: int = 1024
    attn_every: int = 4
    H: int = 16
    dk: int = 128
    dv: int = 128
    Hq: int = 8
    Hkv: int = 2
    D: int = 256
    F: int = 3584
    ctx: int = 4096
    kinds: tuple | None = None  # explicit per-layer "gdn" / "attn", overrides attn_every

    def layer_kinds(self) -> list[str]:
        if self.kinds is not None:
            return list(self.kinds)
        return ["attn" if (i + 1) % self.attn_every == 0 else "gdn" for i in range(self.layers)]

    @property
    def scale(self) -> float:
        return self.D ** -0.5


def validate(cfg: ToyConfig) -> None:
    kinds = cfg.layer_kinds()
    if len(kinds) != cfg.layers or any(k not in ("gdn", "attn") for k in kinds):
        raise ValueError("kinds must hold 'gdn' or 'attn' for each layer")
    for name, k in (("d", cfg.d), ("H*dv", cfg.H * cfg.dv), ("Hq*D", cfg.Hq * cfg.D), ("F", cfg.F)):
        if k % QK:
            raise ValueError(f"{name}={k} must be a multiple of {QK}")
    if cfg.dk not in (32, 64, 128, 256) or cfg.D not in (64, 128, 256):
        raise ValueError("dk must be in {32,64,128,256} and D in {64,128,256}")
    if cfg.Hq % cfg.Hkv or cfg.Hq // cfg.Hkv > 8 or cfg.Hq > 32:
        raise ValueError("need Hq % Hkv == 0, Hq / Hkv <= 8, Hq <= 32")


def matrix_specs(cfg: ToyConfig) -> list[dict]:
    """Ternary matrices in launch order: {name, layer, N, K}."""
    out = []
    d, F = cfg.d, cfg.F
    for i, kind in enumerate(cfg.layer_kinds()):
        if kind == "gdn":
            out += [
                dict(name=f"l{i}.qkv", layer=i, N=cfg.H * (2 * cfg.dk + cfg.dv), K=d),
                dict(name=f"l{i}.ab", layer=i, N=2 * cfg.H, K=d),
                dict(name=f"l{i}.out", layer=i, N=d, K=cfg.H * cfg.dv),
            ]
        else:
            out += [
                dict(name=f"l{i}.q", layer=i, N=cfg.Hq * cfg.D, K=d),
                dict(name=f"l{i}.kv", layer=i, N=2 * cfg.Hkv * cfg.D, K=d),
                dict(name=f"l{i}.o", layer=i, N=d, K=cfg.Hq * cfg.D),
            ]
        out += [
            dict(name=f"l{i}.gate", layer=i, N=F, K=d),
            dict(name=f"l{i}.up", layer=i, N=F, K=d),
            dict(name=f"l{i}.down", layer=i, N=d, K=F),
        ]
    return out


def _align(n: int) -> int:
    return (n + ARENA_ALIGN - 1) // ARENA_ALIGN * ARENA_ALIGN


def arena_layout(cfg: ToyConfig) -> tuple[dict, int]:
    """Byte offsets (256 B aligned) in one arena: ternary q and s per matrix, fp16 norm weights per layer."""
    off = 0
    ent: dict = {}
    for m in matrix_specs(cfg):
        N, K = m["N"], m["K"]
        qb, sb = N * K // 4, N * (K // QK) * 2
        ent[m["name"]] = dict(N=N, K=K, q_off=off, s_off=_align(off + qb))
        off = _align(ent[m["name"]]["s_off"] + sb)
    for i in range(cfg.layers):
        for tag in ("n1", "n2"):
            ent[f"l{i}.{tag}"] = dict(w_off=off, n=cfg.d)
            off = _align(off + cfg.d * 2)
    return ent, off


def n_gdn_layers(cfg: ToyConfig) -> int:
    return sum(k == "gdn" for k in cfg.layer_kinds())


def ref_gdn_step(S, qkv, ab, H: int, dk: int, dv: int):
    """fp32 reference of cr_gdn_step. S [H,dv,dk]; returns (S_new, o [H*dv]). Inputs are not modified."""
    S = S.float().clone()
    qkv = qkv.float()
    ab = ab.float()
    q = qkv[: H * dk].reshape(H, dk)
    k = qkv[H * dk : 2 * H * dk].reshape(H, dk)
    v = qkv[2 * H * dk :].reshape(H, dv)
    q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    alpha = torch.sigmoid(ab[:H]).reshape(H, 1, 1)
    beta = torch.sigmoid(ab[H : 2 * H]).reshape(H, 1, 1)
    u = alpha * torch.einsum("hvk,hk->hv", S, k).reshape(H, dv, 1)
    S_new = alpha * S + beta * (v.reshape(H, dv, 1) - u) * k.reshape(H, 1, dk)
    o = torch.einsum("hvk,hk->hv", S_new, q).reshape(H * dv)
    return S_new, o


def ref_attn_decode(q, Kc, Vc, Hq: int, Hkv: int, D: int, scale: float):
    """fp32 reference of cr_attn_decode. Kc, Vc [T,Hkv,D]; returns o [Hq*D]."""
    grp = Hq // Hkv
    qf = q.float().reshape(Hq, D)
    Kf = Kc.float()
    Vf = Vc.float()
    out = torch.empty(Hq, D, dtype=torch.float32)
    for h in range(Hq):
        hk = h // grp
        p = torch.softmax((Kf[:, hk, :] @ qf[h]) * scale, dim=0)
        out[h] = p @ Vf[:, hk, :]
    return out.reshape(Hq * D).to(q.device)


class ToyHybrid:
    """One decode token per step(); every launch is a raw-pointer call on the current stream."""

    def __init__(self, cfg: ToyConfig, device="cuda", variant: str = "lut", seed: int = 0):
        validate(cfg)
        from kernels.build import load_ext

        if variant not in ("lut", "fp4"):
            raise ValueError("variant must be 'lut' or 'fp4'")
        self.cfg, self.variant = cfg, variant
        self.dev = torch.device(device)
        self.ext = load_ext()
        self.vi = 0 if variant == "lut" else 1
        self.kinds = cfg.layer_kinds()
        self.layout, total = arena_layout(cfg)
        self.arena = torch.zeros(total, dtype=torch.uint8, device=self.dev)
        self._fill_weights(seed)

        dev = self.dev
        gen = torch.Generator(device=dev)
        gen.manual_seed(seed + 7)
        h16 = dict(dtype=torch.float16, device=dev)
        self.ng = n_gdn_layers(cfg)
        self.state = torch.randn(max(self.ng, 1), cfg.H, cfg.dv, cfg.dk, dtype=torch.float32, device=dev, generator=gen) * 0.02
        self.x0 = torch.randn(cfg.d, generator=gen, **h16)
        self.x = self.x0.clone()
        self.xn = torch.zeros(cfg.d, **h16)
        self.mix = torch.zeros(cfg.d, **h16)
        self.qkv = torch.zeros(cfg.H * (2 * cfg.dk + cfg.dv), **h16)
        self.ab = torch.zeros(2 * cfg.H, **h16)
        self.og = torch.zeros(cfg.H * cfg.dv, **h16)
        self.qa = torch.zeros(cfg.Hq * cfg.D, **h16)
        self.kva = torch.zeros(2 * cfg.Hkv * cfg.D, **h16)
        self.oa = torch.zeros(cfg.Hq * cfg.D, **h16)
        self.g = torch.zeros(cfg.F, **h16)
        self.u = torch.zeros(cfg.F, **h16)
        self.act = torch.zeros(cfg.F, **h16)
        self.kc: dict[int, torch.Tensor] = {}
        self.vc: dict[int, torch.Tensor] = {}
        for i, kind in enumerate(self.kinds):
            if kind == "attn":
                shp = (cfg.ctx, cfg.Hkv, cfg.D)
                self.kc[i] = (torch.randn(*shp, generator=gen, **h16) * 0.5).contiguous()
                self.vc[i] = (torch.randn(*shp, generator=gen, **h16) * 0.5).contiguous()
        self.state0 = self.state.clone()
        self.n_kernels = 0
        self.plan = self._build_plan()

    # -- setup --
    def _fill_weights(self, seed: int) -> None:
        for idx, m in enumerate(matrix_specs(self.cfg)):
            e = self.layout[m["name"]]
            N, K = e["N"], e["K"]
            q, s, _ = random_ternary(N, K, seed=seed * 100003 + idx, device=self.dev)
            qb, sb = N * K // 4, N * (K // QK) * 2
            self.arena[e["q_off"] : e["q_off"] + qb].view(torch.int32).copy_(q.reshape(-1))
            self.arena[e["s_off"] : e["s_off"] + sb].view(torch.float16).copy_(s.reshape(-1))
        gen = torch.Generator(device=self.dev)
        gen.manual_seed(seed + 11)
        for i in range(self.cfg.layers):
            for tag in ("n1", "n2"):
                e = self.layout[f"l{i}.{tag}"]
                w = (1.0 + 0.1 * torch.randn(self.cfg.d, generator=gen, device=self.dev)).to(torch.float16)
                self.arena[e["w_off"] : e["w_off"] + self.cfg.d * 2].view(torch.float16).copy_(w)

    def weight_arena(self) -> torch.Tensor:
        return self.arena

    def state_arena(self) -> torch.Tensor:
        return self.state

    def matrix_views(self, name: str):
        """(q int32 [N,K/16], s fp16 [N,K/128]) views into the arena."""
        e = self.layout[name]
        N, K = e["N"], e["K"]
        q = self.arena[e["q_off"] : e["q_off"] + N * K // 4].view(torch.int32).reshape(N, K // 16)
        s = self.arena[e["s_off"] : e["s_off"] + N * (K // QK) * 2].view(torch.float16).reshape(N, K // QK)
        return q, s

    def reset(self) -> None:
        self.x.copy_(self.x0)
        self.state.copy_(self.state0)

    # -- launch plan: list of (fn, args, kernels); stream handle is appended at call time --
    def _build_plan(self) -> list:
        c, ext, vi = self.cfg, self.ext, self.vi
        base = self.arena.data_ptr()
        P = lambda t: t.data_ptr()  # noqa: E731
        plan: list = []

        def gemv(name, xin, yout):
            e = self.layout[name]
            plan.append((ext.gemv, (vi, base + e["q_off"], base + e["s_off"], P(xin), P(yout), e["N"], e["K"]), 1))

        def norm(i, tag):
            e = self.layout[f"l{i}.{tag}"]
            plan.append((ext.rmsnorm, (P(self.x), base + e["w_off"], P(self.xn), c.d, EPS), 1))

        gi = 0
        for i, kind in enumerate(self.kinds):
            norm(i, "n1")
            if kind == "gdn":
                gemv(f"l{i}.qkv", self.xn, self.qkv)
                gemv(f"l{i}.ab", self.xn, self.ab)
                plan.append((ext.gdn_step, (P(self.state[gi]), P(self.qkv), P(self.ab), P(self.og), c.H, c.dk, c.dv), 1))
                gemv(f"l{i}.out", self.og, self.mix)
                gi += 1
            else:
                gemv(f"l{i}.q", self.xn, self.qa)
                gemv(f"l{i}.kv", self.xn, self.kva)
                plan.append((ext.attn_decode, (P(self.qa), P(self.kc[i]), P(self.vc[i]), P(self.oa), c.Hq, c.Hkv, c.D, c.ctx, c.scale), 2))
                gemv(f"l{i}.o", self.oa, self.mix)
            plan.append((ext.add_, (P(self.x), P(self.mix), c.d), 1))
            norm(i, "n2")
            gemv(f"l{i}.gate", self.xn, self.g)
            gemv(f"l{i}.up", self.xn, self.u)
            plan.append((ext.silu_mul, (P(self.g), P(self.u), P(self.act), c.F), 1))
            gemv(f"l{i}.down", self.act, self.mix)
            plan.append((ext.add_, (P(self.x), P(self.mix), c.d), 1))
        self.n_kernels = sum(p[2] for p in plan)
        return plan

    def step(self) -> None:
        st = torch.cuda.current_stream(self.dev).cuda_stream
        for fn, args, _ in self.plan:
            fn(*args, st)

    # -- traffic accounting --
    def weight_bytes(self) -> int:
        return sum(m["N"] * m["K"] // 4 + m["N"] * (m["K"] // QK) * 2 for m in matrix_specs(self.cfg))

    def extra_bytes(self) -> int:
        c = self.cfg
        kv = sum(2 * c.ctx * c.Hkv * c.D * 2 for k in self.kinds if k == "attn")
        st = sum(2 * c.H * c.dv * c.dk * 4 for k in self.kinds if k == "gdn")
        return kv + st


def _capture(fn, warmup: int = 3) -> torch.cuda.CUDAGraph:
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g


def _time_us(fn, iters: int, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        torch.cuda.nvtx.range_push("cr_decode")
        fn()
        torch.cuda.nvtx.range_pop()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) * 1000.0 / iters


def measure(cfg: ToyConfig, variant: str, tokens: int) -> dict:
    """Eager and graph us/token for one config."""
    m = ToyHybrid(cfg, variant=variant)
    eager = _time_us(m.step, tokens)
    g = _capture(m.step)
    graph = _time_us(g.replay, tokens)
    total = m.weight_bytes() + m.extra_bytes()
    out = dict(
        total_bytes_per_token=total,
        us_per_token_eager=eager,
        us_per_token_graph=graph,
        us_per_layer_graph=graph / cfg.layers,
        launches_per_token=m.n_kernels,
        weight_bytes_per_token=m.weight_bytes(),
        extra_bytes_per_token=m.extra_bytes(),
        eff_gbps=total / graph / 1e3,
    )
    del g, m
    torch.cuda.empty_cache()
    return out


def floor_config(cfg: ToyConfig) -> ToyConfig:
    """Same layer kinds and launch list at tiny sizes: bytes are negligible, time is launch floor."""
    return dataclasses.replace(cfg, d=128, F=128, H=1, dk=32, dv=128, Hq=2, Hkv=1, D=64, ctx=32, kinds=tuple(cfg.layer_kinds()))


def run(cfg: ToyConfig, variant: str, tokens: int) -> dict:
    from kernels.bench_gemv import l2_bytes

    main = measure(cfg, variant, tokens)
    homo = {}
    for kind in ("gdn", "attn"):
        c2 = dataclasses.replace(cfg, kinds=tuple([kind] * cfg.layers))
        homo[kind] = measure(c2, variant, tokens)["us_per_layer_graph"]
    main["us_per_layer_gdn"] = homo["gdn"]
    main["us_per_layer_attn"] = homo["attn"]
    floor = measure(floor_config(cfg), variant, tokens)["us_per_layer_graph"]
    net_us = main["us_per_token_graph"] - cfg.layers * floor
    main["floor_us_per_layer"] = floor
    main["eff_gbps_net"] = main["total_bytes_per_token"] / net_us / 1e3 if net_us > 0 else None
    l2 = l2_bytes()
    main["l2_bytes"] = l2
    main["l2_assisted"] = bool(main["total_bytes_per_token"] < 2 * l2)
    return main


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    d = ToyConfig()
    for f in ("layers", "d", "attn_every", "H", "dk", "dv", "Hq", "Hkv", "D", "F", "ctx"):
        flag = "--" + f.replace("_", "-")
        ap.add_argument(flag, dest=f, type=int, default=getattr(d, f))
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--variant", choices=("lut", "fp4"), default="lut")
    ap.add_argument("--result-exp", default="p0_toy", help="empty string disables the result file")
    ap.add_argument("--quick", action="store_true", help="tiny sizes for smoke runs")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, no CUDA")
    a = ap.parse_args(argv)

    cfg = ToyConfig(**{f: getattr(a, f) for f in ("layers", "d", "attn_every", "H", "dk", "dv", "Hq", "Hkv", "D", "F", "ctx")})
    tokens = a.tokens
    if a.quick:
        cfg = dataclasses.replace(cfg, layers=min(cfg.layers, 4), ctx=min(cfg.ctx, 512))
        tokens = min(tokens, 20)
    validate(cfg)
    layout, total = arena_layout(cfg)
    plan = dict(config=dataclasses.asdict(cfg), variant=a.variant, tokens=tokens, arena_bytes=total, n_matrices=len(matrix_specs(cfg)))
    if a.dry_run:
        print(json.dumps(plan, indent=1, default=str))
        return 0
    if not torch.cuda.is_available():
        print("toy_hybrid: no CUDA device", file=sys.stderr)
        return 2

    row = run(cfg, a.variant, tokens)
    notes = [
        "eager = precomputed launch list called from Python; graph = same list captured once",
        "eff_gbps counts ternary weights+scales, attention K/V reads and GDN state read+write",
        "us_per_layer_gdn / us_per_layer_attn: homogeneous models with the same layer count, graph mode",
        "floor_us_per_layer: same layer kinds and count at tiny sizes (d=F=H*dv=128, ctx=32), graph mode",
        "eff_gbps_net: total bytes / (graph us - layers * floor); use floor_us_per_layer with eff_gbps_net for projections",
        "weights are random, residual values are not meaningful",
    ]
    if row["l2_assisted"]:
        notes.append("total_bytes_per_token < 2x L2: numbers are L2-assisted, do not use for a DRAM bandwidth fit")
    plan.update(l2_bytes=row["l2_bytes"], total_bytes_per_token=row["total_bytes_per_token"])
    print(json.dumps(row, indent=1))
    if a.result_exp:
        from infra.results import write_result

        path = write_result(a.result_exp, None, dict(config=plan, rows=[row], notes=notes, counters_ok=None))
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
