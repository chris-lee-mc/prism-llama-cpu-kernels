"""E3d: does a FULLY resident ternary draft decode step get faster, and does residency survive the target's verify pass?

Draft = toy hybrid decode step (GDN + single-query attention + ternary MLP/projections + 48K-vocab head) with EVERYTHING
it reads (weights, head, GDN state, KV, scratch) in one arena, sized to D MiB by scaling width, layers and F. One CUDA graph
per draft step; "on" puts an access policy window over the arena on all graph kernel nodes, limit = min(D, device max).
Target verify = a separate graph streaming V bytes of ternary weights with Bonsai 2 27B shapes (GEMV batch 1 = bandwidth proxy).
Scenarios, each off vs on, timed in interleaved rounds (paired deltas, round spread = noise):
  S1 steady: K consecutive draft steps, us/step.
  S2 cycle:  [verify, K draft steps] repeated; per-draft-step time by index (first step after verify vs later) and cycle time.
  S3 cycle:  as S2, but the verify graph marks its own weights Streaming (one window per layer, set during capture).
PDL is a process-wide knob (CR_GEMV_PDL read once), so --pdl both runs one child process per setting and merges the rows.
The verify graph uses the same knob. Projection: tokens/cycle = (1 - a^(K+1)) / (1 - a), cycle = verify + K * draft step,
speedup = tokens/cycle * verify_off / cycle (baseline = target-only decode = verify_off per token).

  python -m exps.e3d_resident_draft [--D 16,32,...] [--K 4,8] [--V MiB] [--pdl 0|1|both] [--head ternary|int8]
        [--rounds 7] [--cycles 30] [--hit-ratio auto|r] [--verify-cols 1|kp1] [--quick] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys

from exps import common, e1_sweep as e1, modelspec
from kernels import toy_hybrid as toy
from kernels.ternary import QK

EXP = "e3d"
MIB = common.MIB
GIB = 1 << 30
ALIGN = 256
DEFAULT_D = [16, 32, 48, 60, 72, 96, 160]
DEFAULT_K = [4, 8]
DEFAULT_V_MIB = 6144
VOCAB = 49152
CTX = 512
WIDTHS = list(range(384, 2049, 128))
LAYER_CHOICES = list(range(4, 65, 4))
PLAN_TOL = 0.015
SCENARIOS = ["S1", "S2", "S3"]
METRICS = ["step_us", "first_us", "later_us", "refill_us", "cycle_us", "verify_us"]
ACCS = (0.6, 0.7, 0.8)
MAX_VERIFY_NODES = 4000
VRAM_RESERVE = 2 * GIB


# ---------------------------------------------------------------- draft sizing (pure)

def align(n: int) -> int:
    return (n + ALIGN - 1) // ALIGN * ALIGN


class _Take:
    def __init__(self, off: int = 0):
        self.off = off

    def __call__(self, nbytes: int) -> int:
        o = self.off
        self.off = align(o + nbytes)
        return o


def head_bytes(head: str, vocab: int, d: int) -> tuple[int, int]:
    """(weight bytes, scale bytes) of the LM head."""
    if head == "ternary":
        return vocab * d // 4, vocab * (d // QK) * 2
    if head == "int8":
        return vocab * d, vocab * 4
    raise ValueError("head must be ternary or int8")


def make_cfg(d: int, layers: int, F: int, ctx: int = CTX) -> toy.ToyConfig:
    return toy.ToyConfig(layers=layers, d=d, attn_every=4, H=min(16, max(4, d // 64)), dk=128, dv=128,
                         Hq=min(16, max(2, 2 * round(d / 256))), Hkv=2, D=128, F=F, ctx=ctx)


def draft_layout(cfg: toy.ToyConfig, vocab: int = VOCAB, head: str = "ternary") -> dict:
    """Offsets in the one arena and the byte split: weights (ternary mats + norms), head, state, kv, scratch, pad."""
    ent, off = toy.arena_layout(cfg)
    t = _Take(off)
    d, kinds = cfg.d, cfg.layer_kinds()
    mats = toy.matrix_specs(cfg)
    wmat = sum(m["N"] * m["K"] // 4 + m["N"] * (m["K"] // QK) * 2 for m in mats)
    lay: dict = {"ent": ent, "final_norm": t(d * 2)}
    qb, sb = head_bytes(head, vocab, d)
    lay["head_q"], lay["head_s"] = t(qb), t(sb)
    ng, na = kinds.count("gdn"), kinds.count("attn")
    state = ng * cfg.H * cfg.dv * cfg.dk * 4
    lay["state"] = t(state)
    kvb = cfg.ctx * cfg.Hkv * cfg.D * 2
    lay["kv"] = {i: (t(kvb), t(kvb)) for i, k in enumerate(kinds) if k == "attn"}
    sizes = [("x0", d), ("x", d), ("xn", d), ("mix", d), ("qkv", cfg.H * (2 * cfg.dk + cfg.dv)), ("ab", 2 * cfg.H),
             ("og", cfg.H * cfg.dv), ("qa", cfg.Hq * cfg.D), ("kva", 2 * cfg.Hkv * cfg.D), ("oa", cfg.Hq * cfg.D),
             ("g", cfg.F), ("u", cfg.F), ("act", cfg.F)]
    if head == "ternary":
        sizes.append(("logits", vocab))
    scratch = 0
    for name, n in sizes:
        lay[name] = t(n * 2)
        scratch += n * 2
    norms = 2 * cfg.layers * d * 2
    lay["total"] = t.off
    by = {"weights": wmat + norms, "head": qb + sb + d * 2, "state": state, "kv": na * 2 * kvb, "scratch": scratch}
    by["pad"] = lay["total"] - sum(by.values())
    by["total"] = lay["total"]
    lay["bytes"] = by
    lay["step_bytes"] = wmat + qb + sb + 2 * state + by["kv"]  # state is read and written
    return lay


def d_pref(d_mib: float) -> float:
    return 1024.0 * (d_mib / 160.0) ** 0.35


def plan_draft(d_mib: float, head: str = "ternary", vocab: int = VOCAB, ctx: int = CTX, tol: float = PLAN_TOL) -> dict:
    """Pick (width, layers, F) whose arena is within tol of d_mib; prefer the width close to d_pref and F near 3.5 d."""
    target = d_mib * MIB
    cands = []
    for d in WIDTHS:
        for L in LAYER_CHOICES:
            u_lo, u_hi = max(2, d // 128), 8 * d // 128
            t_lo = draft_layout(make_cfg(d, L, 128 * u_lo, ctx), vocab, head)["total"]
            t_hi = draft_layout(make_cfg(d, L, 128 * u_hi, ctx), vocab, head)["total"]
            slope = (t_hi - t_lo) / (u_hi - u_lo)
            u0 = int(round(u_lo + (target - t_lo) / slope))
            for u in {min(u_hi, max(u_lo, u0 + j)) for j in (-1, 0, 1)}:
                tot = draft_layout(make_cfg(d, L, 128 * u, ctx), vocab, head)["total"]
                err = abs(tot - target) / target
                score = abs(math.log(d / d_pref(d_mib))) + 0.3 * abs(math.log(128 * u / (3.5 * d)))
                cands.append((err, score, d, L, u))
    ok = [c for c in cands if c[0] <= tol]
    err, _, d, L, u = min(ok, key=lambda c: (c[1], c[0])) if ok else min(cands)
    cfg = make_cfg(d, L, 128 * u, ctx)
    lay = draft_layout(cfg, vocab, head)
    return {"d_mib": d_mib, "cfg": cfg, "layout": lay, "err": err, "within_tol": err <= tol}


def cfg_json(cfg: toy.ToyConfig) -> dict:
    return {k: getattr(cfg, k) for k in ("layers", "d", "attn_every", "H", "dk", "dv", "Hq", "Hkv", "D", "F", "ctx")}


# ---------------------------------------------------------------- verify sizing (pure)

def layer_shapes(cfg: dict, kind: str) -> list[tuple[int, int]]:
    """Ternary matrices (N, K) of one decoder layer; GDN b and a projections are merged into one."""
    g = lambda n: cfg[n]  # noqa: E731
    hid, inter = g("hidden_size"), g("intermediate_size")
    if kind == "linear_attention":
        key, val = g("linear_num_key_heads") * g("linear_key_head_dim"), g("linear_num_value_heads") * g("linear_value_head_dim")
        m = [(2 * key + val, hid), (val, hid), (2 * g("linear_num_value_heads"), hid), (hid, val)]
    else:
        nh, hd = g("num_attention_heads"), g("head_dim")
        gate = 2 if cfg.get("attn_output_gate") else 1
        m = [(nh * hd * gate, hid), (2 * g("num_key_value_heads") * hd, hid), (hid, nh * hd)]
    return m + [(inter, hid), (inter, hid), (hid, inter)]


def plan_verify(v_bytes: int, model: str = "27B") -> list[tuple[int, int]]:
    """Matrices of Bonsai 2 27B layers in order, repeated if needed, until the bytes are closest to v_bytes."""
    cfg = modelspec.TEXT_CFG[model]
    per_layer = [layer_shapes(cfg, k) for k in modelspec.layer_types(cfg)]
    out, acc = [], 0
    while True:
        for mats in per_layer:
            for nk in mats:
                nb = sum(e1.layer_bytes(*nk))
                if acc + nb > v_bytes:
                    if acc + nb - v_bytes <= v_bytes - acc:
                        out.append(nk)
                    return out
                out.append(nk)
                acc += nb


def choose_verify_mib(req_mib: float, free_bytes: int | None, draft_mib: float, reserve: int = VRAM_RESERVE) -> tuple[float, bool]:
    """(V MiB, scaled_down): shrink V so that V + draft + reserve fits the free VRAM."""
    if free_bytes is None:
        return req_mib, False
    cap = (free_bytes - reserve) / MIB - draft_mib
    cap = max(64.0, cap)
    return (cap, True) if cap < req_mib else (req_mib, False)


def window_params(arena_bytes: int, wmax: int, pmax: int, ratio) -> dict:
    """Window over the arena start (clamped to the device max), limit min(D, max), auto hitRatio = min(1, limit / window)."""
    from harness import l2persist
    limit = min(arena_bytes, pmax) if pmax else 0
    nbytes = min(arena_bytes, wmax) if wmax else arena_bytes
    r = l2persist.suggest_hit_ratio(nbytes, limit) if ratio in (None, "auto") else float(ratio)
    return {"window_bytes": nbytes, "clamped": nbytes < arena_bytes, "limit": limit, "hit_ratio": r}


# ---------------------------------------------------------------- cells, summaries, deltas, projection (pure)

def make_cells(scenarios, ks, s3_ok: bool = True) -> list[tuple[str, int, str]]:
    """(scenario, K, mode) for every cell; S3 dropped when the driver ignores mid-capture window changes."""
    return [(s, k, m) for s in scenarios if s != "S3" or s3_ok for k in ks for m in ("off", "on")]


def summarize_cycles(raw: list[tuple[float, list[float], float]]) -> dict:
    """raw = [(verify_us, [step_us per index], cycle_us)] -> medians over cycles."""
    k = len(raw[0][1])
    by = [statistics.median(r[1][i] for r in raw) for i in range(k)]
    mean = statistics.median(statistics.fmean(r[1]) for r in raw)
    later = statistics.median(statistics.fmean(r[1][1:]) for r in raw) if k > 1 else by[0]
    return {"verify_us": statistics.median(r[0] for r in raw), "by_index": by, "first_us": by[0], "later_us": later,
            "step_us": mean, "refill_us": by[0] - later, "cycle_us": statistics.median(r[2] for r in raw)}


def _spread(xs):
    return (max(xs) - min(xs)) if len(xs) >= 2 else None


def _key(r: dict) -> tuple:
    return (r["D_mib"], r.get("pdl"), r["scenario"], r["K"])


def add_deltas(rows: list[dict]) -> None:
    """For each on row: delta_<m> = off - on (positive = on faster), gain_<m>, noise_<m>, above_<m> (paired per round)."""
    off = {_key(r): r for r in rows if r["mode"] == "off"}
    for r in rows:
        o = off.get(_key(r))
        if r["mode"] != "on" or o is None:
            continue
        for m in METRICS:
            p = e1.paired(o.get(m + "_all"), r.get(m + "_all"))
            if p is None:
                continue
            r["delta_" + m] = p["delta"]
            r["delta_all_" + m] = p["all"]
            r["noise_" + m] = p["noise"]
            r["above_" + m] = p["above"]
            if o.get(m):
                r["gain_" + m] = p["delta"] / o[m]


def add_cross(rows: list[dict]) -> None:
    """S3 rows: s3_gain_<m> = S2 - S3 (positive = streaming verify helps). S2 rows: first_minus_steady_us vs the S1 step."""
    by = {(*_key(r), r["mode"]): r for r in rows}
    for r in rows:
        if r["scenario"] == "S3":
            s2 = by.get((r["D_mib"], r.get("pdl"), "S2", r["K"], r["mode"]))
            for m in ("step_us", "first_us", "later_us", "cycle_us"):
                p = e1.paired(s2.get(m + "_all") if s2 else None, r.get(m + "_all"))
                if p is not None:
                    r["s3_delta_" + m] = p["delta"]
                    r["s3_noise_" + m] = p["noise"]
        elif r["scenario"] == "S2":
            s1 = by.get((r["D_mib"], r.get("pdl"), "S1", r["K"], r["mode"]))
            p = e1.paired(r.get("first_us_all"), s1.get("step_us_all") if s1 else None)
            if p is not None:
                r["first_minus_steady_us"] = p["delta"]
                r["first_minus_steady_noise_us"] = p["noise"]


def tokens_per_cycle(a: float, k: int) -> float:
    return k + 1.0 if a >= 1.0 else (1.0 - a ** (k + 1)) / (1.0 - a)


def spec_speedup(a: float, k: int, verify_us: float, step_us: float, base_us: float) -> float:
    return tokens_per_cycle(a, k) * base_us / (verify_us + k * step_us)


def projection(rows: list[dict], accs=ACCS) -> list[dict]:
    """Speculative speedup off vs on per (D, pdl, K, a, source). Baseline = verify time of the S2 off row.

    Sources: S1 (steady draft step, verify from the S2 row of the same mode), S2 and S3 (their own steps and verify).
    """
    by = {(*_key(r), r["mode"]): r for r in rows if r.get("step_us") is not None}
    out = []
    for (d, pdl, src, k, mode) in sorted(by, key=str):
        base = by.get((d, pdl, "S2", k, "off"))
        if mode != "off" or base is None or not base.get("verify_us"):
            continue
        res = {}
        for m in ("off", "on"):
            r = by.get((d, pdl, src, k, m))
            v = by.get((d, pdl, "S2" if src == "S1" else src, k, m))
            if r is not None and v is not None and v.get("verify_us"):
                res[m] = (r["step_us"], v["verify_us"])
        if len(res) < 2:
            continue
        for a in accs:
            sp = {m: spec_speedup(a, k, res[m][1], res[m][0], base["verify_us"]) for m in res}
            out.append({"D_mib": d, "pdl": pdl, "K": k, "acc": a, "source": src, "speedup_off": sp["off"],
                        "speedup_on": sp["on"], "speedup_gain": sp["on"] / sp["off"] - 1.0,
                        "cycle_off_us": res["off"][1] + k * res["off"][0], "cycle_on_us": res["on"][1] + k * res["on"][0],
                        "tokens_per_cycle": tokens_per_cycle(a, k)})
    return out


# ---------------------------------------------------------------- GPU side: models

class DraftModel:
    """Toy hybrid decode step with every tensor in one uint8 arena; step() launches raw-pointer ops on the current stream."""

    def __init__(self, cfg: toy.ToyConfig, vocab: int = VOCAB, head: str = "ternary", variant: str = "lut", seed: int = 0,
                 device: str = "cuda"):
        import torch
        from kernels.build import load_ext
        toy.validate(cfg)
        self.torch, self.cfg, self.vocab, self.head = torch, cfg, vocab, head
        self.dev = torch.device(device)
        self.ext = load_ext()
        self.vi = 0 if variant == "lut" else 1
        self.lay = draft_layout(cfg, vocab, head)
        self.total = self.lay["total"]
        self.bytes = self.lay["bytes"]
        self.step_bytes = self.lay["step_bytes"]
        self.arena = torch.zeros(self.total, dtype=torch.uint8, device=self.dev)
        self.base = self.arena.data_ptr()
        self._views()
        self._fill(seed)
        self.plan = self._build_plan()

    def _v(self, off, dtype, *shape):
        n = 1
        for s in shape:
            n *= s
        return self.arena[off:off + n * dtype.itemsize].view(dtype).reshape(*shape)

    def _views(self):
        torch, c, lay = self.torch, self.cfg, self.lay
        h = torch.float16
        for name in ("x0", "x", "xn", "mix", "qkv", "ab", "og", "qa", "kva", "oa", "g", "u", "act"):
            n = {"x0": c.d, "x": c.d, "xn": c.d, "mix": c.d, "qkv": c.H * (2 * c.dk + c.dv), "ab": 2 * c.H, "og": c.H * c.dv,
                 "qa": c.Hq * c.D, "kva": 2 * c.Hkv * c.D, "oa": c.Hq * c.D, "g": c.F, "u": c.F, "act": c.F}[name]
            setattr(self, name, self._v(lay[name], h, n))
        self.ng = self.cfg.layer_kinds().count("gdn")
        self.state = self._v(lay["state"], torch.float32, self.ng, c.H, c.dv, c.dk)
        self.logits = self._v(lay["logits"], h, self.vocab) if self.head == "ternary" else None
        d, V = c.d, self.vocab
        if self.head == "ternary":
            self.hq = self._v(lay["head_q"], torch.int32, V, d // 16)
            self.hs = self._v(lay["head_s"], h, V, d // QK)
        else:
            self.hq = self._v(lay["head_q"], torch.int8, V, d)
            self.hs = self._v(lay["head_s"], torch.float32, V)

    def _fill(self, seed: int):
        torch, c, lay = self.torch, self.cfg, self.lay
        from kernels.ternary import random_ternary
        for idx, m in enumerate(toy.matrix_specs(c)):
            e = lay["ent"][m["name"]]
            N, K = e["N"], e["K"]
            q, s, _ = random_ternary(N, K, seed=seed * 100003 + idx, device=self.dev)
            self.arena[e["q_off"]:e["q_off"] + N * K // 4].view(torch.int32).copy_(q.reshape(-1))
            self.arena[e["s_off"]:e["s_off"] + N * (K // QK) * 2].view(torch.float16).copy_(s.reshape(-1))
        gen = torch.Generator(device=self.dev)
        gen.manual_seed(seed + 11)
        for i in range(c.layers):
            for tag in ("n1", "n2"):
                w = (1.0 + 0.1 * torch.randn(c.d, generator=gen, device=self.dev)).to(torch.float16)
                self._v(lay["ent"][f"l{i}.{tag}"]["w_off"], torch.float16, c.d).copy_(w)
        self._v(lay["final_norm"], torch.float16, c.d).copy_((1.0 + 0.1 * torch.randn(c.d, generator=gen, device=self.dev)).half())
        if self.head == "ternary":
            q, s, _ = random_ternary(self.vocab, c.d, seed=seed + 5, device=self.dev)
            self.hq.copy_(q)
            self.hs.copy_(s)
        else:
            for r0 in range(0, self.vocab, 8192):
                r1 = min(self.vocab, r0 + 8192)
                self.hq[r0:r1].copy_(torch.randint(-127, 128, (r1 - r0, c.d), generator=gen, device=self.dev, dtype=torch.int8))
            self.hs.copy_((torch.rand(self.vocab, generator=gen, device=self.dev) + 0.5) * (4.0 / (127.0 * math.sqrt(c.d))))
        self.x0.copy_(torch.randn(c.d, generator=gen, device=self.dev).half())
        if self.ng:
            self.state.copy_(torch.randn(self.state.shape, generator=gen, device=self.dev) * 0.02)
        for i, (ko, vo) in lay["kv"].items():
            shp = (c.ctx, c.Hkv, c.D)
            self._v(ko, torch.float16, *shp).copy_((torch.randn(shp, generator=gen, device=self.dev) * 0.5).half())
            self._v(vo, torch.float16, *shp).copy_((torch.randn(shp, generator=gen, device=self.dev) * 0.5).half())

    def _build_plan(self) -> list:
        c, ext, vi, lay, base = self.cfg, self.ext, self.vi, self.lay, self.base
        P = lambda t: t.data_ptr()  # noqa: E731
        plan: list = []

        def gemv(name, xin, yout):
            e = lay["ent"][name]
            args = (vi, base + e["q_off"], base + e["s_off"], P(xin), P(yout), e["N"], e["K"])
            plan.append(lambda st, a=args: ext.gemv(*a, st))

        def op(fn, *args):
            plan.append(lambda st, f=fn, a=args: f(*a, st))

        def norm(w_off, xout):
            op(ext.rmsnorm, P(self.x), base + w_off, P(xout), c.d, toy.EPS)

        gi = 0
        for i, kind in enumerate(c.layer_kinds()):
            norm(lay["ent"][f"l{i}.n1"]["w_off"], self.xn)
            if kind == "gdn":
                gemv(f"l{i}.qkv", self.xn, self.qkv)
                gemv(f"l{i}.ab", self.xn, self.ab)
                op(ext.gdn_step, base + lay["state"] + gi * c.H * c.dv * c.dk * 4, P(self.qkv), P(self.ab), P(self.og), c.H, c.dk, c.dv)
                gemv(f"l{i}.out", self.og, self.mix)
                gi += 1
            else:
                gemv(f"l{i}.q", self.xn, self.qa)
                gemv(f"l{i}.kv", self.xn, self.kva)
                ko, vo = lay["kv"][i]
                op(ext.attn_decode, P(self.qa), base + ko, base + vo, P(self.oa), c.Hq, c.Hkv, c.D, c.ctx, c.scale)
                gemv(f"l{i}.o", self.oa, self.mix)
            op(ext.add_, P(self.x), P(self.mix), c.d)
            norm(lay["ent"][f"l{i}.n2"]["w_off"], self.xn)
            gemv(f"l{i}.gate", self.xn, self.g)
            gemv(f"l{i}.up", self.xn, self.u)
            op(ext.silu_mul, P(self.g), P(self.u), P(self.act), c.F)
            gemv(f"l{i}.down", self.act, self.mix)
            op(ext.add_, P(self.x), P(self.mix), c.d)
        norm(lay["final_norm"], self.xn)
        if self.head == "ternary":
            args = (vi, base + lay["head_q"], base + lay["head_s"], P(self.xn), P(self.logits), self.vocab, c.d)
            plan.append(lambda st, a=args: ext.gemv(*a, st))
        else:
            plan.append(lambda st: self._int8_head())
        return plan

    def _int8_head(self):
        from exps import int8_head
        self.logits = int8_head.int8_gemv(self.hq, self.hs, self.xn, out_dtype=self.torch.float16)

    def step(self) -> None:
        st = self.torch.cuda.current_stream().cuda_stream
        self.x.copy_(self.x0)  # new token embedding: keeps the residual bounded across replays
        for fn in self.plan:
            fn(st)

    def check(self) -> dict:
        """One eager step; head logits vs the fp32 reference on the final normed activation."""
        torch = self.torch
        from kernels import ternary
        from exps import int8_head
        self.step()
        torch.cuda.synchronize()
        y = self.logits.float()
        if self.head == "ternary":
            ref = ternary.ref_gemv(self.hq, self.hs, self.xn)
        else:
            ref = int8_head.int8_gemv_ref(self.hq, self.hs, self.xn)
        fin = bool(torch.isfinite(y).all().item())
        err = (y - ref).abs().max().item() if fin else float("nan")
        tol = 2e-3 if self.head == "ternary" else 1e-2
        ok = fin and bool(torch.allclose(y, ref, rtol=tol, atol=tol * ref.abs().max().item()))
        return {"finite": fin, "ok": ok, "max_abs_err": err}


class Verify(e1.Layers):
    """Streaming weight set (Bonsai 27B shapes) in its own arena."""

    def launch_v(self, stream: int, cols: int = 1, stream_windows: bool = False) -> None:
        from harness import l2persist
        xp, yp = self.x.data_ptr(), self.y.data_ptr()
        for i, (n, k) in enumerate(self.shapes):
            q, s = self.arena.ptrs(i)
            if stream_windows:
                set_streaming_window(self.arena.base + self.arena.offs[i], e1.layer_stride(n, k), stream)
            for _ in range(cols):
                self.ternary.gemv_ptr(self.variant, q, s, xp, yp, n, k, stream)
        if stream_windows:
            l2persist.clear_stream_window(stream)


def set_streaming_window(ptr: int, nbytes: int, stream: int) -> None:
    from harness import l2persist
    l2persist.set_stream_window(ptr, nbytes, 1.0, stream=stream, hit_prop=l2persist.ACCESS_STREAMING,
                                miss_prop=l2persist.ACCESS_STREAMING)


# ---------------------------------------------------------------- GPU side: graphs and timing

def free_vram() -> int | None:
    try:
        import torch
        return int(torch.cuda.mem_get_info()[0])
    except Exception:
        return None


def device_cc_major() -> int | None:
    try:
        import torch
        return int(torch.cuda.get_device_capability()[0])
    except Exception:
        return None


def build_draft_graph(model, ratio: float | None, win_bytes: int):
    """Draft step graph; ratio not None puts the persisting window over the arena on every kernel node."""
    return e1.build_graph(model.step, ratio, model.base, win_bytes)


def build_verify_graph(ver, cols: int, stream_windows: bool):
    from harness import l2persist, timing
    import torch
    l2persist.reset()
    common.clear_capture_windows()

    def fn():
        ver.launch_v(torch.cuda.current_stream().cuda_stream, cols, stream_windows)

    g = timing.capture_graph(fn, keep_graph=True)
    info = {"window_readback": common.window_readback(g, graph=True), "kernel_nodes": len(ver.shapes) * cols}
    g.instantiate()
    common.clear_capture_windows()
    return g, info


def probe_midcapture(ver) -> dict:
    """Window A set before capture, window B set inside it: the captured node must read back B (else S3 is skipped)."""
    import torch
    from harness import l2persist, timing
    try:
        s = torch.cuda.Stream()
        n, k = ver.shapes[0]
        a_ptr = ver.arena.base + ver.arena.offs[-1]
        b_ptr = ver.arena.base + ver.arena.offs[0]
        set_streaming_window(a_ptr, e1.layer_stride(*ver.shapes[-1]), s.cuda_stream)

        def fn():
            st = torch.cuda.current_stream().cuda_stream
            set_streaming_window(b_ptr, e1.layer_stride(n, k), st)
            q, sc = ver.arena.ptrs(0)
            ver.ternary.gemv_ptr(ver.variant, q, sc, ver.x.data_ptr(), ver.y.data_ptr(), n, k, st)
            l2persist.clear_stream_window(st)

        g = timing.capture_graph(fn, stream=s, keep_graph=True)
        rb = common.window_readback(g, graph=True)
        l2persist.clear_stream_window(s.cuda_stream)
        common.clear_capture_windows()
        ok = bool(rb) and rb.get("base_ptr") == b_ptr and rb.get("hit_prop") == "streaming"
        return {"supported": ok, "readback": rb, "expected_base_ptr": b_ptr}
    except Exception as e:
        return {"supported": False, "error": f"{type(e).__name__}: {e}"}


def set_persist(limit: int) -> int:
    return e1.set_persist(limit)


def nvml_state():
    return e1.nvml_state()


def calibrate(variant: str, shapes, tag: str, l2: int) -> dict:
    return e1.calibrate(variant, shapes, tag, l2)


def time_steady(dg, k: int, bursts: int) -> float:
    """Median over bursts of (time of k back-to-back replays) / k, us."""
    import torch
    for _ in range(2 * k):
        dg.replay()
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(bursts + 1)]
    ev[0].record()
    for b in range(bursts):
        for _ in range(k):
            dg.replay()
        ev[b + 1].record()
    ev[-1].synchronize()
    return statistics.median(ev[b].elapsed_time(ev[b + 1]) * 1e3 / k for b in range(bursts))


def time_cycles(dg, vg, k: int, cycles: int, warm: int = 3) -> list[tuple[float, list[float], float]]:
    """[(verify_us, [draft step us by index], cycle_us)] for cycles of [verify, k draft steps]; events between graphs."""
    import torch
    for _ in range(warm):
        vg.replay()
        for _ in range(k):
            dg.replay()
    ev = [[torch.cuda.Event(enable_timing=True) for _ in range(k + 2)] for _ in range(cycles)]
    for e in ev:
        e[0].record()
        vg.replay()
        e[1].record()
        for i in range(k):
            dg.replay()
            e[2 + i].record()
    ev[-1][-1].synchronize()
    return [(e[0].elapsed_time(e[1]) * 1e3, [e[1 + i].elapsed_time(e[2 + i]) * 1e3 for i in range(k)],
             e[0].elapsed_time(e[k + 1]) * 1e3) for e in ev]


# ---------------------------------------------------------------- run

def dump_json(path: str, doc: dict) -> None:
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="ascii") as f:
        json.dump(doc, f, default=str)
    os.replace(tmp, path)


def run_matrix(args, pdl: int) -> dict:
    """All cells for one PDL setting. Returns {config, rows, notes}; also snapshots to --emit or the partial file."""
    import torch
    from harness import l2persist
    os.environ["CR_GEMV_PDL"] = str(pdl)  # read once at the first GEMV launch
    notes: list[str] = []
    tag = common.gpu_tag(args)
    args.gpu = tag
    dev = l2persist.log_device(0)
    l2 = int(dev.get("l2_bytes") or 96 * MIB)
    pmax = int(dev.get("persisting_max_bytes") or 0)
    wmax = l2persist.max_window_bytes()
    budget = common.Budget(args.budget_min if args.budget_min else (8 if args.quick else 26))
    cc = device_cc_major()
    config = {"design": "resident_draft", "pdl": pdl, "pdl_effective": bool(pdl and cc is not None and cc >= 9), "D_mib": args.D,
              "K": args.K, "scenarios": args.scenarios, "rounds": args.rounds, "cycles": args.cycles, "bursts": args.bursts,
              "head": args.head, "vocab": args.vocab, "ctx": args.ctx, "variant": args.variant, "hit_ratio_arg": args.hit_ratio,
              "verify_cols_arg": args.verify_cols, "quick": args.quick, "device": dev, "window_max_bytes": wmax}
    rows: list[dict] = []

    def snapshot():
        doc = {"config": config, "rows": rows, "notes": notes}
        if args.emit:
            dump_json(args.emit, doc)
        else:
            common.write_partial(EXP, args, config, rows, notes, False)

    plans = {d: plan_draft(d, args.head, args.vocab, args.ctx) for d in args.D}
    if args.calib:
        try:
            config["calibration"] = calibrate(args.variant, [(17408, 5120), (5120, 17408), (args.vocab, 1024)], tag, l2)
            print("[e3d] calibration %s" % config["calibration"], flush=True)
        except Exception as e:
            notes.append(f"calibration failed: {type(e).__name__}: {e}")
        torch.cuda.empty_cache()

    v_mib, scaled = choose_verify_mib(args.V, free_vram(), max(args.D))
    shapes = plan_verify(int(v_mib * MIB))
    ver = Verify(shapes, args.variant, seed=77)
    config.update(verify_requested_mib=args.V, verify_mib=v_mib, verify_scaled_down=scaled, verify_matrices=len(shapes),
                  verify_bytes=e1.set_bytes(shapes), verify_arena_bytes=ver.arena.nbytes, verify_model="Bonsai 2 27B shapes")
    if scaled:
        notes.append(f"VRAM short: verify set scaled from {args.V:.0f} to {v_mib:.0f} MiB")
    vg: dict = {}

    def vgraph(kind: str, cols: int):
        if (kind, cols) not in vg:
            vg[(kind, cols)] = build_verify_graph(ver, cols, kind == "stream")
        return vg[(kind, cols)]

    probe = probe_midcapture(ver) if "S3" in args.scenarios else None
    config["s3_probe"] = probe
    s3_ok = bool(probe and probe.get("supported"))
    if probe is not None and not s3_ok:
        notes.append("S3 skipped: stream window changed inside graph capture was not honored: %s" % probe)
    cells = make_cells(args.scenarios, args.K, s3_ok)
    cols_for = (lambda k: 1) if args.verify_cols == "1" else (lambda k: k + 1)
    if args.verify_cols != "1" and len(shapes) * (max(args.K) + 1) > MAX_VERIFY_NODES:
        cols_for = lambda k: 1  # noqa: E731
        notes.append("verify-cols kp1 needs too many graph nodes; used 1 column")
    config["verify_cols"] = {str(k): cols_for(k) for k in args.K}
    snapshot()

    for d in args.D:
        if budget.expired(60):
            notes.append(f"time budget used up before D={d}; this and larger D skipped")
            break
        pl = plans[d]
        mrows: dict = {}
        try:
            model = DraftModel(pl["cfg"], args.vocab, args.head, args.variant, seed=d)
            chk = model.check()
            wp = window_params(model.total, wmax, pmax, args.hit_ratio)
            graphs = {"off": build_draft_graph(model, None, 0), "on": build_draft_graph(model, wp["hit_ratio"], wp["window_bytes"])}
            limits = {"off": 0, "on": wp["limit"]}
            for (scn, k, mode) in cells:
                kind = "stream" if scn == "S3" else "plain"
                mrows[(scn, k, mode)] = {
                    "D_mib": d, "D_bytes": model.total, "cfg": cfg_json(pl["cfg"]), "bytes": model.bytes,
                    "step_bytes": model.step_bytes, "plan_err": pl["err"], "pdl": pdl, "scenario": scn, "K": k, "mode": mode,
                    "verify": None if scn == "S1" else kind, "verify_cols": None if scn == "S1" else cols_for(k),
                    "hit_ratio": wp["hit_ratio"] if mode == "on" else None, "window_bytes": wp["window_bytes"] if mode == "on" else 0,
                    "window_clamped": wp["clamped"] and mode == "on", "persist_limit_bytes": limits[mode],
                    "window_readback": graphs[mode][1].get("window_readback"), "kernel_nodes": graphs[mode][1].get("kernel_nodes"),
                    "finite": chk["finite"], "check_ok": chk["ok"], "max_abs_err": chk["max_abs_err"]}
            order = list(mrows)
            for rnd in range(args.rounds):
                if rnd >= 3 and budget.expired(30):
                    notes.append(f"D={d}: stopped after {rnd} rounds (time budget)")
                    break
                j = rnd % len(order)
                for key in order[j:] + order[:j]:
                    scn, k, mode = key
                    r = mrows[key]
                    set_persist(limits[mode])
                    dg = graphs[mode][0]
                    if scn == "S1":
                        r.setdefault("step_us_all", []).append(time_steady(dg, k, args.bursts))
                        continue
                    v = vgraph("stream" if scn == "S3" else "plain", cols_for(k))[0]
                    s = summarize_cycles(time_cycles(dg, v, k, args.cycles))
                    for m in METRICS:
                        r.setdefault(m + "_all", []).append(s[m])
                    r.setdefault("by_index_all", []).append(s["by_index"])
            set_persist(0)
            config.setdefault("nvml_per_D", {})[str(d)] = nvml_state()
            for r in mrows.values():
                for m in METRICS:
                    a = r.get(m + "_all") or []
                    r[m] = statistics.median(a) if a else None
                    r[m + "_spread"] = _spread(a)
                if r.get("by_index_all"):
                    r["by_index"] = [statistics.median(c) for c in zip(*r["by_index_all"])]
                if r.get("step_us"):
                    r["gbps_eff"] = r["step_bytes"] / (r["step_us"] * 1e3)
                if r.get("verify_us"):
                    r["verify_gbps"] = config["verify_bytes"] * r["verify_cols"] / (r["verify_us"] * 1e3)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            notes.append(f"D={d} failed: {err}")
            for (scn, k, mode) in cells:
                r = mrows.setdefault((scn, k, mode), {"D_mib": d, "pdl": pdl, "scenario": scn, "K": k, "mode": mode})
                r["error"] = err
                r.setdefault("step_us", None)
        set_persist(0)
        rows.extend(mrows.values())
        snapshot()
        for r in mrows.values():
            print("[e3d] pdl=%d D=%d %s K=%d %s step=%s first=%s later=%s cycle=%s" % (
                pdl, d, r["scenario"], r["K"], r["mode"], r.get("step_us"), r.get("first_us"), r.get("later_us"),
                r.get("cycle_us")), flush=True)
        graphs = model = None
        torch.cuda.empty_cache()
    return {"config": config, "rows": rows, "notes": notes}


def child_argv(a, pdl: int, emit: str, budget_min: float, calib: bool) -> list[str]:
    argv = common.py_module_argv("exps.e3d_resident_draft", "--pdl", str(pdl), "--emit", emit, "--D", ",".join(map(str, a.D)),
                                 "--K", ",".join(map(str, a.K)), "--V", str(a.V), "--rounds", str(a.rounds),
                                 "--cycles", str(a.cycles), "--bursts", str(a.bursts), "--head", a.head,
                                 "--vocab", str(a.vocab), "--ctx", str(a.ctx), "--variant", a.variant,
                                 "--hit-ratio", str(a.hit_ratio), "--verify-cols", a.verify_cols,
                                 "--scenarios", ",".join(a.scenarios), "--budget-min", "%.2f" % budget_min)
    if a.quick:
        argv.append("--quick")
    if not calib:
        argv.append("--no-calib")
    if a.gpu:
        argv += ["--gpu", a.gpu]
    return argv


def run_both(a, pdls: list[int]) -> dict:
    """One child process per PDL setting (the knob is read once per process); merge the emitted rows."""
    import tempfile
    budget = common.Budget(a.budget_min if a.budget_min else (8 if a.quick else 26))
    docs, notes, rcs = {}, [], {}
    tmp = tempfile.mkdtemp(prefix="cr-e3d-")
    for i, pdl in enumerate(pdls):
        emit = os.path.join(tmp, f"pdl{pdl}.json")
        child_min = max(1.0, budget.left() / 60.0 / (len(pdls) - i) - 0.3)
        argv = child_argv(a, pdl, emit, child_min, calib=(i == 0) and a.calib)
        try:
            rc = subprocess.run(argv, cwd=str(common.EXP_DIR), timeout=child_min * 60 + 180).returncode
        except subprocess.TimeoutExpired:
            rc = -9
        rcs[pdl] = rc
        if rc != 0:
            notes.append(f"pdl={pdl} child exit {rc}")
        try:
            with open(emit, encoding="ascii") as f:
                docs[pdl] = json.load(f)
        except (OSError, ValueError):
            notes.append(f"pdl={pdl} child left no rows")
    if not docs:
        return {"config": {"child_exit": rcs}, "rows": [], "notes": notes}
    first = docs[min(docs)]
    config = dict(first["config"])
    config.update(pdl_modes=sorted(docs), child_exit=rcs, pdl_configs={str(p): d["config"] for p, d in docs.items()})
    rows = [r for p in sorted(docs) for r in docs[p]["rows"]]
    for p in sorted(docs):
        notes += ["[pdl=%d] %s" % (p, n) for n in docs[p]["notes"]]
    return {"config": config, "rows": rows, "notes": notes}


def finalize(args, doc: dict) -> int:
    rows, notes = doc["rows"], doc["notes"]
    add_deltas(rows)
    add_cross(rows)
    proj = projection(rows)
    notes.append("Deltas are off - on per round (positive = on faster); noise = half range of the paired deltas. "
                 "S2/S3 first_us = first draft step after the verify pass, later_us = mean of the other steps; refill_us = first - later. "
                 "Window: arena start, clamped to the device window max; hitRatio auto = min(1, limit / window). "
                 "PDL knob also applies to the verify graph. Verify uses batch-1 GEMV as a bandwidth proxy.")
    if any(r.get("finite") is False or r.get("check_ok") is False for r in rows):
        notes.append("WARNING: some draft outputs not finite or not matching the reference")
    common.write_result(EXP, args, doc["config"], rows, notes, False, extra={"projection": proj})
    return 0 if rows else 1


# ---------------------------------------------------------------- CLI

def parse_list(text: str, cast=int) -> list:
    return [cast(t) for t in text.split(",") if t.strip()]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="exps.e3d_resident_draft", description=__doc__.split("\n")[0])
    common.add_common_args(ap)
    ap.add_argument("--D", type=parse_list, default=None, help="draft working set sizes in MiB")
    ap.add_argument("--K", type=parse_list, default=None, help="draft steps per cycle")
    ap.add_argument("--V", type=float, default=None, help="verify weights MiB (default 6144, scaled down if VRAM is short)")
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--cycles", type=int, default=None, help="verify cycles per timing (S2, S3)")
    ap.add_argument("--bursts", type=int, default=None, help="K-step bursts per timing (S1)")
    ap.add_argument("--head", choices=["ternary", "int8"], default="ternary")
    ap.add_argument("--vocab", type=int, default=VOCAB)
    ap.add_argument("--ctx", type=int, default=CTX, help="draft KV length")
    ap.add_argument("--variant", choices=["lut", "fp4"], default="lut")
    ap.add_argument("--pdl", choices=["0", "1", "both"], default="both")
    ap.add_argument("--hit-ratio", default="auto", help="auto = min(1, limit / window), or a number in (0, 1]")
    ap.add_argument("--verify-cols", choices=["1", "kp1"], default="1", help="GEMV launches per verify matrix (kp1 = K+1)")
    ap.add_argument("--scenarios", default="S1,S2,S3")
    ap.add_argument("--no-calib", dest="calib", action="store_false")
    ap.add_argument("--emit", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    q = a.quick
    a.D = a.D or ([16, 72] if q else list(DEFAULT_D))
    a.K = a.K or ([4] if q else list(DEFAULT_K))
    a.V = a.V or (1024.0 if q else float(DEFAULT_V_MIB))
    a.rounds = a.rounds or (3 if q else 7)
    a.cycles = a.cycles or (12 if q else 30)
    a.bursts = a.bursts or (15 if q else 40)
    a.scenarios = [s for s in a.scenarios.split(",") if s]
    if any(s not in SCENARIOS for s in a.scenarios):
        ap.error("scenarios must be from S1,S2,S3")
    if a.hit_ratio != "auto" and not 0.0 < float(a.hit_ratio) <= 1.0:
        ap.error("--hit-ratio must be auto or in (0, 1]")
    return a


def plan_json(a) -> dict:
    ds = []
    for d in a.D:
        p = plan_draft(d, a.head, a.vocab, a.ctx)
        ds.append({"D_mib": d, "cfg": cfg_json(p["cfg"]), "bytes": p["layout"]["bytes"], "step_bytes": p["layout"]["step_bytes"],
                   "plan_err": p["err"], "within_tol": p["within_tol"]})
    shapes = plan_verify(int(a.V * MIB))
    pdls = [0, 1] if a.pdl == "both" else [int(a.pdl)]
    return {"exp": EXP, "D": ds, "K": a.K, "pdl": pdls, "scenarios": a.scenarios, "rounds": a.rounds, "cycles": a.cycles,
            "bursts": a.bursts, "head": a.head, "vocab": a.vocab, "hit_ratio": a.hit_ratio, "verify_cols": a.verify_cols,
            "verify": {"V_mib": a.V, "matrices": len(shapes), "bytes": e1.set_bytes(shapes)},
            "cells_per_D": len(make_cells(a.scenarios, a.K)), "variant": a.variant, "calib": a.calib, "quick": a.quick}


def main(argv=None) -> int:
    a = parse_args(argv)
    if a.dry_run:
        common.print_plan(plan_json(a))
        return 0
    if not common.have_cuda():
        print("error: no CUDA device (use --dry-run for the plan)", file=sys.stderr)
        return 2
    with common.term_exits():
        if a.emit:  # child: one PDL setting, rows go to the emit file
            doc = run_matrix(a, int(a.pdl))
            dump_json(a.emit, doc)
            return 0
        a.gpu = common.gpu_tag(a)
        if a.pdl == "both":
            cc = device_cc_major()
            if cc is not None and cc < 9:
                doc = run_matrix(a, 0)
                doc["notes"].append("PDL needs sm_90+; only the no-PDL run was done")
            else:
                doc = run_both(a, [0, 1])
        else:
            doc = run_matrix(a, int(a.pdl))
        return finalize(a, doc)


if __name__ == "__main__":
    sys.exit(main())
