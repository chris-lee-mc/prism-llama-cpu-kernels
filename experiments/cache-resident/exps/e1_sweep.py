"""E1: two-tier working-set sweep with persisting L2 (PLAN section 4).

One decode token = GEMVs over a hot set W (the persistence candidate, [4096 x 4096] ternary layers in one arena),
then GEMVs over a fixed cold set C (default 512 MiB, read-only, Bonsai 2 27B MLP shapes, own arena), all with the
P0.4 cr_gemv kernel in one CUDA graph. "on" modes put an access policy window over the hot arena on the graph
kernel nodes (accesses outside the window, i.e. the cold set, stay normal). No subtraction anywhere:
  us_per_token  whole token (hot + cold), gross.
  hot_us/cold_us  split: a 1-token hot graph and a 1-token cold graph replayed in turn with events between them.
  C=0 rows: hot only (natural L2 residency baseline); there hot_us = us_per_token.
Modes are timed in interleaved rounds (each round runs every mode once, order rotated), so off-vs-on deltas
are paired per round; the round spread is the noise estimate. Every GEMV reads a fixed random x, so outputs stay
finite; the last hot and cold outputs are checked against the fp32 reference.
A one-off calibration (copy bandwidth, kernels.bench_gemv on the hot and cold shapes) and a W=1024 chain
diagnostic (fixed x vs the old ping-pong chain) are stored in config. ncu is not used (no counters on RunPod).

  python -m exps.e1_sweep [--W 16,32,...] [--C 512] [--modes off,0.5,0.75,1.0,1.0L] [--tokens 200] [--rounds 5]
                          [--variant lut|fp4] [--no-calib] [--quick] [--dry-run]
Mode "<r>" = hitRatio r with persisting limit = device max; "<r>L" = limit min(W, max).
"""
from __future__ import annotations

import argparse
import statistics
import sys

from exps import common

EXP = "e1"
DEFAULT_W = [8, 16, 24, 32, 48, 56, 64, 80, 96, 128, 192, 256, 512, 1024]
DEFAULT_MODES = ["off", "0.5", "0.75", "1.0", "1.0L"]
HOT_SHAPE = (4096, 4096)
COLD_SHAPES = [(17408, 5120), (17408, 5120), (5120, 17408)]  # 27B MLP gate, up, down
DEFAULT_C_MIB = 512
ALIGN = 256
MAX_NODES = 2000   # kernel nodes per graph
MAX_TPG = 20       # tokens per graph
STOP_GAIN = 0.20
STOP_GPUS = ("rtx5090", "pro6000")
DEFAULT_PERSIST_MIB = 72
DIAG_W = 1024
TOL = 2e-3


# ---------------------------------------------------------------- pure helpers

def round_up(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def layer_bytes(n: int, k: int) -> tuple[int, int]:
    """(q bytes, scale bytes) of one ternary layer."""
    return n * k // 4, n * (k // 128) * 2


def layer_stride(n: int, k: int) -> int:
    q, s = layer_bytes(n, k)
    return round_up(q + s, ALIGN)


def plan_hot(w_mib: float, shape=HOT_SHAPE) -> int:
    return max(1, int(round(w_mib * common.MIB / sum(layer_bytes(*shape)))))


def plan_cold(c_mib: float, shapes=COLD_SHAPES) -> list[tuple[int, int]]:
    """Cycle the shapes; count = round(C / mean layer bytes). C=0 -> []."""
    if c_mib <= 0:
        return []
    mean = sum(sum(layer_bytes(*s)) for s in shapes) / len(shapes)
    n = max(1, int(round(c_mib * common.MIB / mean)))
    return [shapes[i % len(shapes)] for i in range(n)]


def set_bytes(shapes) -> int:
    return sum(sum(layer_bytes(*s)) for s in shapes)


def parse_modes(text: str) -> list[tuple[str, float | None, str | None]]:
    """'off,0.5,1.0L' -> [('off', None, None), ('0.5', 0.5, 'max'), ('1.0L', 1.0, 'W')]. Ratios in (0, 1]."""
    out = []
    for tok in [t.strip() for t in text.split(",") if t.strip()]:
        if tok == "off":
            out.append(("off", None, None))
            continue
        policy = "W" if tok.endswith("L") else "max"
        r = float(tok.rstrip("L"))
        if not 0.0 < r <= 1.0:
            raise ValueError(f"hit ratio must be in (0, 1], got {tok}")
        out.append((tok, r, policy))
    return out


def parse_w(text: str) -> list[int]:
    return [int(t) for t in text.split(",") if t.strip()]


def limit_for(policy: str | None, hot_bytes: int, persist_max: int) -> int:
    if policy is None:
        return 0
    return min(hot_bytes, persist_max) if policy == "W" else persist_max


def window_plan(arena_bytes: int, window_max: int, limit: int, hit_ratio: float) -> dict:
    """Window over the hot arena start, clamped to the device window max."""
    nbytes = min(arena_bytes, window_max) if window_max else arena_bytes
    return {"window_bytes": nbytes, "clamped": nbytes < arena_bytes, "hit_ratio": hit_ratio,
            "persisted_bytes_est": min(limit, int(nbytes * hit_ratio))}


def tokens_per_graph(nodes_per_token: int, max_nodes: int = MAX_NODES, cap: int = MAX_TPG) -> int:
    return max(1, min(cap, max_nodes // max(1, nodes_per_token)))


def replays_for(tokens: int, tpg: int) -> int:
    return max(1, -(-tokens // tpg))


def _spread(xs):
    return (max(xs) - min(xs)) if len(xs) >= 2 else None


def paired(off_all, on_all) -> dict | None:
    """Per-round paired delta off - on. Noise = half range of the paired deltas (None with < 3 rounds)."""
    d = [a - b for a, b in zip(off_all or [], on_all or []) if a is not None and b is not None]
    if not d:
        return None
    med = statistics.median(d)
    noise = (max(d) - min(d)) / 2 if len(d) >= 3 else None
    return {"delta": med, "all": d, "noise": noise, "above": None if noise is None else abs(med) > noise}


def add_deltas(rows: list[dict]) -> None:
    """Add delta_vs_off_us / gain_vs_off (whole token) and hot_delta_vs_off_us / hot_gain_vs_off to each non-off row."""
    for r in rows:
        if r["mode"] == "off":
            continue
        off = next((o for o in rows if o["W_mib"] == r["W_mib"] and o["C_mib"] == r["C_mib"] and o["mode"] == "off"), None)
        if off is None:
            continue
        for key, pre in (("us_per_token", ""), ("hot_us", "hot_")):
            p = paired(off.get(key + "_all"), r.get(key + "_all"))
            if p is None or not off.get(key):
                continue
            r[pre + "delta_vs_off_us"] = p["delta"]
            r[pre + "delta_all"] = p["all"]
            r[pre + "delta_noise_us"] = p["noise"]
            r[pre + "delta_above_noise"] = p["above"]
            r[pre + "gain_vs_off"] = p["delta"] / off[key]


def best_gain(rows: list[dict], w: int, c_mib: int) -> tuple[float, str] | None:
    """Best hot gain of any on mode at (W, C): the hot split for C > 0, the whole token for C = 0."""
    g = [(r["hot_gain_vs_off"], r["mode"]) for r in rows
         if r["W_mib"] == w and r["C_mib"] == c_mib and r["mode"] != "off" and r.get("hot_gain_vs_off") is not None]
    return max(g) if g else None


def stop_check(rows: list[dict], gpu: str, persist_mib: float | None, c_mib: int) -> tuple[bool | None, str]:
    """PLAN stop rule: 'on' gain >= 20% for some W <= persisting max on rtx5090 / pro6000. (met, note); met None = not applicable."""
    pm = persist_mib or DEFAULT_PERSIST_MIB
    ws = sorted({r["W_mib"] for r in rows if r["W_mib"] <= pm and r["C_mib"] == c_mib})
    got = [(w, *bg) for w in ws if (bg := best_gain(rows, w, c_mib))]
    if not got:
        return None, f"stop rule: no on/off pairs at W <= {pm:.0f} MiB, C={c_mib}"
    w, g, m = max(got, key=lambda t: t[1])
    ok = g >= STOP_GAIN
    applies = any(gpu.startswith(x) for x in STOP_GPUS)
    txt = "best hot gain %.1f%% at W=%d mode %s (W <= %.0f MiB, C=%d)" % (100 * g, w, m, pm, c_mib)
    if not applies:
        return None, "stop rule n/a on %s: %s" % (gpu, txt)
    if ok:
        return True, "stop rule met (>= 20%%): " + txt
    return False, "STOP: persisting 'on' gain < 20%% for all W <= %.0f MiB (%s); debug P0.2 before continuing" % (pm, txt)


# ---------------------------------------------------------------- GPU side

class Arena:
    """Ternary layers of the given shapes in one 256 B aligned buffer; one random instance per (shape, pool slot)."""

    def __init__(self, shapes, seed: int = 0, pool: int = 4, device: str = "cuda"):
        import torch
        from kernels import ternary
        self.shapes = list(shapes)
        self.offs, o = [], 0
        for n, k in self.shapes:
            self.offs.append(o)
            o += layer_stride(n, k)
        self.buf = torch.zeros(max(o, ALIGN), dtype=torch.uint8, device=device)
        self.base = self.buf.data_ptr()
        self.nbytes = o
        self.bytes = set_bytes(self.shapes)
        cache = {}
        for i, (n, k) in enumerate(self.shapes):
            key = (n, k, i % pool)
            if key not in cache:
                q, s, _ = ternary.random_ternary(n, k, seed=seed + 7919 * len(cache), device=device)
                cache[key] = (q.contiguous().view(torch.uint8).reshape(-1), s.contiguous().view(torch.uint8).reshape(-1))
            qv, sv = cache[key]
            o = self.offs[i]
            self.buf[o:o + qv.numel()].copy_(qv)
            self.buf[o + qv.numel():o + qv.numel() + sv.numel()].copy_(sv)
        del cache

    def qs(self, i):
        """(q int32 [N, K/16], s fp16 [N, K/128]) views of layer i."""
        import torch
        n, k = self.shapes[i]
        qb, sb = layer_bytes(n, k)
        o = self.offs[i]
        q = self.buf[o:o + qb].view(torch.int32).view(n, k // 16)
        s = self.buf[o + qb:o + qb + sb].view(torch.float16).view(n, k // 128)
        return q, s

    def ptrs(self, i):
        n, k = self.shapes[i]
        o = self.base + self.offs[i]
        return o, o + layer_bytes(n, k)[0]


class Layers:
    """An arena plus a fixed random fp16 x (rms 1) and an fp16 y. Every layer reads x and writes y."""

    def __init__(self, shapes, variant: str, seed: int = 0, device: str = "cuda"):
        import torch
        from kernels import ternary
        self.torch, self.ternary, self.variant = torch, ternary, variant
        self.arena = Arena(shapes, seed=seed, device=device)
        self.shapes = self.arena.shapes
        kmax = max((k for _, k in self.shapes), default=128)
        nmax = max((n for n, _ in self.shapes), default=1)
        g = torch.Generator(device=device).manual_seed(seed + 1)
        self.x = torch.randn(kmax, device=device, dtype=torch.float32, generator=g).half()
        self.y = torch.zeros(max(nmax, kmax), device=device, dtype=torch.float16)
        self.y2 = torch.zeros_like(self.y)

    def launch(self, stream: int) -> None:
        xp, yp = self.x.data_ptr(), self.y.data_ptr()
        for i, (n, k) in enumerate(self.shapes):
            q, s = self.arena.ptrs(i)
            self.ternary.gemv_ptr(self.variant, q, s, xp, yp, n, k, stream)

    def launch_pingpong(self, stream: int) -> None:
        """Old E1 chain: layer i reads the output of layer i-1 (y and y2 alternate). Square shapes only."""
        b = [self.y.data_ptr(), self.y2.data_ptr()]
        for i, (n, k) in enumerate(self.shapes):
            q, s = self.arena.ptrs(i)
            self.ternary.gemv_ptr(self.variant, q, s, b[i % 2], b[(i + 1) % 2], n, k, stream)

    def check(self) -> dict:
        """Run the last layer alone and compare y with the fp32 reference."""
        torch = self.torch
        if not self.shapes:
            return {"finite": True, "ok": True, "max_abs_err": 0.0}
        i = len(self.shapes) - 1
        n, k = self.shapes[i]
        q, s = self.arena.qs(i)
        qp, sp = self.arena.ptrs(i)
        self.ternary.gemv_ptr(self.variant, qp, sp, self.x.data_ptr(), self.y.data_ptr(), n, k)
        torch.cuda.synchronize()
        y = self.y[:n].float()
        ref = self.ternary.ref_gemv(q, s, self.x[:k])
        err = (y - ref).abs().max().item()
        fin = bool(torch.isfinite(y).all().item())
        ok = fin and bool(torch.allclose(y, ref, rtol=TOL, atol=TOL * ref.abs().max().item()))
        return {"finite": fin, "ok": ok, "max_abs_err": err, "y_rms": y.pow(2).mean().sqrt().item()}


def build_graph(fn, ratio: float | None, win_ptr: int, win_bytes: int) -> tuple[object, dict]:
    """Capture fn into a graph; for ratio not None set the window (hot arena) on every kernel node."""
    from harness import l2persist, timing
    l2persist.reset()
    common.clear_capture_windows()
    g = timing.capture_graph(fn, keep_graph=True)
    info = {"kernel_nodes": None}
    if ratio is not None:
        n = l2persist.apply_window_to_graph(g, win_ptr, win_bytes, ratio)
        if n <= 0:
            raise RuntimeError("no kernel nodes found in graph")
        info["kernel_nodes"] = n
    info["window_readback"] = common.window_readback(g, graph=True)
    g.instantiate()
    return g, info


def set_persist(limit: int) -> int:
    import torch
    from harness import l2persist
    torch.cuda.synchronize()
    l2persist.reset()
    return l2persist.set_limit(limit)


def time_graph(g, tpg: int, tokens: int) -> float:
    from harness import timing
    return timing.replay_time_us(g, iters=replays_for(tokens, tpg), warmup=2) / tpg


def time_split(hot_g, cold_g, tokens: int) -> tuple[float, float]:
    """Median us of the hot and the cold part of one token (1-token graphs replayed in turn, events between)."""
    import torch
    for _ in range(2):
        hot_g.replay()
        cold_g.replay()
    ev = [[torch.cuda.Event(enable_timing=True) for _ in range(3)] for _ in range(tokens)]
    for e in ev:
        e[0].record()
        hot_g.replay()
        e[1].record()
        cold_g.replay()
        e[2].record()
    ev[-1][2].synchronize()
    hot = statistics.median(e[0].elapsed_time(e[1]) * 1e3 for e in ev)
    cold = statistics.median(e[1].elapsed_time(e[2]) * 1e3 for e in ev)
    return hot, cold


def calibrate(variant: str, shapes, tag: str, l2: int) -> dict:
    """Copy bandwidth and the standalone P0.4 bench (separate weight copies, rotating) for the given shapes."""
    import torch
    from harness import timing
    from infra.gpus import peak_dram_gbps
    from kernels import bench_gemv
    out = {"peak_gbps": peak_dram_gbps(tag)}
    n = 512 * common.MIB
    a = torch.empty(n, dtype=torch.uint8, device="cuda")
    b = torch.empty_like(a)
    us = timing.cuda_time_us(lambda: b.copy_(a), iters=20, warmup=3)
    out["copy_gbps"] = 2 * n / us / 1e3
    del a, b
    torch.cuda.empty_cache()
    gap = bench_gemv.gap_estimate_us()
    out["graph_node_gap_us"] = gap
    out["bench"] = []
    for nk in dict.fromkeys(shapes):
        r = bench_gemv.bench_one(nk[0], nk[1], variant, 200, gap, out["peak_gbps"], l2)
        out["bench"].append({k: r[k] for k in ("N", "K", "us_cold", "us_hot", "gbps_cold", "pct_peak_cold", "ok")})
    return out


def diag_chain(variant: str, w_mib: int) -> dict:
    """W-sized hot chain alone: fixed x vs the old ping-pong chain (y -> x), per layer us and GB/s."""
    import torch
    from harness import timing
    lay = Layers([HOT_SHAPE] * plan_hot(w_mib), variant, seed=3)
    L = len(lay.shapes)
    tpg = tokens_per_graph(L)
    out = {"W_mib": w_mib, "L": L, "tpg": tpg}
    nb = sum(layer_bytes(*HOT_SHAPE))
    common.clear_capture_windows()
    for name, launch in (("fixed_x", lay.launch), ("pingpong", lay.launch_pingpong)):
        def fn(launch=launch):
            st = torch.cuda.current_stream().cuda_stream
            for _ in range(tpg):
                launch(st)
        g = timing.capture_graph(fn)
        us = time_graph(g, tpg, 50) / L
        out[name] = {"us_per_layer": us, "gbps": nb / us / 1e3}
        if name == "pingpong":
            out[name]["finite"] = bool(torch.isfinite(lay.y.float()).all().item() and torch.isfinite(lay.y2.float()).all().item())
        del g
    del lay
    torch.cuda.empty_cache()
    return out


def run(args) -> int:
    import torch
    from harness import l2persist
    notes: list[str] = []
    if not torch.cuda.is_available():
        print("error: no CUDA device (use --dry-run for the plan)", file=sys.stderr)
        return 2
    tag = common.gpu_tag(args)
    args.gpu = tag
    dev = l2persist.log_device(0)
    l2 = int(dev.get("l2_bytes") or 96 * common.MIB)
    pmax = int(dev.get("persisting_max_bytes") or 0)
    wmax = l2persist.max_window_bytes()
    cold_shapes = plan_cold(args.C)
    budget = common.Budget(args.budget_min if args.budget_min else (8 if args.quick else 30))
    config = {"design": "hotcold", "W_mib": args.W, "C_mib": args.C, "modes": [m for m, _, _ in args.modes],
              "tokens": args.tokens, "rounds": args.rounds, "variant": args.variant, "hot_shape": list(HOT_SHAPE),
              "cold_shapes": [list(s) for s in cold_shapes], "cold_bytes": set_bytes(cold_shapes),
              "max_nodes_per_graph": MAX_NODES, "quick": args.quick, "device": dev, "window_max_bytes": wmax}
    rows: list[dict] = []

    def save():
        common.write_partial(EXP, args, config, rows, notes, False)

    if args.calib:
        try:
            config["calibration"] = calibrate(args.variant, [HOT_SHAPE] + list(dict.fromkeys(cold_shapes)), tag, l2)
            print("[e1] calibration %s" % config["calibration"], flush=True)
        except Exception as e:
            notes.append(f"calibration failed: {type(e).__name__}: {e}")
        try:
            config["diag"] = diag_chain(args.variant, min(DIAG_W, max(args.W)))
            print("[e1] diag %s" % config["diag"], flush=True)
        except Exception as e:
            notes.append(f"chain diag failed: {type(e).__name__}: {e}")
        torch.cuda.empty_cache()

    cold = Layers(cold_shapes, args.variant, seed=1000) if cold_shapes else None
    common.clear_capture_windows()
    cold_g = None
    if cold is not None:
        cold_g, _ = build_graph(lambda: cold.launch(torch.cuda.current_stream().cuda_stream), None, 0, 0)
    cold_check = cold.check() if cold is not None else None
    config["cold_check"] = cold_check
    Cs = [args.C, 0] if cold is not None else [0]

    for w in args.W:
        if budget.expired(60):
            notes.append(f"time budget used up before W={w}; this and larger W skipped")
            break
        hot = Layers([HOT_SHAPE] * plan_hot(w), args.variant, seed=w)
        L, Lc = len(hot.shapes), len(cold_shapes)
        tpg = {args.C: tokens_per_graph(L + Lc), 0: tokens_per_graph(L)}
        mrows, graphs = {}, {}
        try:
            for label, ratio, policy in args.modes:
                limit = limit_for(policy, hot.arena.bytes, pmax)
                plan = window_plan(hot.arena.nbytes, wmax, limit, ratio or 0.0) if ratio else None
                wb = plan["window_bytes"] if plan else 0
                gs = {}
                for c in Cs:
                    def fn(c=c):
                        st = torch.cuda.current_stream().cuda_stream
                        for _ in range(tpg[c]):
                            hot.launch(st)
                            if c:
                                cold.launch(st)
                    gs[c] = build_graph(fn, ratio, hot.arena.base, wb)
                if cold is not None:
                    gs["hot1"] = build_graph(lambda: hot.launch(torch.cuda.current_stream().cuda_stream), ratio, hot.arena.base, wb)
                graphs[label] = (gs, limit)
                for c in Cs:
                    mrows[(label, c)] = {
                        "W_mib": w, "C_mib": c, "L": L, "L_cold": Lc if c else 0, "hot_bytes": hot.arena.bytes,
                        "cold_bytes": set_bytes(cold_shapes) if c else 0, "mode": label, "hit_ratio": ratio,
                        "limit_policy": policy, "window_bytes": wb, "window_clamped": bool(plan and plan["clamped"]),
                        "window_readback": gs[c][1].get("window_readback"), "kernel_nodes": gs[c][1].get("kernel_nodes"),
                        "tokens_per_graph": tpg[c], "us_per_token_all": [], "hot_us_all": [], "cold_us_all": []}
            order = [m for m, _, _ in args.modes]
            for rnd in range(args.rounds):
                k = rnd % len(order)
                for label in order[k:] + order[:k]:
                    gs, limit = graphs[label]
                    got = set_persist(limit)
                    for c in Cs:
                        r = mrows[(label, c)]
                        r["persist_limit_bytes"] = got
                        r["us_per_token_all"].append(time_graph(gs[c][0], tpg[c], args.tokens))
                        if c:
                            h, cu = time_split(gs["hot1"][0], cold_g, args.tokens)
                            r["hot_us_all"].append(h)
                            r["cold_us_all"].append(cu)
                        else:
                            r["hot_us_all"].append(r["us_per_token_all"][-1])
            set_persist(0)
            chk = hot.check()
            for r in mrows.values():
                for key in ("us_per_token", "hot_us", "cold_us"):
                    a = r[key + "_all"]
                    r[key] = statistics.median(a) if a else None
                    r[key + "_spread"] = _spread(a)
                if not r["cold_us_all"]:
                    r["cold_us"] = None
                r["gbps_eff"] = (r["hot_bytes"] + r["cold_bytes"]) / (r["us_per_token"] * 1e3)
                r["hot_gbps_eff"] = r["hot_bytes"] / (r["hot_us"] * 1e3)
                r["cold_gbps_eff"] = r["cold_bytes"] / (r["cold_us"] * 1e3) if r["cold_us"] else None
                r.update(finite=chk["finite"] and (cold_check or {"finite": True})["finite"], check_ok=chk["ok"],
                         max_abs_err=chk["max_abs_err"])
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            notes.append(f"W={w} failed: {err}")
            for label, ratio, policy in args.modes:
                for c in Cs:
                    r = mrows.setdefault((label, c), {"W_mib": w, "C_mib": c, "L": L, "mode": label, "hit_ratio": ratio,
                                                      "limit_policy": policy})
                    r["error"] = err
                    for key in ("us_per_token", "hot_us", "cold_us"):
                        r.setdefault(key, None)
        set_persist(0)
        rows.extend(mrows.values())
        save()
        for (label, c), r in mrows.items():
            print("[e1] W=%d C=%d mode=%s us/token=%s hot=%s cold=%s" % (w, c, label, r.get("us_per_token"), r.get("hot_us"),
                                                                         r.get("cold_us")), flush=True)
        del graphs, hot
        torch.cuda.empty_cache()

    add_deltas(rows)
    met, note = stop_check(rows, tag, pmax / common.MIB if pmax else None, args.C if cold is not None else 0)
    config["stop_rule_met"] = met
    notes.append(note)
    if cold is not None:
        notes.append("C=0 (hot only, for reference): " + stop_check(rows, tag, pmax / common.MIB if pmax else None, 0)[1])
    notes.append("us_per_token: whole token, gross, no subtraction. hot_us / cold_us: split from 1-token hot and cold graphs "
                 "replayed in turn with events between (includes one graph launch gap each). Deltas are paired per round; "
                 "delta_noise_us is half the range of the paired deltas. Window = hot arena only, set on all kernel nodes; "
                 "cold reads fall outside it (normal).")
    if any(r.get("finite") is False or r.get("check_ok") is False for r in rows):
        notes.append("WARNING: some GEMV outputs not finite or not matching the reference")
    common.write_result(EXP, args, config, rows, notes, False)
    return 0


# ---------------------------------------------------------------- CLI

def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="exps.e1_sweep", description=__doc__.split("\n")[0])
    common.add_common_args(ap)
    ap.add_argument("--W", type=parse_w, default=None, help="hot set sizes in MiB, comma separated")
    ap.add_argument("--C", type=int, default=DEFAULT_C_MIB, help="cold set MiB streamed after the hot set (0 = none)")
    ap.add_argument("--modes", default=",".join(DEFAULT_MODES))
    ap.add_argument("--tokens", type=int, default=200, help="tokens per timing (per mode per round)")
    ap.add_argument("--rounds", type=int, default=5, help="interleaved rounds; the median is reported")
    ap.add_argument("--variant", choices=["lut", "fp4"], default="lut")
    ap.add_argument("--no-calib", dest="calib", action="store_false", help="skip the calibration and chain diag")
    a = ap.parse_args(argv)
    if a.W is None:
        a.W = [16, 64] if a.quick else list(DEFAULT_W)
    if a.quick:
        a.tokens = min(a.tokens, 50)
        a.rounds = min(a.rounds, 3)
        if a.modes == ",".join(DEFAULT_MODES):
            a.modes = "off,1.0"
    a.modes = parse_modes(a.modes)
    return a


def plan_json(a) -> dict:
    cold = plan_cold(a.C)
    ws = []
    for w in a.W:
        L = plan_hot(w)
        ws.append({"W_mib": w, "L": L, "hot_bytes": L * sum(layer_bytes(*HOT_SHAPE)),
                   "tokens_per_graph": tokens_per_graph(L + len(cold)), "tokens_per_graph_c0": tokens_per_graph(L)})
    return {"exp": EXP, "modes": [m for m, _, _ in a.modes], "tokens": a.tokens, "rounds": a.rounds,
            "hot_shape": list(HOT_SHAPE), "C_mib": a.C, "cold_layers": len(cold), "cold_bytes": set_bytes(cold),
            "layers": ws, "variant": a.variant, "calib": a.calib, "quick": a.quick}


def main(argv=None) -> int:
    a = parse_args(argv)
    if a.dry_run:
        common.print_plan(plan_json(a))
        return 0
    with common.term_exits():
        return run(a)


if __name__ == "__main__":
    sys.exit(main())
