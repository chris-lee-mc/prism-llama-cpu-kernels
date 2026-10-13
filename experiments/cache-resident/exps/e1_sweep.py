"""E1: synthetic working-set sweep with persisting L2 (PLAN section 4).

A chain of L ternary GEMV layers [4096 x 4096] lives in one 256 B aligned arena. One CUDA graph runs 10
tokens (10*L cr_gemv launches); 100 replays = 1000 tokens. Modes: off, or a persisting window over the
arena with hitRatio 0.5 / 0.75 / 1.0. Per (W, mode) an ncu pass (separate process, --inner) gives the
L2 hit rate and DRAM bytes per token.

Without anything else in the loop a W below the L2 size stays L2 resident even with persisting off. So after
each token the graph runs a streaming kernel over a buffer of 2x L2 (a torch add_, not matched by the cr_gemv
ncu filter), as the rest of a model (KV, other weights) would. Its cost is measured once and subtracted:
us_per_token is the net value, us_per_token_gross the raw one; gross_delta_vs_off_us is the off-minus-on difference of
the raw values, with its noise. With counters the GEMV-only ncu time (ncu_us_per_token, no pollution in it) also
gives ncu_delta_vs_off_us / ncu_gain_vs_off, and the stop rule uses it. W <= 72 MiB get more timing repeats (--small-w-repeats).
--no-pollute turns the pollution off.
Phase 1 times every (W, mode); phase 2 spends the remaining budget on ncu, W nearest the persisting max
and the L2 size first. ncu uses application replay only (no kernel replay fallback).

  python -m exps.e1_sweep [--W 16,32,...] [--modes off,0.5,0.75,1.0] [--tokens 1000] [--variant auto|lut|fp4]
                          [--window-method graph|stream] [--no-pollute] [--quick] [--no-ncu] [--dry-run]
"""
from __future__ import annotations

import argparse
import sys

from exps import common

EXP = "e1"
DEFAULT_W = [16, 32, 48, 64, 80, 96, 128, 192, 256, 512, 1024]
DEFAULT_MODES = ["off", "0.5", "0.75", "1.0"]
N = K = 4096
TOKENS_PER_GRAPH = 10
ALIGN = 256
GAIN = 1.177  # rescale group scales so the activation variance stays near 1 through the chain
STOP_GAIN = 0.20
STOP_W_MIB = 72
STOP_GPUS = ("rtx5090", "pro6000")
OFF_HIT_MAX = 0.5  # stop rule only counts W whose persisting-off hit rate is below this
POLLUTE_MULT = 2.0
SMALL_W_MIB = 72  # at or below this W the off/on delta is small next to run-to-run noise: more repeats
SMALL_W_REPEATS = 7
DEFAULT_L2_BYTES = 96 * common.MIB


# ---------------------------------------------------------------- pure helpers

def round_up(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def layer_bytes(n: int = N, k: int = K) -> tuple[int, int]:
    """(q bytes, scale bytes) of one ternary layer."""
    return n * k // 4, n * (k // 128) * 2


def layer_stride(n: int = N, k: int = K) -> int:
    q, s = layer_bytes(n, k)
    return round_up(q + s, ALIGN)


def plan_layers(w_mib: float, n: int = N, k: int = K) -> int:
    q, s = layer_bytes(n, k)
    return max(1, int(round(w_mib * common.MIB / (q + s))))


def parse_modes(text: str) -> list[tuple[str, float | None]]:
    """'off,0.5' -> [('off', None), ('0.5', 0.5)]. Ratios must be in (0, 1]."""
    out = []
    for tok in [t.strip() for t in text.split(",") if t.strip()]:
        if tok == "off":
            out.append(("off", None))
            continue
        r = float(tok)
        if not 0.0 < r <= 1.0:
            raise ValueError(f"hit ratio must be in (0, 1], got {tok}")
        out.append((tok, r))
    return out


def parse_w(text: str) -> list[int]:
    return [int(t) for t in text.split(",") if t.strip()]


def graph_tokens(tokens: int) -> int:
    """Round tokens up to a whole number of graph replays."""
    return max(1, -(-tokens // TOKENS_PER_GRAPH)) * TOKENS_PER_GRAPH


def window_plan(arena_bytes: int, window_max: int, persist_max: int, hit_ratio: float) -> dict:
    """Window over the arena start: clamped to the device window max."""
    nbytes = min(arena_bytes, window_max) if window_max else arena_bytes
    return {"window_bytes": nbytes, "clamped": nbytes < arena_bytes, "hit_ratio": hit_ratio,
            "persisted_bytes_est": min(persist_max, int(nbytes * hit_ratio))}


def pollute_bytes_for(l2_bytes: int | None, mult: float = POLLUTE_MULT) -> int:
    """Size of the streaming buffer run between tokens: mult x L2, 0 = off."""
    if mult <= 0:
        return 0
    return round_up(int((l2_bytes or DEFAULT_L2_BYTES) * mult), ALIGN)


def repeats_for(w: int, base: int, small: int = SMALL_W_REPEATS) -> int:
    """Timing repeats for a W: small W get at least `small` (the off/on delta is near the noise floor there)."""
    return max(base, small) if w <= SMALL_W_MIB else base


def gross_deltas(rows: list[dict]) -> None:
    """Add gross_delta_vs_off_us (gross off - gross mode, no pollution subtraction) and its noise to each non-off row.

    delta_noise_us is the larger max-min spread of the two rows' repeats; None when either has < 3 repeats.
    delta_above_noise says whether the delta clears it.
    """
    def spread(r):
        a = r.get("us_per_token_all") or []
        return max(a) - min(a) if len(a) >= 3 else None

    for r in rows:
        off = next((o for o in rows if o["W_mib"] == r["W_mib"] and o["mode"] == "off"), None)
        if r["mode"] == "off" or off is None or r.get("us_per_token_gross") is None or off.get("us_per_token_gross") is None:
            continue
        r["gross_delta_vs_off_us"] = off["us_per_token_gross"] - r["us_per_token_gross"]
        sp = [x for x in (spread(off), spread(r)) if x is not None]
        r["delta_noise_us"] = max(sp) if len(sp) == 2 else None
        r["delta_above_noise"] = None if r["delta_noise_us"] is None else r["gross_delta_vs_off_us"] > r["delta_noise_us"]


def ncu_order(rows: list[dict], persist_mib: float | None, l2_mib: float | None) -> list[dict]:
    """ncu job order: W nearest the persisting max or the L2 size first, then W, then the mode order of the rows."""
    marks = [m for m in (persist_mib, l2_mib) if m]
    pos = {id(r): i for i, r in enumerate(rows)}
    return sorted(rows, key=lambda r: (min((abs(r["W_mib"] - m) for m in marks), default=0.0), r["W_mib"], pos[id(r)]))


def _gain_basis(rows: list[dict], w: int) -> tuple[float, float, str] | None:
    """(off value, best on value, basis) for one W. ncu GEMV-only time when off and some on row have it, else net event time."""
    def pick(key, need_counters):
        off = [r[key] for r in rows if r["W_mib"] == w and r["mode"] == "off" and r.get(key) and (r.get("counters_ok") or not need_counters)]
        on = [r[key] for r in rows if r["W_mib"] == w and r["mode"] != "off" and r.get(key) and (r.get("counters_ok") or not need_counters)]
        return (off[0], min(on)) if off and on else None

    for key, need, basis in (("ncu_us_per_token", True, "ncu"), ("us_per_token", False, "net")):
        got = pick(key, need)
        if got:
            return got[0], got[1], basis
    return None


def ncu_deltas(rows: list[dict]) -> None:
    """Add ncu_delta_vs_off_us (GEMV-only ncu time, off minus mode, no pollution kernel in it) to each non-off row that has both."""
    for r in rows:
        off = next((o for o in rows if o["W_mib"] == r["W_mib"] and o["mode"] == "off"), None)
        if r["mode"] == "off" or off is None or not (r.get("counters_ok") and off.get("counters_ok")):
            continue
        if r.get("ncu_us_per_token") and off.get("ncu_us_per_token"):
            r["ncu_delta_vs_off_us"] = off["ncu_us_per_token"] - r["ncu_us_per_token"]
            r["ncu_gain_vs_off"] = r["ncu_delta_vs_off_us"] / off["ncu_us_per_token"]


def stop_flag(rows: list[dict], gpu: str, pollute: bool = True) -> str | None:
    """PLAN stop rule: rtx5090 'on' gain < 20% for all W <= 72 MiB -> debug P0.2.

    Skipped without pollution, and for W already L2 resident with persisting off (nothing left to gain there).
    The gain uses the ncu GEMV-only time when counters worked for the W, else the net event time.
    """
    if not pollute or not any(gpu.startswith(g) for g in STOP_GPUS):
        return None
    resident = {r["W_mib"] for r in rows if r["mode"] == "off" and (r.get("l2_hit_rate") or 0.0) >= OFF_HIT_MAX}
    small = sorted({r["W_mib"] for r in rows if r["W_mib"] <= STOP_W_MIB} - resident)
    if not small:
        return None
    best, bases = [], set()
    for w in small:
        got = _gain_basis(rows, w)
        if got:
            best.append((got[0] - got[1]) / got[0])
            bases.add(got[2])
    if best and all(g < STOP_GAIN for g in best):
        return ("STOP: persisting 'on' gain < 20%% for all W <= %d MiB (best %.1f%%, basis %s); debug P0.2 before continuing"
                % (STOP_W_MIB, 100.0 * max(best), "+".join(sorted(bases))))
    return None


def inner_argv(w: int, mode: str, variant: str, method: str, pollute_bytes: int = 0) -> list[str]:
    extra = ["--pollute-bytes", str(pollute_bytes)] if pollute_bytes else []
    return common.py_module_argv("exps.e1_sweep", "--inner", "--W", str(w), "--modes", mode, "--variant", variant,
                                 "--window-method", method, *extra)


def ncu_args(L: int) -> dict:
    """ncu filter: skip 3 tokens, profile 5 tokens (of the 10 in one replay)."""
    return dict(kernel_regex="cr_gemv", launch_skip=L * 3, launch_count=L * 5)


# ---------------------------------------------------------------- GPU side

def schedule(L: int, n_tokens: int):
    """Yield (layer, src_buf, dst_buf): layer g reads buf[g % 2] and writes buf[(g + 1) % 2], across tokens."""
    g = 0
    for _ in range(n_tokens):
        for i in range(L):
            yield i, g % 2, (g + 1) % 2
            g += 1


class Chain:
    """L ternary GEMV layers in one arena plus two ping-pong fp16 activation buffers.

    pollute_bytes > 0 adds a buffer that is streamed (add_) on the current stream after the last layer of each token.
    """

    def __init__(self, L: int, variant: str, seed: int = 0, pool: int = 4, device: str = "cuda", pollute_bytes: int = 0):
        import torch
        from kernels import ternary
        self.torch, self.ternary = torch, ternary
        self.L, self.variant = L, variant
        qb, sb = layer_bytes()
        self.qb, self.sb, self.stride = qb, sb, layer_stride()
        self.arena = torch.zeros(L * self.stride, dtype=torch.uint8, device=device)
        for p in range(min(L, pool)):
            q, s, _ = ternary.random_ternary(N, K, seed=seed + p, device=device)
            s = (s.float() * GAIN).half()
            qv = q.contiguous().view(torch.uint8).reshape(-1)
            sv = s.contiguous().view(torch.uint8).reshape(-1)
            for i in range(p, L, pool):
                o = i * self.stride
                self.arena[o:o + qb].copy_(qv)
                self.arena[o + qb:o + qb + sb].copy_(sv)
        g = torch.Generator(device=device).manual_seed(seed)
        self.bufs = [torch.randn(K, device=device, dtype=torch.float32, generator=g).half(),
                     torch.zeros(N, device=device, dtype=torch.float16)]
        self.base = self.arena.data_ptr()
        self.bytes_per_token = L * (qb + sb)
        self.arena_bytes = self.arena.numel()
        self.pollute_bytes = pollute_bytes
        self.pbuf = torch.zeros(pollute_bytes // 4, dtype=torch.int32, device=device) if pollute_bytes else None

    def tokens(self, n_tokens: int, stream: int) -> None:
        """Launch n_tokens * L GEMVs; layer g reads buf[g % 2] and writes buf[(g + 1) % 2]."""
        bp = [b.data_ptr() for b in self.bufs]
        for i, src, dst in schedule(self.L, n_tokens):
            o = self.base + i * self.stride
            self.ternary.gemv_ptr(self.variant, o, o + self.qb, bp[src], bp[dst], N, K, stream)
            if self.pbuf is not None and i == self.L - 1:
                self.pbuf.add_(1)

    def finite(self) -> bool:
        return bool(self.torch.isfinite(self.bufs[0].float()).all().item() and
                    self.torch.isfinite(self.bufs[1].float()).all().item())


def choose_variant(requested: str) -> tuple[str, dict]:
    """auto: time lut and fp4 over a pool of layers larger than L2 and take the faster."""
    if requested != "auto":
        return requested, {"requested": requested}
    import torch
    from harness import timing
    l2 = getattr(torch.cuda.get_device_properties(0), "L2_cache_size", 0) or (96 * common.MIB)
    L = max(4, min(120, int(3 * l2 / sum(layer_bytes())) + 1))
    chain = Chain(L, "lut")
    times = {}
    for v in ("lut", "fp4"):
        chain.variant = v
        st = torch.cuda.current_stream().cuda_stream
        times[v] = timing.cuda_time_us(lambda: chain.tokens(1, st), iters=10, warmup=3)
    chain = None
    torch.cuda.empty_cache()
    best = min(times, key=times.get)
    return best, {"requested": "auto", "pool_layers": L, "us_per_pass": times}


def build_graph(chain: Chain, method: str, ratio: float | None, notes: list):
    """Capture one 10-token graph. Returns (graph, info). ratio None = persisting off.

    info["window_readback"] is the window the driver reports for the captured kernels (off rows: must be empty).
    """
    import torch
    from harness import l2persist, timing
    info = {"window_bytes": 0, "persist_limit_bytes": 0, "window_method_used": None, "clamped": False,
            "window_readback": None}
    l2persist.reset()
    common.clear_capture_windows()

    def fn():
        chain.tokens(TOKENS_PER_GRAPH, torch.cuda.current_stream().cuda_stream)

    if ratio is None:
        l2persist.set_limit(0)
        st = torch.cuda.Stream()
        common.clear_capture_windows(st)
        try:
            g = timing.capture_graph(fn, stream=st, keep_graph=True)
            info["window_readback"] = common.window_readback(g, graph=True)
            g.instantiate()
        except Exception:
            g = timing.capture_graph(fn, stream=st)
        info["persist_limit_bytes"] = l2persist.get_limit()
        return g, info
    info["persist_limit_bytes"] = l2persist.set_limit(None)
    plan = window_plan(chain.arena_bytes, l2persist.max_window_bytes(), l2persist.max_persisting_bytes(), ratio)
    info.update(window_bytes=plan["window_bytes"], clamped=plan["clamped"])

    if method == "graph":
        try:
            g = timing.capture_graph(fn, keep_graph=True)
            n = l2persist.apply_window_to_graph(g, chain.arena, plan["window_bytes"], ratio)
            if n <= 0:
                raise RuntimeError("no kernel nodes found in graph")
            info["window_readback"] = common.window_readback(g, graph=True)
            g.instantiate()
            info["window_method_used"] = "graph"
            info["kernel_nodes"] = n
            return g, info
        except Exception as e:
            notes.append(f"graph window failed ({type(e).__name__}: {e}); using stream window")
    st = torch.cuda.Stream()
    try:
        l2persist.set_stream_window(chain.arena, plan["window_bytes"], ratio, stream=st)
        info["window_readback"] = common.window_readback(st)
        g = timing.capture_graph(fn, stream=st)
    finally:
        common.clear_capture_windows(st)
    info["window_method_used"] = "stream"
    return g, info


def measure(chain: Chain, graph, tokens: int, repeats: int) -> dict:
    from harness import timing
    replays = graph_tokens(tokens) // TOKENS_PER_GRAPH
    per = [timing.replay_time_us(graph, iters=replays, warmup=5) / TOKENS_PER_GRAPH for _ in range(repeats)]
    return {"us_per_token": sorted(per)[len(per) // 2], "us_per_token_min": min(per), "us_per_token_all": per,
            "finite": chain.finite()}


def measure_pollution(chain: Chain, repeats: int) -> float:
    """Microseconds per token spent in the pollution kernel (back to back, one graph of TOKENS_PER_GRAPH launches)."""
    if chain.pbuf is None:
        return 0.0
    from harness import timing

    def fn():
        for _ in range(TOKENS_PER_GRAPH):
            chain.pbuf.add_(1)

    g = timing.capture_graph(fn)
    per = [timing.replay_time_us(g, iters=20, warmup=3) / TOKENS_PER_GRAPH for _ in range(repeats)]
    return sorted(per)[len(per) // 2]


def run_inner(args) -> int:
    """ncu target: build, warm up outside the profiled region, then replay once between profiler start/stop."""
    import torch
    from harness import l2persist
    variant, _ = choose_variant(args.variant)
    label, ratio = args.modes[0]  # parse_args already turned --modes into (label, ratio) tuples
    chain = Chain(plan_layers(args.W[0]), variant, pollute_bytes=args.pollute_bytes or 0)
    graph, _ = build_graph(chain, args.window_method, ratio, [])
    graph.replay()
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    graph.replay()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    l2persist.reset()
    return 0


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
    l2_bytes = int(dev.get("l2_bytes") or DEFAULT_L2_BYTES)
    pbytes = pollute_bytes_for(l2_bytes, args.pollute_mult) if args.pollute else 0
    variant, vinfo = choose_variant(args.variant)
    ncu_ok, ncu_note = common.counters_available(args)
    if ncu_note:
        notes.append(ncu_note)
    budget = common.Budget(args.budget_min if args.budget_min else (8 if args.quick else 24))
    config = {"W_mib": args.W, "modes": [m for m, _ in args.modes], "tokens": graph_tokens(args.tokens),
              "tokens_per_graph": TOKENS_PER_GRAPH, "repeats": args.repeats, "small_w_repeats": args.small_w_repeats,
              "small_w_mib": SMALL_W_MIB, "variant_used": variant,
              "variant_choice": vinfo, "window_method": args.window_method, "layer_shape": [N, K],
              "pollute": bool(pbytes), "pollute_bytes": pbytes, "quick": args.quick, "device": dev}
    rows: list[dict] = []

    def save():
        common.write_partial(EXP, args, config, rows, notes, False)

    # phase 1: timing for every (W, mode)
    pol_us = None
    for w in args.W:
        if budget.expired(30):
            notes.append(f"time budget used up before W={w}; this and larger W skipped")
            break
        L = plan_layers(w)
        chain = Chain(L, variant, pollute_bytes=pbytes)
        if pol_us is None:
            pol_us = measure_pollution(chain, args.repeats) if pbytes else 0.0
            config["pollute_us_per_token"] = pol_us
        for label, ratio in args.modes:
            row = {"W_mib": w, "L": L, "bytes_per_token": chain.bytes_per_token, "mode": label,
                   "hit_ratio": ratio, "window_bytes": 0, "persist_limit_bytes": 0, "us_per_token": None,
                   "gbps_eff": None, "l2_hit_rate": None, "dram_bytes_per_token": None, "ncu_us_per_token": None,
                   "counters_ok": False,
                   "pollute_bytes": pbytes, "pollute_us_per_token": pol_us}
            try:
                graph, info = build_graph(chain, args.window_method, ratio, notes)
                row.update(window_bytes=info["window_bytes"], persist_limit_bytes=info["persist_limit_bytes"],
                           window_method_used=info["window_method_used"], window_clamped=info["clamped"],
                           window_readback=info.get("window_readback"))
                m = measure(chain, graph, args.tokens, repeats_for(w, args.repeats, args.small_w_repeats))
                row.update(m)
                gross = m["us_per_token"]
                net = gross - pol_us
                if net <= 0:
                    notes.append(f"W={w} mode={label}: pollution cost {pol_us:.2f} us >= {gross:.2f} us/token; net not subtracted")
                    net = gross
                row.update(us_per_token=net, us_per_token_gross=gross)
                row["gbps_eff"] = chain.bytes_per_token / (net * 1e3)
                del graph
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"
                notes.append(f"W={w} mode={label} failed: {row['error']}")
            l2persist.reset()
            l2persist.set_limit(0)
            rows.append(row)
            save()
            print("[e1] W=%d mode=%s us/token=%s" % (w, label, row["us_per_token"]), flush=True)
        del chain
        torch.cuda.empty_cache()

    gross_deltas(rows)

    # phase 2: ncu, most informative W first
    if ncu_ok:
        for row in rows:
            if row.get("us_per_token") is None:
                row["ncu_error"] = "skipped: no timing"
        jobs = ncu_order([r for r in rows if r.get("us_per_token") is not None],
                         (dev.get("persisting_max_bytes") or 0) / common.MIB, l2_bytes / common.MIB)
        for row in jobs:
            if budget.expired(60):
                row["ncu_error"] = "skipped: time budget"
                continue
            res = common.ncu_run(inner_argv(row["W_mib"], row["mode"], variant, args.window_method, pbytes), 5,
                                 profile_from_start_off=True, app_replay=True, kernel_fallback=False,
                                 timeout=budget.timeout(600, reserve_s=30), env=common.inner_env(), **ncu_args(row["L"]))
            st = common.ncu_stats(res)
            row.update(l2_hit_rate=st["l2_hit_rate"], dram_bytes_per_token=st["dram_bytes_per_token"],
                       counters_ok=st["counters_ok"], ncu_us_per_token=st["ncu_us_per_token"], ncu_error=st["ncu_error"],
                       ncu_replay_mode=st["ncu_replay_mode"],
                       ncu_passes=st["ncu_passes"], ncu_n_kernels=st["n_kernels"],
                       ncu_n_kernels_ok=(st["n_kernels"] == 5 * row["L"]) if st["counters_ok"] else None)
            save()
            print("[e1] ncu W=%d mode=%s hit=%s dram/tok=%s passes=%s" % (
                row["W_mib"], row["mode"], row["l2_hit_rate"], row["dram_bytes_per_token"], row["ncu_passes"]), flush=True)
        if any(r.get("ncu_n_kernels_ok") is False for r in rows):
            notes.append("ncu_n_kernels != 5*L on some rows: --launch-skip/--launch-count may count launches outside the profiler range")
        if any(r.get("ncu_error") == "skipped: time budget" for r in rows):
            notes.append("ncu skipped for some rows: time budget")
    ncu_deltas(rows)
    counters_all = bool(ncu_ok and rows and all(r["counters_ok"] for r in rows))
    flag = stop_flag(rows, tag, bool(pbytes))
    if flag:
        notes.append(flag)
    if pbytes:
        notes.append("us_per_token is net of a separately measured pollution constant; the on/off delta is also given gross "
                     "(gross_delta_vs_off_us, no subtraction) with its repeat spread (delta_noise_us). Small-W deltas are "
                     "near the noise floor: trust rows with delta_above_noise true. ncu_us_per_token is the GEMV-only kernel time "
                     "(no pollution kernel in it); "
                     "the stop rule and ncu_gain_vs_off use it when counters worked")
    if not pbytes:
        notes.append("no pollution (--no-pollute): W below L2 size is L2 resident with persisting off; stop rule not applied")
    common.write_result(EXP, args, config, rows, notes, counters_all)
    return 0


# ---------------------------------------------------------------- CLI

def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="exps.e1_sweep", description=__doc__.split("\n")[0])
    common.add_common_args(ap)
    ap.add_argument("--W", type=parse_w, default=None, help="working set sizes in MiB, comma separated")
    ap.add_argument("--modes", default=",".join(DEFAULT_MODES))
    ap.add_argument("--tokens", type=int, default=1000)
    ap.add_argument("--repeats", type=int, default=3, help="timing repeats; the median is reported")
    ap.add_argument("--small-w-repeats", type=int, default=SMALL_W_REPEATS,
                    help=f"timing repeats for W <= {SMALL_W_MIB} MiB (at least --repeats)")
    ap.add_argument("--variant", choices=["auto", "lut", "fp4"], default="auto")
    ap.add_argument("--window-method", choices=["graph", "stream"], default="graph")
    ap.add_argument("--no-pollute", dest="pollute", action="store_false",
                    help="do not stream a buffer of pollute-mult x L2 between tokens")
    ap.add_argument("--pollute-mult", type=float, default=POLLUTE_MULT)
    ap.add_argument("--pollute-bytes", type=int, default=0, help="pollution buffer size for --inner (set by the driver)")
    ap.add_argument("--inner", action="store_true", help="ncu target (internal)")
    a = ap.parse_args(argv)
    if a.W is None:
        a.W = [16, 64] if a.quick else list(DEFAULT_W)
    if a.quick:
        a.tokens = min(a.tokens, 100)
        a.repeats = a.small_w_repeats = 1
        if a.modes == ",".join(DEFAULT_MODES):
            a.modes = "off,1.0"
    a.modes = parse_modes(a.modes)
    return a


def plan_json(a) -> dict:
    from harness import counters
    ws = []
    for w in a.W:
        L = plan_layers(w)
        ws.append({"W_mib": w, "L": L, "bytes_per_token": L * sum(layer_bytes()),
                   "launches_per_graph": L * TOKENS_PER_GRAPH, "ncu_cmd": counters.build_ncu_cmd(
                       inner_argv(w, "1.0", a.variant, a.window_method), **ncu_args(L),
                       extra=("--profile-from-start", "off", "--replay-mode", "application"))})
    return {"exp": EXP, "modes": [m for m, _ in a.modes], "tokens": graph_tokens(a.tokens),
            "layer_shape": [N, K], "layer_bytes": sum(layer_bytes()), "layers": ws, "variant": a.variant,
            "window_method": a.window_method, "no_ncu": a.no_ncu, "quick": a.quick,
            "pollute": {"enabled": a.pollute, "mult": a.pollute_mult, "bytes": "mult x device L2, known at run time"}}


def main(argv=None) -> int:
    a = parse_args(argv)
    if a.dry_run:
        common.print_plan(plan_json(a))
        return 0
    if a.inner:
        return run_inner(a)
    with common.term_exits():
        return run(a)


if __name__ == "__main__":
    sys.exit(main())
