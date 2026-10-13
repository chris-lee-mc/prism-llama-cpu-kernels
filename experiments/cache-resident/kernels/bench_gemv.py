"""P0.4 ternary GEMV benchmark: achieved GB/s versus bytes moved (weights + scales + x + y).

python -m kernels.bench_gemv [--shapes 4096x4096,8192x8192,4096x16384,16384x4096,28672x8192] [--variants lut,fp4]
    [--iters 200] [--result-exp p0_gemv]       (shape = NxK)

Timing uses CUDA graphs of back-to-back launches, so Python launch cost is excluded but each kernel node still
pays a small inter-kernel gap. `gap_us` estimates that gap with a tiny kernel; `us_*_net` subtracts it.
Cold: the launches rotate over a pool of weight copies larger than 4x L2. Hot: one copy repeated.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys

import torch

from kernels import ternary as T

DEFAULT_SHAPES = "4096x4096,8192x8192,4096x16384,16384x4096,28672x8192"
TOL = 2e-3
L2_FALLBACK = 128 << 20
TRIALS = 7
TARGET_MS = 30.0


def parse_shapes(text: str) -> list[tuple[int, int]]:
    out = []
    for tok in text.split(","):
        tok = tok.strip().lower()
        if tok:
            n, k = tok.split("x")
            out.append((int(n), int(k)))
    return out


def l2_bytes(dev: int = 0) -> int:
    v = getattr(torch.cuda.get_device_properties(dev), "L2_cache_size", 0)
    return int(v) if v else L2_FALLBACK


def _graph_time_us(launch, n_launch: int, reps: int, trials: int = 1, target_ms: float = 0.0) -> list[float]:
    """us per launch, one value per trial. Each trial times `reps` replays of a graph holding n_launch launches.
    With target_ms > 0, reps grows so one trial lasts at least that long."""
    launch(0)
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        launch(0)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(n_launch):
            launch(i)
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def trial(n: int) -> float:
        a.record()
        for _ in range(n):
            g.replay()
        b.record()
        b.synchronize()
        return a.elapsed_time(b)

    if target_ms > 0:
        est_ms = max(trial(reps), 1e-3)
        reps = max(reps, math.ceil(target_ms * reps / est_ms))
    return [trial(reps) * 1000.0 / (reps * n_launch) for _ in range(trials)]


def gap_estimate_us(n_launch: int = 64, reps: int = 20) -> float:
    """Per-node cost of a graph of tiny kernels (launch gap floor)."""
    ext = T._ext()
    a = torch.zeros(256, dtype=torch.float16, device="cuda")
    b = torch.zeros(256, dtype=torch.float16, device="cuda")

    def launch(i):
        ext.add_(a.data_ptr(), b.data_ptr(), 256, torch.cuda.current_stream().cuda_stream)

    return statistics.median(_graph_time_us(launch, n_launch, reps, trials=3))


def bench_one(N: int, K: int, variant: str, iters: int, gap_us: float, peak: float | None, l2: int) -> dict:
    q, s, _ = T.random_ternary(N, K, seed=N * 31 + K, device="cuda")
    gen = torch.Generator(device="cuda")
    gen.manual_seed(5)
    x = torch.randn(K, generator=gen, device="cuda").to(torch.float16)
    y = torch.empty(N, dtype=torch.float16, device="cuda")
    ref = T.ref_gemv(q, s, x)
    T.gemv_ptr(variant, q.data_ptr(), s.data_ptr(), x.data_ptr(), y.data_ptr(), N, K)
    torch.cuda.synchronize()
    err = (y.float() - ref).abs().max().item()
    ok = bool(torch.allclose(y.float(), ref, rtol=TOL, atol=TOL * ref.abs().max().item()))

    wbytes = q.numel() * 4 + s.numel() * 2
    free = torch.cuda.mem_get_info()[0]
    n_pool = max(2, -(-4 * l2 // wbytes) + 1)
    n_pool = min(n_pool, 1024, max(2, int(free * 0.5) // wbytes))
    pool = [(q, s)] + [(q.clone(), s.clone()) for _ in range(n_pool - 1)]
    pp = [(a.data_ptr(), b.data_ptr()) for a, b in pool]
    xp, yp = x.data_ptr(), y.data_ptr()

    def cold(i):
        qp, sp = pp[i % n_pool]
        T.gemv_ptr(variant, qp, sp, xp, yp, N, K)

    def hot(i):
        T.gemv_ptr(variant, pp[0][0], pp[0][1], xp, yp, N, K)

    n_launch = max(n_pool, 32)
    cold_t = _graph_time_us(cold, n_launch, max(3, iters // n_launch), TRIALS, TARGET_MS)
    hot_t = _graph_time_us(hot, 32, max(3, iters // 32), TRIALS, TARGET_MS)
    us_cold, us_hot = statistics.median(cold_t), statistics.median(hot_t)
    nb = T.bytes_moved(N, K)
    row = dict(
        N=N, K=K, variant=variant, bytes=nb, pool_copies=n_pool,
        us_cold=us_cold, us_hot=us_hot, us_cold_min=min(cold_t), us_hot_min=min(hot_t), trials=TRIALS,
        gbps_cold=nb / us_cold / 1e3, gbps_hot=nb / us_hot / 1e3,
        pct_peak_cold=(nb / us_cold / 1e3 / peak) if peak else None,
        gap_us=gap_us,
        us_cold_net=max(us_cold - gap_us, 1e-3),
        pct_peak_cold_net=(nb / max(us_cold - gap_us, 1e-3) / 1e3 / peak) if peak else None,
        max_abs_err=err, ok=ok,
    )
    del pool, pp
    torch.cuda.empty_cache()
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shapes", default=DEFAULT_SHAPES)
    ap.add_argument("--variants", default="lut,fp4")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--result-exp", default="p0_gemv", help="empty string disables the result file")
    ap.add_argument("--quick", action="store_true", help="small shapes and few iterations for smoke runs")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, no CUDA")
    a = ap.parse_args(argv)

    shapes = parse_shapes("1024x1024,4096x4096" if a.quick else a.shapes)
    variants = [v for v in a.variants.split(",") if v]
    iters = 20 if a.quick else a.iters
    for v in variants:
        if v not in ("lut", "fp4"):
            ap.error(f"unknown variant {v}")
    plan = dict(shapes=shapes, variants=variants, iters=iters)
    if a.dry_run:
        print(json.dumps(plan))
        return 0
    if not torch.cuda.is_available():
        print("bench_gemv: no CUDA device", file=sys.stderr)
        return 2

    from infra.gpus import peak_dram_gbps, tag_from_name

    name = torch.cuda.get_device_name(0)
    tag = tag_from_name(name)
    peak = peak_dram_gbps(tag)
    l2 = l2_bytes()
    T._ext()
    gap = gap_estimate_us()
    print(f"device {name} tag {tag} peak {peak} GB/s L2 {l2 >> 20} MiB graph-node gap ~{gap:.2f} us")
    rows = []
    print(f"{'N':>6} {'K':>6} {'var':>4} {'bytes':>11} {'us_cold':>8} {'us_hot':>8} {'GB/s cold':>10} {'GB/s hot':>9} {'%peak':>6} {'%net':>6} {'err':>9} ok")
    for N, K in shapes:
        for v in variants:
            r = bench_one(N, K, v, iters, gap, peak, l2)
            rows.append(r)
            pc = "n/a" if r["pct_peak_cold"] is None else f"{100 * r['pct_peak_cold']:.1f}"
            pn = "n/a" if r["pct_peak_cold_net"] is None else f"{100 * r['pct_peak_cold_net']:.1f}"
            print(f"{N:>6} {K:>6} {v:>4} {r['bytes']:>11} {r['us_cold']:>8.2f} {r['us_hot']:>8.2f} {r['gbps_cold']:>10.1f} {r['gbps_hot']:>9.1f} {pc:>6} {pn:>6} {r['max_abs_err']:>9.2e} {r['ok']}")

    best = [r for r in rows if r["N"] == 4096 and r["K"] == 4096 and r["pct_peak_cold"] is not None]
    target = None
    if best:
        top = max(best, key=lambda r: r["pct_peak_cold"])
        target = bool(top["pct_peak_cold"] >= 0.70)
        print(f"target >= 70% of peak at 4096x4096: best {top['variant']} {100 * top['pct_peak_cold']:.1f}% -> {'MET' if target else 'NOT MET'}")
    notes = [
        f"us_cold, us_hot: median of {TRIALS} trials, each >= {TARGET_MS:.0f} ms of replays; *_min is the fastest trial",
        "us_cold: rotating pool of weight copies > 4x L2, CUDA graph of back-to-back launches (includes graph node gap)",
        "us_hot: one weight copy repeated",
        "gap_us: per-node cost of a graph of tiny kernels; *_net subtracts it",
        "pct_peak_* uses infra.gpus.peak_dram_gbps; None when the GPU tag is unknown",
        f"target_met_4096x4096_70pct: {target}",
    ]
    all_ok = all(r["ok"] for r in rows)
    if a.result_exp:
        from infra.results import write_result

        path = write_result(a.result_exp, None, dict(config=dict(plan, tag=tag, peak_gbps=peak, l2_bytes=l2), rows=rows, notes=notes, counters_ok=None))
        print(f"wrote {path}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
