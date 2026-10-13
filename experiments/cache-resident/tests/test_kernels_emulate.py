"""Python transliteration of the CUDA kernels' index math, checked against the reference.

No GPU is needed. These tests catch layout, swizzle, tail and split mistakes in the design; they cannot catch
CUDA-specific bugs (those need the on-GPU tests in test_kernels_gpu.py).
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from kernels import ternary as T
from kernels.toy_hybrid import ref_attn_decode

LUT_ROW = 16
LUT_HALF = 33 * LUT_ROW
LUT_FLOATS = 2 * LUT_HALF
T_VAL = np.array([0.0, 1.0, -1.0, 0.0], dtype=np.float32)


def _f32(a):
    return np.asarray(a, dtype=np.float32)


# ---------- variant (a): LUT ----------
def lut_build_pair(lut, base, x0, x1, r):
    """Mirror of lut_build_pair. Returns the 4 float4 chunk start offsets that were written."""
    v0 = [0.0, x0, -x0, 0.0]
    v1 = [x1, x1 + x0, x1 - x0, x1]
    v2 = [-x1, x0 - x1, -x0 - x1, -x1]
    starts = []
    for c, v in enumerate((v0, v1, v2, v0)):
        off = base + 4 * (c ^ r)
        lut[off : off + 4] = _f32(v)
        starts.append(off)
    return starts


def emulate_lut(q, s, x, warps=8, gb=4, check_banks=False):
    N = q.shape[0]
    K = x.numel()
    G = K // 128
    qw = q.numpy().astype(np.int64) & 0xFFFFFFFF
    sf = _f32(s.float().numpy())
    xf = _f32(x.float().numpy())
    y = np.zeros(N, dtype=np.float32)
    max_ways = 1
    for blk in range((N + 15) // 16):
        part = np.zeros((warps, 16), dtype=np.float32)
        for warp in range(warps):
            lut = np.zeros(LUT_FLOATS, dtype=np.float32)
            acc = np.zeros(32, dtype=np.float32)
            for g0 in range(warp, G, warps * gb):
                for u in range(gb):
                    g = g0 + warps * u
                    if g >= G:
                        break
                    for lane in range(32):
                        rb = (lane >> 1) & 3
                        xa = xf[2 * (g * 64 + lane) : 2 * (g * 64 + lane) + 2]
                        xb = xf[2 * (g * 64 + 32 + lane) : 2 * (g * 64 + 32 + lane) + 2]
                        lut_build_pair(lut, lane * LUT_ROW, xa[0], xa[1], rb)
                        lut_build_pair(lut, LUT_HALF + lane * LUT_ROW, xb[0], xb[1], rb)
                    if check_banks:
                        max_ways = max(max_ways, _store_conflicts())
                    for lane in range(32):
                        hid = lane >> 4
                        row = blk * 16 + (lane & 15)
                        if row >= N:
                            continue
                        total = np.float32(0)
                        for wd in range(4):
                            w = int(qw[row, g * 8 + hid * 4 + wd])
                            a = np.float32(0)
                            b = np.float32(0)
                            for j in range(0, 8, 2):
                                n0 = ((w >> (4 * j)) & 15) ^ ((j >> 1) << 2)
                                n1 = ((w >> (4 * (j + 1))) & 15) ^ (((j + 1) >> 1) << 2)
                                a += lut[hid * LUT_HALF + (8 * wd + j) * LUT_ROW + n0]
                                b += lut[hid * LUT_HALF + (8 * wd + j + 1) * LUT_ROW + n1]
                            total += a + b
                        acc[lane] = np.float32(sf[row, g]) * total + acc[lane]
                    if check_banks:
                        max_ways = max(max_ways, _lookup_conflicts(qw, N, blk, g))
            tot = acc[:16] + acc[16:]
            part[warp] = tot
        rows = np.arange(16) + blk * 16
        ok = rows < N
        y[rows[ok]] = part.sum(0)[ok]
    return y, max_ways


def _store_conflicts():
    """Worst bank multiplicity of the build stores: 8-lane phases of 128-bit stores, for both pairs and every chunk step."""
    worst = 1
    for pair_base in (0, LUT_HALF):
        for c in range(4):
            for phase in range(4):
                banks = []
                for lane in range(phase * 8, phase * 8 + 8):
                    rb = (lane >> 1) & 3
                    off = pair_base + lane * LUT_ROW + 4 * (c ^ rb)
                    banks += [(off + i) % 32 for i in range(4)]
                worst = max(worst, max(banks.count(b) for b in set(banks)))
    return worst


def _lookup_conflicts(qw, N, blk, g):
    """Worst number of distinct addresses hitting one bank among the 32 lanes of a lookup, over all (wd, j)."""
    worst = 1
    for wd in range(4):
        for j in range(8):
            addrs = set()
            for lane in range(32):
                row = blk * 16 + (lane & 15)
                if row >= N:
                    continue
                hid = lane >> 4
                w = int(qw[row, g * 8 + hid * 4 + wd])
                nib = ((w >> (4 * j)) & 15) ^ ((j >> 1) << 2)
                addrs.add(hid * LUT_HALF + (8 * wd + j) * LUT_ROW + nib)
            per_bank = {}
            for a in addrs:
                per_bank.setdefault(a % 32, set()).add(a)
            worst = max(worst, max(len(v) for v in per_bank.values()))
    return worst


# ---------- variant (b): FP4 ----------
M32 = 0xFFFFFFFF


def to_e2m1(w):
    b0 = w & 0x55555555
    b1 = (w >> 1) & 0x55555555
    nz = b0 ^ b1
    ng = b1 & ~b0 & M32
    e = (((nz & 0x11111111) << 1) | ((ng & 0x11111111) << 3)) & M32
    o = ((((nz >> 2) & 0x11111111) << 1) | (((ng >> 2) & 0x11111111) << 3)) & M32
    return e, o


def e2m1_value(n):
    m = n & 7
    v = 0.5 * m if m < 2 else math.ldexp(1.0 + 0.5 * (m & 1), (m >> 1) - 1)
    return -v if n & 8 else v


def emulate_fp4(q, s, x, warps=8, gb=4):
    N = q.shape[0]
    K = x.numel()
    G = K // 128
    qw = q.numpy().astype(np.int64) & 0xFFFFFFFF
    sf = _f32(s.float().numpy())
    xf = _f32(x.float().numpy())
    tab = _f32([e2m1_value(n) for n in range(16)])
    y = np.zeros(N, dtype=np.float32)
    for blk in range((N + 15) // 16):
        part = np.zeros((warps, 16), dtype=np.float32)
        for warp in range(warps):
            acc = np.zeros(32, dtype=np.float32)
            for g0 in range(warp, G, warps * gb):
                for u in range(gb):
                    g = g0 + warps * u
                    if g >= G:
                        break
                    stage = np.zeros(256, dtype=np.int64)
                    xs = np.zeros((64, 2), dtype=np.float32)
                    for lane in range(32):
                        hid = lane >> 4
                        row = blk * 16 + (lane & 15)
                        words = [int(qw[row, g * 8 + hid * 4 + wd]) if row < N else 0 for wd in range(4)]
                        e = []
                        for w in words:
                            e += list(to_e2m1(w))
                        stage[lane * 4 : lane * 4 + 4] = e[:4]
                        stage[128 + lane * 4 : 128 + lane * 4 + 4] = e[4:]
                        xs[lane] = xf[2 * (g * 64 + lane) : 2 * (g * 64 + lane) + 2]
                        xs[lane + 32] = xf[2 * (g * 64 + 32 + lane) : 2 * (g * 64 + 32 + lane) + 2]
                    for lane in range(32):
                        hid = lane >> 4
                        row = blk * 16 + (lane & 15)
                        if row >= N:
                            continue
                        sw = list(stage[lane * 4 : lane * 4 + 4]) + list(stage[128 + lane * 4 : 128 + lane * 4 + 4])
                        total = np.float32(0)
                        for wd in range(4):
                            ev, od = int(sw[2 * wd]), int(sw[2 * wd + 1])
                            a = np.float32(0)
                            b = np.float32(0)
                            for j in range(0, 8, 2):
                                xa = xs[hid * 32 + 8 * wd + j]
                                xb = xs[hid * 32 + 8 * wd + j + 1]
                                a += tab[(ev >> (4 * j)) & 15] * xa[0] + tab[(od >> (4 * j)) & 15] * xa[1]
                                b += tab[(ev >> (4 * (j + 1))) & 15] * xb[0] + tab[(od >> (4 * (j + 1))) & 15] * xb[1]
                            total += a + b
                        acc[lane] = np.float32(sf[row, g]) * total + acc[lane]
            part[warp] = acc[:16] + acc[16:]
        rows = np.arange(16) + blk * 16
        ok = rows < N
        y[rows[ok]] = part.sum(0)[ok]
    return y


def _case(N, K, seed=0):
    q, s, t = T.random_ternary(N, K, seed=seed)
    g = torch.Generator().manual_seed(seed + 1)
    x = torch.randn(K, generator=g).to(torch.float16)
    return q, s, x, T.ref_gemv(q, s, x).numpy()


SHAPES = [(1, 128), (7, 384), (19, 1024), (33, 256), (16, 128 * 40), (5, 128 * 9)]


@pytest.mark.parametrize("N,K", SHAPES)
def test_emulated_lut_matches_reference(N, K):
    q, s, x, ref = _case(N, K)
    y, _ = emulate_lut(q, s, x)
    np.testing.assert_allclose(y, ref, rtol=1e-4, atol=1e-4 * np.abs(ref).max())


@pytest.mark.parametrize("N,K", SHAPES)
def test_emulated_fp4_matches_reference(N, K):
    q, s, x, ref = _case(N, K)
    y = emulate_fp4(q, s, x)
    np.testing.assert_allclose(y, ref, rtol=1e-4, atol=1e-4 * np.abs(ref).max())


@pytest.mark.parametrize("warps,gb", [(4, 2), (8, 1), (2, 4)])
def test_emulated_lut_other_tunings(warps, gb):
    q, s, x, ref = _case(21, 128 * 13, seed=3)
    y, _ = emulate_lut(q, s, x, warps=warps, gb=gb)
    np.testing.assert_allclose(y, ref, rtol=1e-4, atol=1e-4 * np.abs(ref).max())


def test_lut_is_bank_conflict_free():
    q, s, x, _ = _case(32, 256, seed=5)
    _, ways = emulate_lut(q, s, x, check_banks=True)
    assert ways == 1


def test_e2m1_table_and_ternary_nibbles():
    vals = [e2m1_value(n) for n in range(16)]
    assert vals[:8] == [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    assert vals[8:] == [-0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
    # codes 00,01,10,11 at weights 0..3 of a word -> nibbles 0x0, 0x2, 0xA, 0x0
    e, o = to_e2m1(0b11_10_01_00)
    assert e & 0xF == 0x0 and (e >> 4) & 0xF == 0xA  # weights 0 and 2
    assert o & 0xF == 0x2 and (o >> 4) & 0xF == 0x0  # weights 1 and 3


def test_to_e2m1_matches_scalar_rule():
    rng = np.random.default_rng(0)
    for w in rng.integers(0, 1 << 32, 50, dtype=np.uint64).tolist():
        e, o = to_e2m1(int(w))
        for i in range(16):
            c = (int(w) >> (2 * i)) & 3
            want = {0: 0x0, 1: 0x2, 2: 0xA, 3: 0x0}[c]
            word, j = (e, i // 2) if i % 2 == 0 else (o, i // 2)
            assert (word >> (4 * j)) & 0xF == want


# ---------- attention split / online softmax ----------
def emulate_attn(q, Kc, Vc, Hq, Hkv, D, scale):
    T_ = Kc.shape[0]
    ns = min(128, (T_ + 31) // 32)
    chunk = -(-T_ // ns)
    nsplit = -(-T_ // chunk)
    grp = Hq // Hkv
    qf = q.float().reshape(Hq, D)
    Kf, Vf = Kc.float(), Vc.float()
    pm = np.full((Hq, nsplit), -np.inf, dtype=np.float64)
    pl = np.zeros((Hq, nsplit))
    pa = np.zeros((Hq, nsplit, D))
    for hk in range(Hkv):
        for sp in range(nsplit):
            t0, t1 = sp * chunk, min(T_, sp * chunk + chunk)
            assert t0 < t1
            m = np.full(grp, -np.inf)
            l = np.zeros(grp)
            acc = np.zeros((grp, D))
            for cb in range(t0, t1, 64):
                n = min(64, t1 - cb)
                sc = (qf[hk * grp : (hk + 1) * grp] @ Kf[cb : cb + n, hk, :].T * scale).double().numpy()
                mn = np.maximum(m, sc.max(1))
                f = np.exp(m - mn)
                p = np.exp(sc - mn[:, None])
                l = l * f + p.sum(1)
                acc = acc * f[:, None] + p @ Vf[cb : cb + n, hk, :].double().numpy()
                m = mn
            pm[hk * grp : (hk + 1) * grp, sp] = m
            pl[hk * grp : (hk + 1) * grp, sp] = l
            pa[hk * grp : (hk + 1) * grp, sp] = acc
    M = pm.max(1, keepdims=True)
    w = np.exp(pm - M)
    L = (w * pl).sum(1)
    o = (w[..., None] * pa).sum(1) / L[:, None]
    return o.reshape(-1), nsplit


@pytest.mark.parametrize("T_,Hq,Hkv,D", [(1, 8, 2, 256), (31, 8, 2, 128), (33, 4, 4, 64), (1000, 8, 2, 256), (5000, 32, 4, 128)])
def test_attn_split_math_matches_reference(T_, Hq, Hkv, D):
    g = torch.Generator().manual_seed(T_)
    q = torch.randn(Hq * D, generator=g).half()
    Kc = (torch.randn(T_, Hkv, D, generator=g) * 0.5).half()
    Vc = (torch.randn(T_, Hkv, D, generator=g) * 0.5).half()
    scale = D ** -0.5
    o, nsplit = emulate_attn(q, Kc, Vc, Hq, Hkv, D, scale)
    assert 1 <= nsplit <= 128
    ref = ref_attn_decode(q, Kc, Vc, Hq, Hkv, D, scale).numpy()
    np.testing.assert_allclose(o, ref, rtol=1e-4, atol=1e-5)
