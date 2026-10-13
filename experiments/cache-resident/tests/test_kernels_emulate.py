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

TILE = 4
PAIR = 12
LUT_GROUP = 64 * PAIR


def _f32(a):
    return np.asarray(a, dtype=np.float32)


def _byte(v, b):
    return (v >> (8 * b)) & 0xFF


def _splits(G, S):
    """Tile starts per split: block `split` takes tiles split, split + S, ... (tile = TILE groups)."""
    T = -(-G // TILE)
    return [[t * TILE for t in range(k, T, S)] for k in range(S)]


def q_chunk(r, c):
    """Mirror of q_chunk: swizzled 16 B slot of chunk c of tile row r."""
    return r * 8 + (c ^ (r & 7))


# ---------- variant (a): LUT ----------
def lut_build(lut, xf, g0, ng):
    """Mirror of x_store (LUT). Returns, per 8-lane phase, the float offsets of the 128-bit stores."""
    phases = []
    for e in range(ng * 192):
        u = e // 192
        p = (e - u * 192) // 3
        c1 = e % 3
        x0, x1 = xf[(g0 + u) * 128 + 2 * p], xf[(g0 + u) * 128 + 2 * p + 1]
        t1 = x1 if c1 == 1 else (-x1 if c1 == 2 else np.float32(0))
        assert 4 * e == u * LUT_GROUP + p * PAIR + 4 * c1
        lut[4 * e : 4 * e + 4] = _f32([t1, t1 + x0, t1 - x0, t1])
        if e % 8 == 0:
            phases.append([])
        phases[-1].append(4 * e)
    return phases


def lut_lookups(w):
    """Mirror of lut_dot: (pair, byte offset) per lookup of one 32-bit word."""
    ev = (w << 2) & 0x3C3C3C3C
    od = (w >> 2) & 0x3C3C3C3C
    out = []
    for b in range(4):
        out.append((2 * b, _byte(ev, b)))
        out.append((2 * b + 1, _byte(od, b)))
    return out


def emulate_lut(q, s, x, warps=4, S=1, check_banks=False):
    N = q.shape[0]
    K = x.numel()
    G = K // 128
    RB = 32 * warps
    qw = q.numpy().astype(np.int64) & 0xFFFFFFFF
    sf = _f32(s.float().numpy())
    xf = _f32(x.float().numpy())
    y = np.zeros(N, dtype=np.float32)
    max_ways = 1
    for rblk in range((N + RB - 1) // RB):
        red = np.zeros((S, RB), dtype=np.float32)
        for split, starts in enumerate(_splits(G, S)):
            acc = np.zeros(RB, dtype=np.float32)
            for g0 in starts:
                ng = min(TILE, G - g0)
                lut = np.zeros(TILE * LUT_GROUP, dtype=np.float32)
                phases = lut_build(lut, xf, g0, ng)
                if check_banks:
                    for ph in phases:
                        banks = [(o + i) % 32 for o in ph for i in range(4)]
                        max_ways = max(max_ways, max(banks.count(b) for b in set(banks)))
                for u in range(ng):
                    g = g0 + u
                    for warp in range(warps):
                        addrs = {}
                        for lane in range(32):
                            rl = warp * 32 + lane
                            row = min(rblk * RB + rl, N - 1)
                            d = np.float32(0)
                            for hh in range(2):
                                for wd in range(4):
                                    w = int(qw[row, g * 8 + hh * 4 + wd])
                                    for k, (pr, off) in enumerate(lut_lookups(w)):
                                        assert off % 4 == 0 and off // 4 < PAIR
                                        a = u * LUT_GROUP + 32 * hh * PAIR + (8 * wd + pr) * PAIR + off // 4
                                        d += lut[a]
                                        addrs.setdefault((hh, wd, k), set()).add(a)
                            acc[rl] = np.float32(sf[row, g]) * d + acc[rl]
                        if check_banks:
                            for v in addrs.values():
                                per_bank = {}
                                for a in v:
                                    per_bank.setdefault(a % 32, set()).add(a)
                                max_ways = max(max_ways, max(len(t) for t in per_bank.values()))
            red[split] = acc
        rows = np.arange(RB) + rblk * RB
        ok = rows < N
        y[rows[ok]] = red.sum(0)[ok]
    return y, max_ways


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


def emulate_fp4(q, s, x, warps=4, S=1):
    N = q.shape[0]
    K = x.numel()
    G = K // 128
    RB = 32 * warps
    qw = q.numpy().astype(np.int64) & 0xFFFFFFFF
    sf = _f32(s.float().numpy())
    xf = _f32(x.float().numpy())
    tab = _f32([e2m1_value(n) for n in range(16)])
    y = np.zeros(N, dtype=np.float32)
    for rblk in range((N + RB - 1) // RB):
        red = np.zeros((S, RB), dtype=np.float32)
        for split, starts in enumerate(_splits(G, S)):
            acc = np.zeros(RB, dtype=np.float32)
            for g0 in starts:
                ng = min(TILE, G - g0)
                xs = np.zeros(TILE * 128, dtype=np.float32)
                xs[: ng * 128] = xf[g0 * 128 : (g0 + ng) * 128]
                for u in range(ng):
                    g = g0 + u
                    for rl in range(RB):
                        row = min(rblk * RB + rl, N - 1)
                        d = np.float32(0)
                        for hh in range(2):
                            sw = []
                            for wd in range(4):
                                sw += list(to_e2m1(int(qw[row, g * 8 + hh * 4 + wd])))
                            xh = xs[u * 128 + 64 * hh :]
                            for wd in range(4):
                                ev, od = sw[2 * wd], sw[2 * wd + 1]
                                el, eh = (ev << 2) & 0x3C3C3C3C, (ev >> 2) & 0x3C3C3C3C
                                ol, oh = (od << 2) & 0x3C3C3C3C, (od >> 2) & 0x3C3C3C3C
                                for b in range(4):
                                    xv = xh[16 * wd + 4 * b : 16 * wd + 4 * b + 4]
                                    for wv, xi in ((el, 0), (ol, 1), (eh, 2), (oh, 3)):
                                        d += tab[_byte(wv, b) // 4] * xv[xi]
                        acc[rl] = np.float32(sf[row, g]) * d + acc[rl]
            red[split] = acc
        rows = np.arange(RB) + rblk * RB
        ok = rows < N
        y[rows[ok]] = red.sum(0)[ok]
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


@pytest.mark.parametrize("warps,S", [(4, 2), (2, 8), (1, 1), (2, 3)])
def test_emulated_split_k(warps, S):
    q, s, x, ref = _case(21, 128 * 13, seed=3)
    y, _ = emulate_lut(q, s, x, warps=warps, S=S)
    np.testing.assert_allclose(y, ref, rtol=1e-4, atol=1e-4 * np.abs(ref).max())
    y = emulate_fp4(q, s, x, warps=warps, S=S)
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


def test_q_tile_swizzle_is_conflict_free():
    # copy: 8 lanes of a phase write the 8 chunks of one row; read: 8 lanes (rows) read the same chunk
    for r in range(256):
        assert sorted(q_chunk(r, c) % 8 for c in range(8)) == list(range(8))
    for base in range(0, 256, 8):
        for c in range(8):
            assert len({q_chunk(base + i, c) % 8 for i in range(8)}) == 8
