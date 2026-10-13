"""Packing, reference math and layout tests (CPU only)."""
from __future__ import annotations

import pytest
import torch

from kernels import ternary as T


@pytest.mark.parametrize("N,K", [(1, 128), (3, 128), (7, 384), (5, 4096), (130, 256)])
def test_pack_unpack_roundtrip(N, K):
    _, _, t = T.random_ternary(N, K, seed=N + K)
    q = T.pack(t)
    assert q.dtype == torch.int32 and tuple(q.shape) == (N, K // 16)
    assert torch.equal(T.unpack(q, K), t)


def test_golden_byte():
    t = torch.zeros(1, 128, dtype=torch.int8)
    t[0, :4] = torch.tensor([1, -1, 0, 1], dtype=torch.int8)
    q = T.pack(t)
    assert q.view(torch.uint8)[0, 0].item() == 0b01_00_10_01 == 0x49
    assert q[0, 0].item() == 0x00000049
    assert (q[0, 1:] == 0).all()


def test_word_bit_positions():
    t = torch.zeros(1, 128, dtype=torch.int8)
    t[0, 15] = -1  # k = 16*0 + 15 -> bits [31:30] of word 0, code 0b10
    t[0, 16] = 1  # word 1, bit 0
    q = T.pack(t)
    assert (q[0, 0].item() & 0xFFFFFFFF) == 0b10 << 30
    assert q[0, 1].item() == 1


def test_codes_follow_decode_rule():
    # decode rule t = (c & 1) - (c >> 1); code 3 decodes to 0
    q = torch.tensor([[0b11_10_01_00]], dtype=torch.int32).repeat(1, 8)
    t = T.unpack(q, 128)
    assert t[0, :4].tolist() == [0, 1, -1, 0]


def test_ref_gemv_matches_dense_matmul():
    N, K = 37, 640
    q, s, t = T.random_ternary(N, K, seed=2)
    x = torch.randn(K, generator=torch.Generator().manual_seed(3)).to(torch.float16)
    ref = T.ref_gemv(q, s, x)
    dense = T.dequant(q, s, K) @ x.float()
    assert torch.allclose(ref, dense, rtol=1e-5, atol=1e-5)
    manual = torch.zeros(N)
    for n in range(N):
        for g in range(K // 128):
            sl = slice(128 * g, 128 * g + 128)
            manual[n] += s[n, g].float() * (t[n, sl].float() * x[sl].float()).sum()
    assert torch.allclose(ref, manual, rtol=1e-5, atol=1e-5)


def test_ref_gemv_chunked_rows(monkeypatch):
    q, s, _ = T.random_ternary(50, 256, seed=4)
    x = torch.randn(256).to(torch.float16)
    full = T.ref_gemv(q, s, x)
    monkeypatch.setattr(T, "_ROW_CHUNK_ELEMS", 256 * 7)
    assert torch.equal(T.ref_gemv(q, s, x), full)


def test_bytes_moved():
    assert T.bytes_moved(4096, 4096) == 4096 * 4096 // 4 + 4096 * 32 * 2 + 2 * 4096 + 2 * 4096
    assert T.bytes_moved(1, 128) == 32 + 2 + 256 + 2


def test_quantize_ternary_is_idempotent_on_ternary_input():
    q, s, t = T.random_ternary(6, 256, seed=7)
    w = T.dequant(q, s, 256)
    q2, s2 = T.quantize_ternary(w)
    assert torch.equal(q2, q)
    q3, _ = T.quantize_ternary(T.dequant(q2, s2, 256))
    assert torch.equal(q3, q)


def test_quantize_ternary_scale_is_group_absmean():
    w = torch.randn(2, 256, generator=torch.Generator().manual_seed(1))
    q, s = T.quantize_ternary(w)
    want = w.reshape(2, 2, 128).abs().mean(-1)
    assert torch.allclose(s.float(), want, rtol=1e-3)
    assert set(T.unpack(q, 256).unique().tolist()) <= {-1, 0, 1}


def test_random_ternary_properties():
    q, s, t = T.random_ternary(64, 512, density=0.5, seed=9)
    assert abs((t != 0).float().mean().item() - 0.5) < 0.03
    assert s.dtype == torch.float16 and tuple(s.shape) == (64, 4)
    assert (s.float() > 0.4 / 512**0.5).all() and (s.float() < 1.6 / 512**0.5).all()
    q2, _, _ = T.random_ternary(64, 512, density=0.5, seed=9)
    assert torch.equal(q, q2)


def test_k_must_be_multiple():
    with pytest.raises(ValueError):
        T.random_ternary(2, 100)
    with pytest.raises(ValueError):
        T.pack(torch.zeros(2, 20, dtype=torch.int8))


def test_gemv_requires_cuda_tensors():
    q, s, _ = T.random_ternary(4, 128)
    with pytest.raises(ValueError):
        T.gemv_lut(q, s, torch.zeros(128, dtype=torch.float16))
    with pytest.raises(ValueError):
        T.gemv_ptr("nope", 0, 0, 0, 0, 1, 128)
