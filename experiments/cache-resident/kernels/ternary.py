"""Ternary weight packing, reference math and GEMV wrappers (layout: SPEC.md section 4.1).

W is [N, K] with t in {-1, 0, +1}. Each weight is a 2-bit code (00 -> 0, 01 -> +1, 10 -> -1, 11 -> 0).
q is int32 [N, K/16]; word j of row n holds k = 16j + i at bits [2i+1 : 2i]. s is fp16 [N, K/128].
"""
from __future__ import annotations

import math

import torch

QK = 128
_ROW_CHUNK_ELEMS = 1 << 24

_EXT = None


def _ext():
    global _EXT
    if _EXT is None:
        from kernels.build import load_ext

        _EXT = load_ext()
    return _EXT


def _check_k(K: int, mult: int) -> None:
    if K <= 0 or K % mult:
        raise ValueError(f"K={K} must be a positive multiple of {mult}")


def pack(t: torch.Tensor) -> torch.Tensor:
    """int8 {-1,0,1} [N,K] -> int32 [N,K/16]. Pure torch, CPU or CUDA."""
    if t.dim() != 2:
        raise ValueError("t must be 2-D [N,K]")
    N, K = t.shape
    _check_k(K, 16)
    t = t.contiguous()
    c = (t == 1).to(torch.uint8) + 2 * (t == -1).to(torch.uint8)
    c = c.reshape(N, K // 4, 4)
    b = c[..., 0] | (c[..., 1] << 2) | (c[..., 2] << 4) | (c[..., 3] << 6)
    return b.contiguous().view(torch.int32).reshape(N, K // 16)


def unpack(q: torch.Tensor, K: int) -> torch.Tensor:
    """int32 [N,K/16] -> int8 [N,K] in {-1,0,1}."""
    _check_k(K, 16)
    N = q.shape[0]
    b = q.contiguous().view(torch.uint8).reshape(N, K // 4, 1)
    sh = torch.tensor([0, 2, 4, 6], dtype=torch.uint8, device=q.device)
    c = (b >> sh) & 3
    t = (c & 1).to(torch.int8) - (c >> 1).to(torch.int8)
    return t.reshape(N, K)


def quantize_ternary(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Absmean per g128: s = mean|w| (min 1e-8), t = round(clamp(w/s, -1, 1)). Returns (q, s fp16)."""
    N, K = w.shape
    _check_k(K, QK)
    wg = w.float().reshape(N, K // QK, QK)
    s = wg.abs().mean(-1).clamp_min(1e-8)
    t = torch.round(torch.clamp(wg / s[..., None], -1, 1)).to(torch.int8).reshape(N, K)
    return pack(t), s.to(torch.float16)


def random_ternary(N: int, K: int, *, density: float = 2 / 3, seed: int = 0, device="cpu"):
    """Random (q, s, t): t int8 [N,K] with the given nonzero density, s ~ U(0.5,1.5)/sqrt(K) fp16 [N,K/128]."""
    _check_k(K, QK)
    dev = torch.device(device)
    g = torch.Generator(device=dev)
    g.manual_seed(int(seed))
    t = torch.empty(N, K, dtype=torch.int8, device=dev)
    rows = max(1, (1 << 26) // K)
    for r0 in range(0, N, rows):
        r1 = min(N, r0 + rows)
        nz = torch.rand(r1 - r0, K, generator=g, device=dev) < density
        sg = torch.rand(r1 - r0, K, generator=g, device=dev) < 0.5
        t[r0:r1] = (1 - 2 * sg.to(torch.int8)) * nz.to(torch.int8)
    s = ((torch.rand(N, K // QK, generator=g, device=dev) + 0.5) / math.sqrt(K)).to(torch.float16)
    return pack(t), s, t


def dequant(q: torch.Tensor, s: torch.Tensor, K: int, dtype=torch.float32) -> torch.Tensor:
    """Dense [N,K] weights = t * s (scale of the group broadcast)."""
    N = q.shape[0]
    t = unpack(q, K).to(torch.float32).reshape(N, K // QK, QK)
    return (t * s.float()[..., None]).reshape(N, K).to(dtype)


def ref_gemv(q: torch.Tensor, s: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """fp32 reference y[n] = sum_g s[n,g] * sum_{k in g} t[n,k] * x[k]. CPU or CUDA, chunked over rows."""
    N = q.shape[0]
    K = x.numel()
    G = K // QK
    xf = x.float().reshape(1, G, QK)
    rows = max(1, _ROW_CHUNK_ELEMS // K)
    out = torch.empty(N, dtype=torch.float32, device=q.device)
    for r0 in range(0, N, rows):
        r1 = min(N, r0 + rows)
        t = unpack(q[r0:r1], K).to(torch.float32).reshape(r1 - r0, G, QK)
        out[r0:r1] = ((t * xf).sum(-1) * s[r0:r1].float()).sum(-1)
    return out


def bytes_moved(N: int, K: int) -> int:
    """Weights + scales + x + y, in bytes."""
    return N * K // 4 + N * (K // QK) * 2 + 2 * K + 2 * N


def _validate(q, s, x, out):
    if not (q.is_cuda and s.is_cuda and x.is_cuda):
        raise ValueError("q, s, x must be CUDA tensors")
    if q.dtype != torch.int32 or q.dim() != 2:
        raise ValueError("q must be int32 [N,K/16]")
    if s.dtype != torch.float16 or x.dtype != torch.float16:
        raise ValueError("s and x must be float16")
    if not (q.is_contiguous() and s.is_contiguous() and x.is_contiguous()):
        raise ValueError("q, s, x must be contiguous")
    N, K = q.shape[0], q.shape[1] * 16
    _check_k(K, QK)
    if x.numel() != K or tuple(s.shape) != (N, K // QK):
        raise ValueError(f"shape mismatch: q {tuple(q.shape)} s {tuple(s.shape)} x {tuple(x.shape)}")
    if q.data_ptr() % 16 or x.data_ptr() % 16:
        raise ValueError("q and x must be 16 B aligned")
    if out is None:
        out = torch.empty(N, dtype=torch.float16, device=q.device)
    elif out.dtype != torch.float16 or out.numel() != N or not out.is_contiguous() or not out.is_cuda:
        raise ValueError("out must be a contiguous cuda float16 [N]")
    return N, K, out


def _gemv(variant: int, q, s, x, out):
    N, K, out = _validate(q, s, x, out)
    stream = torch.cuda.current_stream(q.device).cuda_stream
    _ext().gemv(variant, q.data_ptr(), s.data_ptr(), x.data_ptr(), out.data_ptr(), N, K, stream)
    return out


def gemv_lut(q, s, x, out=None) -> torch.Tensor:
    """Variant (a): shared-memory LUT. fp16 [N], launched on the current stream."""
    return _gemv(0, q, s, x, out)


def gemv_fp4(q, s, x, out=None) -> torch.Tensor:
    """Variant (b): E2M1 nibbles staged in shared memory, table decode, fp32 dot."""
    return _gemv(1, q, s, x, out)


def gemv_ptr(variant: str, q_ptr: int, s_ptr: int, x_ptr: int, y_ptr: int, N: int, K: int, stream: int = 0) -> None:
    """Raw-pointer launch for graph capture. stream 0 means the torch current stream (0 when that is the default stream)."""
    vi = {"lut": 0, "fp4": 1}.get(variant)
    if vi is None:
        raise ValueError(f"variant must be 'lut' or 'fp4', got {variant!r}")
    if not stream:
        stream = torch.cuda.current_stream().cuda_stream
    _ext().gemv(vi, int(q_ptr), int(s_ptr), int(x_ptr), int(y_ptr), int(N), int(K), int(stream))
