"""INT8 LM head: per-row symmetric quantization and a batch-1 GEMV (Triton kernel cr_int8_gemv, torch fallback).

Replaces lm_head only. Used by E3 to test persisting an INT8 or vocab-pruned head.
"""
from __future__ import annotations

import torch

BLOCK_N = 16
BLOCK_K = 256
_KERNEL = None


def quantize_rows_int8(w: torch.Tensor, chunk: int = 32768) -> tuple[torch.Tensor, torch.Tensor]:
    """w [N, K] float -> (q int8 [N, K], scale fp32 [N]); w ~ q * scale[:, None]."""
    q = torch.empty(w.shape, dtype=torch.int8, device=w.device)
    scale = torch.empty(w.shape[0], dtype=torch.float32, device=w.device)
    for s in range(0, w.shape[0], chunk):
        blk = w[s:s + chunk].float()
        sc = (blk.abs().amax(dim=1) / 127.0).clamp_min(1e-12)
        q[s:s + chunk] = torch.round(blk / sc[:, None]).clamp_(-127, 127).to(torch.int8)
        scale[s:s + chunk] = sc
    return q, scale


def int8_gemv_ref(q: torch.Tensor, scale: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """fp32 reference: y[n] = scale[n] * sum_k q[n, k] * x[k]."""
    return (q.float() @ x.float().reshape(-1)) * scale


def _build_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def cr_int8_gemv(q_ptr, s_ptr, x_ptr, y_ptr, N, K, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        rmask = rows < N
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            ks = k0 + tl.arange(0, BLOCK_K)
            kmask = ks < K
            w = tl.load(q_ptr + rows[:, None].to(tl.int64) * K + ks[None, :], mask=rmask[:, None] & kmask[None, :], other=0)
            xv = tl.load(x_ptr + ks, mask=kmask, other=0.0).to(tl.float32)
            acc += tl.sum(w.to(tl.float32) * xv[None, :], axis=1)
        sc = tl.load(s_ptr + rows, mask=rmask, other=0.0)
        tl.store(y_ptr + rows, (acc * sc).to(y_ptr.dtype.element_ty), mask=rmask)

    return triton, cr_int8_gemv


def triton_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("triton") is not None


def int8_gemv(q: torch.Tensor, scale: torch.Tensor, x: torch.Tensor, out_dtype=torch.bfloat16) -> torch.Tensor:
    """y [N] = dequant(q) @ x. Triton on CUDA when available, torch otherwise."""
    global _KERNEL
    x = x.reshape(-1).contiguous()
    if not (q.is_cuda and triton_available()):
        return int8_gemv_ref(q, scale, x).to(out_dtype)
    if _KERNEL is None:
        _KERNEL = _build_kernel()
    triton, kern = _KERNEL
    n, k = q.shape
    y = torch.empty(n, dtype=out_dtype, device=q.device)
    kern[(triton.cdiv(n, BLOCK_N),)](q, scale, x, y, n, k, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, num_warps=4)
    return y


class Int8Head(torch.nn.Module):
    """lm_head with INT8 rows and fp32 row scales; tensors may be views into a shared arena. Batch 1 only."""

    def __init__(self, q: torch.Tensor, scale: torch.Tensor, out_dtype=torch.bfloat16):
        super().__init__()
        self.q, self.scale, self.out_dtype = q, scale, out_dtype

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        y = int8_gemv(self.q, self.scale, h[:, -1, :], self.out_dtype)
        return y.view(1, 1, -1)


class Bf16Head(torch.nn.Module):
    """lm_head with a plain weight [V, H]; the weight may be a view into a shared arena."""

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = weight

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(h[:, -1:, :], self.weight)
