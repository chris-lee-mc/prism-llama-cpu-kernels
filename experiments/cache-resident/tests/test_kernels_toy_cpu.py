"""ToyHybrid construction and launch plan with a mocked extension on CPU tensors (no GPU)."""
from __future__ import annotations

import dataclasses
import sys
import types

import pytest
import torch

from kernels import build as B
from kernels import ternary as T
from kernels import toy_hybrid as TH


class FakeExt:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def f(*args):
            self.calls.append((name, args))

        return f


@pytest.fixture
def fake(monkeypatch):
    ext = FakeExt()
    monkeypatch.setattr(B, "load_ext", lambda verbose=False: ext)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda dev=None: types.SimpleNamespace(cuda_stream=77))
    return ext


def test_plan_counts_and_pointers(fake):
    cfg = TH.ToyConfig(layers=4, ctx=64)
    m = TH.ToyHybrid(cfg, device="cpu", variant="lut")
    assert m.n_kernels == 3 * 12 + 13
    assert len(m.plan) == 4 * 12
    m.step()
    names = [c[0] for c in fake.calls]
    assert names.count("gemv") == 4 * 6
    assert names.count("gdn_step") == 3 and names.count("attn_decode") == 1
    assert names.count("rmsnorm") == 8 and names.count("add_") == 8 and names.count("silu_mul") == 4
    assert all(c[1][-1] == 77 for c in fake.calls)
    base = m.arena.data_ptr()
    for name, args in fake.calls:
        if name == "gemv":
            vi, qp, sp, xp, yp, N, K, _ = args
            assert vi == 0 and (qp - base) % 256 == 0 and (sp - base) % 256 == 0 and N >= 1 and K % 128 == 0
            assert base <= qp < base + m.arena.numel()


def test_fp4_variant_index(fake):
    m = TH.ToyHybrid(TH.ToyConfig(layers=1, ctx=64), device="cpu", variant="fp4")
    m.step()
    assert all(c[1][0] == 1 for c in fake.calls if c[0] == "gemv")


def test_arena_matrices_decode_to_random_ternary(fake):
    cfg = TH.ToyConfig(layers=1, ctx=64)
    m = TH.ToyHybrid(cfg, device="cpu", seed=3)
    q, s = m.matrix_views("l0.ab")
    q0, s0, t0 = T.random_ternary(32, 1024, seed=3 * 100003 + 1)
    assert torch.equal(q, q0) and torch.equal(s, s0)
    assert torch.equal(T.unpack(q, 1024), t0)


def test_state_arena_is_contiguous_and_reset(fake):
    m = TH.ToyHybrid(TH.ToyConfig(layers=8, ctx=64), device="cpu")
    assert m.state_arena().is_contiguous() and m.state_arena().shape[0] == 6
    m.state.zero_()
    m.x.zero_()
    m.reset()
    assert torch.equal(m.x, m.x0) and torch.equal(m.state, m.state0)


def test_weight_and_extra_bytes(fake):
    cfg = dataclasses.replace(TH.ToyConfig(layers=2, ctx=64), kinds=("gdn", "attn"))
    m = TH.ToyHybrid(cfg, device="cpu")
    assert m.weight_bytes() == sum(T.bytes_moved(x["N"], x["K"]) - 2 * x["K"] - 2 * x["N"] for x in TH.matrix_specs(cfg))
    assert m.extra_bytes() == 2 * 64 * 2 * 256 * 2 + 2 * 16 * 128 * 128 * 4
