"""CPU references, toy model layout and CLI dry runs (no GPU)."""
from __future__ import annotations

import dataclasses
import subprocess
import sys

import pytest
import torch

from kernels import toy_hybrid as TH


def test_ref_gdn_matches_matrix_form():
    H, dk, dv = 3, 32, 16
    g = torch.Generator().manual_seed(0)
    S = torch.randn(H, dv, dk, generator=g)
    qkv = torch.randn(H * (2 * dk + dv), generator=g)
    ab = torch.randn(2 * H, generator=g)
    S1, o = TH.ref_gdn_step(S, qkv, ab, H, dk, dv)
    q = qkv[: H * dk].reshape(H, dk)
    k = qkv[H * dk : 2 * H * dk].reshape(H, dk)
    v = qkv[2 * H * dk :].reshape(H, dv)
    for h in range(H):
        qh = q[h] / torch.sqrt((q[h] ** 2).sum() + 1e-6)
        kh = k[h] / torch.sqrt((k[h] ** 2).sum() + 1e-6)
        a, b = torch.sigmoid(ab[h]), torch.sigmoid(ab[H + h])
        want = a * S[h] @ (torch.eye(dk) - b * torch.outer(kh, kh)) + b * torch.outer(v[h], kh)
        assert torch.allclose(S1[h], want, atol=1e-5)
        assert torch.allclose(o[h * dv : (h + 1) * dv], want @ qh, atol=1e-5)


def test_ref_gdn_hand_case():
    # H=1, dk=2 (not a kernel size, reference only), zero state: S' = beta v k^T, o = S' q
    S = torch.zeros(1, 1, 2)
    qkv = torch.tensor([1.0, 0.0, 0.0, 1.0, 5.0])  # q=(1,0) k=(0,1) v=(5)
    ab = torch.zeros(2)  # alpha = beta = 0.5
    S1, o = TH.ref_gdn_step(S, qkv, ab, 1, 2, 1)
    assert torch.allclose(S1, torch.tensor([[[0.0, 2.5]]]), atol=1e-4)
    assert torch.allclose(o, torch.tensor([0.0]), atol=1e-4)


def test_ref_attn_hand_case():
    # one kv head, two q heads, D=2, T=2. Equal scores -> o is the mean of V.
    q = torch.zeros(4)
    Kc = torch.randn(2, 1, 2)
    Vc = torch.tensor([[[1.0, 2.0]], [[3.0, 6.0]]])
    o = TH.ref_attn_decode(q, Kc, Vc, 2, 1, 2, 1.0)
    assert o.shape == (4,)
    assert torch.allclose(o, torch.tensor([2.0, 4.0, 2.0, 4.0]), atol=1e-6)


def test_ref_attn_peaked_softmax():
    q = torch.tensor([10.0, 0.0])
    Kc = torch.tensor([[[1.0, 0.0]], [[-1.0, 0.0]]])
    Vc = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    o = TH.ref_attn_decode(q, Kc, Vc, 1, 1, 2, 1.0)
    assert torch.allclose(o, torch.tensor([1.0, 0.0]), atol=1e-6)


def test_layer_kinds_default_mirrors_qwen35():
    kinds = TH.ToyConfig().layer_kinds()
    assert len(kinds) == 24 and kinds.count("attn") == 6 and kinds.count("gdn") == 18
    assert kinds[3] == "attn" and kinds[0] == "gdn"


def test_matrix_shapes_and_arena_alignment():
    cfg = TH.ToyConfig(layers=4)
    TH.validate(cfg)
    specs = {m["name"]: m for m in TH.matrix_specs(cfg)}
    assert (specs["l0.qkv"]["N"], specs["l0.qkv"]["K"]) == (16 * 384, 1024)
    assert (specs["l0.ab"]["N"], specs["l0.out"]["K"]) == (32, 2048)
    assert (specs["l3.q"]["N"], specs["l3.kv"]["N"], specs["l3.o"]["K"]) == (2048, 1024, 2048)
    assert (specs["l0.gate"]["N"], specs["l0.down"]["K"]) == (3584, 3584)
    layout, total = TH.arena_layout(cfg)
    spans = []
    for name, e in layout.items():
        if "q_off" in e:
            assert e["q_off"] % 256 == 0 and e["s_off"] % 256 == 0
            spans += [(e["q_off"], e["q_off"] + e["N"] * e["K"] // 4), (e["s_off"], e["s_off"] + e["N"] * (e["K"] // 128) * 2)]
        else:
            assert e["w_off"] % 256 == 0
            spans.append((e["w_off"], e["w_off"] + e["n"] * 2))
    spans.sort()
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))
    assert spans[-1][1] <= total


def test_validate_rejects_bad_dims():
    with pytest.raises(ValueError):
        TH.validate(TH.ToyConfig(d=1000))
    with pytest.raises(ValueError):
        TH.validate(TH.ToyConfig(dk=100))
    with pytest.raises(ValueError):
        TH.validate(dataclasses.replace(TH.ToyConfig(layers=2), kinds=("gdn",)))


def test_homogeneous_kinds():
    cfg = dataclasses.replace(TH.ToyConfig(layers=5), kinds=("attn",) * 5)
    assert TH.n_gdn_layers(cfg) == 0
    assert all(".qkv" not in m["name"] for m in TH.matrix_specs(cfg))


@pytest.mark.parametrize("mod", ["kernels.bench_gemv", "kernels.toy_hybrid"])
def test_dry_run_exits_zero(mod, exp_dir):
    r = subprocess.run([sys.executable, "-W", "ignore", "-m", mod, "--dry-run"], cwd=exp_dir, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip()


def test_no_cuda_exit_code(exp_dir):
    if torch.cuda.is_available():
        pytest.skip("has CUDA")
    r = subprocess.run([sys.executable, "-W", "ignore", "-m", "kernels.bench_gemv", "--quick"], cwd=exp_dir, capture_output=True, text=True, timeout=120)
    assert r.returncode == 2


def test_modules_import_without_gpu():
    import importlib

    for m in ("kernels", "kernels.build", "kernels.ternary", "kernels.bench_gemv", "kernels.toy_hybrid"):
        importlib.import_module(m)


def test_nvcc_flags():
    from kernels.build import nvcc_flags

    f = nvcc_flags((12, 0))
    assert "-gencode=arch=compute_120,code=sm_120" in f and "-gencode=arch=compute_120,code=compute_120" in f
    f = nvcc_flags((12, 0), fp4_mma=True)
    assert any("sm_120a" in x for x in f) and "-DCR_FP4_MMA" in f
    assert any("sm_86" in x for x in nvcc_flags((8, 6)))
