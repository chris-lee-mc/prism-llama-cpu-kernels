"""On-GPU tests (run on a pod with `pytest -m gpu tests/test_kernels_gpu.py`)."""
from __future__ import annotations

import dataclasses
import subprocess
import sys

import pytest
import torch

from kernels import ternary as T
from kernels import toy_hybrid as TH

pytestmark = pytest.mark.gpu

SHAPES = [(1, 128), (7, 384), (4096, 4096), (11008, 4096), (4096, 11008), (33, 128 * 40)]
TOL = 2e-3


def _rand(N, K, seed=0):
    q, s, _ = T.random_ternary(N, K, seed=seed, device="cuda")
    g = torch.Generator(device="cuda").manual_seed(seed + 1)
    x = torch.randn(K, generator=g, device="cuda").to(torch.float16)
    return q, s, x


def _close(y, ref):
    return torch.allclose(y.float(), ref, rtol=TOL, atol=TOL * ref.abs().max().item())


@pytest.mark.parametrize("variant", ["lut", "fp4"])
@pytest.mark.parametrize("N,K", SHAPES)
def test_gemv_matches_ref(variant, N, K):
    q, s, x = _rand(N, K, N + K)
    fn = T.gemv_lut if variant == "lut" else T.gemv_fp4
    y = fn(q, s, x)
    torch.cuda.synchronize()
    assert y.dtype == torch.float16 and y.shape == (N,)
    assert _close(y, T.ref_gemv(q, s, x))


@pytest.mark.parametrize("variant", ["lut", "fp4"])
def test_gemv_unaligned_scales_and_output(variant):
    N, K = 45, 512
    q, s, x = _rand(N, K, 3)
    sb = torch.empty(N * (K // 128) + 8, dtype=torch.float16, device="cuda")
    sb[1 : 1 + s.numel()] = s.reshape(-1)
    su = sb[1 : 1 + s.numel()].reshape(s.shape)
    yb = torch.empty(N + 8, dtype=torch.float16, device="cuda")
    yu = yb[3 : 3 + N]
    assert su.data_ptr() % 16 != 0 and yu.data_ptr() % 16 != 0
    fn = T.gemv_lut if variant == "lut" else T.gemv_fp4
    fn(q, su, x, out=yu)
    torch.cuda.synchronize()
    assert _close(yu, T.ref_gemv(q, s, x))


def test_variants_agree():
    q, s, x = _rand(1024, 2048, 9)
    a, b = T.gemv_lut(q, s, x), T.gemv_fp4(q, s, x)
    assert torch.allclose(a.float(), b.float(), rtol=1e-2, atol=1e-3 * a.float().abs().max().item())


@pytest.mark.parametrize("variant", ["lut", "fp4"])
def test_gemv_ptr_in_graph_equals_eager(variant):
    N, K = 1024, 1024
    q, s, x = _rand(N, K, 4)
    y_e = torch.empty(N, dtype=torch.float16, device="cuda")
    y_g = torch.zeros(N, dtype=torch.float16, device="cuda")
    T.gemv_ptr(variant, q.data_ptr(), s.data_ptr(), x.data_ptr(), y_e.data_ptr(), N, K)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        T.gemv_ptr(variant, q.data_ptr(), s.data_ptr(), x.data_ptr(), y_g.data_ptr(), N, K)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    y_g.zero_()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        T.gemv_ptr(variant, q.data_ptr(), s.data_ptr(), x.data_ptr(), y_g.data_ptr(), N, K)
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(y_e, y_g)


def test_bad_args_raise():
    q, s, x = _rand(4, 128)
    ext = T._ext()
    y = torch.empty(4, dtype=torch.float16, device="cuda")
    with pytest.raises(RuntimeError):
        ext.gemv(0, q.data_ptr(), s.data_ptr(), x.data_ptr(), y.data_ptr(), 4, 100, 0)
    with pytest.raises(RuntimeError):
        ext.gemv(5, q.data_ptr(), s.data_ptr(), x.data_ptr(), y.data_ptr(), 4, 128, 0)


# ---- toy ops vs CPU references ----
def _h(*shape, seed=0, scale=1.0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(*shape, generator=g, device="cuda") * scale).to(torch.float16)


@pytest.mark.parametrize("dk,dv,H", [(128, 128, 16), (64, 40, 3), (256, 32, 2), (32, 9, 1)])
def test_gdn_step_matches_ref(dk, dv, H):
    ext = T._ext()
    g = torch.Generator(device="cuda").manual_seed(1)
    S = torch.randn(H, dv, dk, generator=g, device="cuda") * 0.1
    qkv = _h(H * (2 * dk + dv), seed=2)
    ab = _h(2 * H, seed=3)
    o = torch.empty(H * dv, dtype=torch.float16, device="cuda")
    S_ref, o_ref = TH.ref_gdn_step(S.cpu(), qkv.cpu(), ab.cpu(), H, dk, dv)
    ext.gdn_step(S.data_ptr(), qkv.data_ptr(), ab.data_ptr(), o.data_ptr(), H, dk, dv, torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    assert torch.allclose(S.cpu(), S_ref, rtol=1e-4, atol=1e-5)
    assert torch.allclose(o.float().cpu(), o_ref, rtol=2e-3, atol=2e-3 * o_ref.abs().max().item())


@pytest.mark.parametrize("T_,Hq,Hkv,D", [(1, 8, 2, 256), (100, 8, 2, 128), (4096, 8, 2, 256), (777, 4, 4, 64), (9000, 32, 4, 128)])
def test_attn_decode_matches_ref(T_, Hq, Hkv, D):
    ext = T._ext()
    q = _h(Hq * D, seed=1)
    Kc = _h(T_, Hkv, D, seed=2, scale=0.5)
    Vc = _h(T_, Hkv, D, seed=3, scale=0.5)
    o = torch.empty(Hq * D, dtype=torch.float16, device="cuda")
    scale = D ** -0.5
    ext.attn_decode(q.data_ptr(), Kc.data_ptr(), Vc.data_ptr(), o.data_ptr(), Hq, Hkv, D, T_, scale, torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    ref = TH.ref_attn_decode(q.cpu(), Kc.cpu(), Vc.cpu(), Hq, Hkv, D, scale)
    got = o.float().cpu()
    mx = ref.abs().max().item()
    assert torch.allclose(got, ref, rtol=1e-2, atol=2e-3 * mx)
    assert (got - ref).abs().max().item() < 1e-3 * mx


def test_elementwise_ops():
    ext = T._ext()
    st = torch.cuda.current_stream().cuda_stream
    n = 3000
    x, w = _h(n, seed=1), (1 + 0.1 * _h(n, seed=2)).to(torch.float16)
    y = torch.empty_like(x)
    ext.rmsnorm(x.data_ptr(), w.data_ptr(), y.data_ptr(), n, 1e-6, st)
    xf = x.float()
    want = xf * torch.rsqrt((xf * xf).mean() + 1e-6) * w.float()
    torch.cuda.synchronize()
    assert torch.allclose(y.float(), want, rtol=2e-3, atol=2e-3)
    g, u = _h(n, seed=3), _h(n, seed=4)
    z = torch.empty_like(g)
    ext.silu_mul(g.data_ptr(), u.data_ptr(), z.data_ptr(), n, st)
    torch.cuda.synchronize()
    assert torch.allclose(z.float(), torch.nn.functional.silu(g.float()) * u.float(), rtol=2e-3, atol=2e-3)
    a = _h(n, seed=5)
    a0 = a.clone()
    ext.add_(a.data_ptr(), u.data_ptr(), n, st)
    torch.cuda.synchronize()
    assert torch.allclose(a.float(), (a0.float() + u.float()), rtol=2e-3, atol=2e-3)


# ---- toy model ----
@pytest.mark.parametrize("variant", ["lut", "fp4"])
def test_toy_graph_equals_eager(variant):
    cfg = TH.ToyConfig(layers=4, ctx=300)
    m = TH.ToyHybrid(cfg, variant=variant)
    m.reset()
    m.step()
    torch.cuda.synchronize()
    x_e, s_e = m.x.clone(), m.state.clone()
    m.reset()
    g = TH._capture(m.step)
    m.reset()
    g.replay()
    torch.cuda.synchronize()
    assert torch.isfinite(x_e.float()).all()
    assert torch.equal(m.x, x_e)
    assert torch.equal(m.state, s_e)
    assert m.n_kernels == 3 * 12 + 13  # 3 gdn layers (12 kernels) + 1 attention layer (13, attention is 2 kernels)


def test_toy_layer_matches_cpu_reference():
    cfg = TH.ToyConfig(layers=1, attn_every=1, ctx=200)  # one attention layer
    m = TH.ToyHybrid(cfg, variant="lut")
    m.reset()
    x0 = m.x.clone()
    m.step()
    torch.cuda.synchronize()

    def lin(name, v):
        q, s = m.matrix_views(name)
        return T.ref_gemv(q.cpu(), s.cpu(), v.cpu().to(torch.float16))

    def norm(v, tag):
        e = m.layout[f"l0.{tag}"]
        w = m.arena[e["w_off"] : e["w_off"] + cfg.d * 2].view(torch.float16).float().cpu()
        vf = v.float()
        return (vf * torch.rsqrt((vf * vf).mean() + TH.EPS) * w).to(torch.float16)

    x = x0.cpu()
    xn = norm(x, "n1")
    qa = lin("l0.q", xn).to(torch.float16)
    oa = TH.ref_attn_decode(qa, m.kc[0].cpu(), m.vc[0].cpu(), cfg.Hq, cfg.Hkv, cfg.D, cfg.scale).to(torch.float16)
    x = (x.float() + lin("l0.o", oa).to(torch.float16).float()).to(torch.float16)
    xn = norm(x, "n2")
    gt, up = lin("l0.gate", xn).to(torch.float16).float(), lin("l0.up", xn).to(torch.float16).float()
    act = (torch.nn.functional.silu(gt) * up).to(torch.float16)
    x = (x.float() + lin("l0.down", act).to(torch.float16).float()).to(torch.float16)
    assert torch.allclose(m.x.float().cpu(), x.float(), rtol=2e-2, atol=2e-2 * x.float().abs().max().item())


def test_toy_measure_small():
    r = TH.measure(TH.ToyConfig(layers=4, ctx=256), "lut", 10)
    assert r["us_per_token_graph"] > 0 and r["us_per_token_eager"] > 0
    assert r["launches_per_token"] == 3 * 12 + 13


def test_bench_quick_cli(exp_dir, tmp_path):
    env = dict(__import__("os").environ, CR_RESULTS_DIR=str(tmp_path))
    r = subprocess.run([sys.executable, "-m", "kernels.bench_gemv", "--quick", "--result-exp", "p0_gemv_test"], cwd=exp_dir, capture_output=True, text=True, timeout=900, env=env)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert list(tmp_path.glob("p0_gemv_test/*/*.json"))


def test_bench_target_on_rtx5090():
    from infra.gpus import peak_dram_gbps, tag_from_name
    from kernels import bench_gemv as B

    tag = tag_from_name(torch.cuda.get_device_name(0))
    if tag != "rtx5090":
        pytest.skip(f"target is defined for rtx5090, this is {tag}")
    gap = B.gap_estimate_us()
    rows = [B.bench_one(4096, 4096, v, 200, gap, peak_dram_gbps(tag), B.l2_bytes()) for v in ("lut", "fp4")]
    assert all(r["ok"] for r in rows)
    assert max(r["pct_peak_cold"] for r in rows) >= 0.70, rows
