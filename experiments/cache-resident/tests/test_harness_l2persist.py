import contextlib
import os
import shutil
import subprocess
import sys
import sysconfig
import types

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness import l2persist as l2  # noqa: E402

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HAS_CUDA = torch.cuda.is_available()
MIB = 1024 * 1024


def test_clamp_window():
    assert l2._clamp_window(10, 100) == (10, False)
    assert l2._clamp_window(100, 100) == (100, False)
    assert l2._clamp_window(101, 100) == (100, True)
    assert l2._clamp_window(5, 0) == (5, False)
    with pytest.raises(ValueError):
        l2._clamp_window(-1, 100)


def test_suggest_hit_ratio():
    assert l2.suggest_hit_ratio(100, 50) == 0.5
    assert l2.suggest_hit_ratio(50, 100) == 1.0
    assert l2.suggest_hit_ratio(0, 100) == 1.0


def test_helpers_validate_inputs():
    with pytest.raises(ValueError):
        l2._resolve_ptr(1234)
    assert l2._resolve_ptr(1234, 64) == (1234, 64)
    with pytest.raises(ValueError):
        l2._resolve_ptr(torch.zeros(4))
    assert l2._stream_handle(77) == 77
    assert l2._stream_handle(types.SimpleNamespace(cuda_stream=88)) == 88
    with pytest.raises(TypeError):
        l2._stream_handle("x")
    with pytest.raises(ValueError):
        l2._check_ratio(1.5)
    assert l2._graph_handle(99) == 99
    with pytest.raises(TypeError):
        l2._graph_handle("x")

    class G:
        def raw_cuda_graph(self):
            raise RuntimeError("template gone")

    with pytest.raises(RuntimeError, match="keep_graph=True"):
        l2._graph_handle(G())


def test_backend_selection(monkeypatch):
    monkeypatch.setenv("CR_L2_BACKEND", "inline")
    assert l2.backend() == "inline"
    monkeypatch.setenv("CR_L2_BACKEND", "cuda-python")
    assert l2.backend() == "cuda-python"
    monkeypatch.setenv("CR_L2_BACKEND", "bogus")
    with pytest.raises(ValueError):
        l2.backend()
    monkeypatch.delenv("CR_L2_BACKEND")
    auto = l2.backend()
    assert auto in ("cuda-python", "inline")
    for v in ("auto", "AUTO", " auto "):
        monkeypatch.setenv("CR_L2_BACKEND", v)
        assert l2.backend() == auto


@pytest.mark.skipif(HAS_CUDA, reason="checks the no-GPU path")
def test_raises_cleanly_without_cuda(tmp_path):
    for fn in (l2.device_info, l2.max_persisting_bytes, l2.max_window_bytes, l2.get_limit, l2.set_limit, l2.reset,
               l2.log_device, l2.get_stream_window, l2.clear_stream_window):
        with pytest.raises(RuntimeError, match="no CUDA device"):
            fn()
    with pytest.raises(RuntimeError, match="no CUDA device"):
        l2.set_stream_window(1234, 64)
    with pytest.raises(RuntimeError, match="no CUDA device"):
        l2.apply_window_to_graph(1, 1234, 64)
    with pytest.raises(RuntimeError, match="no CUDA device"):
        with l2.persist(1234, 64):
            pass
    assert l2.main([]) == 1


class FakeBE:
    name = "fake"

    def __init__(self, persisting_max=72 * MIB, window_max=64 * MIB, graph_nodes=3):
        self.pmax, self.wmax, self.graph_nodes = persisting_max, window_max, graph_nodes
        self.calls = []
        self.limit = 0
        self.window = [0, 0, 0, 0], 0.0

    def max_persisting_bytes(self, dev):
        return self.pmax

    def max_window_bytes(self, dev):
        return self.wmax

    def l2_bytes(self, dev):
        return 96 * MIB

    def get_limit(self):
        return self.limit

    def set_limit(self, n):
        self.calls.append(("set_limit", n))
        self.limit = n

    def reset(self):
        self.calls.append(("reset",))

    def set_stream_window(self, stream, ptr, nbytes, ratio, hp, mp):
        self.calls.append(("set_window", stream, ptr, nbytes, ratio, hp, mp))
        self.window = [ptr, nbytes, hp, mp], ratio

    def get_stream_window(self, stream):
        return self.window

    def apply_window_to_graph(self, graph, ptr, nbytes, ratio, hp, mp):
        self.calls.append(("graph", graph, ptr, nbytes, ratio, hp, mp))
        self.node_window = [ptr, nbytes, hp, mp], ratio
        return self.graph_nodes

    node_window = None

    def get_graph_window(self, graph):
        self.calls.append(("get_graph", graph))
        return self.node_window


@pytest.fixture
def fake(monkeypatch):
    be = FakeBE()
    monkeypatch.setattr(l2, "_be", lambda: be)
    monkeypatch.setattr(l2.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(l2.torch.cuda, "get_device_properties", lambda d: types.SimpleNamespace(name="Fake 5090", major=12, minor=0))
    monkeypatch.setattr(l2.torch.cuda, "device", lambda d: contextlib.nullcontext())
    return be


def test_set_limit_clamps_to_device_max(fake):
    assert l2.set_limit() == 72 * MIB
    assert l2.set_limit(10 * MIB) == 10 * MIB
    assert l2.set_limit(10_000 * MIB) == 72 * MIB
    assert l2.set_limit(-5) == 0


def test_set_stream_window_clamps_and_reports(fake):
    w = l2.set_stream_window(0xA000, 128 * MIB, 0.75, stream=5)
    assert w == {"base_ptr": 0xA000, "num_bytes": 64 * MIB, "hit_ratio": 0.75, "hit_prop": "persisting",
                 "miss_prop": "streaming", "clamped": True}
    assert fake.calls[-1] == ("set_window", 5, 0xA000, 64 * MIB, 0.75, l2.ACCESS_PERSISTING, l2.ACCESS_STREAMING)
    w = l2.set_stream_window(0xA000, 1 * MIB, stream=5)
    assert w["clamped"] is False and w["num_bytes"] == 1 * MIB and w["hit_ratio"] == 1.0
    assert l2.get_stream_window(stream=5) == {"base_ptr": 0xA000, "num_bytes": MIB, "hit_ratio": 1.0,
                                              "hit_prop": "persisting", "miss_prop": "streaming"}
    l2.clear_stream_window(stream=5)
    assert fake.calls[-1] == ("set_window", 5, 0, 0, 0.0, l2.ACCESS_NORMAL, l2.ACCESS_NORMAL)
    with pytest.raises(ValueError):
        l2.set_stream_window(0xA000, 64, 2.0, stream=5)


def test_persist_context_order_and_cleanup_on_error(fake):
    with l2.persist(0xB000, 4 * MIB, 0.5, stream=9) as w:
        assert w["num_bytes"] == 4 * MIB
        assert fake.limit == 72 * MIB
    kinds = [c[0] for c in fake.calls]
    assert kinds == ["set_limit", "set_window", "set_window", "reset", "set_limit"]
    assert fake.calls[2][3] == 0 and fake.calls[-1] == ("set_limit", 0) and fake.limit == 0
    fake.calls.clear()
    with pytest.raises(KeyError):
        with l2.persist(0xB000, 4 * MIB, stream=9, limit=8 * MIB):
            assert fake.limit == 8 * MIB
            raise KeyError("x")
    assert [c[0] for c in fake.calls][-3:] == ["set_window", "reset", "set_limit"] and fake.limit == 0


def test_get_graph_window_readback(fake):
    assert l2.get_graph_window(1234) is None
    l2.apply_window_to_graph(1234, 0xC000, MIB, 0.5)
    w = l2.get_graph_window(types.SimpleNamespace(raw_cuda_graph=lambda: 1234))
    assert w == {"base_ptr": 0xC000, "num_bytes": MIB, "hit_ratio": 0.5, "hit_prop": "persisting", "miss_prop": "streaming"}
    assert fake.calls[-1] == ("get_graph", 1234)


def test_apply_window_to_graph_args(fake):
    n = l2.apply_window_to_graph(1234, 0xC000, 100 * MIB, 0.9)
    assert n == 3
    assert fake.calls[-1] == ("graph", 1234, 0xC000, 64 * MIB, 0.9, l2.ACCESS_PERSISTING, l2.ACCESS_STREAMING)
    g = types.SimpleNamespace(raw_cuda_graph=lambda: 4321)
    l2.apply_window_to_graph(g, 0xC000, MIB)
    assert fake.calls[-1][1] == 4321


def test_device_info_and_log(fake, capsys):
    info = l2.log_device(0)
    assert info == {"name": "Fake 5090", "cc": "12.0", "l2_bytes": 96 * MIB, "persisting_max_bytes": 72 * MIB,
                    "window_max_bytes": 64 * MIB, "persisting_limit_bytes": 0, "backend": "fake"}
    line = capsys.readouterr().out.strip()
    assert "sm_120" in line and "persisting_max=72.0 MiB" in line and "window_max=64.0 MiB" in line and "\n" not in line


# ---- cuda-python backend logic against a stub runtime (no driver needed)

class _Err:
    cudaSuccess = 0
    cudaErrorInvalidValue = 1


def _stub_rt(node_types):
    rt = types.SimpleNamespace()
    rt.cudaError_t = _Err
    rt.cudaGetErrorName = lambda e: (0, b"cudaErrorInvalidValue")
    rt.cudaGetErrorString = lambda e: (0, b"invalid argument")
    rt.cudaAccessProperty = lambda v: v
    rt.cudaGraphNodeType = types.SimpleNamespace(cudaGraphNodeTypeKernel="K")
    rt.cudaKernelNodeAttrID = types.SimpleNamespace(cudaLaunchAttributeAccessPolicyWindow="attr_k")
    rt.cudaStreamAttrID = types.SimpleNamespace(cudaStreamAttributeAccessPolicyWindow="attr_s")
    rt.cudaKernelNodeAttrValue = lambda: types.SimpleNamespace(accessPolicyWindow=types.SimpleNamespace())
    rt.cudaStreamAttrValue = lambda: types.SimpleNamespace(accessPolicyWindow=types.SimpleNamespace())
    nodes = ["n%d" % i for i in range(len(node_types))]
    rt.set_calls = []

    def get_nodes(graph, numNodes=0):
        return (0, nodes if numNodes else [], len(nodes))

    rt.cudaGraphGetNodes = get_nodes
    rt.cudaGraphNodeGetType = lambda n: (0, node_types[nodes.index(n)])
    rt.cudaGraphKernelNodeSetAttribute = lambda n, a, v: (rt.set_calls.append((n, a, v.accessPolicyWindow)) or (0,))
    def get_attr(n, a):
        w = types.SimpleNamespace(base_ptr=0xE000, num_bytes=123, hitRatio=0.25, hitProp=2, missProp=1)
        rt.get_calls.append((n, a))
        return (0, types.SimpleNamespace(accessPolicyWindow=w))

    rt.get_calls = []
    rt.cudaGraphKernelNodeGetAttribute = get_attr
    rt.cudaStreamSetAttribute = lambda s, a, v: (rt.set_calls.append((s, a, v.accessPolicyWindow)) or (0,))
    return rt


def _stub_backend(rt):
    be = object.__new__(l2._CudaPythonBackend)
    be.rt = rt
    return be


def test_cudapython_graph_only_kernel_nodes():
    rt = _stub_rt(["K", "memcpy", "K", "empty", "K"])
    n = _stub_backend(rt).apply_window_to_graph(7, 0xD000, 4 * MIB, 0.5, 2, 1)
    assert n == 3
    assert [c[0] for c in rt.set_calls] == ["n0", "n2", "n4"]
    w = rt.set_calls[0][2]
    assert (w.base_ptr, w.num_bytes, w.hitRatio, w.hitProp, w.missProp) == (0xD000, 4 * MIB, 0.5, 2, 1)
    assert rt.set_calls[0][1] == "attr_k"


def test_cudapython_get_graph_window_first_kernel_node():
    rt = _stub_rt(["memcpy", "K", "K"])
    vals, ratio = _stub_backend(rt).get_graph_window(7)
    assert vals == [0xE000, 123, 2, 1] and ratio == 0.25
    assert rt.get_calls == [("n1", "attr_k")]
    assert _stub_backend(_stub_rt(["memcpy"])).get_graph_window(7) is None


def test_cudapython_graph_empty_and_stream_attr():
    rt = _stub_rt([])
    be = _stub_backend(rt)
    assert be.apply_window_to_graph(7, 1, 1, 1.0, 2, 1) == 0
    be.set_stream_window(11, 0x10, 32, 1.0, 2, 1)
    assert rt.set_calls[-1][:2] == (11, "attr_s")


def test_cudapython_error_message():
    be = _stub_backend(_stub_rt([]))
    with pytest.raises(RuntimeError, match="myfunc failed: cudaErrorInvalidValue invalid argument"):
        be._chk((1,), "myfunc")
    assert be._chk((0,), "f") is None
    assert be._chk((0, 5), "f") == 5
    assert be._chk((0, 5, 6), "f") == (5, 6)


def test_cudapython_real_bindings_enums_and_struct():
    pytest.importorskip("cuda.bindings.runtime")
    be = l2._CudaPythonBackend()
    assert be._attr_id("stream") is not None and be._attr_id("kernel") is not None
    val = be.rt.cudaStreamAttrValue()
    be._fill(val.accessPolicyWindow, 0x1000, 4096, 0.25, 2, 1)
    w = val.accessPolicyWindow
    assert (w.base_ptr, w.num_bytes, w.hitRatio) == (0x1000, 4096, 0.25)
    assert int(w.hitProp) == 2 and int(w.missProp) == 1


@pytest.mark.slow
def test_inline_source_syntax():
    gxx = shutil.which("g++")
    if gxx is None:
        pytest.skip("g++ missing")
    import torch.utils.cpp_extension as ce

    cuda_inc = "/usr/local/cuda-12.8/include"
    if not os.path.isdir(cuda_inc):
        pytest.skip("CUDA headers missing")
    inc = ["-I" + p for p in ce.include_paths()] + ["-I" + sysconfig.get_paths()["include"], "-I" + cuda_inc]
    src = os.path.join(EXP, "harness", "csrc", "l2persist_ext.cpp")
    last = None
    for std in ("-std=c++20", "-std=c++17"):
        last = subprocess.run([gxx, std, "-fsyntax-only", "-DTORCH_EXTENSION_NAME=cr_l2persist_ext"] + inc + [src],
                              capture_output=True, text=True)
        if last.returncode == 0:
            break
    assert last.returncode == 0, last.stderr[-2000:]


def test_inline_source_covers_api():
    text = open(os.path.join(EXP, "harness", "csrc", "l2persist_ext.cpp")).read()
    for sym in ["cudaDevAttrMaxPersistingL2CacheSize", "cudaDevAttrMaxAccessPolicyWindowSize", "cudaLimitPersistingL2CacheSize",
                "cudaStreamAttributeAccessPolicyWindow", "cudaCtxResetPersistingL2Cache", "cudaGraphKernelNodeSetAttribute",
                "cudaKernelNodeAttributeAccessPolicyWindow", "cudaGraphNodeTypeKernel", "cudaGraphKernelNodeGetAttribute"]:
        assert sym in text, sym


# ---- on a GPU (pod): run for both backends

@pytest.mark.gpu
@pytest.mark.skipif(not HAS_CUDA, reason="needs a CUDA device")
@pytest.mark.parametrize("be_name", ["cuda-python", "inline"])
def test_gpu_roundtrip(be_name, monkeypatch):
    monkeypatch.setenv("CR_L2_BACKEND", be_name)
    info = l2.log_device(0)
    cc = tuple(int(x) for x in info["cc"].split("."))
    assert info["l2_bytes"] > 0 and info["backend"] == be_name
    if cc >= (8, 0):
        assert info["persisting_max_bytes"] > 0 and info["window_max_bytes"] > 0
    pmax = info["persisting_max_bytes"]
    assert 0 <= l2.set_limit(None) <= pmax
    x = torch.zeros(1 << 18, device="cuda")
    s = torch.cuda.Stream()
    w = l2.set_stream_window(x, hit_ratio=0.5, stream=s)
    rb = l2.get_stream_window(s)
    assert rb["num_bytes"] == w["num_bytes"] and rb["base_ptr"] == x.data_ptr()
    assert rb["hit_prop"] == "persisting" and rb["miss_prop"] == "streaming" and rb["hit_ratio"] == pytest.approx(0.5)
    l2.clear_stream_window(s)
    assert l2.get_stream_window(s)["num_bytes"] == 0
    l2.reset()
    l2.set_limit(0)
    with l2.persist(x, stream=s) as w2:
        assert w2["num_bytes"] == x.numel() * 4
    assert l2.get_limit() == 0


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_CUDA, reason="needs a CUDA device")
@pytest.mark.parametrize("be_name", ["cuda-python", "inline"])
def test_gpu_graph_nodes(be_name, monkeypatch):
    monkeypatch.setenv("CR_L2_BACKEND", be_name)
    x = torch.arange(1 << 16, device="cuda", dtype=torch.float32)
    y = torch.zeros_like(x)
    g = torch.cuda.CUDAGraph(keep_graph=True)
    s = torch.cuda.Stream()
    with torch.cuda.graph(g, stream=s):
        torch.mul(x, 2.0, out=y)
        torch.add(y, 1.0, out=y)
    l2.set_limit(None)
    try:
        n = l2.apply_window_to_graph(g, x, hit_ratio=1.0)
        assert n >= 2
        w = l2.get_graph_window(g)
        assert w["num_bytes"] == x.numel() * 4 and w["base_ptr"] == x.data_ptr()
        assert w["hit_prop"] == "persisting" and w["miss_prop"] == "streaming" and w["hit_ratio"] == pytest.approx(1.0)
        g.instantiate()
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(y, x * 2.0 + 1.0)
    finally:
        l2.reset()
        l2.set_limit(0)


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_CUDA, reason="needs a CUDA device")
def test_gpu_timing_helpers():
    from harness import timing

    x = torch.ones(1 << 20, device="cuda")
    assert timing.cuda_time_us(lambda: x.add_(1.0), iters=20, warmup=2) > 0
    g = timing.capture_graph(lambda: x.add_(1.0), keep_graph=True)
    g.instantiate()
    assert timing.replay_time_us(g, iters=20, warmup=2) > 0
