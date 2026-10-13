"""Persisting-L2 control (PLAN P0.2).

Two backends with the same surface: `cuda-python` (cuda.bindings.runtime) and `inline`
(torch load() of csrc/l2persist_ext.cpp). CR_L2_BACKEND forces one; unset or "auto" tries cuda-python first.

Limit and stream window act on the current device. All functions raise RuntimeError("no CUDA device")
without CUDA; importing this module never needs a GPU.

Graph capture: a stream window set before capture is copied into captured kernel nodes. Or build the
graph with keep_graph=True and call apply_window_to_graph() before graph.instantiate(). Which one the
driver honors on sm_120 is decided on the GPU (p0_l2_check --method, E1 --window-method).
"""
from __future__ import annotations

import contextlib
import json
import os
import sys

import torch

ACCESS_NORMAL = 0
ACCESS_STREAMING = 1
ACCESS_PERSISTING = 2
_PROP_NAMES = {ACCESS_NORMAL: "normal", ACCESS_STREAMING: "streaming", ACCESS_PERSISTING: "persisting"}

_BACKENDS: dict = {}


# ---------------------------------------------------------------- pure helpers

def _clamp_window(nbytes: int, max_window: int) -> tuple[int, bool]:
    """Clamp a window size to the device maximum. Returns (nbytes, clamped)."""
    nbytes = int(nbytes)
    if nbytes < 0:
        raise ValueError("nbytes must be >= 0")
    if max_window > 0 and nbytes > max_window:
        return int(max_window), True
    return nbytes, False


def suggest_hit_ratio(nbytes: int, persist_limit: int) -> float:
    """hitRatio that keeps the persisted bytes within the set-aside size (avoids self-thrash)."""
    if nbytes <= 0 or persist_limit <= 0:
        return 1.0
    return min(1.0, persist_limit / float(nbytes))


def _check_ratio(hit_ratio: float) -> float:
    hit_ratio = float(hit_ratio)
    if not (0.0 <= hit_ratio <= 1.0):
        raise ValueError("hit_ratio must be in [0, 1]")
    return hit_ratio


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device")


def _resolve_ptr(ptr, nbytes=None) -> tuple[int, int]:
    """Tensor or int device pointer -> (base_ptr, nbytes)."""
    if isinstance(ptr, torch.Tensor):
        if ptr.device.type != "cuda":
            raise ValueError("tensor must be on a CUDA device")
        size = ptr.numel() * ptr.element_size()
        return int(ptr.data_ptr()), int(size if nbytes is None else nbytes)
    if nbytes is None:
        raise ValueError("nbytes is required when ptr is an int")
    return int(ptr), int(nbytes)


def _stream_handle(stream) -> int:
    if stream is None:
        return int(torch.cuda.current_stream().cuda_stream)
    if isinstance(stream, int):
        return stream
    if hasattr(stream, "cuda_stream"):
        return int(stream.cuda_stream)
    raise TypeError("stream must be torch.cuda.Stream, int handle, or None")


def _graph_handle(graph) -> int:
    if isinstance(graph, int):
        return graph
    if hasattr(graph, "raw_cuda_graph"):
        try:
            return int(graph.raw_cuda_graph())
        except RuntimeError as e:
            raise RuntimeError("graph template not live: build torch.cuda.CUDAGraph(keep_graph=True) (%s)" % e)
    raise TypeError("graph must be torch.cuda.CUDAGraph(keep_graph=True) or int cudaGraph_t")


# ---------------------------------------------------------------- backends

class _CudaPythonBackend:
    name = "cuda-python"

    def __init__(self):
        try:
            from cuda.bindings import runtime as rt
        except ImportError:
            from cuda import cudart as rt  # cuda-python < 12.8 layout
        self.rt = rt

    def _chk(self, res, what):
        rt = self.rt
        err, vals = res[0], res[1:]
        if err != rt.cudaError_t.cudaSuccess:
            try:
                name = rt.cudaGetErrorName(err)[1].decode()
                msg = rt.cudaGetErrorString(err)[1].decode()
            except Exception:
                name, msg = str(err), ""
            raise RuntimeError("%s failed: %s %s" % (what, name, msg))
        if not vals:
            return None
        return vals[0] if len(vals) == 1 else vals

    def _enum(self, cls_name, *names):
        cls = getattr(self.rt, cls_name)
        for n in names:
            v = getattr(cls, n, None)
            if v is not None:
                return v
        raise RuntimeError("cuda-python lacks %s.%s" % (cls_name, names[0]))

    def _attr_id(self, kind):
        if kind == "stream":
            return self._enum("cudaStreamAttrID", "cudaStreamAttributeAccessPolicyWindow",
                              "cudaLaunchAttributeAccessPolicyWindow")
        return self._enum("cudaKernelNodeAttrID", "cudaKernelNodeAttributeAccessPolicyWindow",
                          "cudaLaunchAttributeAccessPolicyWindow")

    def _dev_attr(self, name, dev):
        a = getattr(self.rt.cudaDeviceAttr, name)
        return int(self._chk(self.rt.cudaDeviceGetAttribute(a, dev), "cudaDeviceGetAttribute(%s)" % name))

    def max_persisting_bytes(self, dev):
        return self._dev_attr("cudaDevAttrMaxPersistingL2CacheSize", dev)

    def max_window_bytes(self, dev):
        return self._dev_attr("cudaDevAttrMaxAccessPolicyWindowSize", dev)

    def l2_bytes(self, dev):
        return self._dev_attr("cudaDevAttrL2CacheSize", dev)

    def get_limit(self):
        lim = self.rt.cudaLimit.cudaLimitPersistingL2CacheSize
        return int(self._chk(self.rt.cudaDeviceGetLimit(lim), "cudaDeviceGetLimit"))

    def set_limit(self, nbytes):
        lim = self.rt.cudaLimit.cudaLimitPersistingL2CacheSize
        self._chk(self.rt.cudaDeviceSetLimit(lim, int(nbytes)), "cudaDeviceSetLimit")

    def reset(self):
        self._chk(self.rt.cudaCtxResetPersistingL2Cache(), "cudaCtxResetPersistingL2Cache")

    def _fill(self, w, ptr, nbytes, hit_ratio, hit_prop, miss_prop):
        ap = self.rt.cudaAccessProperty
        w.base_ptr = int(ptr)
        w.num_bytes = int(nbytes)
        w.hitRatio = float(hit_ratio)
        w.hitProp = ap(hit_prop)
        w.missProp = ap(miss_prop)

    def set_stream_window(self, stream, ptr, nbytes, hit_ratio, hit_prop, miss_prop):
        val = self.rt.cudaStreamAttrValue()
        self._fill(val.accessPolicyWindow, ptr, nbytes, hit_ratio, hit_prop, miss_prop)
        self._chk(self.rt.cudaStreamSetAttribute(stream, self._attr_id("stream"), val), "cudaStreamSetAttribute")

    def get_stream_window(self, stream):
        val = self._chk(self.rt.cudaStreamGetAttribute(stream, self._attr_id("stream")), "cudaStreamGetAttribute")
        w = val.accessPolicyWindow
        return [int(w.base_ptr), int(w.num_bytes), int(w.hitProp), int(w.missProp)], float(w.hitRatio)

    def _kernel_nodes(self, graph):
        rt = self.rt
        n = self._chk(rt.cudaGraphGetNodes(graph), "cudaGraphGetNodes(count)")
        count_nodes = n[1] if isinstance(n, (list, tuple)) else 0
        if not count_nodes:
            return []
        res = rt.cudaGraphGetNodes(graph, count_nodes)
        nodes, got = self._chk(res, "cudaGraphGetNodes")
        out = []
        for node in list(nodes)[: int(got)]:
            ntype = self._chk(rt.cudaGraphNodeGetType(node), "cudaGraphNodeGetType")
            if ntype == rt.cudaGraphNodeType.cudaGraphNodeTypeKernel:
                out.append(node)
        return out

    def apply_window_to_graph(self, graph, ptr, nbytes, hit_ratio, hit_prop, miss_prop):
        rt = self.rt
        nodes = self._kernel_nodes(graph)
        if not nodes:
            return 0
        val = rt.cudaKernelNodeAttrValue()
        self._fill(val.accessPolicyWindow, ptr, nbytes, hit_ratio, hit_prop, miss_prop)
        attr = self._attr_id("kernel")
        for node in nodes:
            self._chk(rt.cudaGraphKernelNodeSetAttribute(node, attr, val), "cudaGraphKernelNodeSetAttribute")
        return len(nodes)

    def get_graph_window(self, graph):
        """([base_ptr, num_bytes, hit_prop, miss_prop], hit_ratio) of the first kernel node, or None."""
        nodes = self._kernel_nodes(graph)
        if not nodes:
            return None
        val = self._chk(self.rt.cudaGraphKernelNodeGetAttribute(nodes[0], self._attr_id("kernel")),
                        "cudaGraphKernelNodeGetAttribute")
        w = val.accessPolicyWindow
        return [int(w.base_ptr), int(w.num_bytes), int(w.hitProp), int(w.missProp)], float(w.hitRatio)


class _InlineBackend:
    name = "inline"

    def __init__(self):
        from torch.utils.cpp_extension import CUDA_HOME, load

        home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or CUDA_HOME or "/usr/local/cuda"
        src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", "l2persist_ext.cpp")
        # plain C++ build: only cudart is needed, so no nvcc and no arch flags
        self.ext = load(name="cr_l2persist_ext", sources=[src], extra_cflags=["-O2"],
                        extra_include_paths=[os.path.join(home, "include")],
                        extra_ldflags=["-L" + os.path.join(home, "lib64"), "-lcudart", "-Wl,-rpath," + os.path.join(home, "lib64")],
                        verbose=bool(os.environ.get("CR_L2_VERBOSE")))

    def max_persisting_bytes(self, dev):
        return int(self.ext.max_persisting_bytes(dev))

    def max_window_bytes(self, dev):
        return int(self.ext.max_window_bytes(dev))

    def l2_bytes(self, dev):
        return int(self.ext.l2_bytes(dev))

    def get_limit(self):
        return int(self.ext.get_limit())

    def set_limit(self, nbytes):
        self.ext.set_limit(int(nbytes))

    def reset(self):
        self.ext.reset()

    def set_stream_window(self, stream, ptr, nbytes, hit_ratio, hit_prop, miss_prop):
        self.ext.set_stream_window(int(stream), int(ptr), int(nbytes), float(hit_ratio), int(hit_prop), int(miss_prop))

    def get_stream_window(self, stream):
        vals, ratio = self.ext.get_stream_window(int(stream))
        return [int(v) for v in vals], float(ratio)

    def apply_window_to_graph(self, graph, ptr, nbytes, hit_ratio, hit_prop, miss_prop):
        return int(self.ext.apply_window_to_graph(int(graph), int(ptr), int(nbytes), float(hit_ratio),
                                                  int(hit_prop), int(miss_prop)))

    def get_graph_window(self, graph):
        r = self.ext.get_graph_window(int(graph))
        if r is None:
            return None
        vals, ratio = r
        return [int(v) for v in vals], float(ratio)


def backend() -> str:
    """Active backend name: CR_L2_BACKEND if cuda-python or inline, else (unset or "auto") cuda-python when importable, else inline."""
    forced = os.environ.get("CR_L2_BACKEND", "").strip().lower()
    if forced in ("cuda-python", "inline"):
        return forced
    if forced not in ("", "auto"):
        raise ValueError("CR_L2_BACKEND must be auto, cuda-python or inline, got %r" % forced)
    try:
        from cuda.bindings import runtime  # noqa: F401
    except ImportError:
        try:
            from cuda import cudart  # noqa: F401
        except ImportError:
            return "inline"
    return "cuda-python"


def _be():
    _require_cuda()
    name = backend()
    be = _BACKENDS.get(name)
    if be is None:
        be = _CudaPythonBackend() if name == "cuda-python" else _InlineBackend()
        _BACKENDS[name] = be
    return be


# ---------------------------------------------------------------- public API

def max_persisting_bytes(dev: int = 0) -> int:
    """cudaDevAttrMaxPersistingL2CacheSize."""
    return _be().max_persisting_bytes(dev)


def max_window_bytes(dev: int = 0) -> int:
    """cudaDevAttrMaxAccessPolicyWindowSize."""
    return _be().max_window_bytes(dev)


def get_limit() -> int:
    """cudaDeviceGetLimit(PersistingL2CacheSize) on the current device."""
    return _be().get_limit()


def set_limit(nbytes: int | None = None) -> int:
    """Set the persisting set-aside, clamped to the device max (None = max). Returns the limit read back."""
    be = _be()
    dev = torch.cuda.current_device()
    cap = be.max_persisting_bytes(dev)
    want = cap if nbytes is None else max(0, min(int(nbytes), cap))
    be.set_limit(want)
    return be.get_limit()


def _window_info(base, nbytes, hit_ratio, clamped, hit_prop=ACCESS_PERSISTING, miss_prop=ACCESS_STREAMING):
    return {"base_ptr": int(base), "num_bytes": int(nbytes), "hit_ratio": float(hit_ratio),
            "hit_prop": _PROP_NAMES[hit_prop], "miss_prop": _PROP_NAMES[miss_prop], "clamped": bool(clamped)}


def set_stream_window(ptr, nbytes: int | None = None, hit_ratio: float = 1.0, stream=None,
                      hit_prop: int = ACCESS_PERSISTING, miss_prop: int = ACCESS_STREAMING) -> dict:
    """Access policy window on a stream (default hitProp Persisting, missProp Streaming). nbytes clamped to the window max."""
    be = _be()
    hit_ratio = _check_ratio(hit_ratio)
    base, size = _resolve_ptr(ptr, nbytes)
    size, clamped = _clamp_window(size, be.max_window_bytes(torch.cuda.current_device()))
    be.set_stream_window(_stream_handle(stream), base, size, hit_ratio, hit_prop, miss_prop)
    return _window_info(base, size, hit_ratio, clamped, hit_prop, miss_prop)


def get_stream_window(stream=None) -> dict:
    """Read back the stream window as the driver reports it."""
    (base, size, hit_prop, miss_prop), ratio = _be().get_stream_window(_stream_handle(stream))
    return {"base_ptr": base, "num_bytes": size, "hit_ratio": ratio,
            "hit_prop": _PROP_NAMES.get(hit_prop, str(hit_prop)), "miss_prop": _PROP_NAMES.get(miss_prop, str(miss_prop))}


def get_graph_window(graph) -> dict | None:
    """Read back the window of the first kernel node of a graph (keep_graph template or int cudaGraph_t); None without kernel nodes."""
    r = _be().get_graph_window(_graph_handle(graph))
    if r is None:
        return None
    (base, size, hit_prop, miss_prop), ratio = r
    return {"base_ptr": base, "num_bytes": size, "hit_ratio": ratio,
            "hit_prop": _PROP_NAMES.get(hit_prop, str(hit_prop)), "miss_prop": _PROP_NAMES.get(miss_prop, str(miss_prop))}


def clear_stream_window(stream=None) -> None:
    """num_bytes = 0, hitProp = missProp = Normal."""
    _be().set_stream_window(_stream_handle(stream), 0, 0, 0.0, ACCESS_NORMAL, ACCESS_NORMAL)


def reset() -> None:
    """cudaCtxResetPersistingL2Cache: turn all persisting lines back to normal."""
    _be().reset()


@contextlib.contextmanager
def persist(ptr, nbytes=None, hit_ratio: float = 1.0, stream=None, limit=None):
    """Set the limit and a stream window; on exit clear the window, reset the cache, set the limit to 0."""
    set_limit(limit)
    try:
        yield set_stream_window(ptr, nbytes, hit_ratio, stream)
    finally:
        try:
            clear_stream_window(stream)
        finally:
            try:
                reset()
            finally:
                set_limit(0)


def apply_window_to_graph(graph, ptr, nbytes: int | None = None, hit_ratio: float = 1.0) -> int:
    """Set the access policy window on every kernel node of a graph. Returns the node count.

    graph: torch.cuda.CUDAGraph built with keep_graph=True, or an int cudaGraph_t. Call graph.instantiate() after.
    """
    be = _be()
    hit_ratio = _check_ratio(hit_ratio)
    base, size = _resolve_ptr(ptr, nbytes)
    size, _ = _clamp_window(size, be.max_window_bytes(torch.cuda.current_device()))
    return int(be.apply_window_to_graph(_graph_handle(graph), base, size, hit_ratio, ACCESS_PERSISTING, ACCESS_STREAMING))


def device_info(dev: int = 0) -> dict:
    """Device-reported L2 numbers: {name, cc, l2_bytes, persisting_max_bytes, window_max_bytes, persisting_limit_bytes, backend}."""
    be = _be()
    props = torch.cuda.get_device_properties(dev)
    with torch.cuda.device(dev):
        limit = be.get_limit()
    return {"name": props.name, "cc": "%d.%d" % (props.major, props.minor), "l2_bytes": be.l2_bytes(dev),
            "persisting_max_bytes": be.max_persisting_bytes(dev), "window_max_bytes": be.max_window_bytes(dev),
            "persisting_limit_bytes": limit, "backend": be.name}


def log_device(dev: int = 0) -> dict:
    """Print one line with the device-reported numbers and return them."""
    info = device_info(dev)
    mib = 1024.0 * 1024.0
    print("[l2persist] %s sm_%s L2=%.1f MiB persisting_max=%.1f MiB window_max=%.1f MiB limit=%.1f MiB backend=%s" % (
        info["name"], info["cc"].replace(".", ""), info["l2_bytes"] / mib, info["persisting_max_bytes"] / mib,
        info["window_max_bytes"] / mib, info["persisting_limit_bytes"] / mib, info["backend"]), flush=True)
    return info


def main(argv=None) -> int:
    try:
        info = log_device(0)
    except RuntimeError as e:
        print(json.dumps({"ok": False, "error": str(e)}))
        return 1
    print(json.dumps(info, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
