"""CUDA event timing and CUDA graph capture helpers."""
from __future__ import annotations

import torch


def cuda_time_us(fn, iters: int = 100, warmup: int = 10, stream=None) -> float:
    """Mean microseconds per call of fn(), measured with CUDA events on `stream` (default: current)."""
    s = stream or torch.cuda.current_stream()
    with torch.cuda.stream(s):
        for _ in range(warmup):
            fn()
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record(s)
        for _ in range(iters):
            fn()
        b.record(s)
    b.synchronize()
    return a.elapsed_time(b) * 1000.0 / iters


def capture_graph(fn, warmup: int = 3, stream=None, keep_graph: bool = False) -> torch.cuda.CUDAGraph:
    """Warm up fn on a side stream, then capture one call.

    keep_graph=True returns an un-instantiated graph so l2persist.apply_window_to_graph can edit it;
    the caller then calls graph.instantiate(). `stream` is the capture stream: pass one when a stream
    window was set on it before capture.
    """
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph(keep_graph=keep_graph)
    with torch.cuda.graph(g, stream=stream):
        fn()
    return g


def replay_time_us(graph, iters: int = 100, warmup: int = 5) -> float:
    """Mean microseconds per replay. A keep_graph graph is instantiated on the first replay if needed."""
    for _ in range(warmup):
        graph.replay()
    a = torch.cuda.Event(enable_timing=True)
    b = torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        graph.replay()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) * 1000.0 / iters
