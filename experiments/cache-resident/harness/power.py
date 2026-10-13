"""NVML power sampling (PLAN P0.3): joules and tokens per joule.

Prefers the NVML total-energy counter delta (mJ). Falls back to trapezoid integration of
nvmlDeviceGetPowerUsage samples taken every interval_s.
"""
from __future__ import annotations

import os
import threading
import time


def nvml_index(dev: int) -> int:
    """Map a CUDA ordinal to the NVML index when CUDA_VISIBLE_DEVICES holds plain integers."""
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if vis:
        parts = [p.strip() for p in vis.split(",")]
        if dev < len(parts) and parts[dev].isdigit():
            return int(parts[dev])
    return dev


def tok_per_joule(tokens: int, joules: float) -> float | None:
    if joules is None or joules <= 0:
        return None
    return tokens / joules


def _trapezoid(samples: list[tuple[float, float]]) -> float:
    """samples: [(t_seconds, watts)] -> joules."""
    j = 0.0
    for (t0, w0), (t1, w1) in zip(samples, samples[1:]):
        j += 0.5 * (w0 + w1) * (t1 - t0)
    return j


class PowerSampler:
    """Context manager: samples power in a thread; stop() returns joules, avg/max watts, method."""

    def __init__(self, dev: int = 0, interval_s: float = 0.010, nvml=None):
        self.dev = dev
        self.interval_s = interval_s
        self._nvml = nvml
        self._own_init = False
        self._handle = None
        self._thread = None
        self._stop = threading.Event()
        self._samples: list[tuple[float, float]] = []
        self._errors = 0
        self._e0 = None
        self._t0 = 0.0
        self.result: dict | None = None

    def _read_w(self):
        return self._nvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0

    def _read_mj(self):
        try:
            return int(self._nvml.nvmlDeviceGetTotalEnergyConsumption(self._handle))
        except Exception:
            return None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("already started")
        if self._nvml is None:
            import pynvml  # nvidia-ml-py
            self._nvml = pynvml
        self._nvml.nvmlInit()
        self._own_init = True
        self._handle = self._nvml.nvmlDeviceGetHandleByIndex(nvml_index(self.dev))
        self._samples = []
        self._errors = 0
        self._stop.clear()
        self.result = None
        self._e0 = self._read_mj()
        self._t0 = time.perf_counter()
        self._sample()
        self._thread = threading.Thread(target=self._run, name="cr-power", daemon=True)
        self._thread.start()

    def _sample(self) -> None:
        try:
            self._samples.append((time.perf_counter(), self._read_w()))
        except Exception:
            self._errors += 1

    def _run(self) -> None:
        nxt = time.perf_counter() + self.interval_s
        while not self._stop.wait(max(0.0, nxt - time.perf_counter())):
            self._sample()
            nxt += self.interval_s

    def stop(self) -> dict:
        if self.result is not None:
            return self.result
        if self._thread is None:
            raise RuntimeError("not started")
        self._stop.set()
        self._thread.join()
        self._sample()
        t1 = time.perf_counter()
        e1 = self._read_mj()
        if self._own_init:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
        dur = t1 - self._t0
        watts = [w for _, w in self._samples]
        trap = _trapezoid(self._samples)
        if len(self._samples) == 1:
            trap = self._samples[0][1] * dur
        if self._e0 is not None and e1 is not None and e1 > self._e0:
            joules, method = (e1 - self._e0) / 1000.0, "energy_counter"
        else:
            joules, method = trap, "trapezoid"
        self.result = {"joules": joules, "avg_w": joules / dur if dur > 0 else 0.0,
                       "max_w": max(watts) if watts else 0.0, "samples": len(self._samples),
                       "duration_s": dur, "method": method, "errors": self._errors}
        self._thread = None
        return self.result

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        return False
