import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness import power  # noqa: E402


class FakeNvml:
    """Constant 100 W. energy: list of mJ values returned in order, or None for not supported."""

    def __init__(self, energy=None, watts=100.0):
        self.energy = list(energy) if energy is not None else None
        self.watts = watts
        self.inited = 0
        self.shutdown = 0
        self.handle_idx = None

    def nvmlInit(self):
        self.inited += 1

    def nvmlShutdown(self):
        self.shutdown += 1

    def nvmlDeviceGetHandleByIndex(self, i):
        self.handle_idx = i
        return "h%d" % i

    def nvmlDeviceGetPowerUsage(self, h):
        return int(self.watts * 1000)

    def nvmlDeviceGetTotalEnergyConsumption(self, h):
        if self.energy is None:
            raise RuntimeError("Not Supported")
        return self.energy.pop(0) if len(self.energy) > 1 else self.energy[0]


def test_energy_counter_method():
    nv = FakeNvml(energy=[1000, 6000])
    with power.PowerSampler(dev=0, interval_s=0.005, nvml=nv) as ps:
        time.sleep(0.05)
    r = ps.result
    assert r["method"] == "energy_counter"
    assert r["joules"] == pytest.approx(5.0)
    assert r["avg_w"] == pytest.approx(5.0 / r["duration_s"])
    assert r["max_w"] == pytest.approx(100.0) and r["samples"] >= 3
    assert nv.inited == 1 and nv.shutdown == 1


def test_trapezoid_when_energy_unsupported():
    nv = FakeNvml(energy=None)
    ps = power.PowerSampler(interval_s=0.010, nvml=nv)
    ps.start()
    time.sleep(0.15)
    r = ps.stop()
    assert r["method"] == "trapezoid"
    assert r["joules"] == pytest.approx(100.0 * r["duration_s"], rel=0.1)
    assert r["avg_w"] == pytest.approx(100.0, rel=0.1)
    assert r["samples"] >= 5
    assert ps.stop() is r


def test_zero_energy_delta_falls_back():
    nv = FakeNvml(energy=[500, 500])
    with power.PowerSampler(interval_s=0.005, nvml=nv) as ps:
        time.sleep(0.04)
    assert ps.result["method"] == "trapezoid" and ps.result["joules"] > 0


def test_sample_errors_counted():
    class Bad(FakeNvml):
        def nvmlDeviceGetPowerUsage(self, h):
            raise RuntimeError("gpu lost")

    with power.PowerSampler(interval_s=0.005, nvml=Bad(energy=[0, 2000])) as ps:
        time.sleep(0.03)
    assert ps.result["errors"] >= 1 and ps.result["samples"] == 0
    assert ps.result["method"] == "energy_counter" and ps.result["joules"] == pytest.approx(2.0)


def test_start_stop_misuse():
    ps = power.PowerSampler(nvml=FakeNvml())
    with pytest.raises(RuntimeError):
        ps.stop()
    ps.start()
    with pytest.raises(RuntimeError):
        ps.start()
    ps.stop()


def test_trapezoid_pure():
    assert power._trapezoid([(0.0, 0.0), (1.0, 10.0), (3.0, 10.0)]) == pytest.approx(25.0)
    assert power._trapezoid([(0.0, 5.0)]) == 0.0


def test_tok_per_joule():
    assert power.tok_per_joule(100, 50.0) == 2.0
    assert power.tok_per_joule(100, 0.0) is None
    assert power.tok_per_joule(100, None) is None


def test_nvml_index_mapping(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert power.nvml_index(0) == 0
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,1")
    assert power.nvml_index(0) == 3 and power.nvml_index(1) == 1
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abc")
    assert power.nvml_index(0) == 0
    nv = FakeNvml(energy=[0, 1000])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    with power.PowerSampler(dev=0, nvml=nv):
        pass
    assert nv.handle_idx == 2
