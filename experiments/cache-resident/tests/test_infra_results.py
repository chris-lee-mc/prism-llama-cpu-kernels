from __future__ import annotations

import json
import math
import re
import sys
import types

import pytest

from infra import results


@pytest.fixture
def fake_env(monkeypatch):
    md = {"gpu_name": "Fake GPU", "git_sha": "abc"}
    monkeypatch.setattr(results, "env_metadata", lambda: dict(md))
    return md


def test_write_result_basic(tmp_path, fake_env):
    p = results.write_result("e1", "rtx5090", {"rows": [1, 2], "x": float("nan"), "y": float("inf")}, root=tmp_path)
    assert p.parent == tmp_path / "e1" / "rtx5090"
    assert re.fullmatch(r"\d{8}T\d{6}Z\.json", p.name)
    doc = json.loads(p.read_text())
    assert doc["schema"] == 1 and doc["exp"] == "e1" and doc["gpu"] == "rtx5090"
    assert doc["env"] == fake_env
    assert doc["data"] == {"rows": [1, 2], "x": None, "y": None}
    assert doc["utc"].endswith("+00:00") and "T" in doc["utc"]
    p.read_bytes().decode("ascii")


def test_collision_suffix(tmp_path, fake_env, monkeypatch):
    monkeypatch.setattr(results, "_stamp", lambda now=None: "20260101T000000Z")
    names = [results.write_result("e", "g", {"i": i}, root=tmp_path).name for i in range(3)]
    assert names == ["20260101T000000Z.json", "20260101T000000Z-1.json", "20260101T000000Z-2.json"]
    d = tmp_path / "e" / "g"
    assert json.loads((d / names[0]).read_text())["data"] == {"i": 0}


def test_gpu_none_uses_tag_and_env_root(results_dir, fake_env, monkeypatch):
    monkeypatch.setenv("CR_GPU_TAG", "rtx3090")
    p = results.write_result("e2", None, {"a": 1})
    assert p.parent == results_dir / "e2" / "rtx3090"
    assert results.results_root() == results_dir


@pytest.mark.parametrize("exp,gpu", [("Bad", "g"), ("a/b", "g"), ("e", "G"), ("e", "a b"), ("..", "g"), ("e", ".."), ("", "g")])
def test_bad_names(tmp_path, fake_env, exp, gpu):
    with pytest.raises(ValueError):
        results.write_result(exp, gpu, {}, root=tmp_path)


def test_numpy_and_odd_values(tmp_path, fake_env):
    np = pytest.importorskip("numpy")
    p = results.write_result("e", "g", {"a": np.float32(1.5), "b": np.arange(3), "c": (1, 2), "d": object, "e": np.nan},
                             root=tmp_path)
    d = json.loads(p.read_text())["data"]
    assert d["a"] == 1.5 and d["b"] == [0, 1, 2] and d["c"] == [1, 2] and d["e"] is None
    assert isinstance(d["d"], str)


def test_run_dir_and_artifact_path(results_dir, monkeypatch):
    monkeypatch.setenv("CR_GPU_TAG", "gx")
    d = results.run_dir("e3")
    assert d == results_dir / "e3" / "gx" and d.is_dir()
    a = results.artifact_path("e3", "plot.png")
    assert a.parent == d / "raw" and a.parent.is_dir()
    assert re.fullmatch(r"\d{8}T\d{6}Z-plot\.png", a.name)
    with pytest.raises(ValueError):
        results.artifact_path("e3", "../x")


def test_env_metadata_never_raises_and_has_keys(monkeypatch):
    md = results.env_metadata()
    for k in ("gpu_name", "gpu_tag", "cc", "sm", "driver", "cuda_runtime", "cuda_driver", "torch", "vram_bytes",
              "sm_count", "l2_bytes", "persisting_l2_max_bytes", "access_policy_window_max_bytes",
              "persisting_l2_limit_bytes", "git_sha", "git_branch", "hostname", "pod_id", "python", "counters_ok"):
        assert k in md
    assert md["counters_ok"] is None
    json.dumps(md)


def test_env_metadata_env_overrides_and_l2persist(monkeypatch):
    monkeypatch.setenv("CR_GIT_SHA", "deadbeef")
    monkeypatch.setenv("RUNPOD_POD_ID", "pod123")
    monkeypatch.setenv("CR_GPU_TAG", "mytag")
    h = types.ModuleType("harness")
    l2 = types.ModuleType("harness.l2persist")
    l2.device_info = lambda dev=0: {"persisting_max_bytes": 100, "window_max_bytes": 200,
                                    "persisting_limit_bytes": 0, "l2_bytes": 300}
    h.l2persist = l2
    monkeypatch.setitem(sys.modules, "harness", h)
    monkeypatch.setitem(sys.modules, "harness.l2persist", l2)
    md = results.env_metadata()
    assert md["git_sha"] == "deadbeef" and md["pod_id"] == "pod123" and md["gpu_tag"] == "mytag"
    assert md["persisting_l2_max_bytes"] == 100 and md["access_policy_window_max_bytes"] == 200
    assert md["persisting_l2_limit_bytes"] == 0
    if md["l2_bytes"] is not None:
        assert isinstance(md["l2_bytes"], int)


def test_env_metadata_survives_failing_l2persist(monkeypatch):
    h = types.ModuleType("harness")
    l2 = types.ModuleType("harness.l2persist")

    def boom(dev=0):
        raise RuntimeError("no CUDA device")
    l2.device_info = boom
    h.l2persist = l2
    monkeypatch.setitem(sys.modules, "harness", h)
    monkeypatch.setitem(sys.modules, "harness.l2persist", l2)
    md = results.env_metadata()
    assert md["persisting_l2_max_bytes"] is None


def test_gpu_tag_no_gpu(monkeypatch):
    monkeypatch.delenv("CR_GPU_TAG", raising=False)
    monkeypatch.setattr(results, "_torch_name", lambda: None)
    monkeypatch.setattr(results, "_smi_name", lambda: None)
    assert results.gpu_tag() == "cpu"
    monkeypatch.setattr(results, "_torch_name", lambda: "NVIDIA GeForce RTX 5090")
    assert results.gpu_tag() == "rtx5090"
