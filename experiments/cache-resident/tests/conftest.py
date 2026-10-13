"""Shared pytest setup: put EXP_DIR on sys.path, skip gpu / nvcc tests when unavailable."""
from __future__ import annotations

import os
import pathlib
import shutil
import sys

import pytest

EXP_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

NVCC = os.environ.get("CR_NVCC") or shutil.which("nvcc") or "/usr/local/cuda-12.8/bin/nvcc"


def _has_cuda() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    cuda = None
    for item in items:
        if "gpu" in item.keywords:
            cuda = _has_cuda() if cuda is None else cuda
            if not cuda:
                item.add_marker(pytest.mark.skip(reason="no CUDA device"))
        if "nvcc" in item.keywords and not os.path.exists(NVCC):
            item.add_marker(pytest.mark.skip(reason="nvcc not found"))


@pytest.fixture
def exp_dir() -> pathlib.Path:
    return EXP_DIR


@pytest.fixture
def results_dir(tmp_path, monkeypatch):
    """Isolated results root; clears RunPod and CR_* env that could leak in."""
    for k in list(os.environ):
        if k.startswith("RUNPOD_") or k in ("CR_GPU_TAG", "CR_CMD", "CR_EXP", "CR_REQUIRE_COUNTERS",
                                           "CR_SESSION_ID", "CR_LEDGER", "CR_START_EPOCH"):
            monkeypatch.delenv(k, raising=False)
    d = tmp_path / "results"
    d.mkdir()
    monkeypatch.setenv("CR_RESULTS_DIR", str(d))
    return d
