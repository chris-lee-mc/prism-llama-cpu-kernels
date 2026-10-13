"""On-GPU smoke: each manifest command with --quick (run on a pod with -m gpu)."""
import importlib.util
import json
import os
import pathlib
import subprocess
import sys

import pytest

EXP = pathlib.Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.gpu


def _run(tmp, mod, *args, timeout=1500):
    env = dict(os.environ, CR_RESULTS_DIR=str(tmp / "results"))
    r = subprocess.run([sys.executable, "-m", mod, "--quick", *args], cwd=EXP, env=env, capture_output=True, text=True, timeout=timeout)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return r


def _newest(tmp, exp):
    files = sorted((tmp / "results" / exp).rglob("*.json"))
    assert files
    return json.loads(files[-1].read_text())


def test_e1_quick_and_plot(tmp_path):
    _run(tmp_path, "exps.e1_sweep", "--no-calib")
    doc = _newest(tmp_path, "e1")
    rows = doc["data"]["rows"]
    assert len(rows) == 8 and all(r["us_per_token"] > 0 and r["finite"] and r["check_ok"] for r in rows)
    from exps import plot_e1
    out = plot_e1.plot(plot_e1.load_latest(tmp_path / "results" / "e1"), tmp_path / "knee.png")
    assert out.stat().st_size > 5000


@pytest.mark.slow
@pytest.mark.skipif(importlib.util.find_spec("transformers") is None, reason="transformers missing")
def test_e2_quick(tmp_path):
    _run(tmp_path, "exps.e2_hybrid_bf16", "--model", "0.8B", "--no-ncu")
    assert len(_newest(tmp_path, "e2_08b")["data"]["rows"]) == 2


@pytest.mark.slow
@pytest.mark.skipif(importlib.util.find_spec("transformers") is None, reason="transformers missing")
def test_e3_quick(tmp_path):
    _run(tmp_path, "exps.e3_residency", "--no-ncu")
    assert _newest(tmp_path, "e3")["data"]["rows"]


def _llama(name):
    from exps.e4_bonsai import llama_bin
    return (llama_bin() / name).exists()


@pytest.mark.slow
@pytest.mark.skipif(not _llama("llama-bench"), reason="llama-bench not built")
def test_e4_quick(tmp_path):
    _run(tmp_path, "exps.e4_bonsai", "--quant", "PQ2_0", "--no-ncu")
    assert _newest(tmp_path, "e4_pq2")["data"]["rows"][0]["tg_tok_s"] > 0


@pytest.mark.slow
@pytest.mark.skipif(not _llama("llama-server"), reason="llama-server not built")
def test_e5_quick(tmp_path):
    _run(tmp_path, "exps.e5_spec", "--k", "4")
    assert _newest(tmp_path, "e5_k4")["data"]["summary"]["n_ok"] > 0
