import json
import pathlib
import subprocess
import sys

import pytest

EXP = pathlib.Path(__file__).resolve().parent.parent
DRIVERS = [
    ("exps.e1_sweep", []),
    ("exps.e2_hybrid_bf16", ["--model", "0.8B"]),
    ("exps.e2_hybrid_bf16", ["--model", "2B"]),
    ("exps.e3_residency", []),
    ("exps.e3d_resident_draft", []),
    ("exps.e4_bonsai", ["--quant", "PQ2_0"]),
    ("exps.e4_bonsai", ["--quant", "PTQ1_0"]),
    ("exps.e5_spec", ["--k", "0"]),
    ("exps.e5_spec", ["--k", "6"]),
]


@pytest.mark.parametrize("mod,extra", DRIVERS)
@pytest.mark.parametrize("quick", [False, True])
def test_dry_run_in_process(mod, extra, quick, capsys):
    import importlib
    m = importlib.import_module(mod)
    rc = m.main(extra + ["--dry-run"] + (["--quick"] if quick else []))
    assert rc == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["exp"]


def test_dry_run_subprocess_e1():
    r = subprocess.run([sys.executable, "-m", "exps.e1_sweep", "--dry-run"], cwd=EXP, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    plan = json.loads(r.stdout)
    assert [x["W_mib"] for x in plan["layers"]] == [8, 16, 24, 32, 48, 56, 64, 80, 96, 128, 192, 256, 512, 1024]
    assert plan["modes"] == ["off", "0.5", "0.75", "1.0", "1.0L"] and plan["C_mib"] == 512


def test_proj_runs_offline(capsys):
    from exps import proj_hybrid
    assert proj_hybrid.main(["--floor-us", "12", "--bw-gbps", "1400"]) == 0
    assert "qwen3.5-0.8b" in capsys.readouterr().out
