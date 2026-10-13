import json
import pathlib
import re
import shutil
import subprocess

import pytest

from infra import gpus

EXP = pathlib.Path(__file__).resolve().parent.parent
MAN = json.loads((EXP / "exps" / "manifest.json").read_text())
KEYS = {"cmd", "gpus", "est_min", "failsafe_min", "build_llama", "disk_gb", "desc"}
LONG_FAILSAFE = {"e4_pdl": 60}  # clone + cold build + downloads + 32 benches; launch needs --allow-long
EXPECTED = {"p0_smoke", "p0_l2", "e1", "e2_08b", "e2_2b", "e3", "e3d", "e4_pq2", "e4_ptq1", "e4_pdl", "e5_base", "e5_k4", "e5_k6", "e5_k8"}


def test_manifest_has_all_experiments():
    assert set(MAN) == EXPECTED


@pytest.mark.parametrize("name", sorted(MAN))
def test_manifest_entry_schema(name):
    e = MAN[name]
    assert set(e) == KEYS
    assert isinstance(e["cmd"], str) and e["cmd"]
    assert e["gpus"] and all(g in gpus.RUNPOD_CANDIDATES for g in e["gpus"])
    assert e["failsafe_min"] <= LONG_FAILSAFE.get(name, 40)
    assert e["est_min"] <= e["failsafe_min"] - 5
    assert isinstance(e["build_llama"], bool)
    assert isinstance(e["disk_gb"], int) and e["disk_gb"] > 0


def test_manifest_llama_builds_only_where_needed():
    assert all(MAN[k]["build_llama"] for k in MAN if k.startswith(("e4_", "e5_")))
    assert not any(MAN[k]["build_llama"] for k in MAN if k.startswith(("p0_", "e1", "e2", "e3")))


@pytest.mark.parametrize("name", sorted(MAN))
def test_manifest_modules_exist(name):
    cmd = MAN[name]["cmd"]
    for mod in re.findall(r"-m\s+([A-Za-z0-9_.]+)", cmd):
        top = mod.split(".")[0]
        assert (EXP / top).is_dir(), mod
        assert (EXP / pathlib.Path(*mod.split("."))).with_suffix(".py").exists() or (EXP / top / "__init__.py").exists(), mod
    m = re.match(r"bash\s+(\S+)", cmd)
    if m:
        assert (EXP / m.group(1)).exists()


def test_result_dirs_match_manifest_keys():
    from exps import e2_hybrid_bf16, e4_bonsai, e5_spec
    assert set(e2_hybrid_bf16.EXP_NAMES.values()) <= set(MAN)
    assert set(e4_bonsai.EXP_NAMES.values()) <= set(MAN)
    from exps import e4_pdl
    assert e4_pdl.EXP in MAN
    assert set(e5_spec.EXP_NAMES.values()) <= set(MAN)
    from exps import e1_sweep, e3_residency, e3d_resident_draft
    assert e1_sweep.EXP in MAN and e3_residency.EXP in MAN and e3d_resident_draft.EXP in MAN


def test_e5_commands_pass_matching_k():
    for k, name in {0: "e5_base", 4: "e5_k4", 6: "e5_k6", 8: "e5_k8"}.items():
        assert MAN[name]["cmd"].endswith(f"--k {k}")


def test_run_p0_syntax():
    if not shutil.which("bash"):
        pytest.skip("no bash")
    subprocess.run(["bash", "-n", str(EXP / "exps" / "run_p0.sh")], check=True)


def test_run_p0_mirrors_each_step_and_skips_slow_gpu_tests(tmp_path):
    """Results are copied into p0_smoke right after their step, even when a later step fails or is killed."""
    if not shutil.which("bash"):
        pytest.skip("no bash")
    shim = tmp_path / "bin" / "python3"
    shim.parent.mkdir()
    shim.write_text('#!/usr/bin/env bash\necho "$@" >> "$SHIM_LOG"\n'
                    'case "$*" in *p0_l2_check*) d=p0_l2;; *bench_gemv*) d=p0_gemv;; *toy_hybrid*) d=p0_toy;; *) d="";; esac\n'
                    '[ -n "$d" ] && mkdir -p "$CR_RESULTS_DIR/$d/fake" && echo "{}" > "$CR_RESULTS_DIR/$d/fake/1.json"\n'
                    '[[ "$*" == *toy_hybrid* ]] && exit 3\nexit 0\n')
    shim.chmod(0o755)
    import os
    env = dict(os.environ, PATH=f"{shim.parent}:{os.environ['PATH']}", CR_RESULTS_DIR=str(tmp_path / "results"),
               CR_GPU_TAG="rtx5090", SHIM_LOG=str(tmp_path / "log.txt"))
    r = subprocess.run(["bash", str(EXP / "exps" / "run_p0.sh")], env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 1, r.stdout + r.stderr          # the failing toy step is reported but does not stop the rest
    for d in ("p0_l2", "p0_gemv", "p0_toy"):
        assert (tmp_path / "results" / "p0_smoke" / d / "fake" / "1.json").exists(), d
    log = (tmp_path / "log.txt").read_text()
    assert "-m gpu and not slow" in log
    assert (tmp_path / "results" / "p0_smoke" / "rtx5090" / "logs" / "pytest_gpu.log").exists()
    # the pytest step runs last, after all copies were made
    assert r.stdout.index("=== p0_toy exit 3") < r.stdout.index("=== pytest_gpu:")
