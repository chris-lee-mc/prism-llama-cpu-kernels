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
OPTIONAL_KEYS = {"result_exp"}          # a chained entry: the exp dir of its first command
LONG_FAILSAFE = {"e4_pdl": 70, "r1": 55, "r2a": 55, "r2b_mc": 55, "r2b_spec": 55, "wrap_a3": 55, "wrap_b": 55, "wrap_ab": 60}  # clone + cold build + downloads + the runs; launch needs --allow-long
EXPECTED = {"p0_smoke", "p0_l2", "e1", "e2_08b", "e2_2b", "e3", "e3d", "e4_pq2", "e4_ptq1", "e4_pdl", "e5_base", "e5_k4", "e5_k6", "e5_k8", "e5_cost", "r1", "r2a", "r2b_mc", "r2b_spec", "wrap_a", "wrap_a3", "wrap_b", "wrap_ab"}


def test_manifest_has_all_experiments():
    assert set(MAN) == EXPECTED


@pytest.mark.parametrize("name", sorted(MAN))
def test_manifest_entry_schema(name):
    e = MAN[name]
    assert KEYS <= set(e) <= KEYS | OPTIONAL_KEYS
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
    from exps import e5_cost, r1, r2a_longctx, r2b_multicol, r2b_spec, wrap_a, wrap_a3, wrap_b
    assert e5_cost.EXP in MAN and r1.EXP in MAN and r2a_longctx.EXP in MAN and r2b_multicol.EXP in MAN and r2b_spec.EXP in MAN
    assert wrap_a.EXP in MAN and wrap_a3.EXP in MAN and wrap_b.EXP in MAN
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


def test_r1_entry_fits_one_session():
    from infra import runpod
    e = MAN["r1"]
    assert e["gpus"] == ["pro6000"] and e["build_llama"] is True
    # est_min counts the downloads and the r1 budget; with bootstrap and the cold llama build it must fit the failsafe (55, --allow-long)
    assert e["failsafe_min"] == 55 and e["est_min"] + runpod.BOOTSTRAP_MIN + runpod.LLAMA_BUILD_MIN <= e["failsafe_min"] - 5
    assert e["cmd"].startswith("python3 -m exps.fetch_models --models pq2,draft_bf16,draft_q8_0 --timeout-min 6 && python3 -m exps.r1")
    from exps import r1
    budget = float(re.search(r"exps\.r1 --budget-min (\d+)", e["cmd"]).group(1))
    assert budget == r1.DEFAULT_TOTAL_MIN and e["est_min"] == 6 + budget


def test_r2a_entry_fits_one_session():
    from infra import runpod
    e = MAN["r2a"]
    assert e["gpus"] == ["pro6000"] and e["build_llama"] is True
    assert e["failsafe_min"] == 55 and e["est_min"] + runpod.BOOTSTRAP_MIN + runpod.LLAMA_BUILD_MIN <= e["failsafe_min"] - 5
    assert e["cmd"].startswith("python3 -m exps.fetch_models --models pq2,draft_bf16 --timeout-min 6 && python3 -m exps.r2a_longctx")
    from exps import r2a_longctx
    budget = float(re.search(r"exps\.r2a_longctx --budget-min (\d+)", e["cmd"]).group(1))
    assert budget == r2a_longctx.DEFAULT_TOTAL_MIN and e["est_min"] == 6 + budget
    assert (EXP / "exps" / "data" / "prompts_r2a.jsonl").exists()


def test_r2b_entries_fit_one_session_each():
    from infra import runpod
    from exps import r2b_multicol, r2b_spec
    for key, head, mod, budget in (("r2b_mc", "python3 -m exps.fetch_models --models pq2 --timeout-min 6 && python3 -m exps.r2b_multicol",
                                    "r2b_multicol", 22.0),
                                   ("r2b_spec", "python3 -m exps.fetch_models --models pq2,draft_bf16 --timeout-min 6 && python3 -m exps.r2b_spec",
                                    "r2b_spec", r2b_spec.DEFAULT_TOTAL_MIN)):
        e = MAN[key]
        assert e["build_llama"] is True and e["failsafe_min"] == 55 and e["gpus"][-1] == "pro6000"
        assert e["est_min"] + runpod.BOOTSTRAP_MIN + runpod.LLAMA_BUILD_MIN <= e["failsafe_min"] - 5
        assert e["cmd"].startswith(head)
        assert float(re.search(r"exps\.%s --budget-min (\d+)" % mod, e["cmd"]).group(1)) == budget and e["est_min"] == 6 + budget
    assert MAN["r2b_spec"]["gpus"] == ["pro6000"]                      # the anchor rows of R2a are PRO 6000 rows
    assert r2b_multicol.parse_args([]).budget_min == 22.0


def test_wrap_entries_fit_one_session_each():
    from infra import runpod
    from exps import wrap_a, wrap_a3, wrap_b
    fetch = "python3 -m exps.fetch_models --models pq2,draft_q8_0 --timeout-min 6 && "
    for key, head, mod, budget, failsafe in (
            ("wrap_a", fetch + "python3 -m exps.wrap_a", "wrap_a", wrap_a.DEFAULT_TOTAL_MIN, 40),
            ("wrap_b", fetch + "python3 -m exps.wrap_b", "wrap_b", wrap_b.DEFAULT_TOTAL_MIN, 55),
            ("wrap_a3", "python3 -m exps.wrap_a3", "wrap_a3", wrap_a3.DEFAULT_TOTAL_MIN, 55)):
        e = MAN[key]
        assert e["build_llama"] is True and e["failsafe_min"] == failsafe and e["cmd"].startswith(head)
        assert e["est_min"] + runpod.BOOTSTRAP_MIN + runpod.LLAMA_BUILD_MIN <= e["failsafe_min"] - 5
        got = float(re.search(r"exps\.%s --budget-min (\d+)" % mod, e["cmd"]).group(1))
        assert got == budget and e["est_min"] == budget + (6 if "fetch_models" in head else 0)
    # A1 and A2 must not land on a Max-Q; B is not PR evidence (Max-Q or 5090); A3 is the Ada check
    assert MAN["wrap_a"]["gpus"] == ["pro6000_600w"] and MAN["wrap_b"]["gpus"] == ["pro6000_maxq", "rtx5090"] and MAN["wrap_a3"]["gpus"] == ["rtx4090"]
    assert not any("Max-Q" in n for n in gpus.RUNPOD_CANDIDATES["pro6000_600w"])
    # wrap_a at the trimmed budget stays inside the 40 min limit that needs no --allow-long
    assert MAN["wrap_a"]["failsafe_min"] <= 40 and MAN["wrap_a"]["est_min"] == 18


def test_wrap_ab_runs_both_in_one_600w_session():
    from infra import runpod
    from exps import wrap_a, wrap_b
    e = MAN["wrap_ab"]
    assert e["gpus"] == ["pro6000_600w"] and e["result_exp"] == "wrap_a" and e["build_llama"] is True
    cmd = e["cmd"]
    assert cmd.index("exps.fetch_models") < cmd.index("exps.wrap_a --budget-min %d" % wrap_a.DEFAULT_TOTAL_MIN) < cmd.index("exps.wrap_b --budget-min %d" % wrap_b.DEFAULT_TOTAL_MIN)
    assert cmd.count("exps.fetch_models") == 1                                         # one download for both
    assert "; a=$?;" in cmd and "exit $((a ? a : b))" in cmd                           # wrap_b runs after a failing wrap_a, the exit code keeps the failure
    assert e["est_min"] == 6 + wrap_a.DEFAULT_TOTAL_MIN + wrap_b.DEFAULT_TOTAL_MIN
    assert e["est_min"] + runpod.BOOTSTRAP_MIN + runpod.LLAMA_BUILD_MIN <= e["failsafe_min"] - 5
    # one session shares the bootstrap and the build: cheaper than two 600 W sessions
    assert e["failsafe_min"] < MAN["wrap_a"]["failsafe_min"] + MAN["wrap_b"]["failsafe_min"]


def test_wrap_a_budget_gate_reserve_is_inside_the_stated_cost():
    from infra import runpod
    # the gate reserves (failsafe + 5) min at the price cap: wrap_a about $1.3 worst case, the 4090 check well under a dollar
    assert runpod.est_cost(runpod.DEFAULT_MAX_PRICE["pro6000_600w"], MAN["wrap_a"]["failsafe_min"]) < 1.35
    assert runpod.est_cost(runpod.DEFAULT_MAX_PRICE["rtx4090"], MAN["wrap_a3"]["failsafe_min"]) < 0.9
