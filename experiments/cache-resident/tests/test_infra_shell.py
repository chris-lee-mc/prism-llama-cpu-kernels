from __future__ import annotations

import hashlib
import os
import pathlib
import re
import shutil
import stat
import subprocess

import pytest

INFRA = pathlib.Path(__file__).resolve().parent.parent / "infra"


@pytest.mark.parametrize("name", ["bootstrap.sh", "pod_entry.sh"])
def test_bash_syntax(name):
    r = subprocess.run(["bash", "-n", str(INFRA / name)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_bootstrap_pinned_content():
    s = (INFRA / "bootstrap.sh").read_text()
    for needle in ("https://download.pytorch.org/whl/cu128", "cuda-nsight-compute-12-8", "/usr/local/bin/ncu",
                   ".cr_bootstrap_ok", "-DGGML_CUDA=ON", "-DCMAKE_CUDA_ARCHITECTURES", "compute_cap", "bootstrap.log",
                   "llama-bench", "llama-server", "llama-cli", "llama-speculative-simple", "llama-perplexity", "llama-quantize", "test-backend-ops", "CR_BUILD_LLAMA",
                   "ccache", "/opt/nvidia/nsight-compute", "cuda-nsight-systems-12-8"):
        assert needle in s, needle
    # llama build must be opt-in
    assert re.search(r'\[ "\$\{CR_BUILD_LLAMA:-0\}" = "1" \]', s)


def test_bootstrap_installs_toolchain_and_sanity_checks():
    s = (INFRA / "bootstrap.sh").read_text()
    for needle in ("python3-dev", "build-essential", "Python.h", "g++", "load_inline", "torch.cuda.is_available()",
                   "import torch, triton"):
        assert needle in s, needle
    assert "sanity:" in s and "MISSING+=" in s


def test_requirements_pinned_content():
    s = (INFRA / "requirements-pod.txt").read_text()
    for needle in ("cuda-python>=12.8,<13", "cupy-cuda12x", "nvidia-ml-py", "pytest", "transformers==5.19.0",
                   "huggingface_hub[hf_transfer]"):
        assert needle in s
    assert not re.search(r"^kernels\b", s, re.M)
    assert "transformers: " in (INFRA / "bootstrap.sh").read_text()


def test_pod_entry_hands_over_to_runner():
    s = (INFRA / "pod_entry.sh").read_text()
    assert "pod_runner.py" in s and "CR_EXP" in s and "CR_CMD" in s and "exec python3" in s


def test_llama_target_grep_pattern():
    pat = r"(^|[. ])llama-bench(:| |$)"
    for line in ("llama-bench: phony", ".. llama-bench", "ninja: llama-bench"):
        assert re.search(pat, line)
    assert not re.search(pat, "llama-bench-impl: phony")


def _host_check(tmp_path, power, env_extra):
    """Run bootstrap.sh with a fake nvidia-smi (power.limit,power.default_limit = power) and a fake python3 (cuInit ok). A matching stamp makes
    the script stop after the host check, so nothing is installed. Returns (exit code, bootstrap.log text)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("nvidia-smi", "#!/bin/sh\necho '%s'\n" % power), ("python3", "#!/bin/sh\nexit 0\n")):
        f = bin_dir / name
        f.write_text(body)
        f.chmod(f.stat().st_mode | stat.S_IXUSR)
    script = INFRA / "bootstrap.sh"
    lines = "".join("%s  %s\n" % (hashlib.sha256(p.read_bytes()).hexdigest(), p) for p in (INFRA / "requirements-pod.txt", script))
    stamp = tmp_path / "stamp"
    stamp.write_text(hashlib.sha256(lines.encode()).hexdigest()[:16] + "-llama0\n")
    env = dict(os.environ, PATH="%s:%s" % (bin_dir, os.environ["PATH"]), CR_RESULTS_DIR=str(tmp_path / "res"), CR_BOOTSTRAP_STAMP=str(stamp), **env_extra)
    env.pop("CR_BUILD_LLAMA", None)
    r = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, timeout=60)
    return r.returncode, (tmp_path / "res" / "_status" / "bootstrap.log").read_text()


@pytest.mark.parametrize("power,ok", [("600.00, 600.00", True), ("598.00, 600.00", True), ("450.00, 600.00", False), ("[N/A], 600.00", False),
                                      ("660.00, 600.00", False)])
def test_bootstrap_power_tol_needs_the_default_limit(tmp_path, power, ok):
    code, log = _host_check(tmp_path, power, {"CR_POWER_TOL": "0.01"})
    if ok:
        assert code == 0 and "host check: power limit,default %s W ok" % power.replace(" ", "") in log and "stamp matches" in log
    else:
        assert code == 3 and "is not the default within 0.01 (or unreadable), stop" in log and "stamp matches" not in log


def test_bootstrap_power_tol_stops_without_nvidia_smi(tmp_path):
    (tmp_path / "bin").mkdir()
    env = dict(os.environ, PATH="%s:/usr/bin:/bin" % (tmp_path / "bin"), CR_RESULTS_DIR=str(tmp_path / "res"), CR_POWER_TOL="0.01",
               CR_BOOTSTRAP_STAMP=str(tmp_path / "stamp"))
    if shutil.which("nvidia-smi", path=env["PATH"]):
        pytest.skip("host has nvidia-smi")
    r = subprocess.run(["bash", str(INFRA / "bootstrap.sh")], env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 3 and "no nvidia-smi, power limit unreadable, stop" in (tmp_path / "res" / "_status" / "bootstrap.log").read_text()


def test_bootstrap_min_power_frac_only_stops_a_capped_host(tmp_path):
    for sub, power, env, want in (("a", "500.00, 600.00", {"CR_MIN_POWER_FRAC": "0.75"}, 0), ("b", "300.00, 600.00", {"CR_MIN_POWER_FRAC": "0.75"}, 3),
                                  ("c", "300.00, 600.00", {}, 0)):                              # c: no gate set, no check
        (tmp_path / sub).mkdir()
        code, log = _host_check(tmp_path / sub, power, env)
        assert code == want and (want == 0 or "capped host" in log), (sub, code, log)
