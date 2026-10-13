from __future__ import annotations

import pathlib
import re
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
                   "llama-bench", "llama-server", "llama-cli", "llama-speculative-simple", "CR_BUILD_LLAMA",
                   "ccache", "/opt/nvidia/nsight-compute"):
        assert needle in s, needle
    # llama build must be opt-in
    assert re.search(r'\[ "\$\{CR_BUILD_LLAMA:-0\}" = "1" \]', s)


def test_requirements_pinned_content():
    s = (INFRA / "requirements-pod.txt").read_text()
    for needle in ("cuda-python>=12.8,<13", "cupy-cuda12x", "nvidia-ml-py", "pytest", "transformers",
                   "huggingface_hub[hf_transfer]"):
        assert needle in s
    assert not re.search(r"^kernels\b", s, re.M)


def test_pod_entry_hands_over_to_runner():
    s = (INFRA / "pod_entry.sh").read_text()
    assert "pod_runner.py" in s and "CR_EXP" in s and "CR_CMD" in s and "exec python3" in s


def test_llama_target_grep_pattern():
    pat = r"(^|[. ])llama-bench(:| |$)"
    for line in ("llama-bench: phony", ".. llama-bench", "ninja: llama-bench"):
        assert re.search(pat, line)
    assert not re.search(pat, "llama-bench-impl: phony")
