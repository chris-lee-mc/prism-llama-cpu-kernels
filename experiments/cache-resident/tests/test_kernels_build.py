"""Offline compile checks: nvcc for sm_86 / sm_89 / sm_120 and g++ syntax of the pybind glue."""
from __future__ import annotations

import pathlib
import re
import subprocess
import sys

import pytest

from conftest import NVCC

EXP_DIR = pathlib.Path(__file__).resolve().parent.parent
CSRC = EXP_DIR / "kernels" / "csrc"


@pytest.mark.nvcc
@pytest.mark.parametrize("src", ["ternary_gemv.cu", "toy_ops.cu"])
@pytest.mark.parametrize("sm", [86, 89, 120])
def test_nvcc_compiles_without_spills(src, sm, tmp_path):
    cmd = [NVCC, "-std=c++17", "-O3", "-c", f"-gencode=arch=compute_{sm},code=sm_{sm}", "-Xptxas", "-v", "-I", str(CSRC), str(CSRC / src), "-o", str(tmp_path / "o.o")]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "warning" not in r.stderr.lower(), r.stderr[-3000:]
    spills = re.findall(r"(\d+) bytes spill stores, (\d+) bytes spill loads", r.stderr)
    assert spills, "ptxas -v output missing"
    assert all(a == "0" and b == "0" for a, b in spills), spills
    names = re.findall(r"Compiling entry function '([^']*)'", r.stderr)
    want = ["cr_gemv_lut", "cr_gemv_fp4"] if src == "ternary_gemv.cu" else ["cr_gdn_step", "cr_attn_decode", "cr_rmsnorm", "cr_silu_mul", "cr_add"]
    for w in want:
        assert any(w in n for n in names), (w, names)


@pytest.mark.nvcc
def test_launchers_are_exported_with_c_linkage(tmp_path):
    for src in ("ternary_gemv.cu", "toy_ops.cu"):
        obj = tmp_path / (src + ".o")
        subprocess.run([NVCC, "-std=c++17", "-c", "-gencode=arch=compute_86,code=sm_86", "-I", str(CSRC), str(CSRC / src), "-o", str(obj)], check=True, capture_output=True)
        syms = subprocess.run(["nm", str(obj)], capture_output=True, text=True).stdout
        want = ["cr_gemv_lut", "cr_gemv_fp4"] if src == "ternary_gemv.cu" else ["cr_gdn_step", "cr_attn_decode", "cr_rmsnorm", "cr_silu_mul", "cr_add"]
        for w in want:
            assert re.search(rf" T {w}$", syms, re.M), (w, syms[:500])


@pytest.mark.nvcc
def test_binding_syntax_with_torch_headers():
    import shutil
    import sysconfig

    if not shutil.which("g++"):
        pytest.skip("no g++")
    from torch.utils.cpp_extension import include_paths

    inc = [f"-I{p}" for p in include_paths()] + [f"-I{sysconfig.get_paths()['include']}", "-I/usr/local/cuda-12.8/include", f"-I{CSRC}"]
    r = subprocess.run(["g++", "-std=c++17", "-fsyntax-only", "-DTORCH_EXTENSION_NAME=cr_ternary_ext", *inc, str(CSRC / "binding.cpp")], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-3000:]


@pytest.mark.nvcc
@pytest.mark.parametrize("cc", [(8, 6), (8, 9), (12, 0)])
def test_build_py_flags_compile(cc, tmp_path):
    from kernels.build import nvcc_flags

    for src in ("ternary_gemv.cu", "toy_ops.cu"):
        cmd = [NVCC, "-c", *nvcc_flags(cc), "-I", str(CSRC), str(CSRC / src), "-o", str(tmp_path / (src + ".o"))]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        assert r.returncode == 0, r.stderr[-3000:]
