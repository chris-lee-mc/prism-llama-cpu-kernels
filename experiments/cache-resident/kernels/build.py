"""Build the cr_ternary_ext torch extension (needs nvcc and a CUDA device for arch detection)."""
from __future__ import annotations

import os
import pathlib

CSRC = pathlib.Path(__file__).resolve().parent / "csrc"
SOURCES = ["binding.cpp", "ternary_gemv.cu", "toy_ops.cu"]
NAME = "cr_ternary_ext"
_EXT = None


def nvcc_flags(cc: tuple[int, int], fp4_mma: bool = False) -> list[str]:
    """nvcc flags for one compute capability. CR_FP4_MMA=1 builds sm_120a and defines CR_FP4_MMA."""
    sm = f"{cc[0]}{cc[1]}"
    if fp4_mma and sm == "120":
        sm = "120a"
    flags = ["-O3", "-std=c++17", "-lineinfo", f"-gencode=arch=compute_{sm},code=sm_{sm}", f"-gencode=arch=compute_{sm},code=compute_{sm}"]
    if fp4_mma:
        flags.append("-DCR_FP4_MMA")
    extra = os.environ.get("CR_NVCC_EXTRA", "").split()
    return flags + extra


def load_ext(verbose: bool = False):
    """Compile (cached by torch) and import the extension for the current device."""
    global _EXT
    if _EXT is not None:
        return _EXT
    import torch
    from torch.utils.cpp_extension import load

    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device: cannot build cr_ternary_ext")
    cc = torch.cuda.get_device_capability()
    fp4_mma = os.environ.get("CR_FP4_MMA", "0") == "1"
    flags = nvcc_flags(cc, fp4_mma)
    _EXT = load(
        name=NAME + ("_mma" if fp4_mma else ""),
        sources=[str(CSRC / s) for s in SOURCES],
        extra_include_paths=[str(CSRC)],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=flags,
        verbose=verbose,
    )
    return _EXT
