#!/usr/bin/env bash
# P0.1 pod bootstrap. Idempotent. Works on RunPod (root) and on plain CUDA hosts (non-root: no apt).
set -uo pipefail

INFRA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(dirname "$INFRA_DIR")"
REPO_DIR="$(cd "$EXP_DIR/../.." && pwd)"
RESULTS_DIR="${CR_RESULTS_DIR:-$EXP_DIR/results}"
mkdir -p "$RESULTS_DIR/_status"
LOG="$RESULTS_DIR/_status/bootstrap.log"
exec > >(tee -a "$LOG") 2>&1

if [ "$(id -u)" = "0" ]; then IS_ROOT=1; STAMP_DIR=/root; else IS_ROOT=0; STAMP_DIR="$HOME"; fi
STAMP="${CR_BOOTSTRAP_STAMP:-$STAMP_DIR/.cr_bootstrap_ok}"
REQ="$INFRA_DIR/requirements-pod.txt"
REQ_HASH="$(sha256sum "$REQ" "$0" | sha256sum | cut -c1-16)-llama${CR_BUILD_LLAMA:-0}"
MISSING=()
T0=$(date +%s)
say() { echo "[bootstrap $(date -u +%H:%M:%S)] $*"; }

# Persistent caches when a volume exists (cache-friendly across pod restarts)
if [ -d /workspace ]; then
  export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/.cache/pip}"
  export CCACHE_DIR="${CCACHE_DIR:-/workspace/.ccache}"
  export HF_HOME="${HF_HOME:-/workspace/hf}"
  mkdir -p "$PIP_CACHE_DIR" "$CCACHE_DIR" "$HF_HOME" 2>/dev/null || true
fi
export DEBIAN_FRONTEND=noninteractive PIP_BREAK_SYSTEM_PACKAGES=1 PIP_DISABLE_PIP_VERSION_CHECK=1

if [ -f "$STAMP" ] && [ "$(cat "$STAMP" 2>/dev/null)" = "$REQ_HASH" ]; then
  say "stamp matches, nothing to do"
  exit 0
fi

# 1) apt packages
# python3-dev and build-essential are only Recommends of python3-pip: torch cpp_extension, triton and cmake need them
APT_PKGS=(git curl cmake ninja-build ccache python3 python3-pip python3-venv python3-dev build-essential)
have_pyh() {
  command -v python3 >/dev/null 2>&1 || return 1
  local inc
  inc="$(python3 -c 'import sysconfig;print(sysconfig.get_paths()["include"])' 2>/dev/null)" || return 1
  [ -f "$inc/Python.h" ]
}
need_apt=()
for p in "${APT_PKGS[@]}"; do
  case "$p" in
    ninja-build) c=ninja ;; python3-pip) c=pip3 ;; python3-venv) c="" ;; build-essential) c=g++ ;;
    python3-dev) c=""; have_pyh || need_apt+=("$p") ;;
    *) c="$p" ;;
  esac
  if [ -n "$c" ] && ! command -v "$c" >/dev/null 2>&1; then need_apt+=("$p"); fi
done
if [ "${#need_apt[@]}" -gt 0 ]; then
  if [ "$IS_ROOT" = "1" ] && command -v apt-get >/dev/null 2>&1; then
    say "apt install: ${need_apt[*]}"
    apt-get update -qq && apt-get install -y -qq --no-install-recommends "${need_apt[@]}" || MISSING+=("apt:${need_apt[*]}")
  else
    say "not root or no apt: skipping apt; missing: ${need_apt[*]}"
    MISSING+=("apt:${need_apt[*]}")
  fi
fi
PY=python3
command -v "$PY" >/dev/null 2>&1 || { say "python3 not found"; exit 1; }

# 2) torch (cu128). Skip if a CUDA >= 12.8 build is already importable.
if "$PY" - <<'PYEOF' 2>/dev/null
import sys, torch
v = torch.version.cuda or "0.0"
a, b = (int(x) for x in v.split(".")[:2])
sys.exit(0 if (a, b) >= (12, 8) else 1)
PYEOF
then
  say "torch with CUDA >= 12.8 already present"
else
  say "installing torch (cu128)"
  "$PY" -m pip install torch --index-url https://download.pytorch.org/whl/cu128 || MISSING+=("pip:torch")
fi

# 3) the rest of the python deps
say "pip install -r requirements-pod.txt"
"$PY" -m pip install -r "$REQ" || MISSING+=("pip:requirements-pod.txt")
say "transformers: $("$PY" -c 'import transformers; print(transformers.__version__)' 2>/dev/null || echo missing)"
"$PY" -c "import triton" 2>/dev/null || "$PY" -m pip install triton || MISSING+=("pip:triton")

# 3b) sanity: torch + CUDA + triton import, and a tiny C++ extension build (needs Python.h and g++)
if ! "$PY" -c 'import torch, triton; assert torch.cuda.is_available()' >/dev/null 2>&1; then
  say "sanity: torch/triton import or cuda.is_available failed"
  MISSING+=("sanity:torch-triton-cuda")
fi
if ! TORCH_EXTENSIONS_DIR="$(mktemp -d)" "$PY" - >/dev/null 2>&1 <<'PYEOF'
from torch.utils.cpp_extension import load_inline
m = load_inline("cr_probe", cpp_sources="int cr_probe() { return 7; }", functions=["cr_probe"])
assert m.cr_probe() == 7
PYEOF
then
  say "sanity: cpp_extension.load_inline failed (Python.h or g++ missing?)"
  MISSING+=("sanity:cpp-extension")
fi

# 4) Nsight Compute
find_ncu() {
  command -v ncu 2>/dev/null && return 0
  local c
  for c in /opt/nvidia/nsight-compute/*/ncu /usr/local/cuda*/bin/ncu /usr/local/cuda/nsight-compute*/ncu; do
    [ -x "$c" ] && { echo "$c"; return 0; }
  done
  return 1
}
NCU="$(find_ncu || true)"
if [ -z "$NCU" ] && [ "$IS_ROOT" = "1" ] && command -v apt-get >/dev/null 2>&1; then
  say "apt install cuda-nsight-compute-12-8"
  apt-get install -y -qq --no-install-recommends cuda-nsight-compute-12-8 || say "ncu apt install failed"
  NCU="$(find_ncu || true)"
fi
if [ -n "$NCU" ]; then
  say "ncu: $NCU"
  NCU_LINK="${CR_NCU_LINK:-/usr/local/bin/ncu}"
  if [ "$IS_ROOT" = "1" ] && [ ! -e "$NCU_LINK" ]; then ln -sf "$NCU" "$NCU_LINK"; fi
else
  say "ncu not found"
  MISSING+=("ncu")
fi

# 5) optional llama.cpp build
if [ "${CR_BUILD_LLAMA:-0}" = "1" ]; then
  CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d . | tr -d ' ')"
  if [ -z "$CC" ]; then
    say "no GPU compute cap, skipping llama.cpp build"
    MISSING+=("llama-build:no-gpu")
  else
    say "building llama.cpp for sm_$CC"
    BUILD="$REPO_DIR/build-cuda"
    cmake -S "$REPO_DIR" -B "$BUILD" -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$CC" \
      -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
      -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache || MISSING+=("llama-cmake")
    TARGETS=()
    for t in llama-bench llama-cli llama-server llama-speculative-simple llama-perplexity llama-quantize test-backend-ops; do
      if cmake --build "$BUILD" --target help 2>/dev/null | grep -Eq "(^|[. ])$t(:| |\$)"; then TARGETS+=("$t"); else say "target $t not found"; fi
    done
    if [ "${#TARGETS[@]}" -gt 0 ]; then
      cmake --build "$BUILD" -j "$(nproc)" --target "${TARGETS[@]}" || MISSING+=("llama-build")
    fi
  fi
fi

if [ "${#MISSING[@]}" -gt 0 ]; then
  say "DONE with missing: ${MISSING[*]} (no stamp written)"
  echo "$(( $(date +%s) - T0 ))s" >> "$LOG"
  exit 0
fi
echo "$REQ_HASH" > "$STAMP" 2>/dev/null || say "cannot write stamp $STAMP"
say "OK in $(( $(date +%s) - T0 ))s"
