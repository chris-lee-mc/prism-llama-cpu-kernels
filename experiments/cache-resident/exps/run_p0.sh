#!/usr/bin/env bash
# Phase 0 smoke on a pod: persisting-L2 check, GEMV bench, toy hybrid, then the fast gpu tests.
# Runs every step even if one fails; exit code is nonzero when any step failed.
# Results of each step are mirrored under results/p0_smoke/ right after the step, so a late kill loses nothing.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
export CR_RESULTS_DIR="${CR_RESULTS_DIR:-$PWD/results}"
GPU="${CR_GPU_TAG:-$(python3 -c 'from infra.results import gpu_tag; print(gpu_tag())' 2>/dev/null || echo cpu)}"
OUT="$CR_RESULTS_DIR/p0_smoke/$GPU"
mkdir -p "$OUT/logs"
rc=0
step() {
    local name="$1"; shift
    echo "=== $name: $*"
    "$@" 2>&1 | tee "$OUT/logs/$name.log"
    local s="${PIPESTATUS[0]}"
    echo "=== $name exit $s"
    [ "$s" -eq 0 ] || rc=1
    if [ -d "$CR_RESULTS_DIR/$name" ]; then
        mkdir -p "$CR_RESULTS_DIR/p0_smoke/$name"
        cp -a "$CR_RESULTS_DIR/$name/." "$CR_RESULTS_DIR/p0_smoke/$name/"
    fi
}
step p0_l2 python3 -m harness.p0_l2_check
step p0_gemv python3 -m kernels.bench_gemv --result-exp p0_gemv
step p0_toy python3 -m kernels.toy_hybrid --result-exp p0_toy
# slow tests (model download, torch.compile, ncu sweeps) have their own manifest entries
step pytest_gpu timeout 600 python3 -m pytest -q -m "gpu and not slow" tests
exit "$rc"
