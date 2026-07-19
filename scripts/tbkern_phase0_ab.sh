#!/usr/bin/env bash
# Phase 0 interleaved A/B runner for the real Q2_0 model.
#
# The runner intentionally writes raw command output and a manifest.  A later
# analysis step should parse the llama-bench rows and report medians/MAD (or a
# confidence interval); this script does not claim statistical significance.
# Existing output is never removed: each invocation gets a UTC run directory.
#
# Required: MODEL=/path/to/model.gguf
# Optional: BUILD, OUT, PPL_FILE, THREADS, RUNS, PROMPT, N_PREDICT,
#           BENCH_PROMPT, BENCH_GEN, BATCH, UBATCH, PPL_CONTEXT, PPL_BATCH,
#           PPL_CHUNKS, PYTHON

set -euo pipefail
IFS=$'\n\t'

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODEL=${MODEL:?set MODEL to the real Q2_0 GGUF path}
BUILD=${BUILD:-"$ROOT/build"}
OUT_ROOT=${OUT:-"$ROOT/prism-bench-results/phase0"}
THREADS=${THREADS:-8}
RUNS=${RUNS:-5}
PROMPT=${PROMPT:-Hello}
N_PREDICT=${N_PREDICT:-16}
BENCH_PROMPT=${BENCH_PROMPT:-0}
BENCH_GEN=${BENCH_GEN:-16}
BATCH=${BATCH:-32}
UBATCH=${UBATCH:-32}
PPL_CONTEXT=${PPL_CONTEXT:-32}
PPL_BATCH=${PPL_BATCH:-32}
PPL_CHUNKS=${PPL_CHUNKS:-1}
PYTHON=${PYTHON:-python3}

[[ -f "$MODEL" ]] || { echo "MODEL does not exist: $MODEL" >&2; exit 2; }
DEBUG="$BUILD/bin/llama-debug"
BENCH="$BUILD/bin/llama-bench"
PPL="$BUILD/bin/llama-perplexity"
for executable in "$DEBUG" "$BENCH"; do
    [[ -x "$executable" ]] || { echo "missing executable: $executable" >&2; exit 2; }
done
if [[ -n "${PPL_FILE:-}" ]]; then
    [[ -f "$PPL_FILE" ]] || { echo "PPL_FILE does not exist: $PPL_FILE" >&2; exit 2; }
    [[ -x "$PPL" ]] || { echo "missing executable: $PPL" >&2; exit 2; }
fi

RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
OUT="$OUT_ROOT/$RUN_ID"
mkdir -p "$OUT/parity" "$OUT/bench" "$OUT/ppl"

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
run_logged() {
    local logfile=$1
    shift
    {
        printf '$'
        printf ' %q' "$@"
        printf '\n'
        "$@"
    } 2>&1 | tee "$logfile"
}

# Keep native Prism deterministic even when the caller has a selector set.
native_env=(env
    -u GGML_TBKERN_Q2_0
    -u GGML_TBKERN_Q2_0_AVX2
    -u GGML_TBKERN_Q2_0_VNNI
    -u GGML_TBKERN_Q2_0_VNNI64
    -u GGML_TBKERN_Q2_0_VNNI64_4R
    -u GGML_TBKERN_Q2_0_VNNI64_NATIVE
    -u GGML_TBKERN_Q2_0_VNNI64_NATIVE_4R
    -u GGML_TBKERN_Q2_0_VNNI64_NATIVE_VBMI)
p9_env=("${native_env[@]}" GGML_TBKERN_Q2_0_VNNI64_4R=1)
p13_env=("${native_env[@]}"
    GGML_TBKERN_Q2_0_VNNI64_NATIVE_4R=1
    GGML_TBKERN_Q2_0_VNNI64_NATIVE_VBMI=1)

env_for() {
    case "$1" in
        native) printf '%s\0' "${native_env[@]}" ;;
        p9) printf '%s\0' "${p9_env[@]}" ;;
        p13) printf '%s\0' "${p13_env[@]}" ;;
        *) echo "unknown variant: $1" >&2; return 2 ;;
    esac
}

variant_command() {
    local variant=$1
    local output_dir=$2
    local logfile=$3
    local -a e
    mapfile -d '' -t e < <(env_for "$variant")
    run_logged "$logfile" "${e[@]}" "$DEBUG" -m "$MODEL" -p "$PROMPT" \
        -t "$THREADS" -n "$N_PREDICT" --save-logits \
        --logits-output-dir "$output_dir"
}

find_artifact() {
    local directory=$1
    local pattern=$2
    local filter=${3:-}
    local -a files
    if [[ -n "$filter" ]]; then
        mapfile -t files < <(find "$directory" -maxdepth 1 -type f -name "$pattern" -print | grep -v "$filter" | sort)
    else
        mapfile -t files < <(find "$directory" -maxdepth 1 -type f -name "$pattern" -print | sort)
    fi
    [[ ${#files[@]} -eq 1 ]] || {
        echo "expected one $pattern artifact in $directory, found ${#files[@]}" >&2
        printf '  %s\n' "${files[@]}" >&2
        return 2
    }
    printf '%s\n' "${files[0]}"
}

log "root=$ROOT model=$MODEL build=$BUILD output=$OUT"
{
    printf 'schema=tbkern.phase0.ab.v1\n'
    printf 'utc_start=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'root=%s\nmodel=%s\nbuild=%s\nrun_dir=%s\n' "$ROOT" "$MODEL" "$BUILD" "$OUT"
    printf 'model_size_bytes=%s\n' "$(stat -c %s "$MODEL")"
    printf 'model_sha256='; sha256sum "$MODEL" | awk '{print $1}'
    if git -C "$ROOT" rev-parse HEAD >/dev/null 2>&1; then
        printf 'git_head='; git -C "$ROOT" rev-parse HEAD
        printf 'git_dirty='; if git -C "$ROOT" diff --quiet; then echo false; else echo true; fi
    else
        printf 'git_head=unavailable\n'
        printf 'git_dirty=unavailable\n'
    fi
    printf 'uname='; uname -a
    printf 'nproc='; nproc
    printf 'cpu_model='; awk -F: '/model name/{gsub(/^ +/,"",$2); print $2; exit}' /proc/cpuinfo
    printf 'cpu_flags='; awk -F: '/flags/{gsub(/^ +/,"",$2); print $2; exit}' /proc/cpuinfo
    printf 'memory='; free -h | tr '\n' ';'; printf '\n'
    for executable in "$DEBUG" "$BENCH" "$PPL"; do
        [[ -x "$executable" ]] || continue
        printf 'sha256_%s=' "$(basename "$executable")"
        sha256sum "$executable" | awk '{print $1}'
    done
    printf 'threads=%s\nruns=%s\nprompt=%s\nn_predict=%s\n' "$THREADS" "$RUNS" "$PROMPT" "$N_PREDICT"
    printf 'bench_prompt=%s\nbench_gen=%s\nbatch=%s\nubatch=%s\n' "$BENCH_PROMPT" "$BENCH_GEN" "$BATCH" "$UBATCH"
    printf 'ppl_file=%s\nppl_context=%s\nppl_batch=%s\nppl_chunks=%s\n' "${PPL_FILE:-}" "$PPL_CONTEXT" "$PPL_BATCH" "$PPL_CHUNKS"
} > "$OUT/manifest.txt"

declare -A logits tokens
for variant in native p9 p13; do
    mkdir -p "$OUT/parity/$variant"
    log "parity variant=$variant"
    variant_command "$variant" "$OUT/parity/$variant" "$OUT/parity/$variant/run.log"
    logits[$variant]=$(find_artifact "$OUT/parity/$variant" '*.bin' 'tokens')
    tokens[$variant]=$(find_artifact "$OUT/parity/$variant" '*tokens*.bin')
done

for variant in p9 p13; do
    log "parity compare native vs $variant"
    run_logged "$OUT/parity/native-vs-$variant.log" "$PYTHON" "$ROOT/scripts/tbkern_compare.py" \
        --native-logits "${logits[native]}" --tbkern-logits "${logits[$variant]}" \
        --native-tokens "${tokens[native]}" --tbkern-tokens "${tokens[$variant]}" \
        --json-out "$OUT/parity/native-vs-$variant.json"
done

bench_variant() {
    local variant=$1 round=$2 logfile=$3
    local -a e
    mapfile -d '' -t e < <(env_for "$variant")
    run_logged "$logfile" "${e[@]}" "$BENCH" -m "$MODEL" \
        -p "$BENCH_PROMPT" -n "$BENCH_GEN" -t "$THREADS" -ngl 0 \
        -b "$BATCH" -ub "$UBATCH" -r 1
}

for ((round = 1; round <= RUNS; round++)); do
    if (( round % 2 )); then order=(native p9 p13); else order=(p13 p9 native); fi
    for variant in "${order[@]}"; do
        log "bench round=$round variant=$variant"
        bench_variant "$variant" "$round" "$OUT/bench/r${round}-${variant}.log"
    done
done

if [[ -n "${PPL_FILE:-}" ]]; then
    for variant in native p9 p13; do
        log "ppl variant=$variant"
        local_env=()
        mapfile -d '' -t local_env < <(env_for "$variant")
        run_logged "$OUT/ppl/$variant.log" "${local_env[@]}" "$PPL" -m "$MODEL" \
            -f "$PPL_FILE" -t "$THREADS" -c "$PPL_CONTEXT" -b "$PPL_BATCH" \
            --chunks "$PPL_CHUNKS"
    done
fi

printf 'utc_end=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/manifest.txt"
log "completed raw artifacts in $OUT"
