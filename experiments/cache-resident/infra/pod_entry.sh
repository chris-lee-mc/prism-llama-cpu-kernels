#!/usr/bin/env bash
# Pod entry: bootstrap, then hand over to the runner. Env: CR_EXP (required), CR_CMD (optional).
set -uo pipefail
INFRA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(dirname "$INFRA_DIR")"
cd "$EXP_DIR" || exit 1
: "${CR_EXP:?CR_EXP is required}"
export CR_RESULTS_DIR="${CR_RESULTS_DIR:-$EXP_DIR/results}"
mkdir -p "$CR_RESULTS_DIR/_status"
# Runner starts first (serves results, tracks phases); it runs bootstrap itself.
ARGS=(--exp "$CR_EXP" --bootstrap "$INFRA_DIR/bootstrap.sh")
if [ -n "${CR_CMD:-}" ]; then ARGS+=(--cmd "$CR_CMD"); fi
exec python3 "$INFRA_DIR/pod_runner.py" "${ARGS[@]}"
