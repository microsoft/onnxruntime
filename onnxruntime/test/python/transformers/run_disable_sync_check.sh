#!/usr/bin/env bash
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
#
# End-to-end check that disabling EP synchronization
# (disable_synchronize_execution_providers=1, PR #28686) removes host-side
# stream/device synchronization from the CUDA EP Run() path.
#
# It profiles profile_disable_sync.py with nsys for both sync=off and sync=on,
# then uses parse_nsys.py to count host synchronization CUDA APIs that occur
# inside the per-run NVTX range.
#
# Expectation:
#   sync=off  -> PASS (0 sync APIs inside the run range)
#   sync=on   -> sync APIs ARE present inside the run range (control/baseline)
#
# This is a developer verification script, not a CI unit test. Requires a CUDA
# build of onnxruntime, a GPU, the `nsys` profiler, and the `nvtx` python pkg.
#
# Usage:
#   ./run_disable_sync_check.sh [python_executable]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${1:-python}"
NVTX_RANGE="ort_run"
WARMUP=5
REPEAT=50
OUTDIR="${OUTDIR:-$(mktemp -d)}"

echo "Output directory: ${OUTDIR}"
echo "Python: ${PY}"

profile_one() {
    local mode="$1"   # off | on
    local base="${OUTDIR}/disable_sync_${mode}"
    echo
    echo "=== Profiling sync=${mode} ==="
    nsys profile -t cuda,nvtx --force-overwrite true -o "${base}" --export=sqlite \
        "${PY}" "${SCRIPT_DIR}/profile_disable_sync.py" \
        --sync "${mode}" --nvtx-range "${NVTX_RANGE}" \
        --warmup "${WARMUP}" --repeat "${REPEAT}" \
        --model "${OUTDIR}/disable_sync_test.onnx" >/dev/null
    echo "--- Host synchronization CUDA APIs inside NVTX range '${NVTX_RANGE}' (sync=${mode}) ---"
    # --skip-first-ranges skips the warmup occurrences of the NVTX range.
    "${PY}" "${SCRIPT_DIR}/parse_nsys.py" "${base}.sqlite" \
        --cuda-api --sync-apis-only --nvtx-range "${NVTX_RANGE}" \
        --skip-first-ranges "${WARMUP}"
}

rc_off=0
profile_one off || rc_off=$?

echo
echo "=== Baseline (control): host synchronization CUDA APIs in range for sync=on ==="
rc_on=0
profile_one on || rc_on=$?

echo
echo "================ SUMMARY ================"
if [[ ${rc_off} -eq 0 ]]; then
    echo "sync=off : PASS (no host synchronization CUDA APIs inside the run)"
else
    echo "sync=off : FAIL (synchronization detected inside the run)"
fi
if [[ ${rc_on} -ne 0 ]]; then
    echo "sync=on  : sync APIs present inside the run (expected baseline behavior)"
else
    echo "sync=on  : no sync APIs detected (unexpected; check profiling capture)"
fi

exit ${rc_off}
