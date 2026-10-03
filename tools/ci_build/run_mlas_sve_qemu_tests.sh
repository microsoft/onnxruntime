#!/bin/bash
# Runs the MLAS unit-test subset under QEMU user-mode emulation with SVE enabled.
#
# Background: build.py enables onnxruntime_USE_SVE unless --no_sve, so SVE kernels
# are compiled everywhere, but every test suite that touches them skips when
# MLAS_CPUIDINFO::GetCPUIDInfo().HasArmSve() is false -- which is the case on all
# CI runners (no SVE hardware). This gap let the bugs fixed in #33040 ship.
# Running the tests under `qemu-aarch64 -cpu max` exposes SVE, so the SVE code
# paths are actually executed. See https://github.com/microsoft/onnxruntime/issues/33069.
#
# Usage: run_mlas_sve_qemu_tests.sh <build-dir>
#   <build-dir>: directory containing the onnxruntime_mlas_test binary
#
# The script sweeps SVE vector lengths (128/256/512/1024/2048 bits) because both
# bugs fixed in #33040 were lane-position dependent. The vector length is set via
# prctl(PR_SVE_SET_VL) from a small Python wrapper that itself runs under QEMU,
# so the prctl is emulated for the guest and sets the vCPU's vector length before
# exec'ing the test binary (the setting survives execve).
#
# Env overrides (mainly for local testing):
#   GTEST_FILTER: gtest filter to use (default: the merged CI filter from #33054)
#   SVE_VLS: space-separated SVE vector lengths in bits to sweep (default: 128 256 512 1024 2048)

set -euo pipefail

BUILD_DIR="${1:?Usage: $0 <build-dir>}"
TEST_BIN="${BUILD_DIR}/onnxruntime_mlas_test"
GTEST_FILTER="${GTEST_FILTER:-*FP16*:*Fp16*:Exp.*:Softmax*:Activation*}"
SVE_VLS="${SVE_VLS:-128 256 512 1024 2048}"

if [[ ! -x "${TEST_BIN}" ]]; then
  echo "ERROR: test binary not found or not executable: ${TEST_BIN}" >&2
  exit 1
fi

# qemu-user provides qemu-aarch64. The CI image is AlmaLinux-based (dnf);
# fall back to apt-get for Debian/Ubuntu environments.
if ! command -v qemu-aarch64 >/dev/null 2>&1; then
  echo "Installing qemu-user..."
  if command -v dnf >/dev/null 2>&1; then
    dnf install -y qemu-user
  elif command -v apt-get >/dev/null 2>&1; then
    apt-get update && apt-get install -y qemu-user
  else
    echo "ERROR: no supported package manager (dnf/apt-get) to install qemu-user" >&2
    exit 1
  fi
fi
if ! command -v qemu-aarch64 >/dev/null 2>&1; then
  echo "ERROR: qemu-aarch64 still not found after install attempt" >&2
  exit 1
fi

# Small helper (written to a temp file below): set the SVE vector length for the
# emulated CPU via prctl, verify it took effect, then exec the test binary.
# IMPORTANT: this script must itself run under `qemu-aarch64` so that the prctl
# is emulated for the guest; a host-side prctl would fail on machines without
# SVE hardware (which is the whole reason we are emulating).
read -r -d '' VL_WRAPPER_PY <<'PYEOF' || true
import ctypes
import os
import sys

PR_SVE_SET_VL = 50
PR_SVE_GET_VL = 51
PR_SVE_VL_INHERIT = 1 << 17

libc = ctypes.CDLL("libc.so.6", use_errno=True)
vl_bits = int(sys.argv[1])
if libc.prctl(PR_SVE_SET_VL, PR_SVE_VL_INHERIT | (vl_bits // 8)) == -1:
    raise OSError(ctypes.get_errno(), "prctl(PR_SVE_SET_VL) failed for %d bits" % vl_bits)
actual = libc.prctl(PR_SVE_GET_VL, 0)
print("SVE vector length: %d bits (requested %d)" % (actual & 0xFFFF, vl_bits), flush=True)
os.execv(sys.argv[2], sys.argv[2:])
PYEOF

WRAPPER_PY="$(mktemp /tmp/sve_vl_wrapper_XXXXXX.py)"
printf '%s\n' "${VL_WRAPPER_PY}" > "${WRAPPER_PY}"
trap 'rm -f "${WRAPPER_PY}"' EXIT

PYTHON3="$(command -v python3)"
if [[ -z "${PYTHON3}" ]]; then
  echo "ERROR: python3 not found (needed for the prctl wrapper)" >&2
  exit 1
fi

echo "Running MLAS tests under qemu-aarch64 -cpu max (SVE enabled)"
echo "Test binary: ${TEST_BIN}"
echo "GTest filter: ${GTEST_FILTER}"

FAILED=0
for VL in ${SVE_VLS}; do
  echo "=== SVE vector length: ${VL} bits ==="
  if ! qemu-aarch64 -cpu max "${PYTHON3}" "${WRAPPER_PY}" "${VL}" \
      "${TEST_BIN}" --gtest_filter="${GTEST_FILTER}"; then
    echo "ERROR: MLAS tests FAILED at SVE VL=${VL} bits" >&2
    FAILED=1
  fi
done

if [[ "${FAILED}" -ne 0 ]]; then
  echo "ERROR: MLAS SVE QEMU tests failed for at least one vector length" >&2
  exit 1
fi
echo "All MLAS SVE QEMU tests passed."
