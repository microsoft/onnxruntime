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
# QEMU's `sve-default-vector-length` CPU property (in bytes), which sets the
# emulated vCPU's default VL for the test process from startup.
#
# NOTE: an earlier revision set the VL via prctl() from a Python wrapper and then
# os.execv()'d the test binary. That is broken: in QEMU user-mode the guest
# execve is passed straight to the host kernel ("at the point of execve the
# process leaves QEMU's control" -- linux-user/syscall.c), so the test binary
# ran natively on the host, the emulated SVE state was lost, and the SVE tests
# silently skipped. Never exec from inside the emulated process; pass the VL
# to QEMU directly instead.
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

# qemu-aarch64 provides the user-mode emulation. Any QEMU >= 6.1 works (the
# sve-default-vector-length CPU property used below dates to 2021).
if ! command -v qemu-aarch64 >/dev/null 2>&1; then
  echo "Installing qemu-aarch64..."
  if command -v dnf >/dev/null 2>&1; then
    # AlmaLinux/RHEL/Fedora. Note: qemu-user is not in the base RHEL-clone
    # repos, so also try EPEL before giving up on distro packages.
    dnf install -y qemu-user || \
      (dnf install -y epel-release && dnf install -y qemu-user) || true
  elif command -v apt-get >/dev/null 2>&1; then
    # Debian/Ubuntu (also handy for local testing).
    apt-get update && apt-get install -y qemu-user || true
  fi
  if ! command -v qemu-aarch64 >/dev/null 2>&1; then
    # Last resort: statically-linked qemu-aarch64 extracted from Debian's
    # qemu-user-static package (arm64 build). The version is resolved
    # dynamically from the Packages index so this doesn't rot.
    echo "Distro packages unavailable; falling back to static qemu-aarch64 from Debian..."
    QEMU_TMP="$(mktemp -d)"
    df -h "${QEMU_TMP}"  # debug: check disk space
    QEMU_DEB_FILENAME="$(curl -fsSL --retry 3 http://ftp.debian.org/debian/dists/stable/main/binary-arm64/Packages.gz \
      | gzip -dc | awk '/^Package: qemu-user-static$/{found=1} found && /^Filename: /{print $2; exit}')"
    echo "Deb filename: ${QEMU_DEB_FILENAME}"  # debug
    if [[ -n "${QEMU_DEB_FILENAME}" ]]; then
      for i in 1 2 3; do
        if curl -fsSL --retry 3 -o "${QEMU_TMP}/qemu.deb" "http://ftp.debian.org/debian/${QEMU_DEB_FILENAME}"; then
          break
        fi
        echo "Download attempt $i failed, retrying..." >&2
        sleep 5
      done
      if [[ -f "${QEMU_TMP}/qemu.deb" ]]; then
        ( cd "${QEMU_TMP}" && ar x qemu.deb data.tar.xz && tar -xf data.tar.xz ./usr/bin/qemu-aarch64-static ) && \
          install -m 755 "${QEMU_TMP}/usr/bin/qemu-aarch64-static" /usr/local/bin/qemu-aarch64 || true
      else
        echo "ERROR: failed to download qemu-user-static .deb after 3 attempts" >&2
      fi
    fi
    rm -rf "${QEMU_TMP}"
  fi
fi
if ! command -v qemu-aarch64 >/dev/null 2>&1; then
  echo "ERROR: qemu-aarch64 still not found after install attempts" >&2
  exit 1
fi

echo "Running MLAS tests under qemu-aarch64 -cpu max (SVE enabled)"
echo "Test binary: ${TEST_BIN}"
echo "GTest filter: ${GTEST_FILTER}"

FAILED=0
for VL in ${SVE_VLS}; do
  echo "=== SVE vector length: ${VL} bits ==="
  # sve-default-vector-length takes bytes. The test binary is launched directly
  # under QEMU (no exec from inside the emulated process), so it stays emulated
  # for its whole lifetime and the VL applies. QEMU fails loudly on an unknown
  # property, so an unsupported qemu-user version errors here instead of
  # silently testing the wrong thing.
  if ! OUTPUT=$(qemu-aarch64 -cpu max,sve-default-vector-length=$((VL / 8)) \
      "${TEST_BIN}" --gtest_filter="${GTEST_FILTER}" 2>&1); then
    echo "${OUTPUT}" >&2
    echo "ERROR: MLAS tests FAILED at SVE VL=${VL} bits" >&2
    FAILED=1
    continue
  fi
  echo "${OUTPUT}"
  # Guard against silent skips: the whole point of this step is executing SVE
  # tests, so fail loudly if no tests actually ran.
  if ! grep -qE "[1-9][0-9]* tests? from [1-9][0-9]* test suites? ran" <<<"${OUTPUT}"; then
    echo "ERROR: no tests ran at SVE VL=${VL} bits (all skipped?) -- refusing silent green" >&2
    FAILED=1
  fi
done

if [[ "${FAILED}" -ne 0 ]]; then
  echo "ERROR: MLAS SVE QEMU tests failed for at least one vector length" >&2
  exit 1
fi
echo "All MLAS SVE QEMU tests passed."
