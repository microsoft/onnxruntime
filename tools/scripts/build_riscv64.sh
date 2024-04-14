#!/bin/bash
# Copyright (c) 2024 SiFive, Inc. All rights reserved.
# Copyright (c) 2024, Phoebe Chen <phoebe.chen@sifive.com>
# Licensed under the MIT License.


# The script is a sample for RISC-V 64-bit cross compilation in
# GNU/Linux, and you should ensure that your environment meets
# ORT requirements. You may need to make changes before using it.

set -e
set -o pipefail

# Get directory this script is in
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OS=$(uname -s)

if [ "$OS" == "Linux" ]; then
    LINUX_DISTRO=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
    if [[ "${LINUX_DISTRO}" == "ubuntu" ]] ;then
        DIR_OS="Linux"
    else
        echo "${LINUX_DISTRO} is not supported"
        return 1
    fi
else
    echo "$OS is not supported"
    return 1
fi

function cleanup {
  if [ -d "$WORK_DIR" ]; then
    rm -rf "$WORK_DIR"
  fi
}

# The riscv toolchain, qemu and other platform related settings.
ORT_ROOT_DIR=$DIR/../..

PREBUILT_DIR="${ORT_ROOT_DIR}/riscv_tools"

read -rp "Enter the riscv tools root path(press enter to use default path:${PREBUILT_DIR}): " INPUT_PATH
if [[ "${INPUT_PATH}" ]]; then
  PREBUILT_DIR=${INPUT_PATH}
fi
echo "The riscv tool prefix path: ${PREBUILT_DIR}"

WORK_DIR=$DIR/.prebuilt

# The prebuit toolchain download from riscv-collab works with Ubuntu.
RISCV_GNU_TOOLCHAIN_URL="https://github.com/riscv-collab/riscv-gnu-toolchain/releases/download"
TOOLCHAIN_VERSION="2023.11.20"
RISCV_TOOLCHAIN_FILE_NAME="riscv64-glibc-ubuntu-22.04-llvm-nightly-2023.11.20-nightly.tar.gz"
RISCV_TOOLCHAIN_FILE_SHA="98d6531b757fac01e065460c19abe8974976c607a8d88631cc5c1529d90ba7ba"

TOOLCHAIN_PATH_PREFIX=${PREBUILT_DIR}

execute () {
  if ! eval "$1"; then
    echo "command:\"$1\" error"
    exit 1
  fi
}

execute "mkdir -p $WORK_DIR"

# Call the cleanup function when this tool exits.
trap cleanup EXIT

# Download and install the toolchain from
# https://github.com/riscv-collab/riscv-gnu-toolchain/releases/download
download_file() {
  local file_name="$1"
  local install_path="$2"
  local file_sha="$3"

  echo "Install $1 to $2"
  if [[ "$(ls -A "$2")" ]]; then
    read -rp "The file already exists. Keep it (y/n)? " replaced
    case ${replaced:0:1} in
      y|Y )
        echo "Skip download $1."
        return
      ;;
      * )
        rm -rf "$2"
      ;;
    esac
  fi

  echo "Download ${file_name} ..."
  mkdir -p "$install_path"
  wget --progress=bar:force:noscroll --directory-prefix="${WORK_DIR}" \
    "${RISCV_GNU_TOOLCHAIN_URL}/${TOOLCHAIN_VERSION}/${file_name}" && \
    echo "${file_sha} ${WORK_DIR}/${file_name}" | sha256sum -c -
  echo "Extract ${file_name} ..."
  tar -C "${install_path}" -xf "${WORK_DIR}/${file_name}" --no-same-owner \
    --strip-components=1
}


read -rp "Install RISCV toolchain(y/n)? " answer
case ${answer:0:1} in
  y|Y )
    download_file "${RISCV_TOOLCHAIN_FILE_NAME}" \
                  "${TOOLCHAIN_PATH_PREFIX}" \
                  "${RISCV_TOOLCHAIN_FILE_SHA}"
  ;;
  * )
    echo "Skip install RISCV toolchain."
  ;;
esac
echo "download finished."


# RISC-V cross compilation in GNU/Linux
RISCV_TOOLCHAIN_ROOT=${TOOLCHAIN_PATH_PREFIX}
RISCV_QEMU_PATH=${TOOLCHAIN_PATH_PREFIX}/bin/qemu-riscv64
python3 "${ORT_ROOT_DIR}"/tools/ci_build/build.py \
    --build_dir "${ORT_ROOT_DIR}/build/${DIR_OS}" \
    --rv64 \
    --parallel \
    --skip_tests \
    --config RelWithDebInfo \
    --cmake_generator=Ninja \
    --riscv_qemu_path="${RISCV_QEMU_PATH}" \
    --riscv_toolchain_root="${RISCV_TOOLCHAIN_ROOT}" "$@"


