#!/usr/bin/env bash
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

REPO_ROOT=$(git rev-parse --show-toplevel)

source "${REPO_ROOT}/qcom/scripts/linux/common.sh"
source "${REPO_ROOT}/qcom/scripts/linux/tools.sh"

set_strict_mode

build_dir="${REPO_ROOT}/build/Linux"
config="Release"
cmake_generator="Ninja"

qairt_sdk_file_path="${build_dir}/${config}/qairt-sdk-path.txt"

function update_needed() {
  if [ -f "${qairt_sdk_file_path}" ]; then
    if [ "$(cat ${qairt_sdk_file_path})" != "${1}" ]; then
      log_debug "New QAIRT SDK detected"
      echo "1"
    else
      log_debug "No updated needed"
    fi
  else
    log_debug "No record of previous QAIRT SDK"
    echo "1"
  fi
}

function save_qairt_sdk_path() {
  echo "${1}" > "${qairt_sdk_file_path}"
}

for i in "$@"; do
  case $i in
    --config=*)
      config="${i#*=}"
      shift
      ;;
    --mode=*)
      mode="${i#*=}"
      shift
      ;;
    --qairt_sdk_root=*)
      qairt_sdk_root="${i#*=}"
      shift
      ;;
    *)
      die "Unknown option: $i"
      ;;
  esac
done

export PATH="$(get_cmake_bindir):$(get_ccache_bindir):$(get_ninja_bindir):${PATH}"

mkdir -p "${build_dir}/${config}"

action_args=()
case "${mode}" in
  build)
    if [ $(update_needed "${qairt_sdk_root}") ]; then
      action_args+=("--update")
      save_qairt_sdk_path "${qairt_sdk_root}"
    fi
    action_args+=("--build")
    ;;
  test)
    action_args+=("--test")
    ;;
  *)
    die "Invalid mode '${mode}'."
esac

cd "${REPO_ROOT}"
./build.sh \
    "${action_args[@]}" \
    --use_cache \
    --use_qnn --qnn_home "${qairt_sdk_root}" \
    --build_shared_lib --config "${config}" \
    --parallel \
    --cmake_generator "${cmake_generator}" \
    --build_dir "${build_dir}"
