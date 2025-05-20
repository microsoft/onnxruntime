#!/usr/bin/env bash
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

REPO_ROOT=$(git rev-parse --show-toplevel)

source "${REPO_ROOT}/qcom/scripts/linux/common.sh"
source "${REPO_ROOT}/qcom/scripts/linux/tools.sh"

set_strict_mode

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

qairt_sdk_root=
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
    --qairt-sdk-root=*)
      qairt_sdk_root="${i#*=}"
      shift
      ;;
    --target-platform=*)
      target_platform="${i#*=}"
      shift
      ;;
    *)
      die "Unknown option: $i"
      ;;
  esac
done

build_dir="${REPO_ROOT}/build/${target_platform}"
config="Release"
cmake_generator="Ninja"

qairt_sdk_file_path="${build_dir}/qairt-sdk-path-${config}.txt"

if [ -z "${qairt_sdk_root}" ]; then
    qairt_sdk_root="$(get_qairt_contentdir)"
fi

export PATH="$(get_cmake_bindir):$(get_ccache_bindir):$(get_ninja_bindir):${PATH}"

mkdir -p "${build_dir}/${config}"

build_is_dirty=
if [ $(update_needed "${qairt_sdk_root}") ]; then
  build_is_dirty=1
  save_qairt_sdk_path "${qairt_sdk_root}"
fi

common_args=(--cmake_generator "${cmake_generator}" \
             --config "${config}" \
             --use_cache --parallel \
             --build_dir "${build_dir}")

action_args=()

case "${target_platform}" in
  linux)
    if [ -n "${build_is_dirty}" ]; then
      action_args+=("--update")
    fi

    qnn_args=(--use_qnn --qnn_home "${qairt_sdk_root}")
    platform_args=(--build_shared_lib)

    case "${mode}" in
      build)
        action_args+=("--build")
        ;;
      test)
        action_args+=("--test")
        ;;
      *)
        die "Invalid mode '${mode}'."
    esac
    ;;

  android)
    if [ -n "${build_is_dirty}" ]; then
      # The ORT Android build doesn't seem to support --update, but our QNN root has changed
      # so we really want to re-run cmake. Blow away the build.
      log_debug "Build is dirty: blowing away ${build_dir}/${config}"
      rm -fr "${build_dir}/${config}"
    fi

    if [ -n "${ANDROID_HOME:-}" -a -n "${ANDROID_NDK_HOME:-}" ]; then
      android_sdk_path="${ANDROID_HOME}"
      android_ndk_path="${ANDROID_NDK_HOME}"
    else
      android_sdk_path="$(get_android_sdk_root)"
      android_ndk_path="$(get_android_ndk_root)"
    fi

    qnn_args=(--use_qnn static_lib --qnn_home "${qairt_sdk_root}")
    platform_args=(--android_sdk_path "${android_sdk_path}" \
                   --android_ndk_path "${android_ndk_path}" \
                   --android_abi "arm64-v8a" \
                   --android_api "27")
    case "${mode}" in
      build)
        action_args+=("--android")
        ;;
      test)
        die "--mode=test not supported with --target_platform=${target_platform}."
        ;;
      *)
        die "Invalid mode '${mode}'."
    esac
    ;;
  *)
    die "Unknown target platform ${target_platform}."
esac

cd "${REPO_ROOT}"
./build.sh \
  "${action_args[@]}" \
  "${common_args[@]}" \
  "${qnn_args[@]}" \
  "${platform_args[@]}"
