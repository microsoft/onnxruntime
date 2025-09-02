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
      log_debug "No update needed"
    fi
  else
    log_debug "No record of previous QAIRT SDK"
    echo "1"
  fi
}

function save_qairt_sdk_path() {
  echo "${1}" > "${qairt_sdk_file_path}"
}

config="Release"
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
    --target-arch=*)
      target_arch="${i#*=}"
      shift
      ;;
    *)
      die "Unknown option: $i"
      ;;
  esac
done

cmake_generator="Ninja"

build_root="${REPO_ROOT}/build"
build_dir="${build_root}/${target_platform}-${target_arch}"

qairt_sdk_file_path="${build_dir}/qairt-sdk-path-${config}.txt"

if [ -z "${qairt_sdk_root}" ]; then
    qairt_sdk_root="$(get_qairt_contentdir)"
fi

cmake_bindir="$(get_cmake_bindir)"
export PATH="${cmake_bindir}:$(get_ccache_bindir):$(get_ninja_bindir):${PATH}"

mkdir -p "${build_dir}/${config}"

build_is_dirty=
if [ $(update_needed "${qairt_sdk_root}") ]; then
  build_is_dirty=1
  save_qairt_sdk_path "${qairt_sdk_root}"
fi

common_args=(--cmake_generator "${cmake_generator}" \
             --config "${config}" \
             --use_cache --parallel \
             --build_dir "${build_dir}" \
             --wheel_name_suffix qcom-internal \
             --compile_no_warning_as_error)

action_args=()
make_test_archive=
run_tests=
test_runner=

case "${target_platform}" in
  linux)
    qnn_args=(--use_qnn --qnn_home "${qairt_sdk_root}")
    platform_args=(--build_shared_lib)

    test_runner="${REPO_ROOT}/qcom/scripts/linux/run_tests.sh"

    case "${mode}" in
      build)
        action_args+=("--build")
        if [ -n "${build_is_dirty}" ]; then
          action_args+=("--update")
        fi
        if [ "${target_arch}" == "aarch64-oe-gcc11.2" ]; then
          toolchain_root="$(get_linux_oe_gcc112_toolchain_root)"
          toolchain_cmake="${REPO_ROOT}/qcom/scripts/linux/linux-aarch64-gcc11.toolchain.cmake"

          # We need $toolchain_root from the toolchain.cmake, but the toolchain.cmake is sometimes
          # evaluated without the project's CMakeCache.txt entries. Pass it through the environment :-/
          export ORT_BUILD_LINUX_TOOLCHAIN_ROOT="${toolchain_root}"

          platform_args+=(--cmake_extra_defines
                          CMAKE_TOOLCHAIN_FILE:FILEPATH="${toolchain_cmake}"
                          ARM64:BOOL=TRUE)
        else
          platform_args+=(--build_wheel)
        fi
        ;;
      test)
        run_tests=1
        ;;
      archive)
        make_test_archive=1
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

    export PATH="$(get_java_bindir):${PATH}"

    if [ -n "${ANDROID_HOME:-}" -a -n "${ANDROID_NDK_HOME:-}" ]; then
      android_sdk_path="${ANDROID_HOME}"
      android_ndk_path="${ANDROID_NDK_HOME}"
    else
      android_sdk_path="$(get_android_sdk_root)"
      android_ndk_path="$(get_android_ndk_root)"
    fi

    qnn_args=(--use_qnn static_lib --qnn_home "${qairt_sdk_root}")
    platform_args=(--build_shared_lib \
                   --android_sdk_path "${android_sdk_path}" \
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
      archive)
        make_test_archive=1
        ;;
      *)
        die "Invalid mode '${mode}'."
    esac
    ;;
  *)
    die "Unknown target platform ${target_platform}."
esac

clean_tools_dir

# Whatever happens, blow away mirror to avoid it showing up in git; it's okay, it's
# very cheap to regenerate.
function scrub_mirror() {
  rm -fr "${REPO_ROOT}/mirror"
}
trap scrub_mirror EXIT

if [ -n "${make_test_archive}" ]; then
  python "${REPO_ROOT}/qcom/scripts/all/archive_tests.py" \
    "--config=${config}" \
    "--target-platform=${target_platform}-${target_arch}" \
    "--qairt-sdk-root=${qairt_sdk_root}"
else
  cd "${REPO_ROOT}"

  # This platform supports running tests on the host. Prep the build directory
  # to run with our ctest wrapper.
  if [ -n "${test_runner}" ]; then
    cp "${test_runner}" "${build_dir}/${config}/"
    cp "${cmake_bindir}/ctest" "${build_dir}/${config}/"
  fi

  if [ "${#action_args[@]}" -gt 0 ]; then

    python "${REPO_ROOT}/qcom/scripts/all/fetch_cmake_deps.py"

    ./build.sh \
      "${action_args[@]}" \
      "${common_args[@]}" \
      "${qnn_args[@]}" \
      "${platform_args[@]}"
  fi

  if [ -n "${run_tests}" ]; then
    onnx_models_root="$(get_onnx_models_dir)"

    cd "${build_dir}/${config}/"

    # Run tests using our ctest wrapper.
    log_info "-=-=-=- Running unit tests -=-=-=-=-"
    "./$(basename ${test_runner})"

    log_info "-=-=-=- Running ONNX model tests -=-=-=-=-"
    "${build_dir}/${config}/onnx_test_runner" \
        -j 1 \
        -e qnn \
        -i "backend_type|cpu" \
        "${build_dir}/${config}/_deps/onnx-src/onnx/backend/test/data/node"

    log_info "-=-=-=- Running onnx/models float32 tests -=-=-=-=-"
    cd "${onnx_models_root}"
    "${build_dir}/${config}/onnx_test_runner" \
        -j 1 \
        -e qnn \
        -i "backend_type|cpu" \
        "testdata/float32"

    log_info "-=-=-=- Running onnx/models qdq tests -=-=-=-=-"
    "${build_dir}/${config}/onnx_test_runner" \
        -j 1 \
        -e qnn \
        -i "backend_type|htp" \
        "testdata/qdq"

    log_info "-=-=-=- Running onnx/models qdq tests with context cache enabled -=-=-=-=-"
    log_debug "Scrubbing old context caches"
    find "testdata/qdq-with-context-cache" -name "*_ctx.onnx" -print -delete
    "${build_dir}/${config}/onnx_test_runner" \
        -j 1 \
        -e qnn \
        -f -i "backend_type|htp" \
        "testdata/qdq-with-context-cache"
  fi
fi
