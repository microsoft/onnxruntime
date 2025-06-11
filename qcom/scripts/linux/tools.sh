# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT


REPO_ROOT=$(git rev-parse --show-toplevel)

source "${REPO_ROOT}/qcom/scripts/linux/common.sh"

#
# Remove outdated packages from th tools directory.
#
function clean_tools_dir() {
    package_manager --clean
}

#
# Get the Android NDK root (ANDROID_NDK_HOME).
# This also installs any other Android SDK packages that ORT needs.
#
function get_android_ndk_root() {
    python \
        "${REPO_ROOT}/qcom/scripts/all/install_ndk.py" \
        "--cli-tools-root=$(get_package_contentdir android_commandlinetools_linux)"
}

#
# Get the Android SDK root (ANDROID_HOME)
#
function get_android_sdk_root() {
    realpath $(get_package_contentdir android_commandlinetools_linux)/..
}

#
# Get the directory containing ccache, installing it if necessary.
#
function get_ccache_bindir() {
    get_package_bindir ccache_linux
}

#
# Get the directory containing CMake binaries, installing them if necessary.
#
function get_cmake_bindir() {
    get_package_bindir cmake_linux
}

#
# Get the directory containing java, installing it if necessary.
#
function get_java_bindir() {
  if [ -n "${JAVA_HOME:-}" ]; then
    log_debug "JAVA_HOME found at ${JAVA_HOME}"
  else
    get_package_bindir java_linux
  fi
}

#
# Get the directory containing ninja, installing it if necessary.
#
function get_ninja_bindir() {
    get_package_bindir ninja_linux
}

function get_package_bindir() {
    local pkg_name="${1}"

    package_manager --install --package="${pkg_name}"
    package_manager --print-bin-dir --package="${pkg_name}"
}

function get_package_contentdir() {
    local pkg_name="${1}"

    package_manager --install --package="${pkg_name}"
    package_manager --print-content-dir --package="${pkg_name}"
}

#
# Get the root of the managed QAIRT installtion.
#
function get_qairt_contentdir() {
    get_package_contentdir qairt
}

function get_tools_dir() {
    local tools_dir="${ORT_BUILD_TOOLS_PATH:-${REPO_ROOT}/build/tools}"

    if [ ! -d "${tools_dir}" ]; then
        mkdir -p "${tools_dir}"
    fi

    echo "${tools_dir}"
}

#
# Run package_manager.py
#
function package_manager() {
    python3 "${REPO_ROOT}/qcom/scripts/all/package_manager.py" --package-root="$(get_tools_dir)" "$@"
}
