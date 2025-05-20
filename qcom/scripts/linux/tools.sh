# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT


REPO_ROOT=$(git rev-parse --show-toplevel)

source "${REPO_ROOT}/qcom/scripts/linux/common.sh"

#
# Get the Android NDK root (ANDROID_NDK_HOME).
# This also installs any other Android SDK packages that ORT needs.
#
function get_android_ndk_root() {
    local build_tools_version="34.0.0"
    local ndk_version="26.2.11394342"
    local platform_version="29"

    local sdk_root=$(get_android_sdk_root)
    local ndk_root="${sdk_root}/ndk/${ndk_version}"

    if [ -d "${ndk_root}" ]; then
        log_debug "NDK found in ${ndk_root}"
    else
        log_debug "Installing NDK into ${ndk_root}"
        local sdkmanager=$(realpath $(get_package_bindir android_commandlinetools_linux)/sdkmanager)
        (yes || true) | "${sdkmanager}" --sdk_root="${sdk_root}" \
            --install \
            "build-tools;${build_tools_version}" \
            "platforms;android-${platform_version}" \
            "ndk;${ndk_version}" > /dev/null

        (yes || true) | "${sdkmanager}" --sdk_root="${sdk_root}" --licenses > /dev/null
    fi

    echo $ndk_root
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
    local tools_dir="${REPO_ROOT}/build/tools"

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
