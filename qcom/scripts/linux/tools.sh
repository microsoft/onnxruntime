# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT


REPO_ROOT=$(git rev-parse --show-toplevel)

source "${REPO_ROOT}/qcom/scripts/linux/common.sh"

#
# Get the directory containing ccache, installing it if necessary.
#
function get_ccache_bindir() {
    local ccache_version="4.11.3"
    local ccache_sha256="7766991b91b3a5a177ab33fa043fe09e72c68586d5a86d20a563a05b74f119c0"

    local ccache_build="${ccache_version}-linux-x86_64"
    local tools_dir="$(get_tools_dir)"
    local ccache_rootdir="${tools_dir}/ccache-${ccache_build}"
    local ccache_bindir="${ccache_rootdir}"

    if [ ! -d "${ccache_rootdir}" ]; then
        log_info "Installing ccache into ${ccache_rootdir}"

        local ccache_tar="ccache-${ccache_build}.tar.xz"
        local ccache_url_base="https://github.com/ccache/ccache/releases/download"
        local ccache_url="${ccache_url_base}/v${ccache_version}/${ccache_tar}"

        curl --location "${ccache_url}" > "${tools_dir}/${ccache_tar}"

        sha256="$(sha256sum "${tools_dir}/${ccache_tar}" | awk '{ print $1; }')"
        if [ "${sha256}" != "${ccache_sha256}" ]; then
            die "${tools_dir}/${ccache_tar} has invalid SHA256 checksum ${sha256}."
        fi

        tar -C "${tools_dir}" -xJf "${tools_dir}/${ccache_tar}"
        rm "${tools_dir}/${ccache_tar}"
    else
        log_debug "ccache found in ${ccache_rootdir}"
    fi

    echo "${ccache_bindir}"
}

#
# Get the directory containing CMake binaries, installing them if necessary.
#
function get_cmake_bindir() {
    local cmake_version="3.31.7"
    local cmake_sha256="14e15d0b445dbeac686acc13fe13b3135e8307f69ccf4c5c91403996ce5aa2d4"

    local cmake_build="${cmake_version}-linux-x86_64"
    local tools_dir="$(get_tools_dir)"
    local cmake_rootdir="${tools_dir}/cmake-${cmake_build}"
    local cmake_bindir="${cmake_rootdir}/bin"

    if [ ! -d "${cmake_rootdir}" ]; then
        log_info "Installing CMake into ${cmake_rootdir}"

        local cmake_tar="cmake-${cmake_build}.tar.gz"
        local cmake_url_base="https://github.com/Kitware/CMake/releases/download"
        local cmake_url="${cmake_url_base}/v${cmake_version}/${cmake_tar}"

        curl --location "${cmake_url}" > "${tools_dir}/${cmake_tar}"

        sha256="$(sha256sum "${tools_dir}/${cmake_tar}" | awk '{ print $1; }')"
        if [ "${sha256}" != "${cmake_sha256}" ]; then
            die "${tools_dir}/${cmake_tar} has invalid SHA256 checksum ${sha256}."
        fi

        tar -C "${tools_dir}" -xzf "${tools_dir}/${cmake_tar}"
        rm "${tools_dir}/${cmake_tar}"
    else
        log_debug "CMake found in ${cmake_rootdir}"
    fi

    echo "${cmake_bindir}"
}

#
# Get the directory containing ninja, installing it if necessary.
#
function get_ninja_bindir() {
    local ninja_version="1.12.1"
    local ninja_sha256="6f98805688d19672bd699fbbfa2c2cf0fc054ac3df1f0e6a47664d963d530255"

    local ninja_build="${ninja_version}-linux-x86_64"
    local tools_dir="$(get_tools_dir)"
    local ninja_rootdir="${tools_dir}/ninja-${ninja_build}"
    local ninja_bindir="${ninja_rootdir}"

    if [ ! -d "${ninja_rootdir}" ]; then
        log_info "Installing ninja into ${ninja_rootdir}"

        local ninja_zip="ninja-linux.zip"
        local ninja_url_base="https://github.com/ninja-build/ninja/releases/download"
        local ninja_url="${ninja_url_base}/v${ninja_version}/${ninja_zip}"

        curl --location "${ninja_url}" > "${tools_dir}/${ninja_zip}"

        sha256="$(sha256sum "${tools_dir}/${ninja_zip}" | awk '{ print $1; }')"
        if [ "${sha256}" != "${ninja_sha256}" ]; then
            die "${tools_dir}/${ninja_zip} has invalid SHA256 checksum ${sha256}."
        fi

        mkdir -p "${ninja_bindir}"
        unzip -qq -d "${ninja_rootdir}" "${tools_dir}/${ninja_zip}"
        rm "${tools_dir}/${ninja_zip}"
    else
        log_debug "ninja found in ${ninja_rootdir}"
    fi

    echo "${ninja_bindir}"
}

function get_tools_dir() {
    local tools_dir="${REPO_ROOT}/build/tools"

    if [ ! -d "${tools_dir}" ]; then
        mkdir -p "${tools_dir}"
    fi

    echo "${tools_dir}"
}
