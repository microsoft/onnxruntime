#!/bin/bash
# Top-level build script called from Dockerfile

# Stop at any error, show all commands
set -exuo pipefail

# Get script directory
MY_DIR=$(dirname "${BASH_SOURCE[0]}")

# Get build utilities
source $MY_DIR/build_utils.sh

# Install newest libtool
check_var ${GIT_ROOT}
check_var ${GIT_HASH}
check_var ${GIT_DOWNLOAD_URL}

fetch_source ${GIT_ROOT}.tar.gz ${GIT_DOWNLOAD_URL}
check_sha256sum ${GIT_ROOT}.tar.gz ${GIT_HASH}
tar -xzf ${GIT_ROOT}.tar.gz
pushd ${GIT_ROOT}
make -j$(nproc) install prefix=/usr/local NO_GETTEXT=1 NO_TCLTK=1 DESTDIR=/manylinux-rootfs CPPFLAGS="${MANYLINUX_CPPFLAGS}" CFLAGS="${MANYLINUX_CFLAGS}" CXXFLAGS="${MANYLINUX_CXXFLAGS}" LDFLAGS="${MANYLINUX_LDFLAGS}"
popd
rm -rf ${GIT_ROOT} ${GIT_ROOT}.tar.gz


# Strip what we can
strip_ /manylinux-rootfs

# Install
cp -rlf /manylinux-rootfs/* /

git version
