#!/bin/bash
# Top-level build script called from Dockerfile

# Stop at any error, show all commands
set -exuo pipefail

# Get script directory
MY_DIR=$(dirname "${BASH_SOURCE[0]}")

# Get build utilities
source $MY_DIR/build_utils.sh

# Install newest swig
check_var ${SWIG_ROOT}
check_var ${SWIG_HASH}
check_var ${SWIG_DOWNLOAD_URL}
check_var ${PCRE_ROOT}
check_var ${PCRE_HASH}
check_var ${PCRE_DOWNLOAD_URL}

fetch_source ${SWIG_ROOT}.tar.gz ${SWIG_DOWNLOAD_URL}
check_sha256sum ${SWIG_ROOT}.tar.gz ${SWIG_HASH}
tar -xzf ${SWIG_ROOT}.tar.gz
pushd ${SWIG_ROOT}
fetch_source ${PCRE_ROOT}.tar.gz ${PCRE_DOWNLOAD_URL}
check_sha256sum ${PCRE_ROOT}.tar.gz ${PCRE_HASH}
export CPPFLAGS="${MANYLINUX_CPPFLAGS}"
export CFLAGS="${MANYLINUX_CFLAGS}"
export CXXFLAGS="${MANYLINUX_CXXFLAGS}"
export LDFLAGS="${MANYLINUX_LDFLAGS}"
./Tools/pcre-build.sh
./configure
make -j$(nproc)
make install DESTDIR=/manylinux-rootfs
popd
rm -rf ${SWIG_ROOT}*

# Strip what we can
strip_ /manylinux-rootfs

# Install
cp -rlf /manylinux-rootfs/* /

swig -version
