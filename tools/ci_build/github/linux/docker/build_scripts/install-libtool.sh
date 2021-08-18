#!/bin/bash
# Top-level build script called from Dockerfile

# Stop at any error, show all commands
set -exuo pipefail

# Get script directory
MY_DIR=$(dirname "${BASH_SOURCE[0]}")

# Get build utilities
source $MY_DIR/build_utils.sh

# Install newest libtool
check_var ${LIBTOOL_ROOT}
check_var ${LIBTOOL_HASH}
check_var ${LIBTOOL_DOWNLOAD_URL}
fetch_source ${LIBTOOL_ROOT}.tar.gz ${LIBTOOL_DOWNLOAD_URL}
check_sha256sum ${LIBTOOL_ROOT}.tar.gz ${LIBTOOL_HASH}
tar -zxf ${LIBTOOL_ROOT}.tar.gz
pushd ${LIBTOOL_ROOT}
DESTDIR=/manylinux-rootfs do_standard_install
popd
rm -rf ${LIBTOOL_ROOT} ${LIBTOOL_ROOT}.tar.gz

# Strip what we can
strip_ /manylinux-rootfs

# Install
cp -rlf /manylinux-rootfs/* /

# Remove temporary rootfs
rm -rf /manylinux-rootfs

hash -r
libtoolize --version
