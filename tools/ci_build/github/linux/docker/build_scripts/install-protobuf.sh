#!/bin/bash
# Top-level build script called from Dockerfile

# Stop at any error, show all commands
set -exuo pipefail

# Get script directory
MY_DIR=$(dirname "${BASH_SOURCE[0]}")

# Get build utilities
source $MY_DIR/build_utils.sh

# Install newest libtool
check_var ${PROTOBUF_ROOT}
check_var ${PROTOBUF_HASH}
check_var ${PROTOBUF_DOWNLOAD_URL}
fetch_source ${PROTOBUF_ROOT}.tar.gz ${PROTOBUF_DOWNLOAD_URL}
check_sha256sum ${PROTOBUF_ROOT}.tar.gz ${PROTOBUF_HASH}
tar -zxf ${PROTOBUF_ROOT}.tar.gz
pushd protobuf-${PROTOBUF_VERSION}
DESTDIR=/manylinux-rootfs do_standard_install
popd
rm -rf ${PROTOBUF_ROOT} ${PROTOBUF_ROOT}.tar.gz

# Strip what we can
strip_ /manylinux-rootfs

# Install
cp -rlf /manylinux-rootfs/* /

# Remove temporary rootfs
rm -rf /manylinux-rootfs

hash -r
protoc --version
