#!/bin/bash
# Top-level build script called from Dockerfile

# Stop at any error, show all commands
set -exuo pipefail

# Get script directory
MY_DIR=$(dirname "${BASH_SOURCE[0]}")

# Get build utilities
source $MY_DIR/build_utils.sh

# Install newest cmake
check_var ${CMAKE_VERSION}
check_var ${CMAKE_HASH}
check_var ${CMAKE_DOWNLOAD_URL}

fetch_source cmake-${CMAKE_VERSION}.tar.gz ${CMAKE_DOWNLOAD_URL}/v${CMAKE_VERSION}
check_sha256sum cmake-${CMAKE_VERSION}.tar.gz ${CMAKE_HASH}
tar -xzf cmake-${CMAKE_VERSION}.tar.gz
pushd cmake-${CMAKE_VERSION}
export CPPFLAGS="${MANYLINUX_CPPFLAGS}"
export CFLAGS="${MANYLINUX_CFLAGS} ${CPPFLAGS}"
export CXXFLAGS="${MANYLINUX_CXXFLAGS} ${CPPFLAGS}"
export LDFLAGS="${MANYLINUX_LDFLAGS}"
./bootstrap --system-curl
make
make install DESTDIR=/manylinux-rootfs
popd
rm -rf cmake-${CMAKE_VERSION}.tar.gz cmake-${CMAKE_VERSION}

# remove help
rm -rf /manylinux-rootfs/usr/local/share/cmake-*/Help

# Strip what we can
strip_ /manylinux-rootfs

# Install
cp -rlf /manylinux-rootfs/* /


cmake --version
