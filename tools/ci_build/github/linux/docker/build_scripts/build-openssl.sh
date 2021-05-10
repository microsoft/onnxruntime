#!/bin/bash
# Top-level build script called from Dockerfile

# Stop at any error, show all commands
set -exuo pipefail

# Get script directory
MY_DIR=$(dirname "${BASH_SOURCE[0]}")

# Get build utilities
source $MY_DIR/build_utils.sh

# Install a more recent openssl
check_var ${OPENSSL_ROOT}
check_var ${OPENSSL_HASH}
check_var ${OPENSSL_DOWNLOAD_URL}

OPENSSL_VERSION=${OPENSSL_ROOT#*-}
OPENSSL_MIN_VERSION=1.1.1

INSTALLED=$(openssl version | head -1 | awk '{ print $2 }')
SMALLEST=$(echo -e "${INSTALLED}\n${OPENSSL_MIN_VERSION}" | sort -t. -k 1,1n -k 2,2n -k 3,3n -k 4,4n | head -1)
if [ "${SMALLEST}" == "${OPENSSL_MIN_VERSION}" ]; then
	echo "skipping installation of openssl ${OPENSSL_VERSION}, system provides openssl ${INSTALLED} which is newer than openssl ${OPENSSL_MIN_VERSION}"
	exit 0
fi

if which yum; then
	yum erase -y openssl-devel
else
	apt-get remove -y libssl-dev
fi

fetch_source ${OPENSSL_ROOT}.tar.gz ${OPENSSL_DOWNLOAD_URL}
check_sha256sum ${OPENSSL_ROOT}.tar.gz ${OPENSSL_HASH}
tar -xzf ${OPENSSL_ROOT}.tar.gz
pushd ${OPENSSL_ROOT}
./config no-shared --prefix=/usr/local/ssl --openssldir=/usr/local/ssl CPPFLAGS="${MANYLINUX_CPPFLAGS}" CFLAGS="${MANYLINUX_CFLAGS} -fPIC" CXXFLAGS="${MANYLINUX_CXXFLAGS} -fPIC" LDFLAGS="${MANYLINUX_LDFLAGS} -fPIC" > /dev/null
make > /dev/null
make install_sw > /dev/null
popd
rm -rf ${OPENSSL_ROOT} ${OPENSSL_ROOT}.tar.gz


/usr/local/ssl/bin/openssl version
