#!/bin/bash
# Top-level build script called from Dockerfile

# Stop at any error, show all commands
set -exuo pipefail

# Get script directory
MY_DIR=$(dirname "${BASH_SOURCE[0]}")

# Get build utilities
source $MY_DIR/build_utils.sh


# Install newest autoconf
check_var ${AUTOCONF_ROOT}
check_var ${AUTOCONF_HASH}
check_var ${AUTOCONF_DOWNLOAD_URL}

AUTOCONF_VERSION=${AUTOCONF_ROOT#*-}
if autoconf --version > /dev/null 2>&1; then
	INSTALLED=$(autoconf --version | head -1 | awk '{ print $NF }')
	SMALLEST=$(echo -e "${INSTALLED}\n${AUTOCONF_VERSION}" | sort -t. -k 1,1n -k 2,2n -k 3,3n -k 4,4n | head -1)
	if [ "${SMALLEST}" == "${AUTOCONF_VERSION}" ]; then
		echo "skipping installation of autoconf ${AUTOCONF_VERSION}, system provides autoconf ${INSTALLED}"
		exit 0
	fi
fi


fetch_source ${AUTOCONF_ROOT}.tar.gz ${AUTOCONF_DOWNLOAD_URL}
check_sha256sum ${AUTOCONF_ROOT}.tar.gz ${AUTOCONF_HASH}
tar -zxf ${AUTOCONF_ROOT}.tar.gz
pushd ${AUTOCONF_ROOT}
DESTDIR=/manylinux-rootfs do_standard_install
popd
rm -rf ${AUTOCONF_ROOT} ${AUTOCONF_ROOT}.tar.gz

# Strip what we can
strip_ /manylinux-rootfs

# Install
cp -rlf /manylinux-rootfs/* /

# Remove temporary rootfs
rm -rf /manylinux-rootfs

hash -r
autoconf --version
