#!/bin/bash
# Top-level build script called from Dockerfile

# Stop at any error, show all commands
set -exuo pipefail

# Get script directory
MY_DIR=$(dirname "${BASH_SOURCE[0]}")

# Get build utilities
source $MY_DIR/build_utils.sh

# Install libcrypt.so.1 and libcrypt.so.2
check_var ${LIBXCRYPT_VERSION}
check_var ${LIBXCRYPT_HASH}
check_var ${LIBXCRYPT_DOWNLOAD_URL}
fetch_source v${LIBXCRYPT_VERSION}.tar.gz ${LIBXCRYPT_DOWNLOAD_URL}
check_sha256sum "v${LIBXCRYPT_VERSION}.tar.gz" "${LIBXCRYPT_HASH}"
tar xfz "v${LIBXCRYPT_VERSION}.tar.gz"
pushd "libxcrypt-${LIBXCRYPT_VERSION}"
./autogen.sh > /dev/null
DESTDIR=/manylinux-rootfs do_standard_install \
	--disable-obsolete-api \
	--enable-hashes=all \
	--disable-werror
# we also need libcrypt.so.1 with glibc compatibility for system libraries
# c.f https://github.com/pypa/manylinux/issues/305#issuecomment-625902928
make clean > /dev/null
sed -r -i 's/XCRYPT_([0-9.])+/-/g;s/(%chain OW_CRYPT_1.0).*/\1/g' lib/libcrypt.map.in
DESTDIR=/manylinux-rootfs/so.1 do_standard_install \
	--disable-xcrypt-compat-files \
	--enable-obsolete-api=glibc \
	--enable-hashes=all \
	--disable-werror
cp -P /manylinux-rootfs/so.1/usr/local/lib/libcrypt.so.1* /manylinux-rootfs/usr/local/lib/
rm -rf /manylinux-rootfs/so.1
popd
rm -rf "v${LIBXCRYPT_VERSION}.tar.gz" "libxcrypt-${LIBXCRYPT_VERSION}"

# Strip what we can
strip_ /manylinux-rootfs

# Install
cp -rlf /manylinux-rootfs/* /

# Remove temporary rootfs
rm -rf /manylinux-rootfs

# Delete GLIBC version headers and libraries
rm -rf /usr/include/crypt.h
find /lib* /usr/lib* \( -name 'libcrypt.a' -o -name 'libcrypt.so' -o -name 'libcrypt.so.*' -o -name 'libcrypt-2.*.so' \) -delete
ldconfig
