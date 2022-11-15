#!/bin/bash
set -e
# Depending on how the compiler has been configured when it was built, sometimes "gcc -dumpversion" shows the full version.
GCC_VERSION=$(gcc -dumpversion | cut -d . -f 1)
#-fstack-clash-protection prevents attacks based on an overlapping heap and stack.
if [ "$GCC_VERSION" -ge 8 ]; then
    CFLAGS="$CFLAGS -fstack-clash-protection"
    CXXFLAGS="$CXXFLAGS -fstack-clash-protection"
fi

ARCH=$(uname -m)

if [ "$ARCH" == "x86_64" ] && [ "$GCC_VERSION" -ge 9 ]; then
    CFLAGS="$CFLAGS -fcf-protection"
    CXXFLAGS="$CXXFLAGS -fcf-protection"
fi
export CFLAGS
export CXXFLAGS

echo "Installing protobuf ..."
protobuf_url=$(grep '^protobuf' /tmp/scripts/deps.txt | cut -d ';' -f 2 | sed 's/\.zip$/\.tar.gz/')
curl -sSL --retry 5 --retry-delay 10 --create-dirs --fail -L -o protobuf_src.tar.gz $protobuf_url
mkdir protobuf
cd protobuf
tar -zxf ../protobuf_src.tar.gz --strip=1
cmake ./cmake -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_INSTALL_SYSCONFDIR=/etc -DCMAKE_POSITION_INDEPENDENT_CODE=ON -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Relwithdebinfo
make -j$(getconf _NPROCESSORS_ONLN)
make install
cd ..
echo "Installing Microsoft GSL ..."
# The deps.txt doesn't have GSL
curl -sSL --retry 5 --retry-delay 10 --create-dirs --fail -L -o gsl_src.tar.gz https://github.com/microsoft/GSL/archive/refs/tags/v4.0.0.tar.gz
mkdir gsl
cd gsl
tar -xf ../gsl_src.tar.gz --strip=1
cmake . -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr
make -j$(getconf _NPROCESSORS_ONLN)
make install
