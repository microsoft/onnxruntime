#!/bin/bash
set -e
aria2c -q -d /tmp/src https://github.com/Kitware/CMake/releases/download/v3.13.2/cmake-3.13.2.tar.gz
tar -xf /tmp/src/cmake-3.13.2.tar.gz -C /tmp/src
cd /tmp/src/cmake-3.13.2
./configure
make
make install
aria2c -q -d /tmp/src https://github.com/protocolbuffers/protobuf/archive/v3.6.1.tar.gz
tar -xf /tmp/src/protobuf-3.6.1.tar.gz -C /tmp/src
cd /tmp/src/protobuf-3.6.1
if [ -f /etc/redhat-release ] ; then
  PB_LIBDIR=lib64
else
  PB_LIBDIR=lib
fi
for build_type in 'Debug' 'Relwithdebinfo'; do
  pushd .
  mkdir build_$build_type
  cd build_$build_type
  cmake -G Ninja ../cmake -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_INSTALL_LIBDIR=$PB_LIBDIR  -DCMAKE_INSTALL_SYSCONFDIR=/etc -DCMAKE_POSITION_INDEPENDENT_CODE=ON -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=$build_type
  ninja
  ninja install
  popd
done
export ONNX_ML=1
INSTALLED_PYTHON_VERSION=$(python3 -c 'import sys; version=sys.version_info[:2]; print("{0}.{1}".format(*version));')
if [ "$INSTALLED_PYTHON_VERSION" = "3.7" ];then
  pip3 install --upgrade setuptools
else
  source /tmp/scripts/install_onnx.sh x86
fi

#The last onnx version will be kept

rm -rf /tmp/src


