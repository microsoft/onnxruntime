#!/bin/bash
set -e

SYS_LONG_BIT=$(getconf LONG_BIT)

echo "Installing azcopy"
if [ $SYS_LONG_BIT = "64" ]; then
  mkdir -p /tmp/azcopy
  aria2c -q -d /tmp/azcopy -o azcopy.tar.gz https://aka.ms/downloadazcopy-v10-linux
  tar --strip 1 -xf /tmp/azcopy/azcopy.tar.gz -C /tmp/azcopy
  cp /tmp/azcopy/azcopy /usr/bin
fi

mkdir -p /tmp/src
aria2c -q -d /tmp/src https://github.com/Kitware/CMake/releases/download/v3.14.5/cmake-3.14.5.tar.gz
tar -xf /tmp/src/cmake-3.14.5.tar.gz -C /tmp/src
cd /tmp/src/cmake-3.14.5
./configure --prefix=/usr --parallel=`nproc` --system-curl --system-zlib --system-expat
make -j`nproc`
make install

#install onnx
export ONNX_ML=1
INSTALLED_PYTHON_VERSION=$(python3 -c 'import sys; version=sys.version_info[:2]; print("{0}.{1}".format(*version));')
if [ "$INSTALLED_PYTHON_VERSION" = "3.7" ];then
  pip3 install --upgrade setuptools
fi
if [ "$INSTALLED_PYTHON_VERSION" = "3.4" ];then
  echo "Python 3.5 and above is needed for running onnx tests!" 1>&2
else
  source ${0/%install_deps\.sh/install_onnx\.sh} $INSTALLED_PYTHON_VERSION
fi

#The last onnx version will be kept
cd /
rm -rf /tmp/src
DISTRIBUTOR=$(lsb_release -i -s)
if [ "$DISTRIBUTOR" = "Ubuntu" ]; then
  apt-get -y remove libprotobuf-dev protobuf-compiler
elif [ "$AUDITWHEEL_PLAT" = "manylinux2010_x86_64" ]; then
  # we did not install protobuf 2.x no need to uninstall
  :
else
  dnf remove -y protobuf-devel protobuf-compiler
fi

