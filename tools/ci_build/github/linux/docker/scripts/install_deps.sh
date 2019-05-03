#!/bin/bash
set -e
#install ninja
aria2c -q -d /tmp https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip
unzip -oq /tmp/ninja-linux.zip -d /usr/bin
rm -f /tmp/ninja-linux.zip
#install protobuf
mkdir -p /tmp/src
mkdir -p /opt/cmake
aria2c -q -d /tmp/src https://github.com/Kitware/CMake/releases/download/v3.13.2/cmake-3.13.2-Linux-x86_64.tar.gz
tar -xf /tmp/src/cmake-3.13.2-Linux-x86_64.tar.gz --strip 1 -C /opt/cmake
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
  /opt/cmake/bin/cmake -G Ninja ../cmake -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_INSTALL_LIBDIR=$PB_LIBDIR  -DCMAKE_INSTALL_SYSCONFDIR=/etc -DCMAKE_POSITION_INDEPENDENT_CODE=ON -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=$build_type
  ninja
  ninja install
  popd
done
export ONNX_ML=1
INSTALLED_PYTHON_VERSION=$(python3 -c 'import sys; version=sys.version_info[:2]; print("{0}.{1}".format(*version));')
if [ "$INSTALLED_PYTHON_VERSION" = "3.7" ];then
  pip3 install --upgrade setuptools
fi
if [ "$INSTALLED_PYTHON_VERSION" = "3.4" ];then
  echo "Python 3.5 and above is needed for running onnx tests!" 1>&2
else
  #Install ONNX
  #5af210ca8a1c73aa6bae8754c9346ec54d0a756e is v1.2.3
  #bae6333e149a59a3faa9c4d9c44974373dcf5256 is v1.3.0
  #9e55ace55aad1ada27516038dfbdc66a8a0763db is v1.4.1
  #7d7bc83d29a328233d3e8affa4c4ea8b3e3599ef is v1.5.0 
    for onnx_version in "5af210ca8a1c73aa6bae8754c9346ec54d0a756e" "bae6333e149a59a3faa9c4d9c44974373dcf5256" "9e55ace55aad1ada27516038dfbdc66a8a0763db" "7d7bc83d29a328233d3e8affa4c4ea8b3e3599ef"; do
    if [ -z ${lastest_onnx_version+x} ]; then
      echo "first pass";
    else
      echo "deleting old onnx-${lastest_onnx_version}";
      pip3 uninstall -y onnx
    fi
    lastest_onnx_version=$onnx_version
    aria2c -q -d /tmp/src  https://github.com/onnx/onnx/archive/$onnx_version.tar.gz
    tar -xf /tmp/src/onnx-$onnx_version.tar.gz -C /tmp/src
    cd /tmp/src/onnx-$onnx_version
    git clone https://github.com/pybind/pybind11.git third_party/pybind11
    python3 setup.py bdist_wheel
    pip3 install -q dist/*
    mkdir -p /data/onnx/$onnx_version
    backend-test-tools generate-data -o /data/onnx/$onnx_version
  done
fi

#The last onnx version will be kept

rm -rf /tmp/src
rm -rf /usr/include/google
rm -rf /usr/lib/libproto*


