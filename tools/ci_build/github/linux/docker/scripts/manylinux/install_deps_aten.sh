#!/bin/bash
set -e -x

# Development tools and libraries
yum -y install \
    graphviz

if [ ! -d "/opt/conda/bin" ]; then
    PYTHON_EXES=("/opt/python/cp38-cp38/bin/python3.8" "/opt/python/cp39-cp39/bin/python3.9" "/opt/python/cp310-cp310/bin/python3.10" "/opt/python/cp311-cp311/bin/python3.11" "/opt/python/cp312-cp312/bin/python3.12")
else
    PYTHON_EXES=("/opt/conda/bin/python")
fi

os_major_version=$(tr -dc '0-9.' < /etc/redhat-release |cut -d \. -f1)

SYS_LONG_BIT=$(getconf LONG_BIT)
mkdir -p /tmp/src
GLIBC_VERSION=$(getconf GNU_LIBC_VERSION | cut -f 2 -d \.)

DISTRIBUTOR=$(lsb_release -i -s)

if [[  ("$DISTRIBUTOR" = "CentOS" || "$DISTRIBUTOR" = "RedHatEnterprise") && $SYS_LONG_BIT = "64" ]]; then
  LIBDIR="lib64"
else
  LIBDIR="lib"
fi

cd /tmp/src
source $(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)/install_shared_deps.sh

cd /tmp/src

if ! [ -x "$(command -v protoc)" ]; then
  source ${0/%install_deps_aten\.sh/..\/install_protobuf.sh}
fi

export ONNX_ML=1
export CMAKE_ARGS="-DONNX_GEN_PB_TYPE_STUBS=OFF -DONNX_WERROR=OFF"

for PYTHON_EXE in "${PYTHON_EXES[@]}"
do
  ${PYTHON_EXE} -m pip install -r ${0/%install_deps_aten\.sh/requirements\.txt}
  if ! [[ ${PYTHON_EXE} = "/opt/python/cp310-cp310/bin/python3.10" ]]; then
    ${PYTHON_EXE} -m pip install -r ${0/%install_deps_aten\.sh/..\/training\/ortmodule\/stage1\/requirements_torch_cpu\/requirements.txt}
  else
    ${PYTHON_EXE} -m pip install torch==2.0.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
  fi
done

cd /tmp/src
GetFile 'https://sourceware.org/pub/valgrind/valgrind-3.16.1.tar.bz2' /tmp/src/valgrind-3.16.1.tar.bz2
tar -jxvf valgrind-3.16.1.tar.bz2
cd valgrind-3.16.1
./configure --prefix=/usr --libdir=/usr/lib64 --enable-only64bit --enable-tls
make -j$(getconf _NPROCESSORS_ONLN)
make install

cd /
rm -rf /tmp/src
