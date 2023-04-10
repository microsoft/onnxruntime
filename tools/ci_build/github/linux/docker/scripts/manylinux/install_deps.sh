#!/bin/bash
set -e -x

# Development tools and libraries
if [ -f /etc/redhat-release ]; then
  yum update && yum -y install graphviz
  os_major_version=$(cat /etc/redhat-release | tr -dc '0-9.'|cut -d \. -f1)
elif [ -f /etc/os-release ]; then
  apt-get update && apt-get install -y graphviz
  os_major_version=$(cat /etc/os-release | tr -dc '0-9.'|cut -d \. -f1)
else
  echo "Unsupported OS"
  exit 1
fi

if [ ! -d "/opt/conda/bin" ]; then
    PYTHON_EXES=("/opt/python/cp38-cp38/bin/python3.8" "/opt/python/cp39-cp39/bin/python3.9" "/opt/python/cp310-cp310/bin/python3.10" "/opt/python/cp311-cp311/bin/python3.11")
else
    PYTHON_EXES=("/opt/conda/bin/python")
fi

SYS_LONG_BIT=$(getconf LONG_BIT)
mkdir -p /tmp/src
GLIBC_VERSION=$(getconf GNU_LIBC_VERSION | cut -f 2 -d \.)

if [[ $SYS_LONG_BIT = "64" ]]; then
  LIBDIR="lib64"
else
  LIBDIR="lib"
fi

cd /tmp/src
source $(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)/install_shared_deps.sh

cd /tmp/src

if ! [ -x "$(command -v protoc)" ]; then
  source ${0/%install_deps.sh/..\/install_protobuf.sh}
fi

export ONNX_ML=1
export CMAKE_ARGS="-DONNX_GEN_PB_TYPE_STUBS=OFF -DONNX_WERROR=OFF"

for PYTHON_EXE in "${PYTHON_EXES[@]}"
do
  ${PYTHON_EXE} -m pip install -r ${0/%install_deps\.sh/requirements\.txt}
done

cd /
rm -rf /tmp/src
