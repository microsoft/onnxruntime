#!/bin/bash
set -e -x

# Development tools and libraries
dnf -y install \
    graphviz

if [ ! -d "/opt/conda/bin" ]; then
    PYTHON_EXES=("/opt/python/cp38-cp38/bin/python3.8" "/opt/python/cp39-cp39/bin/python3.9" "/opt/python/cp310-cp310/bin/python3.10" "/opt/python/cp311-cp311/bin/python3.11")
else
    PYTHON_EXES=("/opt/conda/bin/python")
fi


SYS_LONG_BIT=$(getconf LONG_BIT)
mkdir -p /tmp/src

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
    ${PYTHON_EXE} -m pip install torch==2.3.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
  fi
done

cd /
rm -rf /tmp/src
