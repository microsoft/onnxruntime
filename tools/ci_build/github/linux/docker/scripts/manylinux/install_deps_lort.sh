#!/bin/bash
set -e -x

# Development tools and libraries
dnf -y install \
    graphviz xz gcc-toolset-13-gcc-c++ gcc-toolset-13-gcc gcc-toolset-13-libstdc++-devel cmake python39-devel git
source /opt/rh/gcc-toolset-13/enable
mkdir -p /tmp/src

cd /tmp/src
source $(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)/install_shared_deps.sh

cd /tmp/src

if ! [ -x "$(command -v protoc)" ]; then
  source ${0/%install_deps_lort\.sh/..\/install_protobuf.sh}
fi

export ONNX_ML=1
export CMAKE_ARGS="-DONNX_GEN_PB_TYPE_STUBS=OFF -DONNX_WERROR=OFF"
PYTHON_EXE=/usr/bin/python3.9

echo "Installing Pytorch requirements"
# This may install PyTorch, which will be overrided by the PyTorch local build below.
# beartype is installed here so that onnxscript installation step won't
# install a version PyTorch doesn't like. Once beartype fixes this problem.
# We can remove this line.
$PYTHON_EXE -m pip install -r /tmp/scripts/lort/requirements.txt

cd /usr/local/
echo "Cloning ONNX Script"
git clone --recursive https://github.com/microsoft/onnxscript.git
cd onnxscript
$PYTHON_EXE -m pip install .
cd ~ && $PYTHON_EXE -c "import onnxscript; print(f'Installed ONNX Script: {onnxscript.__version__}')"

cd /usr/local
echo "Cloning Pytorch"
git clone --recursive https://github.com/pytorch/pytorch.git
cd pytorch

echo "Building and installing Pytorch"
VERBOSE=1 BUILD_LAZY_TS_BACKEND=1 $PYTHON_EXE setup.py install
cd ~ && $PYTHON_EXE -c "import torch; print(f'Installed Pytorch: {torch.__version__}')"

cd /
rm -rf /tmp/src
