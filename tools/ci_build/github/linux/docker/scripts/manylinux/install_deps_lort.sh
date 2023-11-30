#!/bin/bash
set -e -x

# Development tools and libraries
yum -y install \
    graphviz

mkdir -p /tmp/src

cd /tmp/src
source $(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)/install_shared_deps.sh

cd /tmp/src

if ! [ -x "$(command -v protoc)" ]; then
  source ${0/%install_deps_lort\.sh/..\/install_protobuf.sh}
fi

export ONNX_ML=1
export CMAKE_ARGS="-DONNX_GEN_PB_TYPE_STUBS=OFF -DONNX_WERROR=OFF"

# This may install PyTorch, which will be overrided by the PyTorch local build below.
/opt/python/cp39-cp39/bin/python3.9 -m pip install transformers

# beartype is installed here so that onnxscript installation step won't
# install a version PyTorch doesn't like. Once beartype fixes this problem.
# We can remove this line.
/opt/python/cp39-cp39/bin/python3.9 -m pip install beartype==0.15.0

cd /usr/local/
echo "Cloning ONNX Script"
git clone --recursive https://github.com/microsoft/onnxscript.git
cd onnxscript
/opt/python/cp39-cp39/bin/python3.9 -m pip install -r requirements-dev.txt
/opt/python/cp39-cp39/bin/python3.9 setup.py install
cd ~ && /opt/python/cp39-cp39/bin/python3.9 -c "import onnxscript; print(f'Installed ONNX Script: {onnxscript.__version__}')"

cd /usr/local
echo "Cloning Pytorch"
git clone --recursive https://github.com/pytorch/pytorch.git
cd pytorch
echo "Installing Pytorch requirements"
/opt/python/cp39-cp39/bin/python3.9 -m pip install -r requirements.txt
/opt/python/cp39-cp39/bin/python3.9 -m pip install flatbuffers cerberus h5py onnx
echo "Building and installing Pytorch"
VERBOSE=1 BUILD_LAZY_TS_BACKEND=1 /opt/python/cp39-cp39/bin/python3.9 setup.py install
cd ~ && /opt/python/cp39-cp39/bin/python3.9 -c "import torch; print(f'Installed Pytorch: {torch.__version__}')"

cd /
rm -rf /tmp/src
