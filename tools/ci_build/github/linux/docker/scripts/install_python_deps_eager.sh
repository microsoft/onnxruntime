#!/bin/bash
set -e -x

PYTHON_EXE="/usr/bin/python3"

export ONNX_ML=1
export CMAKE_ARGS="-DONNX_GEN_PB_TYPE_STUBS=OFF -DONNX_WERROR=OFF"
${PYTHON_EXE} -m pip install -r ${0/%install_python_deps\.sh/requirements\.txt}
${PYTHON_EXE} -m pip install -r ${0/%install_python_deps.sh/training\/ortmodule\/stage1\/requirements_torch_eager_cpu.txt}