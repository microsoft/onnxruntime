#!/bin/bash
set -e -x
pushd .
PYTHON_EXES=("/opt/python/cp311-cp311/bin/python3.11" "/opt/python/cp312-cp312/bin/python3.12" "/opt/python/cp313-cp313/bin/python3.13" "/opt/python/cp313-cp313t/bin/python3.13" "/opt/python/cp314-cp314/bin/python3.14" "/opt/python/cp314-cp314t/bin/python3.14")
CURRENT_DIR=$(pwd)
if ! [ -x "$(command -v protoc)" ]; then
  $CURRENT_DIR/install_protobuf.sh
fi
popd
export ONNX_ML=1
export CMAKE_ARGS="-DONNX_GEN_PB_TYPE_STUBS=ON -DONNX_WERROR=OFF"

for PYTHON_EXE in "${PYTHON_EXES[@]}"
do
  PIP_REQUIREMENTS=(-r requirements.txt)
  if [[ "${PYTHON_EXE}" == */cp313-cp313t/* ]]; then
    # mypy 1.19+ dependencies do not support free-threaded CPython 3.13.
    PIP_REQUIREMENTS+=("mypy<1.19")
  fi
  "${PYTHON_EXE}" -m pip install "${PIP_REQUIREMENTS[@]}"
done
