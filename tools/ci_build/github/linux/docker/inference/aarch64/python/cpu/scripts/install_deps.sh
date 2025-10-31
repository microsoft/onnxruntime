#!/bin/bash
set -e -x
pushd .
PYTHON_EXES=("/opt/python/cp311-cp311/bin/python3.11" "/opt/python/cp312-cp312/bin/python3.12" "/opt/python/cp313-cp313/bin/python3.13" "/opt/python/cp314-cp314/bin/python3.14")
CURRENT_DIR=$(pwd)
if ! [ -x "$(command -v protoc)" ]; then
  $CURRENT_DIR/install_protobuf.sh
fi
popd
export ONNX_ML=1
export CMAKE_ARGS="-DONNX_GEN_PB_TYPE_STUBS=ON -DONNX_WERROR=OFF"

for PYTHON_EXE in "${PYTHON_EXES[@]}"
do
  if [[ "$PYTHON_VER" = "/opt/python/cp314-cp314/bin/python3.14" ]]; then
    ${PYTHON_EXE} -m pip install --upgrade pip
    ${PYTHON_EXE} -m pip install onnxscript ml_dtypes onnx_ir onnx_weekly typing_extensions packaging numpy --no-deps
  fi

  ${PYTHON_EXE} -m pip install -r requirements.txt
done
