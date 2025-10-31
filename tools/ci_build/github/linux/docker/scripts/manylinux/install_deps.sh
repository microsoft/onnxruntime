#!/bin/bash
set -e -x

# Development tools and libraries
if [ -f /etc/redhat-release ]; then
  dnf -y install graphviz
elif [ -f /etc/os-release ]; then
  apt-get update && apt-get install -y graphviz
else
  echo "Unsupported OS"
  exit 1
fi

# Install dotnet
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PARENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." &> /dev/null && pwd)"
# ShellCheck is unable to follow dynamic paths, such as source "$somedir/file".
# shellcheck disable=SC1091
source "$PARENT_DIR/install_dotnet.sh"

if [ ! -d "/opt/conda/bin" ]; then
    PYTHON_EXES=("/opt/python/cp311-cp311/bin/python3.11" "/opt/python/cp312-cp312/bin/python3.12" "/opt/python/cp313-cp313/bin/python3.13"  "/opt/python/cp314-cp314/bin/python3.14")
else
    PYTHON_EXES=("/opt/conda/bin/python")
fi

mkdir -p /tmp/src

cd /tmp/src
# shellcheck disable=SC1091
source "$LOCAL_DIR/install_shared_deps.sh"

cd /tmp/src

export ONNX_ML=1
export CMAKE_ARGS="-DONNX_GEN_PB_TYPE_STUBS=OFF -DONNX_WERROR=OFF"

for PYTHON_EXE in "${PYTHON_EXES[@]}"
do
  # For python 3.14, install onnxscript without dependencies (It depends on onnx, which does not suppport python 3.14 yet)
  # This can be removed once onnx supports python 3.14: update requirements.txt to replace onnx_weekly with onnx.
  if [[ "$PYTHON_VER" = "/opt/python/cp314-cp314/bin/python3.14" ]]; then
    ${PYTHON_EXE} -m pip install --upgrade pip
    ${PYTHON_EXE} -m pip install onnxscript ml_dtypes onnx_ir onnx_weekly typing_extensions packaging --no-deps
  fi

  ${PYTHON_EXE} -m pip install -r "${0/%install_deps\.sh/requirements\.txt}"
done

cd /
rm -rf /tmp/src
