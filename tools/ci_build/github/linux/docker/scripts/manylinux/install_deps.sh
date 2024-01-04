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
    PYTHON_EXES=("/opt/python/cp38-cp38/bin/python3.8" "/opt/python/cp39-cp39/bin/python3.9" "/opt/python/cp310-cp310/bin/python3.10" "/opt/python/cp311-cp311/bin/python3.11")
else
    PYTHON_EXES=("/opt/conda/bin/python")
fi

mkdir -p /tmp/src

cd /tmp/src
# shellcheck disable=SC1091
source "$LOCAL_DIR/install_shared_deps.sh"

cd /tmp/src

if ! [ -x "$(command -v protoc)" ]; then
# shellcheck disable=SC1091
  source "$PARENT_DIR/install_protobuf.sh"
fi

export ONNX_ML=1
export CMAKE_ARGS="-DONNX_GEN_PB_TYPE_STUBS=OFF -DONNX_WERROR=OFF"

for PYTHON_EXE in "${PYTHON_EXES[@]}"
do
  ${PYTHON_EXE} -m pip install -r "${0/%install_deps\.sh/requirements\.txt}"
done

cd /
rm -rf /tmp/src
