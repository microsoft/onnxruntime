#!/bin/bash
set -e -x

# Development tools and libraries
yum -y install \
    graphviz

# Download a file from internet
function GetFile {
  local uri=$1
  local path=$2
  local force=${3:-false}
  local download_retries=${4:-5}
  local retry_wait_time_seconds=${5:-30}

  if [[ -f $path ]]; then
    if [[ $force = false ]]; then
      echo "File '$path' already exists. Skipping download"
      return 0
    else
      rm -rf $path
    fi
  fi

  if [[ -f $uri ]]; then
    echo "'$uri' is a file path, copying file to '$path'"
    cp $uri $path
    return $?
  fi

  echo "Downloading $uri"
  # Use aria2c if available, otherwise use curl
  if command -v aria2c > /dev/null; then
    aria2c -q -d $(dirname $path) -o $(basename $path) "$uri"
  else
    curl "$uri" -sSL --retry $download_retries --retry-delay $retry_wait_time_seconds --create-dirs -o "$path" --fail
  fi

  return $?
}


os_major_version=$(cat /etc/redhat-release | tr -dc '0-9.'|cut -d \. -f1)

SYS_LONG_BIT=$(getconf LONG_BIT)
mkdir -p /tmp/src
GLIBC_VERSION=$(getconf GNU_LIBC_VERSION | cut -f 2 -d \.)

DISTRIBUTOR=$(lsb_release -i -s)

if [[ "$DISTRIBUTOR" = "CentOS" && $SYS_LONG_BIT = "64" ]]; then
  LIBDIR="lib64"
else
  LIBDIR="lib"
fi

cd /tmp/src

echo "Installing Ninja"
GetFile https://github.com/ninja-build/ninja/archive/v1.10.0.tar.gz /tmp/src/ninja-linux.tar.gz
tar -zxf ninja-linux.tar.gz
cd ninja-1.10.0
cmake -Bbuild-cmake -H.
cmake --build build-cmake
mv ./build-cmake/ninja /usr/bin
echo "Installing Node.js"
GetFile https://nodejs.org/dist/v12.16.3/node-v12.16.3-linux-x64.tar.gz /tmp/src/node-v12.16.3-linux-x64.tar.gz
tar --strip 1 -xf /tmp/src/node-v12.16.3-linux-x64.tar.gz -C /usr

cd /tmp/src
GetFile https://downloads.gradle-dn.com/distributions/gradle-6.3-bin.zip /tmp/src/gradle-6.3-bin.zip
unzip /tmp/src/gradle-6.3-bin.zip
mv /tmp/src/gradle-6.3 /usr/local/gradle

if ! [ -x "$(command -v protoc)" ]; then
  source ${0/%install_deps.sh/..\/install_protobuf.sh}
fi

export ONNX_ML=1
export CMAKE_ARGS="-DONNX_GEN_PB_TYPE_STUBS=OFF -DONNX_WERROR=OFF"

if [ -d "/opt/python" ]; then
    for PREFIX in $(find /opt/python/ -mindepth 1 -maxdepth 1 \( -name 'cp*' \)); do
	PY_VER=$(${PREFIX}/bin/python -c "import sys; print('.'.join(str(v) for v in sys.version_info[:2]))")
        echo "Install packages for $PY_VER"
	if [ "$PY_VER" == "3.10" ]; then
	    ${PREFIX}/bin/python -m pip install -r ${0/%install_deps\.sh/310\/requirements\.txt}
	else
	    ${PREFIX}/bin/python -m pip install -r ${0/%install_deps\.sh/requirements\.txt}
	fi

    done

elif [ -d "/opt/conda/bin" ]; then
    PYTHON_EXES=("/opt/conda/bin/python")
    ${PYTHON_EXE} -m pip install -r ${0/%install_deps\.sh/requirements\.txt}
    ${PYTHON_EXE} -m pip install -r ${0/%install_deps\.sh/..\/training\/ortmodule\/stage1\/requirements_torch_cpu.txt}
fi

cd /
rm -rf /tmp/src
