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

echo "Installing azcopy"
mkdir -p /tmp/azcopy
GetFile https://aka.ms/downloadazcopy-v10-linux /tmp/azcopy/azcopy.tar.gz
tar --strip 1 -xf /tmp/azcopy/azcopy.tar.gz -C /tmp/azcopy
cp /tmp/azcopy/azcopy /usr/bin

echo "Installing Ninja"
GetFile https://github.com/ninja-build/ninja/archive/v1.10.0.tar.gz /tmp/src/ninja-linux.tar.gz
tar -zxf ninja-linux.tar.gz
cd ninja-1.10.0
cmake -Bbuild-cmake -H.
cmake --build build-cmake
mv ./build-cmake/ninja /usr/bin

echo "Installing Node.js"
GetFile https://nodejs.org/dist/v16.14.2/node-v16.14.2-linux-x64.tar.gz /tmp/src/node-v16.14.2-linux-x64.tar.gz
tar --strip 1 -xf /tmp/src/node-v16.14.2-linux-x64.tar.gz -C /usr

echo "Installing gradle"
cd /tmp/src
GetFile https://downloads.gradle-dn.com/distributions/gradle-6.3-bin.zip /tmp/src/gradle-6.3-bin.zip
unzip /tmp/src/gradle-6.3-bin.zip
mv /tmp/src/gradle-6.3 /usr/local/gradle

if ! [ -x "$(command -v protoc)" ]; then
  source ${0/%install_deps_lort\.sh/..\/install_protobuf.sh}
fi

export ONNX_ML=1
export CMAKE_ARGS="-DONNX_GEN_PB_TYPE_STUBS=OFF -DONNX_WERROR=OFF"

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
