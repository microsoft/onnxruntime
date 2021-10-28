#!/bin/bash
set -e -x

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

echo "Installing cmake"
GetFile https://github.com/Kitware/CMake/releases/download/v3.21.1/cmake-3.21.1-linux-`uname -m`.tar.gz /tmp/src/cmake-3.21.1-linux-`uname -m`.tar.gz
tar -zxf /tmp/src/cmake-3.21.1-linux-`uname -m`.tar.gz --strip=1 -C /usr

echo "Installing Ninja"
GetFile https://github.com/ninja-build/ninja/archive/v1.10.0.tar.gz /tmp/src/ninja-linux.tar.gz
tar -zxf ninja-linux.tar.gz
cd ninja-1.10.0
cmake -Bbuild-cmake -H.
cmake --build build-cmake
mv ./build-cmake/ninja /usr/bin

echo "Installing Node.js"
CPU_ARCH=`uname -m`
if [[ "$CPU_ARCH" = "x86_64" ]]; then
  NODEJS_ARCH=x64
elif [[ "$CPU_ARCH" = "aarch64" ]]; then
  NODEJS_ARCH=arm64
else
  NODEJS_ARCH=$CPU_ARCH
fi
GetFile https://nodejs.org/dist/v14.18.1/node-v14.18.1-linux-${NODEJS_ARCH}.tar.gz /tmp/src/node-v14.18.1-linux-${NODEJS_ARCH}.tar.gz
tar --strip 1 -xf /tmp/src/node-v14.18.1-linux-${NODEJS_ARCH}.tar.gz -C /usr

cd /tmp/src
GetFile https://downloads.gradle-dn.com/distributions/gradle-6.3-bin.zip /tmp/src/gradle-6.3-bin.zip
unzip /tmp/src/gradle-6.3-bin.zip
mv /tmp/src/gradle-6.3 /usr/local/gradle

if ! [ -x "$(command -v protoc)" ]; then
  GetFile https://github.com/protocolbuffers/protobuf/archive/v3.16.0.tar.gz /tmp/src/v3.16.0.tar.gz
  tar -xf /tmp/src/v3.16.0.tar.gz -C /tmp/src
  cd /tmp/src/protobuf-3.16.0
  cmake ./cmake -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_INSTALL_SYSCONFDIR=/etc -DCMAKE_POSITION_INDEPENDENT_CODE=ON -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Relwithdebinfo -DCMAKE_INSTALL_LIBDIR=lib64
  make -j$(getconf _NPROCESSORS_ONLN)
  make install
fi

export ONNX_ML=1
export CMAKE_ARGS="-DONNX_GEN_PB_TYPE_STUBS=OFF -DONNX_WERROR=OFF"

pip3 install -r ${0/%install_deps\.sh/requirements\.txt}


cd /
rm -rf /tmp/src
