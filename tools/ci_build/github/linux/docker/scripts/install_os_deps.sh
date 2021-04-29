#!/bin/bash
set -e -x

SCRIPT_DIR="$( dirname "${BASH_SOURCE[0]}" )"
INSTALL_DEPS_DISTRIBUTED_SETUP=false

while getopts d:m parameter_Option
do case "${parameter_Option}"
in
d) DEVICE_TYPE=${OPTARG};;
m) INSTALL_DEPS_DISTRIBUTED_SETUP=true;;
esac
done

DEVICE_TYPE=${DEVICE_TYPE:=Normal}

#Download a file from internet
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

SYS_LONG_BIT=$(getconf LONG_BIT)
mkdir -p /tmp/src
GLIBC_VERSION=$(getconf GNU_LIBC_VERSION | cut -f 2 -d \.)

DISTRIBUTOR=$(lsb_release -i -s)

if [[ "$DISTRIBUTOR" = "CentOS" && $SYS_LONG_BIT = "64" ]]; then
  LIBDIR="lib64"
else
  LIBDIR="lib"
fi
if [[ $SYS_LONG_BIT = "64" && "$GLIBC_VERSION" -gt "9" ]]; then
  echo "Installing azcopy"
  mkdir -p /tmp/azcopy
  GetFile https://aka.ms/downloadazcopy-v10-linux /tmp/azcopy/azcopy.tar.gz
  tar --strip 1 -xf /tmp/azcopy/azcopy.tar.gz -C /tmp/azcopy
  cp /tmp/azcopy/azcopy /usr/bin
  echo "Installing cmake"
  GetFile https://github.com/Kitware/CMake/releases/download/v3.18.2/cmake-3.18.2-Linux-x86_64.tar.gz /tmp/src/cmake-3.18.2-Linux-x86_64.tar.gz
  tar -zxf /tmp/src/cmake-3.18.2-Linux-x86_64.tar.gz --strip=1 -C /usr
  echo "Installing Node.js"
  GetFile https://nodejs.org/dist/v12.16.3/node-v12.16.3-linux-x64.tar.xz /tmp/src/node-v12.16.3-linux-x64.tar.xz
  tar -xf /tmp/src/node-v12.16.3-linux-x64.tar.xz --strip=1 -C /usr
else
  echo "Installing cmake"
  GetFile https://github.com/Kitware/CMake/releases/download/v3.18.2/cmake-3.18.2.tar.gz /tmp/src/cmake-3.18.2.tar.gz
  tar -xf /tmp/src/cmake-3.18.2.tar.gz -C /tmp/src
  pushd .
  cd /tmp/src/cmake-3.18.2
  ./bootstrap --prefix=/usr --parallel=$(getconf _NPROCESSORS_ONLN) --system-bzip2 --system-curl --system-zlib --system-expat
  make -j$(getconf _NPROCESSORS_ONLN)
  make install
  popd
fi

GetFile https://downloads.gradle-dn.com/distributions/gradle-6.3-bin.zip /tmp/src/gradle-6.3-bin.zip
cd /tmp/src
unzip gradle-6.3-bin.zip
mv /tmp/src/gradle-6.3 /usr/local/gradle

if ! [ -x "$(command -v protoc)" ]; then
  source ${0/%install_os_deps\.sh/install_protobuf\.sh}
fi

export ONNX_ML=1
export CMAKE_ARGS="-DONNX_GEN_PB_TYPE_STUBS=OFF -DONNX_WERROR=OFF"
if [ $DEVICE_TYPE = "gpu" ]; then
  if [[ $INSTALL_DEPS_DISTRIBUTED_SETUP = true ]]; then
    source ${0/%install_os_deps.sh/install_openmpi.sh}
  fi
fi

cd /
rm -rf /tmp/src
rm -rf /usr/include/google
rm -rf /usr/$LIBDIR/libproto*
