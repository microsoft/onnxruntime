#!/bin/bash
set -e -x

SCRIPT_DIR="$( dirname "${BASH_SOURCE[0]}" )"
INSTALL_DEPS_DISTRIBUTED_SETUP=false

while getopts p:d:v:tmur parameter_Option
do case "${parameter_Option}"
in
p) echo "Python version is no longer accepted as an input to this script. Ignoring the input argument -p.";;
d) DEVICE_TYPE=${OPTARG};;
v) echo "Cuda version is no longer accepted as an input to this script. Ignoring the input argument -v.";;
t) echo "Installing python training dependencies argument is no longer accepted as an input to this script. Ignoring the input argument -t.";;
m) INSTALL_DEPS_DISTRIBUTED_SETUP=true;;
u) echo "Installing ortmodule python dependencies argument is no longer accepted as an input to this script. Ignoring the input argument -u.";;
r) echo "Installing ROCM python dependencies argument is no longer accepted as an input to this script. Ignoring the input argument -r.";;
esac
done

DEVICE_TYPE=${DEVICE_TYPE:=Normal}

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
  GetFile https://github.com/Kitware/CMake/releases/download/v3.26.3/cmake-3.26.3-Linux-x86_64.tar.gz /tmp/src/cmake-3.26.3-Linux-x86_64.tar.gz
  tar -zxf /tmp/src/cmake-3.26.3-Linux-x86_64.tar.gz --strip=1 -C /usr
  echo "Installing Node.js"
  GetFile https://nodejs.org/dist/v16.14.2/node-v16.14.2-linux-x64.tar.xz /tmp/src/node-v16.14.2-linux-x64.tar.xz
  tar -xf /tmp/src/node-v16.14.2-linux-x64.tar.xz --strip=1 -C /usr
else
  echo "Installing cmake"
  GetFile https://github.com/Kitware/CMake/releases/download/v3.26.3/cmake-3.26.3.tar.gz /tmp/src/cmake-3.26.3.tar.gz
  tar -xf /tmp/src/cmake-3.26.3.tar.gz -C /tmp/src
  pushd .
  cd /tmp/src/cmake-3.26.3
  ./bootstrap --prefix=/usr --parallel=$(getconf _NPROCESSORS_ONLN) --system-bzip2 --system-curl --system-zlib --system-expat
  make -j$(getconf _NPROCESSORS_ONLN)
  make install
  popd
fi

cd /tmp/src

if ! [ -x "$(command -v protoc)" ]; then
  source ${0/%install_os_deps\.sh/install_protobuf\.sh}
fi

if [ $DEVICE_TYPE = "gpu" ]; then
  if [[ $INSTALL_DEPS_DISTRIBUTED_SETUP = true ]]; then
    source ${0/%install_os_deps.sh/install_openmpi.sh}
  fi
fi

cd /
rm -rf /tmp/src
