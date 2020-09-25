#!/bin/bash
set -e -x

SCRIPT_DIR="$( dirname "${BASH_SOURCE[0]}" )"

while getopts p:d:x: parameter_Option
do case "${parameter_Option}"
in
p) PYTHON_VER=${OPTARG};;
d) DEVICE_TYPE=${OPTARG};;
x) BUILD_EXTR_PAR=${OPTARG};;
esac
done

echo "Python version=$PYTHON_VER"

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

if [[ "$PYTHON_VER" = "3.5" && -d "/opt/python/cp35-cp35m"  ]]; then
   PYTHON_EXE="/opt/python/cp35-cp35m/bin/python3.5"
elif [[ "$PYTHON_VER" = "3.6" && -d "/opt/python/cp36-cp36m"  ]]; then
   PYTHON_EXE="/opt/python/cp36-cp36m/bin/python3.6"
elif [[ "$PYTHON_VER" = "3.7" && -d "/opt/python/cp37-cp37m"  ]]; then
   PYTHON_EXE="/opt/python/cp37-cp37m/bin/python3.7"
elif [[ "$PYTHON_VER" = "3.8" && -d "/opt/python/cp38-cp38"  ]]; then
   PYTHON_EXE="/opt/python/cp38-cp38/bin/python3.8"
else
   PYTHON_EXE="/usr/bin/python${PYTHON_VER}"
fi

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
  source ${0/%install_deps\.sh/install_protobuf\.sh}
fi

export ONNX_ML=1
export CMAKE_ARGS="-DONNX_GEN_PB_TYPE_STUBS=OFF -DONNX_WERROR=OFF"
${PYTHON_EXE} -m pip install -r ${0/%install_deps\.sh/requirements\.txt}
if [ $DEVICE_TYPE = "gpu" ]; then
    if [[ $BUILD_EXTR_PAR = *--enable_training* ]]; then
      ${PYTHON_EXE} -m pip install -r ${0/%install_deps.sh/training\/requirements.txt}

      if [[ $BUILD_EXTR_PAR = *--enable_training_python_frontend_e2e_tests* || $BUILD_EXTR_PAR = *enable_training_pipeline_e2e_tests* ]]; then
        source ${0/%install_deps.sh/install_openmpi.sh}
      fi
    fi
fi

cd /
rm -rf /tmp/src
rm -rf /usr/include/google
rm -rf /usr/$LIBDIR/libproto*
