#!/bin/bash
set -e


while getopts p:d: parameter_Option
do case "${parameter_Option}"
in
p) PYTHON_VER=${OPTARG};;
d) DEVICE_TYPE=${OPTARG};;
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
  GetFile https://github.com/Kitware/CMake/releases/download/v3.13.5/cmake-3.13.5-Linux-x86_64.tar.gz /tmp/src/cmake-3.13.5-Linux-x86_64.tar.gz  
  tar -zxf /tmp/src/cmake-3.13.5-Linux-x86_64.tar.gz --strip=1 -C /usr
else
  echo "Installing cmake"
  GetFile https://github.com/Kitware/CMake/releases/download/v3.13.5/cmake-3.13.5.tar.gz /tmp/src/cmake-3.13.5.tar.gz 
  tar -xf /tmp/src/cmake-3.13.5.tar.gz -C /tmp/src
  pushd .
  cd /tmp/src/cmake-3.13.5
  ./bootstrap --prefix=/usr --parallel=$(getconf _NPROCESSORS_ONLN) --system-bzip2 --system-curl --system-zlib --system-expat
  make -j$(getconf _NPROCESSORS_ONLN)
  make install
  popd
fi

GetFile https://downloads.gradle-dn.com/distributions/gradle-6.2-bin.zip /tmp/src/gradle-6.2-bin.zip
cd /tmp/src
unzip gradle-6.2-bin.zip
mv /tmp/src/gradle-6.2 /usr/local/gradle

if ! [ -x "$(command -v protoc)" ]; then
  source ${0/%install_deps\.sh/install_protobuf\.sh}
fi


#Don't update 'wheel' to the latest version. see: https://github.com/pypa/auditwheel/issues/102
${PYTHON_EXE} -m pip install -r ${0/%install_deps\.sh/requirements\.txt}
if [ $DEVICE_TYPE = "Normal" ]; then
    ${PYTHON_EXE} -m pip install sympy==1.1.1
fi


#install onnx
export ONNX_ML=1
if [ "$PYTHON_VER" = "3.4" ];then
  echo "Python 3.5 and above is needed for running onnx tests!" 1>&2
else
  source ${0/%install_deps\.sh/install_onnx\.sh} $PYTHON_VER
fi

#The last onnx version will be kept
cd /
rm -rf /tmp/src

if [ "$DISTRIBUTOR" = "Ubuntu" ]; then
  apt-get -y remove libprotobuf-dev protobuf-compiler
elif [ "$DISTRIBUTOR" = "CentOS" ]; then
  rm -rf /usr/include/google
  rm -rf /usr/$LIBDIR/libproto*
else
  dnf remove -y protobuf-devel protobuf-compiler
fi

