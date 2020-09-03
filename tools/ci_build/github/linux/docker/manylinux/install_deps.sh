#!/bin/bash
set -e -x

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

PYTHON_EXES=("/opt/python/cp35-cp35m/bin/python3.5" "/opt/python/cp36-cp36m/bin/python3.6" "/opt/python/cp37-cp37m/bin/python3.7" "/opt/python/cp38-cp38/bin/python3.8")
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
echo "Installing cmake"
GetFile https://github.com/Kitware/CMake/releases/download/v3.18.2/cmake-3.18.2-Linux-x86_64.tar.gz /tmp/src/cmake-3.18.2-Linux-x86_64.tar.gz
tar -zxf /tmp/src/cmake-3.18.2-Linux-x86_64.tar.gz --strip=1 -C /usr
echo "Installing Ninja"
GetFile https://github.com/ninja-build/ninja/archive/v1.10.0.tar.gz /tmp/src/ninja-linux.tar.gz
tar -zxf ninja-linux.tar.gz
cd ninja-1.10.0
cmake -Bbuild-cmake -H.
cmake --build build-cmake
mv ./build-cmake/ninja /usr/bin
if [[ "$os_major_version" == "6" ]]; then
  echo "Installing Node.js from source"
  GetFile https://nodejs.org/dist/v12.16.3/node-v12.16.3.tar.xz /tmp/src/node-v12.16.3.tar.xz
  tar -xf /tmp/src/node-v12.16.3.tar.xz
  cd node-v12.16.3
  LDFLAGS=-lrt /opt/python/cp27-cp27m/bin/python configure --ninja
  LDFLAGS=-lrt make -j$(getconf _NPROCESSORS_ONLN)
  LDFLAGS=-lrt make install
else
  echo "Installing Node.js from source"
  GetFile https://nodejs.org/dist/v12.16.3/node-v12.16.3-linux-x64.tar.gz /tmp/src/node-v12.16.3-linux-x64.tar.gz
  tar --strip 1 -xf /tmp/src/node-v12.16.3-linux-x64.tar.gz -C /usr
fi
cd /tmp/src
GetFile https://downloads.gradle-dn.com/distributions/gradle-6.3-bin.zip /tmp/src/gradle-6.3-bin.zip
unzip /tmp/src/gradle-6.3-bin.zip
mv /tmp/src/gradle-6.3 /usr/local/gradle

if ! [ -x "$(command -v protoc)" ]; then
  source ${0/%install_deps\.sh/install_protobuf\.sh}
fi

export ONNX_ML=1

for PYTHON_EXE in "${PYTHON_EXES[@]}"
do
  ${PYTHON_EXE} -m pip install -r ${0/%install_deps\.sh/requirements\.txt}
  onnx_version="c443abd2acad2411103593600319ff81a676afbc"
  onnx_tag="onnxtip"
  GetFile https://github.com/onnx/onnx/archive/$onnx_version.tar.gz /tmp/src/$onnx_version.tar.gz
  tar -xf /tmp/src/$onnx_version.tar.gz -C /tmp/src
  cd /tmp/src/onnx-$onnx_version
  if [ ! -d "third_party/pybind11/pybind11" ]; then
    git clone https://github.com/pybind/pybind11.git third_party/pybind11
  fi
  # We need to make the adjustment only for CentOS6 OR we substitue this only for
  # ${PYTHON_EXE} where we'd need to escape slashes
  # Make sure we do not hit pyhon2 as on CentOS 6 it does not work
  ESCAPED_PY=$(echo "${PYTHON_EXE}" | sed 's/\//\\\//g')
  sed "1,1 s/\/usr\/bin\/env python/$ESCAPED_PY/" /tmp/src/onnx-$onnx_version/tools/protoc-gen-mypy.py > /tmp/src/onnx-$onnx_version/tools/repl_protoc-gen-mypy.py
  chmod a+w /tmp/src/onnx-$onnx_version/tools/protoc-gen-mypy.py
  mv /tmp/src/onnx-$onnx_version/tools/repl_protoc-gen-mypy.py /tmp/src/onnx-$onnx_version/tools/protoc-gen-mypy.py
  mkdir -p /data/onnx/${onnx_tag}
  ${PYTHON_EXE} -m pip install .
  cd /tmp  
  ${PYTHON_EXE} -m onnx.backend.test.cmd_tools generate-data -o /data/onnx/$onnx_tag
done


cd /
rm -rf /tmp/src

