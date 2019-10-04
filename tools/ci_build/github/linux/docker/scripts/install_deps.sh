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


SYS_LONG_BIT=$(getconf LONG_BIT)
mkdir -p /tmp/src
if [ $SYS_LONG_BIT = "64" ]; then
  echo "Installing azcopy"
  mkdir -p /tmp/azcopy
  aria2c -q -d /tmp/azcopy -o azcopy.tar.gz https://aka.ms/downloadazcopy-v10-linux
  tar --strip 1 -xf /tmp/azcopy/azcopy.tar.gz -C /tmp/azcopy
  cp /tmp/azcopy/azcopy /usr/bin
  echo "Installing cmake"
  aria2c -q -d /tmp/src "https://github.com/Kitware/CMake/releases/download/v3.15.4/cmake-3.15.4-Linux-x86_64.tar.gz"
  tar -zxf /tmp/src/cmake-3.15.4-Linux-x86_64.tar.gz --strip=1 -C /usr
else
  echo "Installing cmake"
  aria2c -q -d /tmp/src https://github.com/Kitware/CMake/releases/download/v3.15.4/cmake-3.15.4.tar.gz
  tar -xf /tmp/src/cmake-3.15.4.tar.gz -C /tmp/src
  cd /tmp/src/cmake-3.15.4
  ./configure --prefix=/usr --parallel=`nproc` --system-curl --system-zlib --system-expat
  make -j`nproc`
  make install
fi



DISTRIBUTOR=$(lsb_release -i -s)
if ! [ -x "$(command -v protoc)" ]; then
  source ${0/%install_deps\.sh/install_protobuf\.sh}
fi

/usr/bin/python${PYTHON_VER} -m pip install --upgrade --force-reinstall numpy==1.15.0
/usr/bin/python${PYTHON_VER} -m pip install --upgrade --force-reinstall requests==2.21.0
if [ $DEVICE_TYPE = "Normal" ]; then
    /usr/bin/python${PYTHON_VER} -m pip install --upgrade --force-reinstall sympy==1.1.1
fi
/usr/bin/python${PYTHON_VER} -m pip install --upgrade scipy

#install onnx
export ONNX_ML=1
INSTALLED_PYTHON_VERSION=$(python3 -c 'import sys; version=sys.version_info[:2]; print("{0}.{1}".format(*version));')
if [ "$INSTALLED_PYTHON_VERSION" = "3.7" ];then
  pip3 install --upgrade setuptools
fi
if [ "$INSTALLED_PYTHON_VERSION" = "3.4" ];then
  echo "Python 3.5 and above is needed for running onnx tests!" 1>&2
else  
  source ${0/%install_deps\.sh/install_onnx\.sh} $INSTALLED_PYTHON_VERSION
fi

#The last onnx version will be kept
cd /
rm -rf /tmp/src

if [ "$DISTRIBUTOR" = "Ubuntu" ]; then
  apt-get -y remove libprotobuf-dev protobuf-compiler
elif [ "$DISTRIBUTOR" = "CentOS" ]; then
  rm -rf /usr/include/google
  rm -rf /usr/lib64/libproto*
elif [ "$AUDITWHEEL_PLAT" = "manylinux2010_x86_64" ]; then
  # we did not install protobuf 2.x no need to uninstall
  :
else
  dnf remove -y protobuf-devel protobuf-compiler
fi

