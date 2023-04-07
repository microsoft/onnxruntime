#!/bin/bash
set -e -x

INSTALL_PREFIX='/usr'
DEP_FILE_PATH='/tmp/scripts/deps.txt'
while getopts "p:d:" parameter_Option
do case "${parameter_Option}"
in
p) INSTALL_PREFIX=${OPTARG};;
d) DEP_FILE_PATH=${OPTARG};;
esac
done

EXTRA_CMAKE_ARGS=""

case "$(uname -s)" in
   Darwin*)
     echo 'Building ONNX Runtime on Mac OS X'
     EXTRA_CMAKE_ARGS="-DCMAKE_OSX_ARCHITECTURES=x86_64;arm64"
     ;;
   Linux*)
    # Depending on how the compiler has been configured when it was built, sometimes "gcc -dumpversion" shows the full version.
    GCC_VERSION=$(gcc -dumpversion | cut -d . -f 1)
    #-fstack-clash-protection prevents attacks based on an overlapping heap and stack.
    if [ "$GCC_VERSION" -ge 8 ]; then
        CFLAGS="$CFLAGS -fstack-clash-protection"
        CXXFLAGS="$CXXFLAGS -fstack-clash-protection"
    fi
    ARCH=$(uname -m)

    if [ "$ARCH" == "x86_64" ] && [ "$GCC_VERSION" -ge 9 ]; then
        CFLAGS="$CFLAGS -fcf-protection"
        CXXFLAGS="$CXXFLAGS -fcf-protection"
    fi
    export CFLAGS
    export CXXFLAGS
    ;;
   *)
    exit 1
esac
mkdir -p $INSTALL_PREFIX
echo "Installing protobuf ..."
protobuf_url=$(grep '^protobuf' $DEP_FILE_PATH | cut -d ';' -f 2 )
if [[ "$protobuf_url" = https* ]]; then
  protobuf_url=$(echo $protobuf_url | sed 's/\.zip$/\.tar.gz/')
  curl -sSL --retry 5 --retry-delay 10 --create-dirs --fail -L -o protobuf_src.tar.gz $protobuf_url
  mkdir protobuf
  cd protobuf
  tar -zxf ../protobuf_src.tar.gz --strip=1
else
  cp $protobuf_url protobuf_src.zip
  unzip protobuf_src.zip
  cd protobuf-*
fi

cmake . -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX -DCMAKE_POSITION_INDEPENDENT_CODE=ON -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release -Dprotobuf_WITH_ZLIB_DEFAULT=OFF -Dprotobuf_BUILD_SHARED_LIBS=OFF $EXTRA_CMAKE_ARGS
make -j$(getconf _NPROCESSORS_ONLN)
make install
cd ..