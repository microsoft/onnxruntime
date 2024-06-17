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



EXTRA_CMAKE_ARGS="-DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_CXX_STANDARD=17"

case "$(uname -s)" in
   Darwin*)
     echo 'Building ONNX Runtime on Mac OS X'
     EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DCMAKE_OSX_ARCHITECTURES=x86_64;arm64"
     GCC_PATH=$(which clang)
     GPLUSPLUS_PATH=$(which clang++)
     ;;
   Linux*)
     SYS_LONG_BIT=$(getconf LONG_BIT)
     DISTRIBUTOR=$(lsb_release -i -s)

     if [[ ("$DISTRIBUTOR" = "CentOS" || "$DISTRIBUTOR" = "RedHatEnterprise") && $SYS_LONG_BIT = "64" ]]; then
       LIBDIR="lib64"
     else
       LIBDIR="lib"
     fi
     EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -DCMAKE_INSTALL_LIBDIR=$LIBDIR"
     # Depending on how the compiler has been configured when it was built, sometimes "gcc -dumpversion" shows the full version.
     GCC_VERSION=$(gcc -dumpversion | cut -d . -f 1)
     #-fstack-clash-protection prevents attacks based on an overlapping heap and stack.
     if [ "$GCC_VERSION" -ge 8 ]; then
        CFLAGS="$CFLAGS -fstack-clash-protection"
        CXXFLAGS="$CXXFLAGS -fstack-clash-protection"
     fi
     ARCH=$(uname -m)
     GCC_PATH=$(which gcc)
     GPLUSPLUS_PATH=$(which g++)
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
mkdir -p "$INSTALL_PREFIX"

if [ -x "$(command -v ninja)" ]; then
  EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS -G Ninja"
fi
echo "Installing abseil ..."
pushd .
absl_url=$(grep '^abseil_cpp' "$DEP_FILE_PATH" | cut -d ';' -f 2 )
if [[ "$absl_url" = https* ]]; then
  absl_url=$(echo $absl_url | sed 's/\.zip$/\.tar.gz/')
  curl -sSL --retry 5 --retry-delay 10 --create-dirs --fail -L -o absl_src.tar.gz $absl_url
  mkdir abseil
  cd abseil
  tar -zxf ../absl_src.tar.gz --strip=1
else
  cp $absl_url absl_src.zip
  unzip absl_src.zip
  cd */
fi

CC=$GCC_PATH CXX=$GPLUSPLUS_PATH  cmake "."  "-DABSL_PROPAGATE_CXX_STD=ON" "-DCMAKE_BUILD_TYPE=Release" "-DBUILD_TESTING=OFF" "-DABSL_USE_EXTERNAL_GOOGLETEST=ON" "-DCMAKE_PREFIX_PATH=$INSTALL_PREFIX" "-DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX" $EXTRA_CMAKE_ARGS
if [ -x "$(command -v ninja)" ]; then
  ninja
  ninja install
else
  make -j$(getconf _NPROCESSORS_ONLN)
  make install
fi
popd

pushd .
echo "Installing protobuf ..."
protobuf_url=$(grep '^protobuf' $DEP_FILE_PATH | cut -d ';' -f 2 )
if [[ "$protobuf_url" = https* ]]; then
  protobuf_url=$(echo "$protobuf_url" | sed 's/\.zip$/\.tar.gz/')
  curl -sSL --retry 5 --retry-delay 10 --create-dirs --fail -L -o protobuf_src.tar.gz "$protobuf_url"
  mkdir protobuf
  cd protobuf
  tar -zxf ../protobuf_src.tar.gz --strip=1
else
  cp $protobuf_url protobuf_src.zip
  unzip protobuf_src.zip
  cd protobuf-*
fi

CC=$GCC_PATH CXX=$GPLUSPLUS_PATH cmake . "-DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX" -DCMAKE_POSITION_INDEPENDENT_CODE=ON -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release -Dprotobuf_WITH_ZLIB_DEFAULT=OFF -Dprotobuf_BUILD_SHARED_LIBS=OFF "-DCMAKE_PREFIX_PATH=$INSTALL_PREFIX" $EXTRA_CMAKE_ARGS -Dprotobuf_ABSL_PROVIDER=package
if [ -x "$(command -v ninja)" ]; then
  ninja
  ninja install
else
  make -j$(getconf _NPROCESSORS_ONLN)
  make install
fi
popd
