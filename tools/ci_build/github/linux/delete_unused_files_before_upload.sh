#!/bin/bash
set -e -x
rm -rf $BUILD_BINARIESDIRECTORY/Release/onnxruntime $BUILD_BINARIESDIRECTORY/Release/pybind11
rm -f $BUILD_BINARIESDIRECTORY/Release/models
rm -rf $BUILD_BINARIESDIRECTORY/Release/vcpkg_installed
find $BUILD_BINARIESDIRECTORY/Release/_deps -mindepth 1 ! -regex "^$BUILD_BINARIESDIRECTORY/Release/_deps/onnx-src\(/.*\)?" -delete
cd $BUILD_BINARIESDIRECTORY/Release
find -executable -type f > $BUILD_BINARIESDIRECTORY/Release/perms.txt
