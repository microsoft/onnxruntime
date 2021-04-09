#!/bin/bash

# This script will run build_aar_package.py to build Android AAR and copy all the artifacts
# to a given folder for publish to Maven Central

set -e
set -x
export PATH=/opt/python/cp37-cp37m/bin:$PATH

# build the AAR package
python3 /onnxruntime_src/tools/ci_build/github/android/build_aar_package.py \
    --build_dir /build \
    --config $BUILD_CONFIG \
    --android_sdk_path /android_home \
    --android_ndk_path /ndk_home \
    /onnxruntime_src/tools/ci_build/github/android/default_mobile_aar_build_settings.json

# Copy the built artifacts to give folder for publishing
PACKAGE_NAME=onnxruntime-mobile
BASE_PATH=/build/aar_out/${BUILD_CONFIG}/com/microsoft/onnxruntime/${PACKAGE_NAME}/${ORT_VERSION}
cp ${BASE_PATH}/${PACKAGE_NAME}-${ORT_VERSION}-javadoc.jar  /home/onnxruntimedev/.artifacts
cp ${BASE_PATH}/${PACKAGE_NAME}-${ORT_VERSION}-sources.jar  /home/onnxruntimedev/.artifacts
cp ${BASE_PATH}/${PACKAGE_NAME}-${ORT_VERSION}.aar          /home/onnxruntimedev/.artifacts
cp ${BASE_PATH}/${PACKAGE_NAME}-${ORT_VERSION}.pom          /home/onnxruntimedev/.artifacts
