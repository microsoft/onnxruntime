#!/bin/bash

# This script will run build_aar_package.py to build Android AAR and copy all the artifacts
# to a given folder for publishing to Maven Central, or building nuget package
# This script is intended to be used in CI build only

set -e
set -x
export PATH=/opt/python/cp37-cp37m/bin:$PATH

# build the AAR package, using the build settings under /home/onnxruntimedev/.build_settings/
# if there is also include_ops_and_types.config exists in the same folder, use it to build with included ops/types
if [ -f "/home/onnxruntimedev/.build_settings/include_ops_and_types.config" ]; then
    python3 /onnxruntime_src/tools/ci_build/github/android/build_aar_package.py \
        --build_dir /build \
        --config $BUILD_CONFIG \
        --android_sdk_path /android_home \
        --android_ndk_path /ndk_home \
        --include_ops_by_config /home/onnxruntimedev/.build_settings/include_ops_and_types.config \
        /home/onnxruntimedev/.build_settings/build_settings.json
else
    python3 /onnxruntime_src/tools/ci_build/github/android/build_aar_package.py \
        --build_dir /build \
        --config $BUILD_CONFIG \
        --android_sdk_path /android_home \
        --android_ndk_path /ndk_home \
        /home/onnxruntimedev/.build_settings/build_settings.json
fi

# Copy the built artifacts to give folder for publishing
BASE_PATH=/build/aar_out/${BUILD_CONFIG}/com/microsoft/onnxruntime/${PACKAGE_NAME}/${ORT_VERSION}
cp ${BASE_PATH}/${PACKAGE_NAME}-${ORT_VERSION}-javadoc.jar  /home/onnxruntimedev/.artifacts
cp ${BASE_PATH}/${PACKAGE_NAME}-${ORT_VERSION}-sources.jar  /home/onnxruntimedev/.artifacts
cp ${BASE_PATH}/${PACKAGE_NAME}-${ORT_VERSION}.aar          /home/onnxruntimedev/.artifacts
cp ${BASE_PATH}/${PACKAGE_NAME}-${ORT_VERSION}.pom          /home/onnxruntimedev/.artifacts

# Copy executables if necessary
if [ "$PUBLISH_EXECUTABLES" == "1" ]; then
    pushd /build/intermediates/executables/${BUILD_CONFIG}
    tar -czvf /home/onnxruntimedev/.artifacts/${PACKAGE_NAME}-${ORT_VERSION}-executables.tgz *
    popd
fi
