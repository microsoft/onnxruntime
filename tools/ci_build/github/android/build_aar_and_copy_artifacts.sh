#!/bin/bash

# This script will run build_aar_package.py to build Android AAR and copy all the artifacts
# to a given folder for publishing to Maven Central, or building nuget package
# This script is intended to be used in CI build only

set -e
set -x
export PATH=/opt/python/cp312-cp312/bin:$PATH

ls /build
ls /build/deps

# User inputs
USE_QNN=${1:-0}  # by default qnn will not be included in package

# build the AAR package, using the build settings under /home/onnxruntimedev/.build_settings/
# if there is also include_ops_and_types.config exists in the same folder, use it to build with included ops/types

BUILD_SCRIPT="/onnxruntime_src/tools/ci_build/github/android/build_aar_package.py"
BUILD_SETTINGS="/home/onnxruntimedev/.build_settings/build_settings.json"
INCLUDE_OPS_CONFIG="/home/onnxruntimedev/.build_settings/include_ops_and_types.config"

ANDROID_SDK_HOME="/android_home"
ANDROID_NDK_HOME="/ndk_home"
QNN_HOME="/qnn_home"


# Base command for building the AAR package
COMMAND="python3 $BUILD_SCRIPT --build_dir /build --config $BUILD_CONFIG --android_sdk_path $ANDROID_SDK_HOME --android_ndk_path $ANDROID_NDK_HOME $BUILD_SETTINGS"

# Check if the include ops config file exists and modify command if it does
if [ -f "$INCLUDE_OPS_CONFIG" ]; then
    COMMAND+=" --include_ops_by_config $INCLUDE_OPS_CONFIG"
fi

# Add qnn path to command
if [ "$USE_QNN" == "1" ]; then
    if [ -d "$QNN_HOME" ]; then
        COMMAND+=" --qnn_path $QNN_HOME"
    else
        echo "Error: QNN directory does not exist."
        exit 1
    fi
fi

# Execute the build command
eval $COMMAND

# Copy the built artifacts to give folder for publishing
BASE_PATH=/build/aar_out/${BUILD_CONFIG}/com/microsoft/onnxruntime/${PACKAGE_NAME}/${ORT_VERSION}${RELEASE_VERSION_SUFFIX}
cp ${BASE_PATH}/${PACKAGE_NAME}-${ORT_VERSION}${RELEASE_VERSION_SUFFIX}-javadoc.jar  /home/onnxruntimedev/.artifacts
cp ${BASE_PATH}/${PACKAGE_NAME}-${ORT_VERSION}${RELEASE_VERSION_SUFFIX}-sources.jar  /home/onnxruntimedev/.artifacts
cp ${BASE_PATH}/${PACKAGE_NAME}-${ORT_VERSION}${RELEASE_VERSION_SUFFIX}.aar          /home/onnxruntimedev/.artifacts
cp ${BASE_PATH}/${PACKAGE_NAME}-${ORT_VERSION}${RELEASE_VERSION_SUFFIX}.pom          /home/onnxruntimedev/.artifacts

# Copy executables if necessary
if [ "$PUBLISH_EXECUTABLES" == "1" ]; then
    pushd /build/intermediates/executables/${BUILD_CONFIG}
    tar -czvf /home/onnxruntimedev/.artifacts/${PACKAGE_NAME}-${ORT_VERSION}${RELEASE_VERSION_SUFFIX}-executables.tgz *
    popd
fi
