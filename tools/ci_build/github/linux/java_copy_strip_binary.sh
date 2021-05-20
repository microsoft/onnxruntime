#!/bin/bash
set -e -o -x

while getopts r:a:l:n:c:h:v: parameter_Option
do case "${parameter_Option}"
in
r) BINARY_DIR=${OPTARG};;
a) ARTIFACT_NAME=${OPTARG};;
c) BUILD_CONFIG=${OPTARG};;
l) LIB_NAME=${OPTARG};;
n) NATIVE_LIB_NAME=${OPTARG};;
h) ARCH=${OPTARG};;
v) VERSION_NUMBER=${OPTARG};;
esac
done

EXIT_CODE=1

uname -a

echo "Version: $VERSION_NUMBER"
NATIVE_FOLDER=ai/onnxruntime/native/$ARCH

mkdir -p $BINARY_DIR/$ARTIFACT_NAME/$NATIVE_FOLDER

echo "Directories created"

echo "Copy debug symbols in a separate file and strip the original binary."

if [[ $LIB_NAME == *.dylib ]]
then
    # ORT LIB
    dsymutil $BINARY_DIR/$BUILD_CONFIG/$LIB_NAME -o $BINARY_DIR/$ARTIFACT_NAME/$NATIVE_FOLDER/$LIB_NAME.dSYM
    cp $BINARY_DIR/$BUILD_CONFIG/$LIB_NAME $BINARY_DIR/$ARTIFACT_NAME/$NATIVE_FOLDER/libonnxruntime.dylib
    strip -S $BINARY_DIR/$ARTIFACT_NAME/$NATIVE_FOLDER/libonnxruntime.dylib
    # JNI Lib
    dsymutil $BINARY_DIR/$BUILD_CONFIG/$NATIVE_LIB_NAME -o $BINARY_DIR/$ARTIFACT_NAME/$NATIVE_FOLDER/$NATIVE_LIB_NAME.dSYM
    cp $BINARY_DIR/$BUILD_CONFIG/$NATIVE_LIB_NAME $BINARY_DIR/$ARTIFACT_NAME/$NATIVE_FOLDER/libonnxruntime4j_jni.dylib
    strip -S $BINARY_DIR/$ARTIFACT_NAME/$NATIVE_FOLDER/libonnxruntime4j_jni.dylib
    # Add custom lib for testing. This should be added to testing.jar
    cp $BINARY_DIR/$BUILD_CONFIG/libcustom_op_library.dylib $BINARY_DIR/$ARTIFACT_NAME
elif [[ $LIB_NAME == *.so ]]
then
    cp $BINARY_DIR/$BUILD_CONFIG/$LIB_NAME $BINARY_DIR/$ARTIFACT_NAME/$NATIVE_FOLDER/libonnxruntime.so
    cp $BINARY_DIR/$BUILD_CONFIG/$NATIVE_LIB_NAME $BINARY_DIR/$ARTIFACT_NAME/$NATIVE_FOLDER/libonnxruntime4j_jni.so
     # Add custom lib
    cp $BINARY_DIR/$BUILD_CONFIG/libcustom_op_library.so $BINARY_DIR/$ARTIFACT_NAME
    # Add cuda provider if it exists
    cp $BINARY_DIR/$BUILD_CONFIG/$LIB_NAME $BINARY_DIR/$ARTIFACT_NAME/$NATIVE_FOLDER/libonnxruntime_providers_shared.so
    cp $BINARY_DIR/$BUILD_CONFIG/$LIB_NAME $BINARY_DIR/$ARTIFACT_NAME/$NATIVE_FOLDER/libonnxruntime_providers_cuda.so
fi

find $BINARY_DIR/$ARTIFACT_NAME -ls
rm -fr $BINARY_DIR/$ARTIFACT_NAME/jar

EXIT_CODE=$?

set -e
exit $EXIT_CODE
