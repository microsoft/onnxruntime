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
mkdir $BINARY_DIR/$ARTIFACT_NAME/jar
echo "Directories created"

# We are only interested in one jar
# Extract to strip symbols
echo "Extract the original jar"
pushd $BINARY_DIR/$ARTIFACT_NAME/jar
jar xf $BINARY_DIR/$BUILD_CONFIG/java/build/libs/onnxruntime-$VERSION_NUMBER.jar
ls -laR
popd

echo "Copy debug symbols in a separate file and strip the original binary."

if [[ $LIB_NAME == *.dylib ]]
then
    # ORT LIB
    dsymutil $BINARY_DIR/$ARTIFACT_NAME/jar/$NATIVE_FOLDER/$LIB_NAME -o $BINARY_DIR/$ARTIFACT_NAME/$NATIVE_FOLDER/$LIB_NAME.dSYM
    strip -S $BINARY_DIR/$ARTIFACT_NAME/jar/$NATIVE_FOLDER/$LIB_NAME
    cp $BINARY_DIR/$ARTIFACT_NAME/jar/$NATIVE_FOLDER/$LIB_NAME $BINARY_DIR/$ARTIFACT_NAME/$NATIVE_FOLDER/libonnxruntime.dylib
    # JNI Lib
    dsymutil $BINARY_DIR/$ARTIFACT_NAME/jar/$NATIVE_FOLDER/$NATIVE_LIB_NAME -o $BINARY_DIR/$ARTIFACT_NAME/$NATIVE_FOLDER/$NATIVE_LIB_NAME.dSYM
    strip -S $BINARY_DIR/$ARTIFACT_NAME/jar/$NATIVE_FOLDER/$NATIVE_LIB_NAME
    cp $BINARY_DIR/$ARTIFACT_NAME/jar/$NATIVE_FOLDER/$NATIVE_LIB_NAME $BINARY_DIR/$ARTIFACT_NAME/$NATIVE_FOLDER/libonnxruntime4j_jni.dylib
elif [[ $LIB_NAME == *.so.* ]]
then
    cp $BINARY_DIR/$ARTIFACT_NAME/jar/$NATIVE_FOLDER/$LIB_NAME $BINARY_DIR/$ARTIFACT_NAME/$NATIVE_FOLDER/libonnxruntime.so
    cp $BINARY_DIR/$ARTIFACT_NAME/jar/$NATIVE_FOLDER/$NATIVE_LIB_NAME $BINARY_DIR/$ARTIFACT_NAME/$NATIVE_FOLDER/libonnxruntime4j_jni.so
fi

find $BINARY_DIR/$ARTIFACT_NAME -ls
rm -fr $BINARY_DIR/$ARTIFACT_NAME/jar

EXIT_CODE=$?

set -e
exit $EXIT_CODE
