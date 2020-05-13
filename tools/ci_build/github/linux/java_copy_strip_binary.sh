#!/bin/bash
set -e -o -x

while getopts r:a:l:n:c:h: parameter_Option
do case "${parameter_Option}"
in
r) BINARY_DIR=${OPTARG};;
a) ARTIFACT_NAME=${OPTARG};;
l) LIB_NAME=${OPTARG};;
n) NATIVE_LIB_NAME=${OPTARG};;
c) BUILD_CONFIG=${OPTARG};;
h) ARCH=${OPTARG};;
esac
done

EXIT_CODE=1

uname -a

VERSION_NUMBER=`cat $SOURCE_DIR/VERSION_NUMBER | xarg`
echo "Version: $VERSION_NUMBER"
NATIVE_FOLDER=ai/onnxruntime/native/$ARCH

mkdir -p $BINARY_DIR/$ARTIFACT_NAME/lib/$NATIVE_FOLDER
mkdir $BINARY_DIR/$ARTIFACT_NAME/jar
echo "Directories created"

# We are only interested in one jar
# Extract to strip symbols
echo "Extract the original jar"
pushd $BINARY_DIR/$ARTIFACT_NAME/jar
jar xf $BINARY_DIR/$BUILD_CONFIG/java/build/libs/onnxruntime-$VERSION_NUMBER.jar
popd

echo "Copy debug symbols in a separate file and strip the original binary."

cp $BINARY_DIR/$ARTIFACT_NAME/jar/$NATIVE_FOLDER/$LIB_NAME $BINARY_DIR/$ARTIFACT_NAME/lib/$NATIVE_FOLDER/$LIB_NAME
cp $BINARY_DIR/$ARTIFACT_NAME/jar/$NATIVE_FOLDER/$NATIVE_LIB_NAME $BINARY_DIR/$ARTIFACT_NAME/lib/$NATIVE_FOLDER/$NATIVE_LIB_NAME

if [[ $LIB_NAME == *.dylib ]]
then
    # ORT LIB
    dsymutil $BINARY_DIR/$ARTIFACT_NAME/jar/$NATIVE_FOLDER/$LIB_NAME -o $BINARY_DIR/$ARTIFACT_NAME/lib/$NATIVE_FOLDER/$LIB_NAME.dSYM
    strip -S $BINARY_DIR/$ARTIFACT_NAME/lib/$NATIVE_FOLDER/$LIB_NAME
    ln -s $LIB_NAME $BINARY_DIR/$ARTIFACT_NAME/lib/$NATIVE_FOLDER/libonnxruntime.dylib
    # JNI Lib
    dsymutil $BINARY_DIR/$ARTIFACT_NAME/jar/$NATIVE_FOLDER/$NATIVE_LIB_NAME -o $BINARY_DIR/$ARTIFACT_NAME/lib/$NATIVE_FOLDER/$NATIVE_LIB_NAME.dSYM
    strip -S $BINARY_DIR/$ARTIFACT_NAME/lib/$NATIVE_FOLDER/$NATIVE_LIB_NAME
    ln -s $NATIVE_LIB_NAME $BINARY_DIR/$ARTIFACT_NAME/lib/$NATIVE_FOLDER/libonnxruntime4j_jni.dylib
elif [[ $LIB_NAME == *.so.* ]]
then
    ln -s $LIB_NAME $BINARY_DIR/$ARTIFACT_NAME/lib/$NATIVE_FOLDER/libonnxruntime.so
    ln -s $NATIVE_LIB_NAME $BINARY_DIR/$ARTIFACT_NAME/lib/$NATIVE_FOLDER/libonnxruntime4j_jni.so
fi

EXIT_CODE=$?

set -e
exit $EXIT_CODE
