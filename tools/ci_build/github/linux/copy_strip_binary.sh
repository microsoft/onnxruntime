#!/bin/bash
set -e -o -x

while getopts r:a:l:c:s:t: parameter_Option
do case "${parameter_Option}"
in
r) BINARY_DIR=${OPTARG};;
a) ARTIFACT_NAME=${OPTARG};;
l) LIB_NAME=${OPTARG};;
c) BUILD_CONFIG=${OPTARG};;
s) SOURCE_DIR=${OPTARG};;
t) COMMIT_ID=${OPTARG};;
esac
done

EXIT_CODE=1

uname -a
cd "$BINARY_DIR"
mv installed/usr/local $ARTIFACT_NAME
mv $ARTIFACT_NAME/include/onnxruntime/* $ARTIFACT_NAME/include
rmdir $ARTIFACT_NAME/include/onnxruntime
# Do not ship onnx_test_runner
rm -rf $ARTIFACT_NAME/bin
echo "Copy debug symbols in a separate file and strip the original binary."
if [[ $LIB_NAME == *.dylib ]]
then
    dsymutil $BINARY_DIR/$ARTIFACT_NAME/lib/$LIB_NAME -o $BINARY_DIR/$ARTIFACT_NAME/lib/$LIB_NAME.dSYM
    strip -S $BINARY_DIR/$ARTIFACT_NAME/lib/$LIB_NAME

    # ORT NuGet packaging expects the unversioned library (libonnxruntime.dylib) to contain the binary content,
    # because the versioned library is excluded by the nuspec generation script.
    # We explicitly overwrite the symlink with the real file to ensure 'nuget pack' (especially on Windows)
    # doesn't pack an empty/broken symlink.
    # Only applies to versioned libonnxruntime libraries (e.g. libonnxruntime.1.24.0.dylib).
    if [[ "$LIB_NAME" =~ ^libonnxruntime\..*\.dylib$ && -L "$BINARY_DIR/$ARTIFACT_NAME/lib/libonnxruntime.dylib" ]]; then
       rm "$BINARY_DIR/$ARTIFACT_NAME/lib/libonnxruntime.dylib"
       cp "$BINARY_DIR/$ARTIFACT_NAME/lib/$LIB_NAME" "$BINARY_DIR/$ARTIFACT_NAME/lib/libonnxruntime.dylib"
    fi

    # copy the CoreML EP header for macOS build (libs with .dylib ext)
    cp $SOURCE_DIR/include/onnxruntime/core/providers/coreml/coreml_provider_factory.h  $BINARY_DIR/$ARTIFACT_NAME/include
else
   # Linux
   mv $ARTIFACT_NAME/lib64 $ARTIFACT_NAME/lib
fi

# copy the README, licence and TPN
cp $SOURCE_DIR/README.md $BINARY_DIR/$ARTIFACT_NAME/README.md
cp $SOURCE_DIR/docs/Privacy.md $BINARY_DIR/$ARTIFACT_NAME/Privacy.md
cp $SOURCE_DIR/LICENSE $BINARY_DIR/$ARTIFACT_NAME/LICENSE
cp $SOURCE_DIR/ThirdPartyNotices.txt $BINARY_DIR/$ARTIFACT_NAME/ThirdPartyNotices.txt
cp $SOURCE_DIR/VERSION_NUMBER $BINARY_DIR/$ARTIFACT_NAME/VERSION_NUMBER


echo $COMMIT_ID > $BINARY_DIR/$ARTIFACT_NAME/GIT_COMMIT_ID

EXIT_CODE=$?

set -e
exit $EXIT_CODE
