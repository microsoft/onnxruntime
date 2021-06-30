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
mkdir $BINARY_DIR/$ARTIFACT_NAME
mkdir $BINARY_DIR/$ARTIFACT_NAME/lib
mkdir $BINARY_DIR/$ARTIFACT_NAME/include
echo "Directories created"
cp $BINARY_DIR/$BUILD_CONFIG/$LIB_NAME $BINARY_DIR/$ARTIFACT_NAME/lib
if [[ -f "$BINARY_DIR/$BUILD_CONFIG/libonnxruntime_providers_cuda.so" ]]; then
    cp $BINARY_DIR/$BUILD_CONFIG/libonnxruntime_providers_shared.so $BINARY_DIR/$ARTIFACT_NAME/lib
    cp $BINARY_DIR/$BUILD_CONFIG/libonnxruntime_providers_cuda.so $BINARY_DIR/$ARTIFACT_NAME/lib
fi
if [[ -f "$BINARY_DIR/$BUILD_CONFIG/libonnxruntime_providers_tensorrt.so" ]]; then
    cp $BINARY_DIR/$BUILD_CONFIG/libonnxruntime_providers_tensorrt.so $BINARY_DIR/$ARTIFACT_NAME/lib
    cp $SOURCE_DIR/include/onnxruntime/core/providers/tensorrt/tensorrt_provider_factory.h  $BINARY_DIR/$ARTIFACT_NAME/include
fi
echo "Copy debug symbols in a separate file and strip the original binary."
if [[ $LIB_NAME == *.dylib ]]
then
    dsymutil $BINARY_DIR/$ARTIFACT_NAME/lib/$LIB_NAME -o $BINARY_DIR/$ARTIFACT_NAME/lib/$LIB_NAME.dSYM
    strip -S $BINARY_DIR/$ARTIFACT_NAME/lib/$LIB_NAME
    ln -s $LIB_NAME $BINARY_DIR/$ARTIFACT_NAME/lib/libonnxruntime.dylib
elif [[ $LIB_NAME == *.so.* ]]
then
    ln -s $LIB_NAME $BINARY_DIR/$ARTIFACT_NAME/lib/libonnxruntime.so
fi
cp $SOURCE_DIR/include/onnxruntime/core/session/onnxruntime_c_api.h  $BINARY_DIR/$ARTIFACT_NAME/include
cp $SOURCE_DIR/include/onnxruntime/core/session/onnxruntime_cxx_api.h  $BINARY_DIR/$ARTIFACT_NAME/include
cp $SOURCE_DIR/include/onnxruntime/core/session/onnxruntime_cxx_inline.h  $BINARY_DIR/$ARTIFACT_NAME/include
cp $SOURCE_DIR/include/onnxruntime/core/providers/cpu/cpu_provider_factory.h  $BINARY_DIR/$ARTIFACT_NAME/include
cp $SOURCE_DIR/include/onnxruntime/core/providers/cuda/cuda_provider_factory.h  $BINARY_DIR/$ARTIFACT_NAME/include
cp $SOURCE_DIR/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h  $BINARY_DIR/$ARTIFACT_NAME/include
cp $SOURCE_DIR/include/onnxruntime/core/session/onnxruntime_run_options_config_keys.h  $BINARY_DIR/$ARTIFACT_NAME/include
cp $SOURCE_DIR/include/onnxruntime/core/framework/provider_options.h  $BINARY_DIR/$ARTIFACT_NAME/include

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
