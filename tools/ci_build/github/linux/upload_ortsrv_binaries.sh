#!/bin/bash
set -e -o -x

while getopts r:i:c:p:b: parameter_Option
do case "${parameter_Option}"
in
r) BINARY_DIR=${OPTARG};;
i) BUILD_ID=${OPTARG};;
c) LAST_COMMIT_ID=${OPTARG};;
p) BUILD_PARAMETERS=${OPTARG};;
b) BLOB_SAS_URL=${OPTARG};;
esac
done

echo ""
echo "bd=$BINARY_DIR bi=$BUILD_ID lci=$LAST_COMMIT_ID bc=$BUILD_COMMAND bsu=$BLOB_SAS_URL"

echo ""
echo "Creating temp folder $BINARY_DIR/$BUILD_ID ... "
mkdir $BINARY_DIR/$BUILD_ID
cp $BINARY_DIR/onnxruntime_server $BINARY_DIR/$BUILD_ID

# echo "Split binary and symbol ..."
# cd $BINARY_DIR/$BUILD_ID
# objcopy --only-keep-debug onnxruntime_server onnxruntime_server.symbol
# strip --strip-debug --strip-unneeded onnxruntime_server
# objcopy --add-gnu-debuglink=onnxruntime_server.symbol onnxruntime_server

echo "Create build info file ..."
echo "Build parameters: $BUILD_PARAMETERS" >> build_info.txt
echo "Last commit id: $LAST_COMMIT_ID" >> build_info.txt

echo "Upload the folder to blob storage ..."
azcopy cp $BINARY_DIR/$BUILD_ID $BLOB_SAS_URL --recursive=true

echo "Done!"