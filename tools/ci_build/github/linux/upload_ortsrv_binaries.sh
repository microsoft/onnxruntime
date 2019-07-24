#!/bin/bash
set -e -o -x

while getopts a:r:i:c:p:b: parameter_Option
do case "${parameter_Option}"
in
a) AZCOPY_DIR=${OPTARG};;
r) BINARY_DIR=${OPTARG};;
i) BUILD_ID=${OPTARG};;
c) LAST_COMMIT_ID=${OPTARG};;
p) BUILD_PARAMETERS=${OPTARG};;
b) BLOB_SAS_URL=${OPTARG};;
esac
done

echo ""
echo "ad=$AZCOPY_DIR bd=$BINARY_DIR bi=$BUILD_ID lci=$LAST_COMMIT_ID bc=$BUILD_PARAMETERS bsu=$BLOB_SAS_URL"

echo ""
echo "Creating temp folder $BINARY_DIR/$BUILD_ID ... "
mkdir $BINARY_DIR/$BUILD_ID
cp $BINARY_DIR/onnxruntime_server $BINARY_DIR/$BUILD_ID
cp $BINARY_DIR/onnxruntime_server.symbol $BINARY_DIR/$BUILD_ID

echo "Create build info file ..."
echo "Build parameters: $BUILD_PARAMETERS" >> $BINARY_DIR/$BUILD_ID/build_info.txt
echo "Last commit id: $LAST_COMMIT_ID" >> $BINARY_DIR/$BUILD_ID/build_info.txt

echo "Upload the folder to blob storage ..."
$AZCOPY_DIR/azcopy cp $BINARY_DIR/$BUILD_ID $BLOB_SAS_URL --recursive=true

echo "Done!"