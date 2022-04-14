TMP_DIR=/tmp/onnxruntime

rm -rf $TMP_DIR
mkdir -p $TMP_DIR
cp -r . $TMP_DIR
cd $TMP_DIR
pwd
ls

sh scripts/amd/build.sh
