#!/bin/bash
set -e -x

cd $BUILD_BINARIESDIRECTORY
files=(whl/*.whl)
FILE_NAME="${files[0]}"
FILE_NAME=$(basename $FILE_NAME)
PYTHON_PACKAGE_NAME=$(echo "$FILE_NAME" | cut -f 1 -d '-')

echo "Package name:$PYTHON_PACKAGE_NAME"

# We assume the machine doesn't have gcc and python development header files
sudo rm -f /build /onnxruntime_src
sudo ln -s $BUILD_SOURCESDIRECTORY /onnxruntime_src
python3 -m pip uninstall -y $$PYTHON_PACKAGE_NAME ort-nightly-gpu ort-nightly onnxruntime onnxruntime-gpu onnxruntime-training onnxruntime-directml ort-nightly-directml onnx -qq
cp $BUILD_SOURCESDIRECTORY/tools/ci_build/github/linux/docker/scripts/manylinux/requirements.txt $BUILD_BINARIESDIRECTORY/requirements.txt
# Test ORT with the latest ONNX release.
sed -i "s/git+http:\/\/github\.com\/onnx\/onnx.*/onnx/" $BUILD_BINARIESDIRECTORY/requirements.txt
python3 -m pip install -r $BUILD_BINARIESDIRECTORY/requirements.txt
python3 -m pip install --find-links $BUILD_BINARIESDIRECTORY/whl $PYTHON_PACKAGE_NAME
ln -s /data/models $BUILD_BINARIESDIRECTORY
cd $BUILD_BINARIESDIRECTORY/Release
# Restore file permissions
xargs -a $BUILD_BINARIESDIRECTORY/Release/perms.txt chmod a+x
python3 $BUILD_SOURCESDIRECTORY/tools/ci_build/build.py \
  --build_dir $BUILD_BINARIESDIRECTORY \
  --config Release --test \
  --skip_submodule_sync \
  --parallel \
  --build_wheel \
  --enable_onnx_tests \
  --enable_pybind --ctest_path ''