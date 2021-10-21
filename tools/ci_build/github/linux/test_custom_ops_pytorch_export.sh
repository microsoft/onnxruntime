#!/bin/bash

pip3 install --user --upgrade pip

pip3 install --user numpy==1.19.0 torch pytest
pip3 install --user /build/Release/dist/*.whl

export PYTHONPATH=/onnxruntime_src/tools:/usr/local/lib/python3.8/site-packages:$PYTHONPATH

python3 -m pytest -v /onnxruntime_src/tools/test/test_custom_ops_pytorch_exporter.py || exit 1

for filename in /onnxruntime_src/onnxruntime/test/python/contrib_ops/onnx_test_* ; do
  cd /build/Release && python3 -m pytest -v $filename || exit 1
done

cd /build/Release && ./onnxruntime_test_all --gtest_filter=ShapeInferenceTests.* || exit 1
