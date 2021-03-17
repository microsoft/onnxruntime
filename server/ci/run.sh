#!/bin/bash
set -e

cd /build
cmake -DCMAKE_BUILD_TYPE=Debug /onnxruntime_src
make -j$(getconf _NPROCESSORS_ONLN)
cd /onnxruntime_src/test
/build/onnxruntime_server_tests
cd /build
python3 server_test/test_main.py /build/onnxruntime_server /build/models/opset8/test_mnist /onnxruntime_src/test/testdata/server /build /build/server_test
