#!/bin/bash
cd /build
cmake -DCMAKE_BUILD_TYPE=Debug /onnxruntime_src
make -j$(getconf _NPROCESSORS_ONLN)
cd /onnxruntime_src/test
../b/onnxruntime_server_tests