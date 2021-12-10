#!/bin/bash

set -ex

./build/Linux/Release/onnxruntime_perf_test -e opencl -r 4 $@
