#!/bin/bash

set -ex

export LIBOPENCL_SO_PATH="/usr/local/cuda-11.1/targets/x86_64-linux/lib/libOpenCL.so"
export ORT_DEFAULT_MAX_VLOG_LEVEL=100

./build/Linux/Release/onnxruntime_perf_test -e opencl -r 4 $@
