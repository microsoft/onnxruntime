#!/bin/bash

set -ex

export ORT_DEFAULT_MAX_VLOG_LEVEL=100

./build/Linux/Debug/onnxruntime_perf_test -e opencl -r 4 $@
