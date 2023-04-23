#!/bin/bash

set -ex

export KERNEL_EXPLORER_BUILD_DIR=build_rocm/Release

HIP_VISIBLE_DEVICES=0 python onnxruntime/python/tools/kernel_explorer/kernels/gemm_test.py N T float16 65536 2304 768 > 0.log &
sleep 2 && HIP_VISIBLE_DEVICES=1 python onnxruntime/python/tools/kernel_explorer/kernels/gemm_test.py N T float16 65536 2304 768 > 1.log
