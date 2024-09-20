#!/bin/sh
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# Directory of the script
dir="$( cd "$( dirname "$0" )" && pwd )"

# Directory of the onnx models
onnx_dir=$dir/sam2_onnx_models

# Directory of the sam2 code by "git clone https://github.com/facebookresearch/segment-anything-2"
sam2_dir=~/segment-anything-2

# model name to benchmark
model=sam2_hiera_large

run_cpu()
{
    repeats=$1

    python3 convert_to_onnx.py  --sam2_dir $sam2_dir --optimize --demo

    echo "Benchmarking SAM2 model $model image encoder for PyTorch ..."
    python3 benchmark_sam2.py --model_type $model --engine torch --sam2_dir $sam2_dir --repeats $repeats --dtype fp32
    python3 benchmark_sam2.py --model_type $model --engine torch --sam2_dir $sam2_dir --repeats $repeats --dtype fp16

    echo "Benchmarking SAM2 model $model image encoder for PyTorch ..."
    python3 benchmark_sam2.py --model_type $model --engine torch --sam2_dir $sam2_dir --repeats $repeats --dtype fp32 --component image_decoder
    python3 benchmark_sam2.py --model_type $model --engine torch --sam2_dir $sam2_dir --repeats $repeats --dtype fp16 --component image_decoder

    echo "Benchmarking SAM2 model $model image encoder for ORT ..."
    python3 benchmark_sam2.py --model_type $model --engine ort --sam2_dir $sam2_dir --repeats $repeats --onnx_path ${onnx_dir}/${model}_image_encoder.onnx --dtype fp32
    python3 benchmark_sam2.py --model_type $model --engine ort --sam2_dir $sam2_dir --repeats $repeats --onnx_path ${onnx_dir}/${model}_image_encoder_fp32_cpu.onnx --dtype fp32

    echo "Benchmarking SAM2 model $model image decoder for ORT ..."
    python3 benchmark_sam2.py --model_type $model --engine ort --sam2_dir $sam2_dir --repeats $repeats --onnx_path ${onnx_dir}/${model}_image_decoder.onnx --component image_decoder
    python3 benchmark_sam2.py --model_type $model --engine ort --sam2_dir $sam2_dir --repeats $repeats --onnx_path ${onnx_dir}/${model}_image_decoder_fp32_cpu.onnx --component image_decoder
}

run_gpu()
{
    repeats=$1

    python3 convert_to_onnx.py  --sam2_dir $sam2_dir --optimize --use_gpu --dtype fp32
    python3 convert_to_onnx.py  --sam2_dir $sam2_dir --optimize --use_gpu --dtype fp16 --demo

    echo "Benchmarking SAM2 model $model image encoder for PyTorch ..."
    python3 benchmark_sam2.py --model_type $model --engine torch --sam2_dir $sam2_dir --repeats $repeats --use_gpu --dtype fp16
    python3 benchmark_sam2.py --model_type $model --engine torch --sam2_dir $sam2_dir --repeats $repeats --use_gpu --dtype bf16
    python3 benchmark_sam2.py --model_type $model --engine torch --sam2_dir $sam2_dir --repeats $repeats --use_gpu --dtype fp32

    echo "Benchmarking SAM2 model $model image decoder for PyTorch ..."
    python3 benchmark_sam2.py --model_type $model --engine torch --sam2_dir $sam2_dir --repeats $repeats --use_gpu --dtype fp16 --component image_decoder
    python3 benchmark_sam2.py --model_type $model --engine torch --sam2_dir $sam2_dir --repeats $repeats --use_gpu --dtype bf16 --component image_decoder
    python3 benchmark_sam2.py --model_type $model --engine torch --sam2_dir $sam2_dir --repeats $repeats --use_gpu --dtype fp32 --component image_decoder

    echo "Benchmarking SAM2 model $model image encoder for ORT ..."
    python3 benchmark_sam2.py --model_type $model --engine ort --sam2_dir $sam2_dir --repeats $repeats --onnx_path ${onnx_dir}/${model}_image_encoder_fp16_gpu.onnx --use_gpu --dtype fp16
    python3 benchmark_sam2.py --model_type $model --engine ort --sam2_dir $sam2_dir --repeats $repeats --onnx_path ${onnx_dir}/${model}_image_encoder_fp32_gpu.onnx --use_gpu --dtype fp32

    echo "Benchmarking SAM2 model $model image decoder for ORT ..."
    python3 benchmark_sam2.py --model_type $model --engine ort --sam2_dir $sam2_dir --repeats $repeats --onnx_path ${onnx_dir}/${model}_image_decoder_fp16_gpu.onnx --component image_decoder --use_gpu --dtype fp16
    python3 benchmark_sam2.py --model_type $model --engine ort --sam2_dir $sam2_dir --repeats $repeats --onnx_path ${onnx_dir}/${model}_image_decoder_fp32_gpu.onnx --component image_decoder --use_gpu
}

if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    run_gpu 1000
else
    run_cpu 100
fi
