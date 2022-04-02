#!/bin/bash

set -e

# Get directory this script is in
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd ${DIR}

python3 ./single_add_gen.py

ORT_CONVERT_ONNX_MODELS_TO_ORT_OPTIMIZATION_LEVEL=basic python3 -m onnxruntime.tools.convert_onnx_models_to_ort --optimization_style=Fixed .
