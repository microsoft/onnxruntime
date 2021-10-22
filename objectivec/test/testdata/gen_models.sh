#!/bin/bash

set -e

# Get directory this script is in
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd ${DIR}

python3 ./single_add_gen.py
python3 -m onnxruntime.tools.convert_onnx_models_to_ort --optimization_level basic .

