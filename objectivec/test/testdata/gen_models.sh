#!/bin/bash

set -e

python ./single_add_gen.py
python -m onnxruntime.tools.convert_onnx_models_to_ort --optimization_level basic .

