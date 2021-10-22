#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This script is a stub that uses the model conversion script from the util subdirectory.
# We do it this way so we can use relative imports in that script, which makes it easy to include
# in the ORT python package (where it must use relative imports)
from util.convert_onnx_models_to_ort import convert_onnx_models_to_ort


if __name__ == '__main__':
    convert_onnx_models_to_ort()
