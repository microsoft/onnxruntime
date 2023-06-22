#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This script is a stub that uses the model conversion script from the util subdirectory.
# We do it this way so we can use relative imports in that script, which makes it easy to include
# in the ORT python package (where it must use relative imports)
from util.convert_onnx_models_to_ort import convert_onnx_models_to_ort, parse_args

if __name__ == "__main__":
    args = parse_args()
    convert_onnx_models_to_ort(
        args.model_path_or_dir,
        output_dir=args.output_dir,
        optimization_styles=args.optimization_style,
        custom_op_library_path=args.custom_op_library,
        target_platform=args.target_platform,
        save_optimized_onnx_model=args.save_optimized_onnx_model,
        allow_conversion_failures=args.allow_conversion_failures,
        enable_type_reduction=args.enable_type_reduction,
    )
