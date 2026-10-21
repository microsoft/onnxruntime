#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Script to apply dynamic 8-bit integer quantization (QuantType.QUInt8)
to the exported ONNX model, optimizing it for CPU/Edge memory footprints.
"""

import argparse
import os
import sys
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType


def quantize_onnx_model(input_path: str, output_path: str) -> None:
    """
    Applies dynamic quantization to an ONNX model.
    
    Args:
        input_path: Path to the source unquantized ONNX model.
        output_path: Path where the quantized ONNX model will be saved.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input model file not found at '{input_path}'. Please run export_model.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"Quantizing model '{input_path}'...")
    print("Applying dynamic 8-bit quantization (weight_type=QUInt8)...")
    
    try:
        # Dynamic quantization is the recommended approach for transformer/NLP models.
        # It quantizes the weights to 8-bit integers, reducing size by ~4x while maintaining high accuracy.
        quantize_dynamic(
            model_input=input_path,
            model_output=output_path,
            weight_type=QuantType.QUInt8
        )
        print(f"Quantization complete. Saved compressed model to '{output_path}'")
    except Exception as e:
        print(f"Quantization failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Verify the quantized model structure
    print("Verifying quantized model structure...")
    try:
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        orig_size_mb = os.path.getsize(input_path) / (1024 * 1024)
        quant_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        compression_ratio = orig_size_mb / quant_size_mb
        
        print(f"Quantized model verification succeeded!")
        print(f"Original Model Size:  {orig_size_mb:.2f} MB")
        print(f"Quantized Model Size: {quant_size_mb:.2f} MB (~{compression_ratio:.1f}x compression)")
    except Exception as e:
        print(f"Quantized model verification failed: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Apply dynamic 8-bit quantization to ONNX models.")
    parser.add_argument(
        "--input",
        type=str,
        default="model.onnx",
        help="Path to the input unquantized ONNX model (default: model.onnx)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="model_quantized.onnx",
        help="Path to the output quantized ONNX model (default: model_quantized.onnx)"
    )
    args = parser.parse_args()

    quantize_onnx_model(args.input, args.output)


if __name__ == "__main__":
    main()
