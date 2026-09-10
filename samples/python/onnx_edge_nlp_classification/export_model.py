#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Script to download a pre-trained text classification model from Hugging Face
and export it to the ONNX format using PyTorch's ONNX exporter.
"""

import argparse
import os
import sys
import shutil
import tempfile
import onnx
from optimum.exporters.onnx import main_export


def export_to_onnx(model_id: str, output_path: str) -> None:
    """
    Downloads a sequence classification model from Hugging Face and exports it to ONNX
    using Hugging Face Optimum. This ensures a clean, validated schema for quantization.
    
    Args:
        model_id: Hugging Face model identifier.
        output_path: Path where the exported ONNX model will be saved.
    """
    print(f"Loading and exporting model '{model_id}' to ONNX format...")
    temp_dir = tempfile.mkdtemp()
    try:
        # Exporter task is set to 'text-classification'
        main_export(
            model_name_or_path=model_id,
            output=temp_dir,
            task="text-classification"
        )
        
        # Check if model.onnx was generated
        exported_file = os.path.join(temp_dir, "model.onnx")
        if not os.path.exists(exported_file):
            files = [f for f in os.listdir(temp_dir) if f.endswith(".onnx")]
            if files:
                exported_file = os.path.join(temp_dir, files[0])
            else:
                raise FileNotFoundError("Could not find any exported .onnx model.")
                
        # Copy the exported model to the final destination
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        shutil.copy(exported_file, output_path)
        
        # Copy separate weights data file if generated
        data_file = exported_file + ".data"
        if os.path.exists(data_file):
            shutil.copy(data_file, output_path + ".data")
            
        print(f"Model exported successfully and saved to: '{output_path}'")
    except Exception as e:
        print(f"Error during ONNX export: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        shutil.rmtree(temp_dir)


    # Verify the model structure
    print("Verifying exported ONNX model structure...")
    try:
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification succeeded. Model is ready for deployment/quantization!")
    except Exception as e:
        print(f"ONNX model verification failed: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Download and export sequence classification models to ONNX for edge inference."
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="bhadresh-savani/distilbert-base-uncased-emotion",
        help="Hugging Face model ID to export (default: bhadresh-savani/distilbert-base-uncased-emotion)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="model.onnx",
        help="Path where the exported model will be saved (default: model.onnx)"
    )
    args = parser.parse_args()
    
    export_to_onnx(args.model_id, args.output_path)


if __name__ == "__main__":
    main()
