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
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import onnx


def export_to_onnx(model_id: str, output_path: str) -> None:
    """
    Downloads a PyTorch model and exports it to ONNX format.
    
    Args:
        model_id: Hugging Face model identifier.
        output_path: Path where the exported ONNX model will be saved.
    """
    print(f"Loading pre-trained model and tokenizer for: '{model_id}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
    except Exception as e:
        print(f"Error loading model from Hugging Face: {e}", file=sys.stderr)
        sys.exit(1)

    # Set model to evaluation mode for inference tracing
    model.eval()

    # Create dummy inputs for tracking/tracing.
    # We trace with a batch of 1 and a short sequence of 5 tokens.
    dummy_text = "Hello ONNX Runtime!"
    inputs = tokenizer(dummy_text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    print(f"Exporting model to ONNX format at '{output_path}'...")
    
    # Configure dynamic axes to support dynamic batch sizes and sequence lengths
    # which is critical for flexible real-world text inference.
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size"}
    }

    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Export the model
        torch.onnx.export(
            model,
            (input_ids, attention_mask),
            output_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=14,  # Opset 14 is widely supported across edge platforms
            do_constant_folding=True
        )
        print("Model exported successfully.")
    except Exception as e:
        print(f"Error during ONNX export: {e}", file=sys.stderr)
        sys.exit(1)

    # Verify the model structure
    print("Verifying exported ONNX model structure...")
    try:
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification succeeded. Model is ready for deployment!")
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
