#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Inference script running text classification using ONNX Runtime.
Tokenizes text inputs, runs the ONNX model session, and computes classification scores.
"""

import argparse
import os
import sys
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

# Default label mapping for the bhadresh-savani/distilbert-base-uncased-emotion model
EMOTION_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]


def softmax(x: np.ndarray) -> np.ndarray:
    """Computes softmax probabilities for raw output logits."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def run_inference(model_path: str, tokenizer_id: str, text: str) -> None:
    """
    Loads the ONNX model, tokenizes the input text, and runs inference.
    
    Args:
        model_path: Path to the exported ONNX model.
        tokenizer_id: Hugging Face model ID to load the tokenizer.
        text: Input string to classify.
    """
    if not os.path.exists(model_path):
        print(f"Error: ONNX model file not found at '{model_path}'. Please run export_model.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading ONNX Runtime InferenceSession for '{model_path}'...")
    try:
        # Load the session with default CPU execution provider for maximum edge compatibility
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    except Exception as e:
        print(f"Error initializing ONNX Runtime session: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading tokenizer: '{tokenizer_id}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    except Exception as e:
        print(f"Error loading tokenizer: {e}", file=sys.stderr)
        sys.exit(1)

    # Tokenize input text. Truncation and padding are enabled to match model inputs.
    inputs = tokenizer(text, truncation=True, max_length=128)
    
    # Format inputs into numpy arrays with the batch dimension (shape: [1, seq_length])
    input_ids = np.array([inputs["input_ids"]], dtype=np.int64)
    attention_mask = np.array([inputs["attention_mask"]], dtype=np.int64)

    # Run the ONNX session
    print("\nRunning inference...")
    try:
        outputs = session.run(
            ["logits"],
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
        )
        logits = outputs[0]
    except Exception as e:
        print(f"Inference execution failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Convert raw logits to probabilities
    probabilities = softmax(logits)[0]

    # Map labels to probabilities
    scores = {label: float(prob) for label, prob in zip(EMOTION_LABELS, probabilities)}

    # Dynamic Intent & Stress Estimation heuristics (CIRVA context matching example)
    stress_score = scores.get("sadness", 0.0) * 0.6 + scores.get("fear", 0.0) * 0.4 + scores.get("anger", 0.0) * 0.3
    if "stressed" in text.lower() or "anxious" in text.lower():
        stress_score = max(stress_score, 0.75)
        
    neutral_score = max(0.0, 1.0 - sum(scores.values()) + scores.get("surprise", 0.0) * 0.2)
    if not any(word in text.lower() for word in ["stressed", "sad", "happy", "angry", "afraid"]):
        neutral_score = max(neutral_score, 0.5)

    # Determine conversational intent category
    intent = "sharing"
    if "?" in text:
        intent = "question"
    elif any(cmd in text.lower() for cmd in ["tell", "do", "call", "send", "show"]):
        intent = "command"

    # Display results
    print(f"\nInput:\n\"{text}\"")
    print("\nOutput:")
    print("Emotion scores:")
    for emotion, score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
        print(f"- {emotion}: {score:.2f}")
    
    print(f"- stress: {stress_score:.2f}")
    print(f"- neutral: {neutral_score:.2f}")
    
    print("\nIntent:")
    print(f"- {intent}")


def main():
    parser = argparse.ArgumentParser(description="Run edge text classification inference using ONNX Runtime.")
    parser.add_argument(
        "--model",
        type=str,
        default="model.onnx",
        help="Path to the ONNX model file (default: model.onnx)"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="bhadresh-savani/distilbert-base-uncased-emotion",
        help="Tokenizer model ID from Hugging Face (default: bhadresh-savani/distilbert-base-uncased-emotion)"
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Input text to classify. If omitted, you will be prompted interactively."
    )
    args = parser.parse_args()

    text = args.text
    if not text:
        # Prompt interactively if no command line text is supplied
        try:
            text = input("Enter text to classify: ")
        except KeyboardInterrupt:
            print("\nExiting.")
            sys.exit(0)

    run_inference(args.model, args.tokenizer, text)


if __name__ == "__main__":
    main()
