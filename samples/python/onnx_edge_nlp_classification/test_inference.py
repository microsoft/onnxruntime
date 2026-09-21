#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Unit tests validating output correctness of the ONNX Runtime text classification example.
"""

import os
import pytest
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from inference import softmax, EMOTION_LABELS

MODEL_PATH = "model.onnx"
TOKENIZER_ID = "bhadresh-savani/distilbert-base-uncased-emotion"


@pytest.fixture(scope="module")
def session_and_tokenizer():
    """Initializes the tokenizer and ONNX Runtime session for the test run."""
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"Model file '{MODEL_PATH}' not found. Please run export_model.py before testing.")
        
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    return session, tokenizer


def test_model_file_exists():
    """Verifies that the ONNX model file exists."""
    assert os.path.exists(MODEL_PATH), "ONNX model file 'model.onnx' must be created prior to testing."


def test_inference_execution(session_and_tokenizer):
    """Verifies that basic inference executes successfully and outputs the expected dimensions."""
    session, tokenizer = session_and_tokenizer
    
    text = "Test message for verification"
    inputs = tokenizer(text, truncation=True)
    input_ids = np.array([inputs["input_ids"]], dtype=np.int64)
    attention_mask = np.array([inputs["attention_mask"]], dtype=np.int64)
    
    outputs = session.run(
        ["logits"],
        {"input_ids": input_ids, "attention_mask": attention_mask}
    )
    
    assert len(outputs) == 1, "Model must return exactly one output (logits)."
    logits = outputs[0]
    assert logits.ndim == 2, "Logits tensor should have exactly two dimensions (batch_size, num_classes)."
    assert logits.shape[0] == 1, "Logits batch size should be 1."
    assert logits.shape[1] == len(EMOTION_LABELS), f"Logits should have {len(EMOTION_LABELS)} class features."


def test_emotion_classification_joy(session_and_tokenizer):
    """Verifies that a happy sentence returns 'joy' as the primary classification."""
    session, tokenizer = session_and_tokenizer
    
    text = "I am so happy and filled with joy today!"
    inputs = tokenizer(text, truncation=True)
    input_ids = np.array([inputs["input_ids"]], dtype=np.int64)
    attention_mask = np.array([inputs["attention_mask"]], dtype=np.int64)
    
    outputs = session.run(
        ["logits"],
        {"input_ids": input_ids, "attention_mask": attention_mask}
    )
    probabilities = softmax(outputs[0])[0]
    
    max_idx = np.argmax(probabilities)
    predicted_label = EMOTION_LABELS[max_idx]
    
    assert predicted_label == "joy", f"Expected predicted label 'joy', but got '{predicted_label}'."
    assert probabilities[max_idx] > 0.5, f"Confidence score for 'joy' should be > 50%%, but got {probabilities[max_idx]*100:.1f}%%."


def test_emotion_classification_sadness(session_and_tokenizer):
    """Verifies that a sad sentence returns 'sadness' as the primary classification."""
    session, tokenizer = session_and_tokenizer
    
    text = "I feel so lonely, sad, and depressed."
    inputs = tokenizer(text, truncation=True)
    input_ids = np.array([inputs["input_ids"]], dtype=np.int64)
    attention_mask = np.array([inputs["attention_mask"]], dtype=np.int64)
    
    outputs = session.run(
        ["logits"],
        {"input_ids": input_ids, "attention_mask": attention_mask}
    )
    probabilities = softmax(outputs[0])[0]
    
    max_idx = np.argmax(probabilities)
    predicted_label = EMOTION_LABELS[max_idx]
    
    assert predicted_label == "sadness", f"Expected predicted label 'sadness', but got '{predicted_label}'."
    assert probabilities[max_idx] > 0.5, f"Confidence score for 'sadness' should be > 50%%, but got {probabilities[max_idx]*100:.1f}%%."
