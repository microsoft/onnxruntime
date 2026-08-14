#!/usr/bin/env python3
"""Run a single MatMulNBits (M=1) ONNX model once on CUDAExecutionProvider and dump the
output tensor to a .npy file. Invoked as a fresh subprocess per (shape, mode) so that the
ORT_MATMULNBITS_M1_PERF_VALIDATION_FORCE_COLS_PER_BLOCK env var override (cached once per
process on the C++ side) takes effect cleanly for that single run.
"""
import argparse
import numpy as np
import onnxruntime as ort

parser = argparse.ArgumentParser()
parser.add_argument("model_path")
parser.add_argument("out_npy")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

sess = ort.InferenceSession(args.model_path, providers=["CUDAExecutionProvider"])
(input_meta,) = sess.get_inputs()
shape = [d if isinstance(d, int) else 1 for d in input_meta.shape]
rng = np.random.default_rng(args.seed)
a = (rng.standard_normal(shape) * 0.1).astype(np.float32)
(y,) = sess.run(None, {input_meta.name: a})
np.save(args.out_npy, y)
print(f"ran {args.model_path} -> {args.out_npy} shape={y.shape}")
