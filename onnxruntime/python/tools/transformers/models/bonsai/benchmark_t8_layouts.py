#!/usr/bin/env python
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import onnx
from benchmark_t8 import checkpoint_paths, load_checkpoint_matrix, options
from onnx import helper, numpy_helper
from safetensors import safe_open
from t8_codec import encode_q2
from transformers import AutoTokenizer

import onnxruntime as ort

VARIANTS = {
    "current": ("current", "1"),
    "tiled": ("tiled", "1"),
    "masks": ("masks", "1"),
    "tiled_reuse2": ("tiled", "2"),
    "masks_reuse2": ("masks", "2"),
}


def create_sessions(model, threads, variants):
    sessions, loads = {}, {}
    for name in variants:
        packing, rows = VARIANTS[name]
        settings = options(threads)
        settings.add_session_config_entry("session.ternary.prepack", packing)
        settings.add_session_config_entry("session.ternary.row_tile", rows)
        start = time.perf_counter()
        sessions[name] = ort.InferenceSession(model, settings, providers=["CPUExecutionProvider"])
        loads[name] = time.perf_counter() - start
    return sessions, loads


def compare_variants(sessions, feeds, warmup, iterations):
    expected = sessions["current"].run(None, feeds)[0]
    if not np.isfinite(expected).all():
        raise ValueError("Non-finite current-kernel output")
    errors = {}
    for name, session in sessions.items():
        actual = session.run(None, feeds)[0]
        if not np.isfinite(actual).all():
            raise ValueError(f"{name}: non-finite output")
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32), err_msg=name)
        errors[name] = float(np.max(np.abs(actual - expected)))
    names = list(sessions)
    samples = {name: [] for name in names}
    for _ in range(warmup):
        for session in sessions.values():
            session.run(None, feeds)
    for iteration in range(iterations):
        offset = iteration % len(names)
        order = names[offset:] + names[:offset]
        if (iteration // len(names)) % 2:
            order.reverse()
        for name in order:
            start = time.perf_counter()
            sessions[name].run(None, feeds)
            samples[name].append(time.perf_counter() - start)
    medians = {name: statistics.median(values) for name, values in samples.items()}
    return {
        "median_ms": {name: value * 1000 for name, value in medians.items()},
        "speedup_over_current": {name: medians["current"] / value for name, value in medians.items()},
        "max_abs_error": errors,
        "bit_identical": True,
        "samples_seconds": samples,
        "warmup": warmup,
        "iterations": iterations,
    }


def matmul_benchmark(args):
    results = []
    with safe_open(args.model_dir / "model.safetensors", framework="np") as weights:
        for path in checkpoint_paths(args.model_dir):
            packed, scales = load_checkpoint_matrix(weights, path)
            n, k = packed.shape[0], packed.shape[1] * 128
            planes, t8_scales = encode_q2(packed, scales)
            units, groups = (n + 4) // 5, k // 128
            for m in args.rows:
                graph = helper.make_graph(
                    [helper.make_node("MatMulTernary", ["A", "B", "S"], ["Y"], domain="com.microsoft", K=k, N=n)],
                    "layout_benchmark",
                    [helper.make_tensor_value_info("A", onnx.TensorProto.FLOAT, [m, k])],
                    [helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [m, n])],
                    [numpy_helper.from_array(planes, "B"), numpy_helper.from_array(t8_scales, "S")],
                )
                model = helper.make_model(
                    graph,
                    opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
                    ir_version=10,
                ).SerializeToString()
                feeds = {"A": np.random.default_rng(7).normal(0, 0.125, (m, k)).astype(np.float32)}
                for threads in args.threads:
                    sessions, loads = create_sessions(model, threads, args.variants)
                    result = compare_variants(sessions, feeds, args.warmup, args.iterations)
                    result.update(
                        {
                            "kind": "matmul",
                            "path": path,
                            "M": m,
                            "N": n,
                            "K": k,
                            "threads": threads,
                            "session_load_seconds": loads,
                            "encoded_plane_bytes": planes.nbytes,
                            "scale_bytes": t8_scales.nbytes,
                            "packed_plane_bytes": {
                                name: units * groups * (40 if VARIANTS[name][0] == "masks" else 32) * 4
                                for name in args.variants
                            },
                        }
                    )
                    results.append(result)
                    print(json.dumps(result), flush=True)
                    del sessions
    return results


def model_benchmark(args):
    if args.model is None:
        raise ValueError("--mode model requires --model")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=False, fix_mistral_regex=True)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    ids = tokenizer(prompt, add_special_tokens=False).input_ids
    results = []
    for threads in args.threads:
        sessions, loads = create_sessions(str(args.model.resolve()), threads, args.variants)
        shape = sessions["current"].get_inputs()[0].shape
        if len(shape) != 2 or not isinstance(shape[1], int) or not 0 < len(ids) <= shape[1]:
            raise ValueError(f"Prompt of {len(ids)} tokens does not fit fixed input shape {shape}")
        length = shape[1]
        input_ids = np.zeros((1, length), dtype=np.int64)
        input_ids[0, : len(ids)] = ids
        mask = np.zeros_like(input_ids)
        mask[0, : len(ids)] = 1
        feeds = {
            "input_ids": input_ids,
            "attention_mask": mask,
            "position_ids": np.arange(length, dtype=np.int64)[None, :],
            "logits_index": np.array([len(ids) - 1], dtype=np.int64),
        }
        result = compare_variants(sessions, feeds, args.warmup, args.iterations)
        result.update(
            {
                "kind": "fixed_window_model",
                "sequence_length": length,
                "prompt_tokens": len(ids),
                "threads": threads,
                "session_load_seconds": loads,
            }
        )
        results.append(result)
        print(json.dumps(result), flush=True)
        del sessions
    return results


def main():
    parser = argparse.ArgumentParser(description="Compare experimental T8 runtime layouts with the unchanged kernel.")
    parser.add_argument("--mode", choices=("matmul", "model"), default="matmul")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--prompt", default="Capital of France? Answer with one word.")
    parser.add_argument("--threads", type=int, nargs="+", default=[8])
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 8, 24])
    parser.add_argument("--variants", choices=tuple(VARIANTS), nargs="+", default=list(VARIANTS))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1 or any(value < 1 for value in args.threads + args.rows):
        raise ValueError("Positive iterations/threads/rows and nonnegative warmup are required")
    if "current" not in args.variants or len(set(args.variants)) != len(args.variants):
        raise ValueError("--variants must include current and must not contain duplicates")
    report = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "onnxruntime": ort.__version__,
        "runtime_path": ort.__file__,
        "model_dir": str(args.model_dir.resolve()),
        "model": str(args.model.resolve()) if args.model else None,
        "variants": {
            name: {"packing": VARIANTS[name][0], "row_tile": int(VARIANTS[name][1])} for name in args.variants
        },
        "provider": "CPUExecutionProvider",
        "activation_dtype": "float32",
        "graph_optimization": "ORT_ENABLE_ALL",
        "intra_op_spinning": False,
        "results": model_benchmark(args) if args.mode == "model" else matmul_benchmark(args),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
