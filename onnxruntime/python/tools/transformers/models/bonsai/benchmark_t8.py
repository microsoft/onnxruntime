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
from onnx import TensorProto, helper, numpy_helper
from safetensors import safe_open
from t8_codec import encode_q2
from transformers import AutoTokenizer

import onnxruntime as ort


def options(threads: int) -> ort.SessionOptions:
    result = ort.SessionOptions()
    result.intra_op_num_threads = threads
    result.add_session_config_entry("session.intra_op.allow_spinning", "0")
    result.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    result.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return result


def compare(q2: ort.InferenceSession, t8: ort.InferenceSession, feeds: dict, warmup: int, iterations: int) -> dict:
    sessions = {"q2": q2, "t8": t8}
    outputs = {name: session.run(None, feeds)[0] for name, session in sessions.items()}
    for name, output in outputs.items():
        if not np.isfinite(output).all():
            raise ValueError(f"{name}: non-finite benchmark output")
    np.testing.assert_allclose(outputs["t8"], outputs["q2"], rtol=3e-4, atol=3e-3)
    maximum_error = float(np.max(np.abs(outputs["t8"] - outputs["q2"])))
    samples = {name: [] for name in sessions}
    for _ in range(warmup):
        for session in sessions.values():
            session.run(None, feeds)
    for iteration in range(iterations):
        order = ("q2", "t8") if iteration % 2 == 0 else ("t8", "q2")
        for name in order:
            start = time.perf_counter()
            sessions[name].run(None, feeds)
            samples[name].append(time.perf_counter() - start)
    medians = {name: statistics.median(times) for name, times in samples.items()}
    return {
        "q2_ms": medians["q2"] * 1000,
        "t8_ms": medians["t8"] * 1000,
        "speedup_q2_over_t8": medians["q2"] / medians["t8"],
        "max_abs_error": maximum_error,
        "samples_seconds": samples,
        "warmup": warmup,
        "iterations": iterations,
    }


def model_benchmark(args) -> list[dict]:
    if args.q2 is None or args.t8 is None:
        raise ValueError("Model benchmarks require --q2 and --t8")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=False, fix_mistral_regex=True)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    ids = tokenizer(prompt, add_special_tokens=False).input_ids
    results = []
    for threads in args.threads:
        start = time.perf_counter()
        q2 = ort.InferenceSession(str(args.q2.resolve()), options(threads), providers=["CPUExecutionProvider"])
        q2_load = time.perf_counter() - start
        start = time.perf_counter()
        t8 = ort.InferenceSession(str(args.t8.resolve()), options(threads), providers=["CPUExecutionProvider"])
        t8_load = time.perf_counter() - start
        shape = q2.get_inputs()[0].shape
        if shape != t8.get_inputs()[0].shape or len(shape) != 2 or not isinstance(shape[1], int):
            raise ValueError(f"Expected identical fixed-window input shapes; got {shape}")
        length = shape[1]
        if not ids or len(ids) > length:
            raise ValueError(f"Prompt has {len(ids)} tokens; fixed window is {length}")
        input_ids = np.zeros((1, length), dtype=np.int64)
        input_ids[0, : len(ids)] = ids
        attention_mask = np.zeros_like(input_ids)
        attention_mask[0, : len(ids)] = 1
        feeds = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": np.arange(length, dtype=np.int64)[None, :],
            "logits_index": np.array([len(ids) - 1], dtype=np.int64),
        }
        logits_q2, logits_t8 = (session.run(["logits"], feeds)[0] for session in (q2, t8))
        token_q2, token_t8 = (int(np.argmax(logits[0, 0])) for logits in (logits_q2, logits_t8))
        if token_q2 != token_t8:
            raise ValueError(f"Greedy token mismatch: Q2={token_q2}, T8={token_t8}")
        distributions = []
        for logits in (logits_q2, logits_t8):
            values = logits[0, 0].astype(np.float64)
            shifted = values - np.max(values)
            distributions.append(shifted - np.log(np.exp(shifted).sum()))
        kl = float(np.sum(np.exp(distributions[0]) * (distributions[0] - distributions[1])))
        if not np.isfinite(kl) or kl > 1e-5:
            raise ValueError(f"Logit KL divergence {kl} must be finite and <= 1e-5")
        result = compare(q2, t8, feeds, args.warmup, args.iterations)
        result.update(
            {
                "kind": "fixed_window_model",
                "threads": threads,
                "sequence_length": length,
                "prompt_tokens": len(ids),
                "greedy_token": token_q2,
                "greedy_text": tokenizer.decode([token_q2]),
                "kl_q2_to_t8": kl,
                "q2_load_seconds": q2_load,
                "t8_load_seconds": t8_load,
                "q2_recomputed_steps_per_second": 1000 / result["q2_ms"],
                "t8_recomputed_steps_per_second": 1000 / result["t8_ms"],
            }
        )
        results.append(result)
        print(json.dumps(result), flush=True)
        del q2, t8
    return results


def checkpoint_paths(model_dir: Path) -> list[str]:
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    head = "model.embed_tokens" if config["text_config"]["tie_word_embeddings"] else "lm_head"
    return [
        "model.layers.0.mlp.up_proj",
        "model.layers.0.mlp.down_proj",
        "model.layers.0.linear_attn.in_proj_qkv",
        head,
    ]


def load_checkpoint_matrix(weights, path: str) -> tuple[np.ndarray, np.ndarray]:
    prefix = "language_model." + path
    words = weights.get_tensor(prefix + ".weight")
    n, k = words.shape[0], words.shape[1] * 16
    packed = words.view(np.uint8).reshape(n, k // 128, 32)
    scales = weights.get_tensor(prefix + ".scales").astype(np.float32)
    biases = weights.get_tensor(prefix + ".biases").astype(np.float32)
    if not np.array_equal(biases, -scales):
        raise ValueError(f"{path}: biases must equal -scales")
    return packed, scales


def matmul_benchmark(args) -> list[dict]:
    results = []
    with safe_open(args.model_dir / "model.safetensors", framework="np") as weights:
        for path in checkpoint_paths(args.model_dir):
            packed, scales = load_checkpoint_matrix(weights, path)
            n, k = packed.shape[0], packed.shape[1] * 128
            planes, t8_scales = encode_q2(packed, scales)
            zeros = np.full((n, (k // 128 + 3) // 4), 0x55, dtype=np.uint8)
            q2_node = helper.make_node(
                "MatMulNBits",
                ["A", "B", "S", "Z"],
                ["Y"],
                domain="com.microsoft",
                K=k,
                N=n,
                bits=2,
                block_size=128,
                accuracy_level=0,
            )
            t8_node = helper.make_node("MatMulTernary", ["A", "B", "S"], ["Y"], domain="com.microsoft", K=k, N=n)
            for m in args.rows:
                input_info = helper.make_tensor_value_info("A", TensorProto.FLOAT, [m, k])
                output_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [m, n])
                models = []
                for node, tensors in (
                    (q2_node, ((packed, "B"), (scales, "S"), (zeros, "Z"))),
                    (t8_node, ((planes, "B"), (t8_scales, "S"))),
                ):
                    graph = helper.make_graph(
                        [node],
                        "matmul",
                        [input_info],
                        [output_info],
                        [numpy_helper.from_array(value, name) for value, name in tensors],
                    )
                    models.append(
                        helper.make_model(
                            graph,
                            opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
                            ir_version=10,
                        ).SerializeToString()
                    )
                feeds = {"A": np.random.default_rng(7).normal(0, 0.125, (m, k)).astype(np.float32)}
                for threads in args.threads:
                    q2, t8 = (
                        ort.InferenceSession(model, options(threads), providers=["CPUExecutionProvider"])
                        for model in models
                    )
                    result = compare(q2, t8, feeds, args.warmup, args.iterations)
                    result.update(
                        {
                            "kind": "matmul",
                            "path": path,
                            "M": m,
                            "N": n,
                            "K": k,
                            "threads": threads,
                            "q2_accuracy_level": 0,
                            "q2_weight_bytes": packed.nbytes + scales.nbytes + zeros.nbytes,
                            "t8_weight_bytes": planes.nbytes + t8_scales.nbytes,
                        }
                    )
                    results.append(result)
                    print(json.dumps(result), flush=True)
                    del q2, t8
                del models
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark native CPU T8 against the existing Bonsai ONNX Q2 format.")
    parser.add_argument("--mode", choices=("model", "matmul"), default="model")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--q2", type=Path)
    parser.add_argument("--t8", type=Path)
    parser.add_argument("--prompt", default="Capital of France? Answer with one word.")
    parser.add_argument("--threads", type=int, nargs="+", default=[8])
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 8, 24])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1 or any(value < 1 for value in args.threads + args.rows):
        raise ValueError("Iterations, thread counts and row counts must be positive; warmup must be nonnegative")
    report = {
        "machine": platform.machine(),
        "platform": platform.platform(),
        "onnxruntime": ort.__version__,
        "runtime_path": ort.__file__,
        "model_dir": str(args.model_dir.resolve()),
        "provider": "CPUExecutionProvider",
        "graph_optimization": "ORT_ENABLE_ALL",
        "activation_dtype": "float32",
        "intra_op_spinning": False,
        "results": (model_benchmark(args) if args.mode == "model" else matmul_benchmark(args)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
