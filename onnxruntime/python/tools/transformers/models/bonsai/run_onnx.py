#!/usr/bin/env python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

import onnxruntime as ort


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run greedy text generation with a converted Bonsai 2 ONNX model.")
    parser.add_argument("--model", type=Path, required=True, help="Converted ONNX model.")
    parser.add_argument(
        "--model-dir", type=Path, required=True, help="Bonsai source directory containing the tokenizer."
    )
    parser.add_argument("--prompt", required=True, help="User prompt.")
    parser.add_argument("--max-new-tokens", type=int, default=1, help="Maximum number of tokens to generate.")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        trust_remote_code=False,
        fix_mistral_regex=True,
    )
    rendered_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    token_ids = tokenizer(rendered_prompt, add_special_tokens=False).input_ids

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    start = time.perf_counter()
    session = ort.InferenceSession(
        str(args.model.resolve()),
        session_options,
        providers=["CPUExecutionProvider"],
    )
    session_seconds = time.perf_counter() - start

    input_shape = session.get_inputs()[0].shape
    if len(input_shape) != 2 or not isinstance(input_shape[1], int):
        raise ValueError(f"Expected a fixed-rank input_ids shape, got {input_shape}")
    sequence_length = input_shape[1]
    if len(token_ids) + args.max_new_tokens > sequence_length:
        raise ValueError(
            f"Prompt has {len(token_ids)} tokens, leaving room for only "
            f"{sequence_length - len(token_ids)} generated tokens"
        )

    eos_token_ids = set(tokenizer.all_special_ids)
    generated: list[int] = []
    run_seconds = 0.0
    for _ in range(args.max_new_tokens):
        active_ids = token_ids + generated
        input_ids = np.zeros((1, sequence_length), dtype=np.int64)
        input_ids[0, : len(active_ids)] = active_ids
        attention_mask = np.zeros((1, sequence_length), dtype=np.int64)
        attention_mask[0, : len(active_ids)] = 1
        position_ids = np.arange(sequence_length, dtype=np.int64)[None, :]
        logits_index = np.array([len(active_ids) - 1], dtype=np.int64)

        start = time.perf_counter()
        logits = session.run(
            ["logits"],
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "logits_index": logits_index,
            },
        )[0]
        run_seconds += time.perf_counter() - start
        next_token = int(np.argmax(logits[0, 0]))
        if next_token in eos_token_ids:
            break
        generated.append(next_token)

    print(tokenizer.decode(generated), flush=True)
    print(f"session_seconds={session_seconds:.3f}", flush=True)
    print(f"run_seconds={run_seconds:.3f}", flush=True)
    print(f"generated_token_ids={generated}", flush=True)


if __name__ == "__main__":
    main()
