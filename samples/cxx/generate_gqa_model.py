# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Generate a minimal ONNX model containing a single GroupQueryAttention op.

This model exercises the WebGPU flash-attention path with Phi-4-like GQA
parameters (24 Q heads, 8 KV heads, head_size=128).

Inputs  (all fp16):
  query              : (batch, seq_len, num_heads * head_size)      -- Q in BSD
  key                : (batch, seq_len, kv_num_heads * head_size)   -- K in BSD
  value              : (batch, seq_len, kv_num_heads * head_size)   -- V in BSD
  past_key           : (batch, kv_num_heads, max_cache, head_size)  -- BNSH
  past_value         : (batch, kv_num_heads, max_cache, head_size)  -- BNSH
  seqlens_k          : (batch,)                                     -- int32
  total_sequence_length : (1,)                                      -- int32

Outputs (all fp16):
  output             : (batch, seq_len, num_heads * head_size)
  present_key        : (batch, kv_num_heads, max_cache, head_size)  -- BNSH
  present_value      : (batch, kv_num_heads, max_cache, head_size)  -- BNSH

Usage:
  pip install onnx
  python generate_gqa_model.py
"""

import argparse

from onnx import TensorProto, helper, save_model


def make_gqa_model(
    batch_size: int = 1,
    num_heads: int = 24,
    kv_num_heads: int = 8,
    head_size: int = 128,
    max_cache: int = 4096,
    output_path: str = "gqa_model.onnx",
):
    hidden_size = num_heads * head_size
    kv_hidden_size = kv_num_heads * head_size

    # --- Inputs ---
    query = helper.make_tensor_value_info(
        "query", TensorProto.FLOAT16, [batch_size, "seq_len", hidden_size]
    )
    key = helper.make_tensor_value_info(
        "key", TensorProto.FLOAT16, [batch_size, "seq_len", kv_hidden_size]
    )
    value = helper.make_tensor_value_info(
        "value", TensorProto.FLOAT16, [batch_size, "seq_len", kv_hidden_size]
    )
    past_key = helper.make_tensor_value_info(
        "past_key",
        TensorProto.FLOAT16,
        [batch_size, kv_num_heads, max_cache, head_size],
    )
    past_value = helper.make_tensor_value_info(
        "past_value",
        TensorProto.FLOAT16,
        [batch_size, kv_num_heads, max_cache, head_size],
    )
    seqlens_k = helper.make_tensor_value_info(
        "seqlens_k", TensorProto.INT32, [batch_size]
    )
    total_sequence_length = helper.make_tensor_value_info(
        "total_sequence_length", TensorProto.INT32, [1]
    )

    # --- Outputs ---
    output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT16, [batch_size, "seq_len", hidden_size]
    )
    present_key = helper.make_tensor_value_info(
        "present_key",
        TensorProto.FLOAT16,
        [batch_size, kv_num_heads, max_cache, head_size],
    )
    present_value = helper.make_tensor_value_info(
        "present_value",
        TensorProto.FLOAT16,
        [batch_size, kv_num_heads, max_cache, head_size],
    )

    # --- GQA Node ---
    gqa_node = helper.make_node(
        "GroupQueryAttention",
        inputs=[
            "query",            # 0
            "key",              # 1
            "value",            # 2
            "past_key",         # 3
            "past_value",       # 4
            "seqlens_k",        # 5
            "total_sequence_length",  # 6
        ],
        outputs=[
            "output",           # 0
            "present_key",      # 1
            "present_value",    # 2
        ],
        domain="com.microsoft",
        name="gqa",
        num_heads=num_heads,
        kv_num_heads=kv_num_heads,
    )

    # --- Graph & Model ---
    graph = helper.make_graph(
        [gqa_node],
        "gqa_graph",
        [query, key, value, past_key, past_value, seqlens_k, total_sequence_length],
        [output, present_key, present_value],
    )

    # opset 13 for the default domain, opset 1 for com.microsoft
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 13),
            helper.make_opsetid("com.microsoft", 1),
        ],
    )

    save_model(model, output_path)
    print(f"Saved {output_path}")
    print(f"  num_heads={num_heads}, kv_num_heads={kv_num_heads}, head_size={head_size}")
    print(f"  hidden_size={hidden_size}, kv_hidden_size={kv_hidden_size}")
    print(f"  max_cache={max_cache}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate GQA ONNX model")
    parser.add_argument("--num_heads", type=int, default=24)
    parser.add_argument("--kv_num_heads", type=int, default=8)
    parser.add_argument("--head_size", type=int, default=128)
    parser.add_argument("--max_cache", type=int, default=4096)
    parser.add_argument("--output", type=str, default="gqa_model.onnx")
    args = parser.parse_args()

    make_gqa_model(
        num_heads=args.num_heads,
        kv_num_heads=args.kv_num_heads,
        head_size=args.head_size,
        max_cache=args.max_cache,
        output_path=args.output,
    )
