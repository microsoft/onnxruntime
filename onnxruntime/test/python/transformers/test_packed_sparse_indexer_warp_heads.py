# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from onnx import TensorProto, helper

import onnxruntime as ort


def run(reference, artifacts, dtype):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    cases = 0
    for capacity, live, ties in [
        (2051, 2051, False),
        (2051, 513, True),
        (65536, 2048, False),
        (65536, 0, True),
        (2051, 31, False),
        (2051, 32, False),
        (2051, 33, False),
        (2051, 2048, True),
        (65536, 32768, False),
        (65536, 32767, True),
    ]:
        rng = np.random.default_rng(481)
        batch, tokens, channels = 2, 130, 128
        past = np.array([live * 4, max(0, live - 1) * 4], dtype=np.int32)
        feed = {
            "query": (np.zeros((tokens, 512)) if ties else rng.normal(size=(tokens, 512))).astype(dtype),
            "key": rng.normal(size=(tokens, channels)).astype(dtype),
            "qn": np.ones(channels, dtype=dtype),
            "kn": np.ones(channels, dtype=dtype),
            "cos": np.ones((32, 64), dtype=dtype),
            "sin": np.zeros((32, 64), dtype=dtype),
            "cumulative": np.array([0, 65, 130], dtype=np.int32),
            "past": past,
            "positions": np.array([0, 0], dtype=np.int64),
            "state": rng.normal(size=(batch, capacity, channels)).astype(dtype),
            "buffer": np.zeros((batch, 7, channels), dtype=dtype),
            "lengths": np.array([[live, 0], [max(0, live - 1), 0]], dtype=np.int32),
        }
        # Position IDs bound RoPE accesses but also set visibility; use identity caches
        # large enough to represent the actual causal positions for both requests.
        feed.pop("positions")
        feed["cos"] = np.ones((max(int(past.max()) + 66, 32), 64), dtype=dtype)
        feed["sin"] = np.zeros_like(feed["cos"])
        names = [
            "query",
            "key",
            "qn",
            "kn",
            "cos",
            "sin",
            "cumulative",
            "past",
            "",
            "",
            "",
            "",
            "state",
            "buffer",
            "",
            "lengths",
        ]
        tensor_type = TensorProto.FLOAT16 if dtype == np.float16 else TensorProto.FLOAT
        inputs = [
            helper.make_tensor_value_info(n, tensor_type if a.dtype == dtype else TensorProto.INT32, a.shape)
            for n, a in feed.items()
        ]
        output_names = ["indices", "counts", "next_state", "next_buffer", "", "next_lengths"]
        outputs = [
            helper.make_tensor_value_info("indices", TensorProto.INT32, [tokens, 2051]),
            helper.make_tensor_value_info("counts", TensorProto.INT32, [tokens]),
            helper.make_tensor_value_info("next_state", tensor_type, [batch, capacity, channels]),
            helper.make_tensor_value_info("next_buffer", tensor_type, [batch, 7, channels]),
            helper.make_tensor_value_info("next_lengths", TensorProto.INT32, [batch, 2]),
        ]
        node = helper.make_node(
            "PackedSparseAttentionIndexer",
            names,
            output_names,
            domain="com.microsoft",
            policy_mode="qsa",
            compress_ratio=4,
            state_capacity=capacity,
            token_budget=2048,
        )
        model = helper.make_model(
            helper.make_graph([node], "indexer_topk", inputs, outputs),
            opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
        )
        model.ir_version = 10
        session = ort.InferenceSession(model.SerializeToString(), options, providers=["CUDAExecutionProvider"])
        assert session.get_providers()[0] == "CUDAExecutionProvider"
        actual = session.run(None, feed)
        path = artifacts / f"indexer_topk_reference_{cases}.npz"
        if reference:
            np.savez(path, *actual)
        else:
            with np.load(path) as expected:
                for i, output in enumerate(actual):
                    np.testing.assert_array_equal(output, expected[f"arr_{i}"])
        print(f"case={cases} capacity={capacity} live={live} ties={ties} counts={actual[1].tolist()}", flush=True)
        cases += 1
    print(f"{cases} cases: {'saved baseline' if reference else 'all outputs bitwise equal to baseline'}")


@pytest.mark.skipif("CUDAExecutionProvider" not in ort.get_available_providers(), reason="CUDA required")
@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_warp_heads_matches_existing_prefill(tmp_path, dtype):
    for flag in ("0", "1"):
        environment = dict(os.environ, ORT_PACKED_SPARSE_INDEXER_WARP_HEADS=flag)
        command = [sys.executable, str(Path(__file__).resolve()), "--artifacts", str(tmp_path), "--dtype", dtype]
        if flag == "0":
            command.append("--reference")
        subprocess.run(command, env=environment, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--dtype", choices=["float16", "float32"], required=True)
    args = parser.parse_args()
    run(args.reference, args.artifacts, np.dtype(args.dtype).type)
