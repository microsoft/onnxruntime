# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""CUDA regression tests for appending to a non-shared FP8 GQA cache."""

import unittest

import numpy as np
import torch
from onnx import TensorProto, helper, numpy_helper

import onnxruntime as ort


@unittest.skipUnless(
    "CUDAExecutionProvider" in ort.get_available_providers()
    and torch.cuda.is_available()
    and torch.cuda.get_device_capability() >= (8, 9),
    "FP8 CUDA device required",
)
class TestGrowingFp8GqaCache(unittest.TestCase):
    def _check_append(self, head_size):
        ort.preload_dlls()
        heads, kv_heads = 32, 2
        info = helper.make_tensor_value_info
        inputs = [
            info("query", TensorProto.FLOAT16, [1, "sequence", heads * head_size]),
            info("key", TensorProto.FLOAT16, [1, "sequence", kv_heads * head_size]),
            info("value", TensorProto.FLOAT16, [1, "sequence", kv_heads * head_size]),
            info("past_key", TensorProto.FLOAT8E4M3FN, [1, kv_heads, "past", head_size]),
            info("past_value", TensorProto.FLOAT8E4M3FN, [1, kv_heads, "past", head_size]),
            info("seqlens", TensorProto.INT32, [1]),
            info("total_length", TensorProto.INT32, [1]),
        ]
        cache_shape = [1, kv_heads, "total", head_size]
        outputs = [
            info("output", TensorProto.FLOAT16, [1, "sequence", heads * head_size]),
            info("present_key", TensorProto.FLOAT8E4M3FN, cache_shape),
            info("present_value", TensorProto.FLOAT8E4M3FN, cache_shape),
            info("key_float", TensorProto.FLOAT, cache_shape),
            info("value_float", TensorProto.FLOAT, cache_shape),
        ]
        nodes = [
            helper.make_node(
                "GroupQueryAttention",
                [
                    "query",
                    "key",
                    "value",
                    "past_key",
                    "past_value",
                    "seqlens",
                    "total_length",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "scale",
                    "scale",
                ],
                ["output", "present_key", "present_value"],
                domain="com.microsoft",
                num_heads=heads,
                kv_num_heads=kv_heads,
                kv_cache_bit_width=8,
                k_quant_type="PER_TENSOR",
                v_quant_type="PER_TENSOR",
            ),
            helper.make_node("Cast", ["present_key"], ["key_float"], to=TensorProto.FLOAT),
            helper.make_node("Cast", ["present_value"], ["value_float"], to=TensorProto.FLOAT),
        ]
        graph = helper.make_graph(
            nodes,
            "growing-fp8-gqa-cache",
            inputs,
            outputs,
            [numpy_helper.from_array(np.array([1.0], dtype=np.float32), "scale")],
        )
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 23), helper.make_opsetid("com.microsoft", 1)],
            ir_version=10,
        )
        options = ort.SessionOptions()
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        session = ort.InferenceSession(
            model.SerializeToString(), sess_options=options, providers=["CUDAExecutionProvider"]
        )
        states = {
            name: ort.OrtValue.ortvalue_from_shape_and_type(
                [1, kv_heads, 0, head_size], TensorProto.FLOAT8E4M3FN, "cuda", 0
            )
            for name in ("past_key", "past_value")
        }
        rng = np.random.default_rng(17)
        total = 0
        reference_caches = [np.empty((1, kv_heads, 0, head_size), dtype=np.float32) for _ in range(2)]
        for sequence in (2, 1, 1, 3):
            binding = session.io_binding()
            values = {}
            for name, count in (("query", heads), ("key", kv_heads), ("value", kv_heads)):
                array = (rng.standard_normal((1, sequence, count * head_size)) * 0.2).astype(np.float16)
                values[name] = array
                binding.bind_ortvalue_input(name, ort.OrtValue.ortvalue_from_numpy(array, "cuda", 0))
            for name, value in states.items():
                binding.bind_ortvalue_input(name, value)
            total += sequence
            binding.bind_cpu_input("seqlens", np.array([total - 1], dtype=np.int32))
            binding.bind_cpu_input("total_length", np.array([total], dtype=np.int32))
            for output in outputs:
                binding.bind_output(output.name, "cuda", 0)
            session.run_with_iobinding(binding)
            binding.synchronize_outputs()
            result = dict(zip((value.name for value in outputs), binding.get_outputs(), strict=True))
            for index, name in enumerate(("key", "value")):
                incoming = torch.from_numpy(values[name]).reshape(1, sequence, kv_heads, head_size)
                quantized = incoming.permute(0, 2, 1, 3).to(torch.float8_e4m3fn).float().numpy()
                reference_caches[index] = np.concatenate([reference_caches[index], quantized], axis=2)
                np.testing.assert_array_equal(
                    result[f"{name}_float"].numpy(),
                    reference_caches[index],
                    err_msg=f"head_size={head_size}, total={total}, cache={name}",
                )
                states[f"past_{name}"] = result[f"present_{name}"]
            self.assertTrue(np.isfinite(result["output"].numpy()).all())
            if sequence == 1:
                query = torch.from_numpy(values["query"]).float().reshape(1, 1, heads, head_size)
                query = query.permute(0, 2, 1, 3)
                key, value = [
                    torch.from_numpy(cache).repeat_interleave(heads // kv_heads, dim=1) for cache in reference_caches
                ]
                scores = torch.matmul(query, key.transpose(-1, -2)) * head_size**-0.5
                expected = torch.matmul(scores.softmax(-1), value).permute(0, 2, 1, 3)
                expected = expected.reshape(1, 1, heads * head_size).numpy()
                np.testing.assert_allclose(result["output"].numpy(), expected, rtol=0.02, atol=0.002)

    def test_head_size_128(self):
        self._check_append(128)

    def test_head_size_256(self):
        self._check_append(256)


if __name__ == "__main__":
    unittest.main()
