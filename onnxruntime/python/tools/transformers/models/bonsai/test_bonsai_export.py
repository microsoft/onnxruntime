# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import torch
from convert_to_onnx import BonsaiSwishAttention, T8Linear, _finalize_t8_weights, _fwht, _gated_delta_rule
from run_onnx import stop_token_ids
from t8_codec import decode_q2
from torch import nn
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention, torch_chunk_gated_delta_rule

import onnxruntime as ort


class RecurrentProjection(nn.Module):
    def __init__(self, normalize=False):
        super().__init__()
        self.normalize = normalize

    def forward(self, query, key, value, decay, beta):
        return _gated_delta_rule(query, key, value, decay, beta, use_qk_l2norm_in_kernel=self.normalize)[0]


class BonsaiExportTest(unittest.TestCase):
    def test_direct_t8_linear_export_and_execution(self):
        rng = np.random.default_rng(7)
        rows, width = 7, 256
        codes = rng.integers(0, 3, (rows, width), dtype=np.uint8)
        packed = (codes.reshape(rows, -1, 4) << np.array([0, 2, 4, 6], dtype=np.uint8)).sum(axis=-1)
        packed = packed.astype(np.uint8).reshape(rows, width // 128, 32)
        scales = rng.integers(1, 32, (rows, width // 128)).astype(np.float32) / 128
        signs = torch.from_numpy(rng.choice(np.array([-1, 1], dtype=np.float32), width))
        activations = torch.from_numpy(rng.normal(size=(1, 3, width)).astype(np.float32))
        dense = (codes.astype(np.float32).reshape(rows, -1, 128) - 1) * scales[:, :, None]
        for block in (0, 128):
            with self.subTest(block=block), tempfile.TemporaryDirectory() as directory:
                module = T8Linear(torch.from_numpy(packed), torch.from_numpy(scales), signs, block)
                np.testing.assert_array_equal(decode_q2(module.planes.numpy().view(np.uint32), rows), packed)
                output = Path(directory) / "direct-t8.onnx"
                torch.onnx.export(
                    module,
                    (activations,),
                    str(output),
                    input_names=["A"],
                    output_names=["Y"],
                    opset_version=21,
                    custom_opsets={"com.microsoft": 1},
                    dynamo=False,
                )
                exported = onnx.load(output)
                _finalize_t8_weights(exported)
                onnx.save(exported, output)
                onnx.checker.check_model(exported)
                self.assertEqual(sum(node.op_type == "MatMulTernary" for node in exported.graph.node), 1)
                self.assertFalse(any(node.op_type == "MatMulNBits" for node in exported.graph.node))
                options = ort.SessionOptions()
                options.intra_op_num_threads = 2
                session = ort.InferenceSession(str(output), options, providers=["CPUExecutionProvider"])
                actual = session.run(None, {"A": activations.numpy()})[0]
                transformed = _fwht(activations * signs, block) if block else activations
                expected = transformed.numpy() @ dense.reshape(rows, width).T
                np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=3e-5)

    def test_recurrent_nonzero_outputs_match_torch_for_both_head_counts(self):
        torch.manual_seed(7)
        for heads in (16, 48):
            with self.subTest(heads=heads), tempfile.TemporaryDirectory() as directory:
                shape, gate_shape = (1, 4, heads, 128), (1, 4, heads)
                query, key, value = (torch.randn(shape) * 0.1 for _ in range(3))
                decay, beta = -torch.rand(gate_shape), torch.sigmoid(torch.randn(gate_shape))
                inputs = query, key, value, decay, beta
                expected = torch_chunk_gated_delta_rule(*inputs, use_qk_l2norm_in_kernel=True)[0]
                output = Path(directory) / "recurrent.onnx"
                torch.onnx.export(
                    RecurrentProjection(normalize=True),
                    inputs,
                    output,
                    input_names=["query", "key", "value", "decay", "beta"],
                    output_names=["output"],
                    opset_version=21,
                    custom_opsets={"com.microsoft": 1},
                    dynamo=False,
                )
                model = onnx.load(output)
                next(node for node in model.graph.node if node.op_type == "LinearAttention").input[3] = ""
                options = ort.SessionOptions()
                options.intra_op_num_threads = 3
                options.add_session_config_entry("session.intra_op.allow_spinning", "0")
                session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
                feeds = {
                    name: tensor.numpy()
                    for name, tensor in zip(["query", "key", "value", "decay", "beta"], inputs, strict=True)
                }
                actual = torch.from_numpy(session.run(None, feeds)[0])
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)

    def test_recurrent_head_count_for_both_models(self):
        for heads in (16, 48):
            with self.subTest(heads=heads), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "recurrent.onnx"
                qkv = torch.zeros((1, 2, heads, 128))
                gates = torch.zeros((1, 2, heads))
                torch.onnx.export(
                    RecurrentProjection(),
                    (qkv, qkv, qkv, gates, gates),
                    output,
                    opset_version=21,
                    custom_opsets={"com.microsoft": 1},
                    dynamo=False,
                )
                model = onnx.load(output)
                graph = model.graph
                node = next(node for node in graph.node if node.op_type == "LinearAttention")
                attributes = {
                    attribute.name: onnx.helper.get_attribute_value(attribute) for attribute in node.attribute
                }
                self.assertEqual(attributes["q_num_heads"], heads)
                self.assertEqual(attributes["kv_num_heads"], heads)
                node.input[3] = ""
                session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
                feeds = {value.name: np.zeros(value.shape, dtype=np.float32) for value in session.get_inputs()}
                self.assertEqual(session.run(None, feeds)[0].shape, (1, 2, heads, 128))

    def test_27b_swish_gate_preserves_attention_and_changes_only_gate(self):
        config = Qwen3_5TextConfig(
            hidden_size=8,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            num_hidden_layers=1,
            intermediate_size=16,
        )
        config._attn_implementation = "eager"
        torch.manual_seed(7)
        sigmoid = Qwen3_5Attention(config, 0).eval()
        swish = BonsaiSwishAttention(config, 0).eval()
        swish.load_state_dict(sigmoid.state_dict())
        sigmoid.o_proj = swish.o_proj = nn.Identity()
        hidden = torch.randn((1, 3, 8))
        positions = (torch.ones((1, 3, 4)), torch.zeros((1, 3, 4)))
        mask = torch.zeros((1, 1, 3, 3))
        base, base_weights = sigmoid(hidden, positions, mask)
        actual, actual_weights = swish(hidden, positions, mask)
        gate = sigmoid.q_proj(hidden).view(1, 3, 2, 8).chunk(2, dim=-1)[1].reshape(1, 3, 8)
        torch.testing.assert_close(actual, base * gate, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(actual_weights, base_weights, rtol=0, atol=0)


class RunOnnxTest(unittest.TestCase):
    def test_stop_token_ids_use_configured_eos_only(self):
        tokenizer = SimpleNamespace(eos_token_id=7, all_special_ids=[1, 2, 3, 7])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(stop_token_ids(root, tokenizer), {7})
            (root / "config.json").write_text(json.dumps({"text_config": {"eos_token_id": 5}}), encoding="utf-8")
            self.assertEqual(stop_token_ids(root, tokenizer), {5})
            (root / "generation_config.json").write_text(json.dumps({"eos_token_id": [4, 6]}), encoding="utf-8")
            self.assertEqual(stop_token_ids(root, tokenizer), {4, 6})
            with self.assertRaisesRegex(ValueError, "No eos_token_id"):
                stop_token_ids(root / "missing", SimpleNamespace(eos_token_id=None))


if __name__ == "__main__":
    unittest.main()
