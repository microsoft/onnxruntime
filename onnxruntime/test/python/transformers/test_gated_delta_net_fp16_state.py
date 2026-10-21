# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import unittest
from unittest.mock import patch

import numpy as np
from onnx import TensorProto, helper
from parameterized import parameterized

import onnxruntime as ort


def make_session(k=128, v=32, *, state_type=TensorProto.FLOAT16, capture=8, uniform=False):
    inputs = [
        helper.make_tensor_value_info(name, TensorProto.FLOAT16, ["tokens", 1, width])
        for name, width in (("q", k), ("k", k), ("v", v))
    ] + [
        helper.make_tensor_value_info("cu", TensorProto.INT32, [2]),
        helper.make_tensor_value_info("decay", TensorProto.FLOAT, ["tokens", 1]),
        helper.make_tensor_value_info("beta", TensorProto.FLOAT, ["tokens", 1]),
        helper.make_tensor_value_info("state", state_type, [1, 1, v, k]),
    ]
    names = ["q", "k", "v", "cu", "decay", "beta", "state"]
    if uniform:
        inputs = [value for value in inputs if value.name != "cu"]
        names[3] = ""
    outputs = [
        helper.make_tensor_value_info("output", TensorProto.FLOAT16, ["tokens", 1, v]),
        helper.make_tensor_value_info("final", state_type, [1, 1, v, k]),
    ]
    if capture:
        names += ["", "", "capture"]
        inputs.append(helper.make_tensor_value_info("capture", TensorProto.INT32, [1]))
        outputs.append(helper.make_tensor_value_info("capsule", TensorProto.FLOAT, [1, capture * (1 + k + v)]))
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "GatedDeltaNet",
                    names,
                    [value.name for value in outputs],
                    domain="com.microsoft",
                    state_update_capacity=capture,
                    scale=1.0,
                )
            ],
            "fp16-state",
            inputs,
            outputs,
        ),
        opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
    )
    model.ir_version = 10
    options = ort.SessionOptions()
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    return ort.InferenceSession(model.SerializeToString(), options, providers=["CUDAExecutionProvider"])


def feed(rows, k=128, v=32, *, capture=8):
    rng = np.random.default_rng(297)
    result = {
        name: rng.normal(0, 0.05, (rows, 1, width)).astype(np.float16) for name, width in (("q", k), ("k", k), ("v", v))
    }
    result.update(
        cu=np.array([0, rows], np.int32),
        decay=np.full((rows, 1), -0.03, np.float32),
        beta=np.full((rows, 1), 0.4, np.float32),
        state=rng.normal(0, 0.05, (1, 1, v, k)).astype(np.float16),
    )
    if capture:
        result["capture"] = np.array([min(rows, capture)], np.int32)
    return result


@unittest.skipUnless("CUDAExecutionProvider" in ort.get_available_providers(), "CUDA required")
class TestGatedDeltaNetFp16State(unittest.TestCase):
    @parameterized.expand([("recurrent", 7), ("chunked", 1024), ("chunked_split", 1024)])
    def test_prefill_engines_allow_inplace_fp16_state(self, engine, rows):
        with patch.dict(os.environ, {"ORT_GDN_PLAN": engine}):
            session = make_session(v=128, capture=0, uniform=True)
        inputs = feed(rows, v=128, capture=0)
        del inputs["cu"]
        expected, final = session.run(None, inputs)
        binding = session.io_binding()
        values = {name: ort.OrtValue.ortvalue_from_numpy(value, "cuda", 0) for name, value in inputs.items()}
        for name, value in values.items():
            binding.bind_ortvalue_input(name, value)
        binding.bind_output("output", "cuda", 0)
        binding.bind_ortvalue_output("final", values["state"])
        session.run_with_iobinding(binding)
        output, state = binding.copy_outputs_to_cpu()
        np.testing.assert_array_equal(output, expected)
        np.testing.assert_array_equal(state, final)
        self.assertTrue(np.isfinite(state).all())

    @parameterized.expand([(128, 32, 1), (128, 32, 2), (128, 32, 7), (24, 13, 7)])
    def test_multi_token_matches_sequential_and_capsule_replay(self, k, v, rows):
        session = make_session(k, v)
        inputs = feed(rows, k, v)
        actual, final, capsule = session.run(None, inputs)
        self.assertEqual(final.dtype, np.float16)
        self.assertEqual(capsule.dtype, np.float32)
        state = inputs["state"].copy()
        outputs = []
        for t in range(rows):
            step = {
                name: value[t : t + 1] for name, value in inputs.items() if name in ("q", "k", "v", "decay", "beta")
            }
            step.update(state=state, cu=np.array([0, 1], np.int32), capture=np.array([1], np.int32))
            output, state, _ = session.run(None, step)
            outputs.append(output)
        np.testing.assert_array_equal(final, state)
        np.testing.assert_array_equal(actual, np.concatenate(outputs))
        # FP32 transition factors are replayed with one FP16 store per token.
        decay = capsule[0, :8]
        keys = capsule[0, 8 : 8 + 8 * k].reshape(8, k)
        delta = capsule[0, 8 + 8 * k :].reshape(8, v)
        state = inputs["state"][0, 0].copy()
        for t in range(rows):
            decayed = state.astype(np.float32) * decay[t]
            state = (
                (
                    keys[t].astype(np.float64)[None, :] * delta[t].astype(np.float64)[:, None]
                    + decayed.astype(np.float64)
                )
                .astype(np.float32)
                .astype(np.float16)
            )
        np.testing.assert_array_equal(final[0, 0], state)

    @parameterized.expand([(1,), (1024,), (32768,)])
    def test_finite_and_close_to_fp32(self, rows):
        inputs = feed(rows, capture=0)
        actual, state = make_session(capture=0).run(None, inputs)
        inputs["state"] = inputs["state"].astype(np.float32)
        expected, reference = make_session(state_type=TensorProto.FLOAT, capture=0).run(None, inputs)
        self.assertTrue(np.isfinite(actual).all())
        self.assertTrue(np.isfinite(state).all())
        np.testing.assert_allclose(actual, expected, atol=0.001, rtol=0.03)
        np.testing.assert_allclose(state, reference, atol=0.001, rtol=0.03)


if __name__ == "__main__":
    unittest.main()
