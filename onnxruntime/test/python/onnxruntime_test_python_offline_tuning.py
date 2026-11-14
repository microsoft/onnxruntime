# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest

import onnx
from onnx import TensorProto, helper

from onnxruntime.tools import offline_tuning


def _make_model() -> onnx.ModelProto:
    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"])],
        "offline_tuning_test",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    return helper.make_model(graph)


def _make_tuning_results(validator_value: str, kernel_id: int) -> list[offline_tuning.TuningResults]:
    return [
        {
            "ep": "ROCMExecutionProvider",
            "validators": {"version": validator_value},
            "results": {"op_sig": {"params_sig": kernel_id}},
        }
    ]


class TestOfflineTuningEmbed(unittest.TestCase):
    def test_embed_requires_overwrite_when_results_exist(self):
        # The tuning results are the first (idx 0) and only metadata entry here, which is the usual case.
        model = _make_model()
        original = _make_tuning_results("1", 1)
        offline_tuning.embed(model, original)

        with self.assertRaises(ValueError):
            offline_tuning.embed(model, _make_tuning_results("2", 2))
        self.assertEqual(offline_tuning.extract(model), original)

    def test_embed_with_overwrite_replaces_results(self):
        model = _make_model()
        offline_tuning.embed(model, _make_tuning_results("1", 1))
        replacement = _make_tuning_results("2", 2)
        offline_tuning.embed(model, replacement, overwrite=True)

        self.assertEqual(offline_tuning.extract(model), replacement)
        self.assertEqual(sum(prop.key == "tuning_results" for prop in model.metadata_props), 1)


if __name__ == "__main__":
    unittest.main()
