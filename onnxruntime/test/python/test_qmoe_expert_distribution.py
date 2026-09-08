# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import onnx

_TOOLS_PYTHON = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "tools", "python"))
if _TOOLS_PYTHON not in sys.path:
    sys.path.insert(0, _TOOLS_PYTHON)

from qmoe_expert_distribution import calculate_qmoe_expert_bytes, iter_routing_events  # noqa: E402


class TestQMoEExpertDistribution(unittest.TestCase):
    def test_prompt_marker_ignores_unrelated_fraction(self):
        event = {
            "node_name": "/layers.0/qmoe",
            "expert_ids": [1],
            "num_rows": 1,
            "top_k": 1,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "routing.log"
            log_path.write_text(
                f"unrelated diagnostic: 9/10\n[qmoe_prompt_runner] 2/3 prompt_start\nmoe_routing {json.dumps(event)}\n",
                encoding="utf-8",
            )

            self.assertEqual(list(iter_routing_events(log_path)), [(2, event)])

    def test_initializer_size_without_external_length(self):
        float_weight = onnx.helper.make_tensor("float_weight", onnx.TensorProto.FLOAT16, [2, 4], [0.0] * 8)
        packed_weight = onnx.helper.make_tensor("packed_weight", onnx.TensorProto.UINT4, [2, 4], [0] * 8)
        node = onnx.helper.make_node(
            "QMoE",
            ["input", "router", "float_weight", "", "", "packed_weight"],
            ["output"],
            name="/layers.0/qmoe",
        )

        expert_bytes = calculate_qmoe_expert_bytes(
            {weight.name: weight for weight in (float_weight, packed_weight)},
            {node.name: node},
            [node.name],
            num_experts=2,
        )

        self.assertEqual(expert_bytes, {node.name: 10})


if __name__ == "__main__":
    unittest.main()
