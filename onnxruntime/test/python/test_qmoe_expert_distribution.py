# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import onnx

_TOOLS_PYTHON = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "tools", "python"))
if _TOOLS_PYTHON not in sys.path:
    sys.path.insert(0, _TOOLS_PYTHON)

from qmoe_expert_distribution import (  # noqa: E402
    calculate_qmoe_expert_bytes,
    iter_routing_events,
    read_distributions,
)


class TestQMoEExpertDistribution(unittest.TestCase):
    def test_analysis_import_does_not_load_matplotlib(self):
        script = "\n".join(
            [
                "import sys",
                "sys.path.insert(0, sys.argv[1])",
                "import qmoe_expert_distribution",
                "assert 'matplotlib' not in sys.modules",
            ]
        )
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                _TOOLS_PYTHON,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_truncated_routing_trace_is_rejected(self):
        event = {"node_name": "/layers.0/qmoe", "expert_ids": [1], "num_rows": 1, "top_k": 1}
        warning = (
            '[W:onnxruntime:, op_kernel_context_internal.h:114] moe_routing_truncated {"dropped_records":1,'
            '"dropped_routing_elements":2,"max_records_per_run":10000,"max_routing_elements_per_run":1000000}'
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "routing.log"
            for warning_index in range(3):
                with self.subTest(warning_index=warning_index):
                    lines = ["[qmoe_prompt_runner] 1/1 prompt_start", f"moe_routing {json.dumps(event)}"]
                    lines.insert(warning_index, warning)
                    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, f"Incomplete routing trace at line {warning_index + 1}"):
                        read_distributions(log_path, num_experts=2)

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
