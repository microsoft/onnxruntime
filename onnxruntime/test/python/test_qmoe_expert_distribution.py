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
    aggregate_rank_thresholds_by_qmoe,
    calculate_qmoe_expert_bytes,
    inference_expert_ids,
    iter_routing_events,
    rank_experts_by_frequency,
    read_distributions,
    write_qmoe_ranked_experts_csv,
)


def _routing_event(expert_ids=None, router_weights=None, num_rows=1, top_k=1):
    expert_ids = [1] if expert_ids is None else expert_ids
    router_weights = [1.0] * len(expert_ids) if router_weights is None else router_weights
    return {
        "node_index": 17,
        "node_type": "QMoE",
        "node_name": "/layers.0/qmoe",
        "expert_ids": expert_ids,
        "router_weights": router_weights,
        "num_rows": num_rows,
        "top_k": top_k,
    }


def _complete_trace(events_by_prompt):
    prompt_count = len(events_by_prompt)
    lines = []
    routing_record_count = 0
    for prompt_index, events in enumerate(events_by_prompt, start=1):
        lines.append(f"[qmoe_prompt_runner] {prompt_index}/{prompt_count} prompt_start")
        lines.extend(f"moe_routing {json.dumps(event)}" for event in events)
        routing_record_count += len(events)
        lines.append(f"[qmoe_prompt_runner] {prompt_index}/{prompt_count} prompt_end")
    lines.append(
        "moe_routing_complete "
        + json.dumps(
            {
                "prompts": prompt_count,
                "prompt_runs": prompt_count,
                "routing_records": routing_record_count,
            }
        )
    )
    return "\n".join(lines) + "\n"


def _set_external_data(initializer, **values):
    initializer.data_location = onnx.TensorProto.EXTERNAL
    del initializer.external_data[:]
    for key, value in values.items():
        entry = initializer.external_data.add()
        entry.key = key
        entry.value = str(value)


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
        event = _routing_event()
        warning = (
            '[W:onnxruntime:, op_kernel_context_internal.h:114] moe_routing_truncated {"dropped_records":1,'
            '"dropped_routing_elements":2,"max_records_per_run":10000,"max_routing_elements_per_run":1000000}'
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "routing.log"
            for warning_index in range(3):
                with self.subTest(warning_index=warning_index):
                    lines = _complete_trace([[event]]).splitlines()
                    lines.insert(warning_index, warning)
                    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, f"Incomplete routing trace at line {warning_index + 1}"):
                        read_distributions(log_path, num_experts=2)

    def test_prompt_marker_ignores_unrelated_fraction(self):
        event = _routing_event()
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "routing.log"
            log_path.write_text("unrelated diagnostic: 9/10\n" + _complete_trace([[event]]), encoding="utf-8")

            self.assertEqual(list(iter_routing_events(log_path)), [(1, event)])

    def test_out_of_range_expert_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "routing.log"
            for expert_id in (-1, 2):
                with self.subTest(expert_id=expert_id):
                    event = _routing_event([expert_id])
                    log_path.write_text(_complete_trace([[event]]), encoding="utf-8")
                    with self.assertRaisesRegex(
                        ValueError,
                        f"Trace contains expert ID {expert_id}, but the model has 2 experts",
                    ):
                        read_distributions(log_path, num_experts=2)

    def test_unnamed_nodes_are_distinguished_by_index(self):
        first_event = _routing_event([0])
        first_event.update(node_index=3, node_name="")
        second_event = _routing_event([1])
        second_event.update(node_index=4, node_name="")
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "routing.log"
            log_path.write_text(_complete_trace([[first_event, second_event]]), encoding="utf-8")

            _, by_qmoe, _, _ = read_distributions(log_path, num_experts=2)

        self.assertEqual(by_qmoe[(3, "QMoE", "")], {0: 1})
        self.assertEqual(by_qmoe[(4, "QMoE", "")], {1: 1})

    def test_router_weight_count_is_validated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "routing.log"
            log_path.write_text(
                _complete_trace([[_routing_event([0, 1], [1.0], num_rows=2)]]),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "expected 2 router_weights, got 1"):
                list(iter_routing_events(log_path))

    def test_incomplete_and_nonsequential_prompt_traces_are_rejected(self):
        event = _routing_event()
        traces = {
            "missing prompt_end": (
                f"[qmoe_prompt_runner] 1/1 prompt_start\nmoe_routing {json.dumps(event)}\n",
                "has no prompt_end",
            ),
            "missing footer": (
                f"[qmoe_prompt_runner] 1/1 prompt_start\nmoe_routing {json.dumps(event)}\n"
                "[qmoe_prompt_runner] 1/1 prompt_end\n",
                "missing moe_routing_complete footer",
            ),
            "overlapping prompts": (
                "[qmoe_prompt_runner] 1/2 prompt_start\n[qmoe_prompt_runner] 2/2 prompt_start\n",
                "started before prompt 1 ended",
            ),
            "skipped prompt": (
                "[qmoe_prompt_runner] 2/2 prompt_start\n",
                "Expected prompt 1, got 2",
            ),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "routing.log"
            for name, (trace, expected_error) in traces.items():
                with self.subTest(name=name):
                    log_path.write_text(trace, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, expected_error):
                        list(iter_routing_events(log_path))

    def test_completion_footer_counts_are_validated(self):
        event = _routing_event()
        trace = _complete_trace([[event]]).replace('"routing_records": 1', '"routing_records": 2')
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "routing.log"
            log_path.write_text(trace, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Routing completion footer mismatch"):
                list(iter_routing_events(log_path))

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
            {(0, node.op_type, node.name): node},
            [(0, node.op_type, node.name)],
            num_experts=2,
        )

        self.assertEqual(expert_bytes, {(0, node.op_type, node.name): 10})

    def test_external_data_length_and_file_range_are_validated(self):
        weight = onnx.helper.make_tensor("weight", onnx.TensorProto.FLOAT16, [2, 4], [0.0] * 8)
        node = onnx.helper.make_node("QMoE", ["input", "router", "weight"], ["output"], name="/layers.0/qmoe")

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model.onnx"
            external_path = Path(temp_dir) / "weights.bin"
            external_path.write_bytes(bytes(16))

            _set_external_data(weight, location=external_path.name, length=8)
            with self.assertRaisesRegex(ValueError, "is 8, expected 16"):
                calculate_qmoe_expert_bytes(
                    {weight.name: weight},
                    {(0, node.op_type, node.name): node},
                    [(0, node.op_type, node.name)],
                    2,
                    model_path=model_path,
                )

            _set_external_data(weight, location=external_path.name, offset=4, length=16)
            with self.assertRaisesRegex(ValueError, "exceeds"):
                calculate_qmoe_expert_bytes(
                    {weight.name: weight},
                    {(0, node.op_type, node.name): node},
                    [(0, node.op_type, node.name)],
                    2,
                    model_path=model_path,
                )

            _set_external_data(weight, location=external_path.name, length=-1)
            with self.assertRaisesRegex(ValueError, "Invalid external_data.length"):
                calculate_qmoe_expert_bytes(
                    {weight.name: weight},
                    {(0, node.op_type, node.name): node},
                    [(0, node.op_type, node.name)],
                    2,
                    model_path=model_path,
                )

    def test_ranking_final_row_and_threshold_outputs(self):
        identity = (17, "QMoE", "/layers.0/qmoe")
        rankings = rank_experts_by_frequency({identity: {0: 2, 1: 2, 2: 1}}, 3)
        self.assertEqual(rankings, {identity: [0, 1, 2]})

        event = _routing_event([2, 0, 1, 2], num_rows=2, top_k=2)
        self.assertEqual(inference_expert_ids(event), [1, 2])

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            log_path = temp_path / "routing.log"
            csv_path = temp_path / "ranked.csv"
            log_path.write_text(_complete_trace([[event]]), encoding="utf-8")
            inference_counts, threshold_totals = aggregate_rank_thresholds_by_qmoe(log_path, {identity: [0, 1, 2]}, 3)
            self.assertEqual(inference_counts, {identity: 1})
            self.assertEqual(threshold_totals[identity], [2, 2, 1])

            write_qmoe_ranked_experts_csv(csv_path, rankings)
            self.assertEqual(
                csv_path.read_text(encoding="utf-8").splitlines(),
                [
                    "node_index,node_type,node_name,expert_ids_by_decreasing_frequency",
                    '17,QMoE,/layers.0/qmoe,"[0,1,2]"',
                ],
            )


if __name__ == "__main__":
    unittest.main()
