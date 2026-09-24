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
    analyze_counter_trace,
    calculate_qmoe_expert_bytes,
    iter_counter_events,
    rank_experts_by_frequency,
    read_distributions,
    selected_expert_ids,
    write_qmoe_ranked_experts_csv,
)


def _counter_event(selected_experts=None, counters=None):
    selected_experts = [1] if selected_experts is None else selected_experts
    counters = [0.0, 1.0] if counters is None else counters
    return {
        "request_id": "",
        "graph_scope": "main",
        "node_index": 17,
        "node_type": "QMoE",
        "node_name": "/layers.0/qmoe",
        "selected_experts": selected_experts,
        "counters": counters,
    }


def _complete_trace(events_by_prompt):
    prompt_count = len(events_by_prompt)
    lines = []
    counter_record_count = 0
    for prompt_index, events in enumerate(events_by_prompt, start=1):
        lines.append(f"[qmoe_prompt_runner] {prompt_index}/{prompt_count} prompt_start")
        lines.extend(f"moe_expert_counters {json.dumps(event)}" for event in events)
        counter_record_count += len(events)
        lines.append(f"[qmoe_prompt_runner] {prompt_index}/{prompt_count} prompt_end")
    lines.append(
        "moe_expert_counters_complete "
        + json.dumps(
            {
                "prompts": prompt_count,
                "prompt_runs": prompt_count,
                "counter_records": counter_record_count,
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

    def test_duplicate_selected_experts_are_rejected(self):
        event = _counter_event([1, 1])
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "counters.log"
            log_path.write_text(_complete_trace([[event]]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "selected_experts must not contain duplicates"):
                read_distributions(log_path, num_experts=2)

    def test_prompt_marker_ignores_unrelated_fraction(self):
        event = _counter_event()
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "routing.log"
            log_path.write_text("unrelated diagnostic: 9/10\n" + _complete_trace([[event]]), encoding="utf-8")

            self.assertEqual(list(iter_counter_events(log_path)), [(1, event)])

    def test_out_of_range_expert_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "routing.log"
            for expert_id in (-1, 2):
                with self.subTest(expert_id=expert_id):
                    event = _counter_event([expert_id])
                    log_path.write_text(_complete_trace([[event]]), encoding="utf-8")
                    with self.assertRaisesRegex(
                        ValueError,
                        f"Trace contains expert ID {expert_id}, but the model has 2 experts",
                    ):
                        read_distributions(log_path, num_experts=2)

    def test_nodes_are_distinguished_by_graph_scope_and_index(self):
        first_event = _counter_event([0])
        first_event.update(node_index=3, node_name="")
        second_event = _counter_event([1])
        second_event.update(graph_scope="main/4/11:then_branch", node_index=3, node_name="")
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "routing.log"
            log_path.write_text(_complete_trace([[first_event, second_event]]), encoding="utf-8")

            _, by_qmoe, _, event_count, completion = analyze_counter_trace(log_path, num_experts=2)

        self.assertEqual(by_qmoe[("main", 3, "QMoE", "")], {0: 1})
        self.assertEqual(by_qmoe[("main/4/11:then_branch", 3, "QMoE", "")], {1: 1})
        self.assertEqual(event_count, 2)
        self.assertEqual(completion["counter_records"], 2)

    def test_counter_count_is_validated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "routing.log"
            log_path.write_text(
                _complete_trace([[_counter_event([0, 1], [1.0])]]),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "contains 1 counters, but the model has 2 experts"):
                read_distributions(log_path, num_experts=2)

    def test_incomplete_and_nonsequential_prompt_traces_are_rejected(self):
        event = _counter_event()
        traces = {
            "missing prompt_end": (
                f"[qmoe_prompt_runner] 1/1 prompt_start\nmoe_expert_counters {json.dumps(event)}\n",
                "has no prompt_end",
            ),
            "missing footer": (
                f"[qmoe_prompt_runner] 1/1 prompt_start\nmoe_expert_counters {json.dumps(event)}\n"
                "[qmoe_prompt_runner] 1/1 prompt_end\n",
                "missing moe_expert_counters_complete footer",
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
                        list(iter_counter_events(log_path))

    def test_completion_footer_counts_are_validated(self):
        event = _counter_event()
        trace = _complete_trace([[event]]).replace('"counter_records": 1', '"counter_records": 2')
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "routing.log"
            log_path.write_text(trace, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Counter completion footer mismatch"):
                list(iter_counter_events(log_path))

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

    def test_external_data_location_and_file_are_required(self):
        weight = onnx.helper.make_tensor("weight", onnx.TensorProto.FLOAT16, [2, 4], [0.0] * 8)
        node = onnx.helper.make_node("QMoE", ["input", "router", "weight"], ["output"], name="/layers.0/qmoe")
        initializers = {weight.name: weight}
        qmoe_nodes = {(0, node.op_type, node.name): node}
        node_identities = [(0, node.op_type, node.name)]

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model.onnx"

            _set_external_data(weight, length=16)
            with self.assertRaisesRegex(ValueError, "has no non-empty location"):
                calculate_qmoe_expert_bytes(
                    initializers,
                    qmoe_nodes,
                    node_identities,
                    2,
                    model_path=model_path,
                )

            _set_external_data(weight, location="missing.bin", length=16)
            with self.assertRaisesRegex(ValueError, "is not a readable regular file"):
                calculate_qmoe_expert_bytes(
                    initializers,
                    qmoe_nodes,
                    node_identities,
                    2,
                    model_path=model_path,
                )

    def test_ranking_final_row_and_threshold_outputs(self):
        identity = ("main", 17, "QMoE", "/layers.0/qmoe")
        rankings = rank_experts_by_frequency({identity: {0: 2, 1: 2, 2: 1}}, 3)
        self.assertEqual(rankings, {identity: [0, 1, 2]})

        event = _counter_event([1, 2], [1.0, 2.0, 1.0])
        self.assertEqual(selected_expert_ids(event), [1, 2])

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
                    "graph_scope,node_index,node_type,node_name,expert_ids_by_decreasing_frequency",
                    'main,17,QMoE,/layers.0/qmoe,"[0,1,2]"',
                ],
            )


if __name__ == "__main__":
    unittest.main()
