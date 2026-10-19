# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

_TOOLS_PYTHON = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "tools", "python"))
if _TOOLS_PYTHON not in sys.path:
    sys.path.insert(0, _TOOLS_PYTHON)

from optimize_qmoe_expert_counters import ReplayEvent, evaluate_alpha, validate_alpha_candidates  # noqa: E402
from qmoe_expert_distribution import (  # noqa: E402
    CUMULATIVE_GLOBAL_POSITION_THRESHOLDS,
    GLOBAL_POSITION_THRESHOLDS,
    aggregate_threshold_counts,
    build_timeline,
    cumulative_moe_counts_at_or_above_threshold,
    cumulative_moe_counts_below_threshold,
    cumulative_threshold_counts,
    cumulative_threshold_fractions,
    inspect_trace,
    parse_counter_trace,
    write_timeline_csv,
)


def _event(node_index, node_name, selected_experts, generation_iteration=1):
    return {
        "request_id": "",
        "graph_scope": "main",
        "node_index": node_index,
        "node_type": "MoE",
        "node_name": node_name,
        "moe_count": 2,
        "total_expert_count": 6,
        "generation_iteration": generation_iteration,
        "selected_experts": selected_experts,
    }


def _selected(expert_id, score, selected_rank, node_rank, global_position):
    return {
        "expert_id": expert_id,
        "score": score,
        "selected_rank": selected_rank,
        "node_rank": node_rank,
        "global_position": global_position,
        "device": "CUDA",
    }


def _trace(events):
    session_options = "Session Options { config_options: { session.moe_expert_counter_alpha: 0.9 } }"
    lines = [
        session_options,
        "[qmoe_prompt_runner] 1/1 prompt_start",
    ]
    lines.extend(f"moe_expert_counters {json.dumps(event)}" for event in events)
    lines.extend(
        [
            "[qmoe_prompt_runner] 1/1 prompt_end",
            "moe_expert_counters_complete "
            + json.dumps({"prompts": 1, "prompt_runs": 1, "counter_records": len(events)}),
        ]
    )
    return "\n".join(lines) + "\n"


class TestQMoEExpertDistribution(unittest.TestCase):
    def test_counter_parameter_optimization(self):
        events = [
            (0, (0,), 0),
            (1, (0,), 0),
            (0, (1, 0), 1),
            (1, (1, 0), 1),
        ]
        replay_events = [
            ReplayEvent(node_index, selected_experts, 1, iteration)
            for node_index, selected_experts, iteration in events
        ]

        results = evaluate_alpha(
            replay_events,
            moe_count=2,
            experts_per_node=2,
            thresholds=(2, 3),
            alpha=0.5,
            selected_expert_count=2,
        )

        self.assertEqual(results[0].threshold, 2)
        self.assertEqual(results[0].selected_experts_at_or_above_threshold, 3)
        self.assertEqual(results[0].selected_expert_count, 4)
        self.assertEqual(results[0].cumulative_share, 0.75)
        self.assertEqual(results[0].moe_invocations_at_or_above_threshold, 2)
        self.assertEqual(results[0].moe_invocation_count, 2)
        self.assertEqual(results[0].cumulative_moe_share, 1.0)
        self.assertEqual(results[0].impacted_invocations_by_node, (1, 1))
        self.assertEqual(results[0].invocations_by_node, (1, 1))
        self.assertEqual(results[0].selected_experts_below_threshold_by_node, (1, 0))
        self.assertEqual(results[1].threshold, 3)
        self.assertEqual(results[1].selected_experts_at_or_above_threshold, 1)
        self.assertEqual(results[1].cumulative_moe_share, 0.5)
        alpha_one_results = evaluate_alpha(
            replay_events,
            moe_count=2,
            experts_per_node=2,
            thresholds=(2,),
            alpha=1.0,
            selected_expert_count=2,
        )
        self.assertEqual(alpha_one_results[0].cumulative_share, 0.5)
        self.assertEqual(validate_alpha_candidates([1.0, 0.0, 0.9]), [0.0, 0.9, 1.0])

    def test_builds_local_and_global_timeline(self):
        events = [
            _event(
                1,
                "/model/layers.0/moe/MoE",
                [_selected(1, 0.1, 1, 1, 1), _selected(0, 0.1, 0, 0, 0)],
            ),
            _event(
                2,
                "/model/layers.1/moe/MoE",
                [_selected(2, 0.1, 1, 1, 4), _selected(1, 0.1, 0, 0, 3)],
            ),
            _event(
                1,
                "/model/layers.0/moe/MoE",
                [_selected(2, 0.1, 1, 1, 1), _selected(1, 0.19, 0, 0, 0)],
            ),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "routing.log"
            csv_path = Path(temp_dir) / "timeline.csv"
            log_path.write_text(_trace(events), encoding="utf-8")
            summary, _, selection_counts, _, _ = inspect_trace(log_path)
            rows = build_timeline(log_path, summary, selected_expert_count=2)
            write_timeline_csv(csv_path, rows)
            with csv_path.open(encoding="utf-8") as stream:
                csv_rows = list(csv.reader(stream))

        self.assertEqual(selection_counts, {2: 3})
        self.assertEqual(rows[-1].minimum_score, 0.1)
        self.assertEqual(rows[0].global_position, 1)
        self.assertEqual(rows[1].global_position, 4)
        self.assertEqual(rows[-1].global_position, 1)
        self.assertEqual(len(csv_rows[0]), len(csv_rows[1]))
        self.assertEqual(
            csv_rows[0][-len(GLOBAL_POSITION_THRESHOLDS) :],
            [f"experts_global_position_ge_{threshold}" for threshold in GLOBAL_POSITION_THRESHOLDS],
        )

    def test_prefill_updates_state_but_is_not_in_decode_timeline(self):
        events = [
            _event(
                1,
                "/model/layers.0/moe/MoE",
                [
                    _selected(0, 0.1, 0, 0, 0),
                    _selected(1, 0.1, 1, 1, 1),
                    _selected(2, 0.1, 2, 2, 2),
                ],
                generation_iteration=0,
            ),
            _event(
                1,
                "/model/layers.0/moe/MoE",
                [_selected(0, 0.19, 0, 0, 0), _selected(2, 0.19, 1, 1, 1)],
            ),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "routing.log"
            log_path.write_text(_trace(events), encoding="utf-8")
            summary, _, _, _, _ = inspect_trace(log_path)
            rows = build_timeline(log_path, summary, selected_expert_count=2)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].minimum_score, 0.19)

    def test_counts_selected_experts_at_or_above_global_position_threshold(self):
        event = _event(
            1,
            "/model/layers.20/moe/MoE",
            [
                _selected(0, 0.2, 0, 0, GLOBAL_POSITION_THRESHOLDS[0]),
                _selected(1, 0.1, 1, 1, GLOBAL_POSITION_THRESHOLDS[0] + 1),
            ],
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "routing.log"
            log_path.write_text(_trace([event]), encoding="utf-8")
            summary, _, _, _, _ = inspect_trace(log_path)
            rows = build_timeline(log_path, summary, selected_expert_count=2)

        self.assertEqual(rows[0].expert_counts_by_global_position_threshold, (2, 0, 0, 0, 0, 0))
        self.assertEqual(
            aggregate_threshold_counts(rows),
            {
                threshold: [(1, 2 if threshold == GLOBAL_POSITION_THRESHOLDS[0] else 0)]
                for threshold in GLOBAL_POSITION_THRESHOLDS
            },
        )
        self.assertEqual(
            cumulative_threshold_counts(rows),
            {
                threshold: [(1, 2 if threshold == CUMULATIVE_GLOBAL_POSITION_THRESHOLDS[0] else 0)]
                for threshold in CUMULATIVE_GLOBAL_POSITION_THRESHOLDS
            },
        )
        self.assertEqual(
            cumulative_threshold_fractions(rows, selected_expert_count=2),
            {
                threshold: [(1, 1.0 if threshold == CUMULATIVE_GLOBAL_POSITION_THRESHOLDS[0] else 0.0)]
                for threshold in CUMULATIVE_GLOBAL_POSITION_THRESHOLDS
            },
        )

    def test_counts_moe_nodes_with_all_expert_positions_below_threshold(self):
        events = [
            _event(
                1,
                "/model/layers.0/moe/MoE",
                [_selected(0, 0.2, 0, 0, 0), _selected(1, 0.1, 1, 1, 1)],
            ),
            _event(
                2,
                "/model/layers.1/moe/MoE",
                [_selected(0, 0.2, 0, 0, 3), _selected(1, 0.1, 1, 1, 4)],
            ),
            _event(
                1,
                "/model/layers.0/moe/MoE",
                [_selected(0, 0.2, 0, 0, 0), _selected(1, 0.1, 1, 1, 1)],
            ),
            _event(
                2,
                "/model/layers.1/moe/MoE",
                [_selected(0, 0.2, 0, 0, 2), _selected(1, 0.1, 1, 1, 3)],
            ),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "routing.log"
            log_path.write_text(_trace(events), encoding="utf-8")
            summary, _, _, _, _ = inspect_trace(log_path)
            rows = build_timeline(log_path, summary, selected_expert_count=2)

        self.assertEqual(
            cumulative_moe_counts_below_threshold(rows, thresholds=(4, 5)),
            {
                4: [(1, 1.0), (2, 1.5)],
                5: [(1, 2), (2, 2)],
            },
        )
        self.assertEqual(
            cumulative_moe_counts_at_or_above_threshold(rows, thresholds=(4, 5)),
            {
                4: [(1, 1.0), (2, 0.5)],
                5: [(1, 0), (2, 0)],
            },
        )

    def test_rejects_missing_global_position(self):
        event = _event(
            1,
            "/model/layers.0/moe/MoE",
            [_selected(0, 0.2, 0, 0, 0)],
        )
        del event["selected_experts"][0]["global_position"]
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "routing.log"
            log_path.write_text(_trace([event]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "global_position must be a non-negative integer"):
                inspect_trace(log_path)

    def test_rejects_invalid_selected_rank(self):
        event = _event(
            1,
            "/model/layers.0/moe/MoE",
            [_selected(0, 0.1, 1, 0, 0)],
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "routing.log"
            log_path.write_text(_trace([event]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "selected_rank values must be consecutive"):
                parse_counter_trace(log_path)

    def test_rejects_incomplete_trace(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "routing.log"
            log_path.write_text(
                "session.moe_expert_counter_alpha: 0.9\n[qmoe_prompt_runner] 1/1 prompt_start\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "has no prompt_end"):
                parse_counter_trace(log_path)

    def test_rejects_out_of_order_generation_markers(self):
        for marker in ("decode 0", "decode 1", "prefill 0\n[qmoe_prompt_runner] decode 2"):
            with self.subTest(marker=marker), tempfile.TemporaryDirectory() as temp_dir:
                log_path = Path(temp_dir) / "routing.log"
                trace = _trace([]).replace(
                    "[qmoe_prompt_runner] 1/1 prompt_start",
                    f"[qmoe_prompt_runner] 1/1 prompt_start\n[qmoe_prompt_runner] {marker}",
                )
                log_path.write_text(trace, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Unexpected generation marker"):
                    parse_counter_trace(log_path)


if __name__ == "__main__":
    unittest.main()
