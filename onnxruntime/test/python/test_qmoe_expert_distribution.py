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

from qmoe_expert_distribution import (  # noqa: E402
    CUMULATIVE_GLOBAL_POSITION_THRESHOLDS,
    GLOBAL_POSITION_THRESHOLDS,
    aggregate_threshold_counts,
    build_timeline,
    cumulative_threshold_counts,
    cumulative_threshold_fractions,
    inspect_trace,
    parse_counter_trace,
    write_timeline_csv,
)


def _event(node_index, node_name, selected_experts):
    return {
        "request_id": "",
        "graph_scope": "main",
        "node_index": node_index,
        "node_type": "MoE",
        "node_name": node_name,
        "moe_count": 2,
        "total_expert_count": 6,
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
    session_options = (
        "Session Options { config_options: { session.moe_expert_counter_beta: 0.1 "
        "session.moe_expert_counter_alpha: 0.9 } }"
    )
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
            cumulative_threshold_fractions(rows),
            {
                threshold: [(1, 0.25 if threshold == CUMULATIVE_GLOBAL_POSITION_THRESHOLDS[0] else 0.0)]
                for threshold in CUMULATIVE_GLOBAL_POSITION_THRESHOLDS
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
                "session.moe_expert_counter_alpha: 0.9 session.moe_expert_counter_beta: 0.1\n"
                "[qmoe_prompt_runner] 1/1 prompt_start\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "has no prompt_end"):
                parse_counter_trace(log_path)


if __name__ == "__main__":
    unittest.main()
