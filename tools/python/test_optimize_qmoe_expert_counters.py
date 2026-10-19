import json
import tempfile
import unittest
from pathlib import Path

from optimize_qmoe_expert_counters import (
    ReplayEvent,
    average_all_experts_below_threshold,
    evaluate_alpha,
    load_replay_trace,
)
from qmoe_expert_activation_correlations import load_activation_trace
from qmoe_expert_distribution import build_timeline, inspect_trace


class TestLoadReplayTrace(unittest.TestCase):
    @staticmethod
    def counter_event(node_index, layer, expert_id, graph_scope="main", expert_offset=None):
        runtime_offset = expert_offset if type(expert_offset) is int and 0 <= expert_offset < 4 else node_index * 2
        event = {
            "request_id": "request",
            "graph_scope": graph_scope,
            "node_index": node_index,
            "node_type": "QMoE",
            "node_name": f"/layers.{layer}/moe",
            "moe_count": 2,
            "total_expert_count": 4,
            "generation_iteration": 1,
            "selected_experts": [
                {
                    "expert_id": expert_id,
                    "score": 0.0,
                    "selected_rank": 0,
                    "node_rank": expert_id,
                    "global_position": runtime_offset + expert_id,
                    "device": "CUDA",
                }
            ],
        }
        if expert_offset is not None:
            event["expert_offset"] = expert_offset
        return event

    @staticmethod
    def load_events(events):
        lines = [
            "[qmoe_prompt_runner] 1/1 prompt_start",
            *(f"moe_expert_counters {json.dumps(event)}" for event in events),
            "[qmoe_prompt_runner] 1/1 prompt_end",
            f"moe_expert_counters_complete {json.dumps({'prompts': 1, 'prompt_runs': 1, 'counter_records': len(events)})}",
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "trace.log"
            path.write_text("\n".join(lines), encoding="utf-8")
            return load_replay_trace(path)

    def test_legacy_replay_preserves_registration_order_not_layer_or_execution_order(self):
        events, identities, moe_count, experts_per_node = self.load_events(
            [self.counter_event(1, 0, 0), self.counter_event(0, 9, 1)]
        )
        self.assertEqual([identity[1] for identity in identities], [0, 1])
        self.assertEqual([event.node_index for event in events], [1, 0])
        result = evaluate_alpha(events, moe_count, experts_per_node, [1], 1.0, selected_expert_count=1)[0]
        self.assertEqual(result.cumulative_share, 1.0)

    def test_explicit_offsets_preserve_subgraph_runtime_order(self):
        events, identities, moe_count, experts_per_node = self.load_events(
            [
                self.counter_event(0, 0, 0, "main/1/then", 2),
                self.counter_event(5, 9, 1, expert_offset=0),
            ]
        )
        self.assertEqual([identity[0] for identity in identities], ["main", "main/1/then"])
        result = evaluate_alpha(events, moe_count, experts_per_node, [1], 1.0, selected_expert_count=1)[0]
        self.assertEqual(result.cumulative_share, 1.0)

    def test_rejects_invalid_runtime_expert_offset(self):
        for offset in (-1, True, 0.5, 1, 4):
            with self.subTest(offset=offset), self.assertRaisesRegex(ValueError, "Invalid runtime expert offset"):
                self.load_events(
                    [
                        self.counter_event(0, 0, 1, expert_offset=offset),
                        self.counter_event(1, 1, 0, expert_offset=2),
                    ]
                )

    def test_rejects_overlapping_or_incomplete_runtime_offsets(self):
        for offset in (0, None):
            with self.subTest(offset=offset), self.assertRaisesRegex(ValueError, "complete, non-overlapping"):
                self.load_events(
                    [
                        self.counter_event(0, 0, 1, expert_offset=0),
                        self.counter_event(1, 1, 0, expert_offset=offset),
                    ]
                )

    def test_rejects_changing_runtime_offset(self):
        with self.assertRaisesRegex(ValueError, "Inconsistent runtime expert offsets"):
            self.load_events(
                [
                    self.counter_event(0, 0, 1, expert_offset=0),
                    self.counter_event(0, 0, 1, expert_offset=2),
                    self.counter_event(1, 1, 0, expert_offset=2),
                ]
            )

    def test_rejects_legacy_subgraph_trace_without_runtime_offsets(self):
        with self.assertRaisesRegex(ValueError, "subgraphs requires"):
            self.load_events([self.counter_event(0, 0, 1), self.counter_event(0, 1, 0, "main/1/then")])

    def test_rejects_trace_missing_declared_moe_node(self):
        event = {
            "request_id": "request",
            "graph_scope": "main",
            "node_index": 0,
            "node_type": "QMoE",
            "node_name": "/layers.0/moe",
            "moe_count": 2,
            "total_expert_count": 6,
            "selected_experts": [
                {
                    "expert_id": 0,
                    "score": 1.0,
                    "selected_rank": 0,
                    "node_rank": 0,
                    "global_position": 0,
                    "device": "CUDA",
                }
            ],
        }
        lines = [
            "[qmoe_prompt_runner] 1/1 prompt_start",
            f"moe_expert_counters {json.dumps(event)}",
            "[qmoe_prompt_runner] 1/1 prompt_end",
            'moe_expert_counters_complete {"prompts": 1, "prompt_runs": 1, "counter_records": 1}',
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            trace_path = Path(temporary_directory) / "trace.log"
            trace_path.write_text("\n".join(lines), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "observed 1 of 2 nodes"):
                load_replay_trace(trace_path)

    def test_generation_markers_exclude_prefill_even_with_eight_selected_experts(self):
        for prefill_counts in ((8, 9), (8, 8)):
            with self.subTest(prefill_counts=prefill_counts), tempfile.TemporaryDirectory() as temporary_directory:
                lines = ["[qmoe_prompt_runner] 1/1 prompt_start"]
                for iteration in (0, 1):
                    phase = "prefill" if iteration == 0 else "decode"
                    lines.append(f"[qmoe_prompt_runner] {phase} {iteration}")
                    for node_index in (0, 1):
                        count = prefill_counts[node_index] if iteration == 0 else 8
                        event = self.counter_event(node_index, node_index, 0)
                        del event["generation_iteration"]
                        event["total_expert_count"] = 20
                        event["expert_offset"] = node_index * 10
                        event["selected_experts"] = [
                            {
                                "expert_id": expert,
                                "score": 0.0,
                                "selected_rank": expert,
                                "node_rank": expert,
                                "global_position": node_index * 10 + expert,
                                "device": "CUDA",
                            }
                            for expert in range(count)
                        ]
                        lines.append(f"moe_expert_counters {json.dumps(event)}")
                lines.extend(
                    [
                        "[qmoe_prompt_runner] 1/1 prompt_end",
                        'moe_expert_counters_complete {"prompts":1,"prompt_runs":1,"counter_records":4}',
                    ]
                )
                path = Path(temporary_directory) / "trace.log"
                path.write_text("\n".join(lines), encoding="utf-8")
                events, _, moe_count, experts_per_node = load_replay_trace(path)
                self.assertEqual([event.generation_iteration for event in events], [0, 0, 1, 1])
                result = evaluate_alpha(events, moe_count, experts_per_node, [9], 0.5)[0]
                self.assertEqual(result.moe_invocation_count, 2)
                self.assertEqual(result.selected_expert_count, 16)
                self.assertEqual(result.selected_experts_at_or_above_threshold, 7)
                self.assertEqual(
                    average_all_experts_below_threshold(events, moe_count, experts_per_node, 0.5, [9]),
                    {9: [8.0, 1.0]},
                )
                activations = load_activation_trace(path)
                self.assertEqual(activations.prompt_activations[0].shape, (1, 2, 10))
                self.assertEqual(len(activations.selected_expert_rows), 16)
                self.assertEqual({row[0] for row in activations.selected_expert_rows}, {1})
                summary, *_ = inspect_trace(path)
                self.assertEqual(len(build_timeline(path, summary)), 2)

    def test_rejects_decode_analysis_without_generation_markers(self):
        events = [self.counter_event(0, 0, 0), self.counter_event(1, 1, 0)]
        for event in events:
            del event["generation_iteration"]
        with self.assertRaisesRegex(ValueError, "prefill/decode generation markers"):
            self.load_events(events)

    def test_average_rejects_incomplete_decode_at_prompt_boundary(self):
        events = [ReplayEvent(0, (0,), 1, 1), ReplayEvent(1, (0,), 2, 1)]
        with self.assertRaisesRegex(ValueError, "incomplete decode iteration"):
            average_all_experts_below_threshold(events, 2, 2, 0.5, [1])

    def test_average_distinguishes_same_decode_iteration_in_different_prompts(self):
        events = [
            ReplayEvent(node_index, (0,), prompt_index, iteration)
            for prompt_index in (1, 2)
            for iteration in (0, 1)
            for node_index in (0, 1)
        ]
        self.assertEqual(
            average_all_experts_below_threshold(events, 2, 2, 0.5, [1]),
            {1: [1.0, 0.0]},
        )

    def test_average_rejects_trace_without_decode(self):
        with self.assertRaisesRegex(ValueError, "no decode events"):
            average_all_experts_below_threshold([ReplayEvent(0, (0,), 1, 0)], 1, 2, 0.5, [1])


if __name__ == "__main__":
    unittest.main()
