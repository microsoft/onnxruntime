import json
import tempfile
import unittest
from pathlib import Path

from optimize_qmoe_expert_counters import evaluate_alpha, load_replay_trace


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


if __name__ == "__main__":
    unittest.main()
