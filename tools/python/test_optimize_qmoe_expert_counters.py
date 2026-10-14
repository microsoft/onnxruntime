import json
import tempfile
import unittest
from pathlib import Path

from optimize_qmoe_expert_counters import load_replay_trace


class TestLoadReplayTrace(unittest.TestCase):
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
