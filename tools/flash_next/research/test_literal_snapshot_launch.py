import json
from pathlib import Path
import tempfile
import unittest

from run_literal_snapshot_v1 import ROOT, command_for, replay_counters


class SnapshotLaunchTests(unittest.TestCase):
    def test_startup_native_and_replay_operator_wiring(self):
        prefix = ROOT / "unused-literal-snapshot-launch-test"
        writer = command_for(prefix, 0, 0, False, write_reference=True)[-1]
        self.assertIn("ORT_WEBGPU_QMOE_FC2_FINAL_MIX=0", writer)
        self.assertIn("native_fixed", writer)
        self.assertIn(" write /evidence/", writer)
        selected = command_for(prefix, 1, 1, True)[-1]
        self.assertIn("ORT_WEBGPU_LITERAL_SNAPSHOT=1", selected)
        self.assertIn("webgpu-literal-snapshot/build-v1/libonnxruntime", selected)
        self.assertIn(" compare /evidence/", selected)
        operator = command_for(prefix, 1, 1, True, True)[-1]
        self.assertIn("ORT_WEBGPU_QMOE_COMMAND_REPLAY=1", operator)
        self.assertNotIn("ORT_WEBGPU_QMOE_COMMAND_REPLAY=0", operator)
        self.assertIn("ORT_WEBGPU_QMOE_SUBGROUP_GATE=1", operator)
        self.assertIn("unused-literal-snapshot-launch-test-stats.jsonl", operator)
        self.assertNotIn("ORT_WEBGPU_QMOE_EARLY_SCRATCH", operator)

    def test_replay_counter_gate_rejects_missing_actual_hits(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stats.jsonl"
            rows = [
                {"kernel_identity": i, "requests": 144, "hits": 143,
                 "captures": 1, "shape_invalidations": 0, "capacity_fallbacks": 0}
                for i in range(32)]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            self.assertEqual(len(replay_counters(path)), 32)
            rows[0]["hits"] = 0
            rows[0]["captures"] = 144
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(RuntimeError, "high-hit"):
                replay_counters(path)


if __name__ == "__main__":
    unittest.main()
