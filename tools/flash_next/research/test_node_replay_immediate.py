from pathlib import Path
import unittest

from build_node_replay_immediate import APPEND, FLAG, HELPER, IMMEDIATE, comparator_source, transform
from run_node_replay_immediate_native import validate_stats


SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-static-node-replay/build-v8/overlay/"
              "core/providers/webgpu/webgpu_context.cc")


class NodeReplayImmediateTests(unittest.TestCase):
    def test_only_append_and_run_change(self):
        source = SOURCE.read_text()
        result = transform(source)
        restored = result.replace(HELPER, "", 1).replace(IMMEDIATE, "", 1).replace(
            "  (void)NodeReplayImmediateEnabled();\n", "", 1)
        self.assertEqual(restored, source)
        self.assertEqual(result.count(IMMEDIATE), 1)

    def test_order_and_submission_boundaries(self):
        self.assertLess(IMMEDIATE.index("EncodeDeferredDispatches(recording)"),
                        IMMEDIATE.index("DispatchCommand(command, recording)"))
        self.assertLess(IMMEDIATE.index("has_unsubmitted_work.store"),
                        IMMEDIATE.index("DispatchCommand(command, recording)"))
        self.assertIn("recording.num_pending_dispatches >= max_num_pending_dispatches_", IMMEDIATE)
        self.assertIn("Flush(manager, recording)", IMMEDIATE)
        self.assertNotIn("deferred_dispatches.push_back", IMMEDIATE)
        self.assertIn("!is_profiling_", IMMEDIATE)

    def test_flag_is_validated_for_replay_and_regular_programs(self):
        source = transform(SOURCE.read_text())
        self.assertIn(f'"{FLAG}", 0', HELPER)
        self.assertIn("mode == 0 || mode == 1", HELPER)
        self.assertIn(f'"{FLAG} must be 0 or 1"', HELPER)
        self.assertLess(source.index(HELPER), source.index(APPEND))
        self.assertIn("  (void)NodeReplayImmediateEnabled();", source)

    def test_comparator_toggles_encoding_not_replay(self):
        source = 'setenv("ORT_WEBGPU_STATIC_NODE_REPLAY", value, 1);'
        result = comparator_source(source, "ORT_WEBGPU_STATIC_NODE_REPLAY")
        self.assertEqual(result, f'setenv("{FLAG}", value, 1);')
        with self.assertRaises(AssertionError):
            comparator_source(source, "missing")

    def test_stats_count_rejected_trial_separately(self):
        row = dict(requests=65, hits=56, captures=8, entries=8, capacity=8)
        validate_stats("qmoe", [row], [{}])
        with self.assertRaises(AssertionError):
            validate_stats("qmoe", [dict(row, requests=64)], [{}])
        row = dict(operator="MatMulNBitsExact", requests=257, hits=254, captures=2,
                   rebinds=2, binding_hits=252, entries=1, capacity=64,
                   binding_entries=2, binding_capacity=64)
        original = [dict(operator="MatMulNBitsExact", requests=128)]
        validate_stats("generic", [row], original)
        with self.assertRaises(AssertionError):
            validate_stats("generic", [dict(row, requests=256)], original)


if __name__ == "__main__":
    unittest.main()
