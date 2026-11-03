from pathlib import Path
import unittest

from build_immediate_node_replay import FLAG, IMMEDIATE, VALIDATE, transform
from run_immediate_node_replay_native import verify_stats


SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-static-node-replay/build-v8/overlay/"
              "core/providers/webgpu/webgpu_context.cc")


class ImmediateNodeReplayTests(unittest.TestCase):
    def test_stats_count_one_explicitly_rejected_attempt_without_a_hit(self):
        row = dict(operator="MatMulNBitsExact", requests=257, entries=1, captures=1, hits=255,
                   rebinds=0, binding_hits=255, capacity=64, binding_entries=1, binding_capacity=64)
        verify_stats([row.copy() for _ in range(9)])
        row["requests"] = 256
        with self.assertRaises(AssertionError):
            verify_stats([row.copy() for _ in range(9)])

    def test_only_two_insertions_and_exact_original_deferred_path(self):
        original = SOURCE.read_text()
        modified = transform(original)
        self.assertEqual(modified.replace(IMMEDIATE, "", 1).replace(VALIDATE, "", 1), original)
        self.assertEqual(modified.count(f'"{FLAG}", 0'), 2)

    def test_preserves_preceding_work_before_direct_dispatch_without_copying_commands(self):
        self.assertLess(IMMEDIATE.index("EncodeDeferredDispatches(recording)"),
                        IMMEDIATE.index("DispatchCommand(command, recording)"))
        self.assertNotIn("deferred_dispatches.push_back", IMMEDIATE)
        self.assertLess(IMMEDIATE.index('"WebGPU node replay command is not ready."'),
                        IMMEDIATE.index("EncodeDeferredDispatches(recording)"))
        self.assertIn("recording.num_pending_dispatches >= max_num_pending_dispatches_", IMMEDIATE)
        self.assertIn("Flush(manager, recording)", IMMEDIATE)
        self.assertIn("has_unsubmitted_work.store(true", IMMEDIATE)

    def test_profile_metadata_and_original_graph_capture_guard_retained(self):
        self.assertIn("is_profiling_ || command.pending_kernel_info.has_value()", IMMEDIATE)
        self.assertIn("pending_kernels.emplace_back(*command.pending_kernel_info)", IMMEDIATE)
        modified = transform(SOURCE.read_text())
        signature = "Status WebGpuContext::AppendNodeCommands("
        body = modified[modified.index(signature):]
        self.assertLess(body.index("CanCaptureNodeCommands(context)"), body.index(IMMEDIATE))


if __name__ == "__main__":
    unittest.main()
