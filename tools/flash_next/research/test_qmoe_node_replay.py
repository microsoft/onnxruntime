from pathlib import Path
import unittest

from build_qmoe_node_replay import transform


ROOT = Path(__file__).parent
REPO = Path("/home/sayanshaw/repos/onnxruntime-webgpu-h200-perf/onnxruntime")
NAMES = ("contrib_ops/webgpu/moe/qmoe.h", "contrib_ops/webgpu/moe/qmoe.cc",
         "core/providers/webgpu/webgpu_context.h", "core/providers/webgpu/webgpu_context.cc",
         "core/providers/webgpu/compute_context.h")


class NodeReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.originals = {name: (REPO / name).read_text() for name in NAMES}
        cls.sources = transform(dict(cls.originals), ROOT)

    def test_existing_core_object_layouts_unchanged(self):
        for name, section in (("core/providers/webgpu/webgpu_context.h", "\n private:"),
                              ("core/providers/webgpu/compute_context.h", "\n protected:")):
            self.assertIn(section, self.originals[name])
            old = self.originals[name].split(section)[-1]
            new = self.sources[name].split(section)[-1]
            self.assertEqual(old, new)
        old_recording = self.originals[NAMES[2]].split("struct CommandRecordingState {")[1].split("\n};")[0]
        new_recording = self.sources[NAMES[2]].split("struct CommandRecordingState {")[1].split("\n};")[0]
        self.assertEqual(old_recording, new_recording)

    def test_uniforms_do_not_return_to_reusable_pool(self):
        text = self.sources["core/providers/webgpu/webgpu_context.cc"]
        self.assertIn("wgpu::Buffer captured_node_uniform;", text)
        self.assertIn("uniform_buffer = captured_node_uniform.Get();", text)
        self.assertIn("!use_uniform_arena && !captured_node_uniform", text)
        self.assertIn('("ORT_WEBGPU_QMOE_COMMAND_REPLAY", 0) == 1', text)

    def test_replay_preserves_deferred_order_and_batching(self):
        text = (ROOT / "node_replay_context.inc").read_text()
        self.assertIn("recording.deferred_dispatches.push_back(command);", text)
        self.assertIn("max_num_pending_dispatches_", text)
        self.assertNotIn("DispatchCommand(", text)
        self.assertNotIn("GraphCaptureState::Replaying", text)
        self.assertIn("command.pending_kernel_info.has_value()", text)

    def test_scratch_is_owned_and_fallback_unchanged(self):
        text = self.sources["contrib_ops/webgpu/moe/qmoe.cc"]
        for index, name in enumerate(("router_values", "indirect_experts", "fc1_outputs",
                                      "fc1_activated", "fc2_outputs")):
            self.assertIn(f"Tensor& {name} = temporary({index},", text)
        self.assertIn("scratch == nullptr ? local_scratch[index] : (*scratch)[index]", text)
        self.assertIn("tensor->DataType() == type && tensor->Shape() == shape", text)
        self.assertIn("ComputeWithWorkspace(context, nullptr)", text)
        self.assertIn("replay_entries_.size() == 8", text)
        self.assertIn("ComputeWithWorkspace(context, &replay_scratch_)", text)
        self.assertIn("replay_workspace_key_ != workspace_key", text)
        self.assertIn("replay_entries_.clear();", text)
        self.assertIn("tensor.reset();", text)
        self.assertIn("std::min_element", text)
        self.assertIn("context.NodeReplayProfilingEnabled() ||", text)
        header = self.sources["contrib_ops/webgpu/moe/qmoe.h"]
        self.assertIn("std::array<std::optional<Tensor>, 5> replay_scratch_;", header)
        self.assertIn("uint64_t last_used = 0;", header)
        self.assertIn("gsl::finally", text)

    def test_key_covers_io_shapes_types_flags_and_profiling(self):
        text = (ROOT / "qmoe_node_replay.inc").read_text()
        for required in ("NodeReplayProfilingEnabled()", "GetElementType()", "GetDims()",
                         "context.InputCount()", "context.Output(0, input->Shape())",
                         "ORT_WEBGPU_QMOE_SUBGROUP_GATE", "ORT_WEBGPU_QMOE_INT4_COOPERATIVE"):
            self.assertIn(required, text)
        self.assertIn('("ORT_WEBGPU_QMOE_COMMAND_REPLAY", 0)', text)
        self.assertIn("enabled == 0 || enabled == 1", text)


if __name__ == "__main__":
    unittest.main()
