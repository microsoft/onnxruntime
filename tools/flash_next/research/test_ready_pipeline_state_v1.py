from pathlib import Path
import subprocess
import tempfile
import unittest

from ready_pipeline_state_transform import HELPER, transform


ROOT = Path(__file__).parent
SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-static-node-replay/build-v8/overlay/core/providers/webgpu/webgpu_context.cc")


class ReadyPipelineStateTests(unittest.TestCase):
    def test_actual_scoped_helper_repetition_resets_nested_disabled_and_foreign_records(self):
        prefix = r'''
#include <cassert>
#include <cstdint>
#include <stdexcept>
using WGPUComputePassEncoder = const void*;
using WGPUComputePipeline = const void*;
namespace webgpu { struct CommandRecordingState {}; }
template<class T> T ParseEnvironmentVariableWithDefault(const char*, T value) { return value; }
#define ORT_ENFORCE(condition, message) do { if (!(condition)) throw std::runtime_error(message); } while (false)
'''
        main = r'''
int main() {
  webgpu::CommandRecordingState first, second;
  int encoder = 0, pipeline = 0, alternate = 0;
  assert(ShouldSetReadyPipeline(first, &encoder, &pipeline));
  {
    ScopedReadyPipelineState outer(first, true);
    assert(ShouldSetReadyPipeline(first, &encoder, &pipeline));
    for (int i = 0; i < 1000; ++i) assert(!ShouldSetReadyPipeline(first, &encoder, &pipeline));
    assert(outer.State().sets == 1 && outer.State().skips == 1000);
    assert(ShouldSetReadyPipeline(second, &encoder, &pipeline));
    assert(!ShouldSetReadyPipeline(first, &encoder, &pipeline));
    assert(ShouldSetReadyPipeline(first, &encoder, &alternate));
    ForgetReadyPipelineState(first);
    assert(ShouldSetReadyPipeline(first, &encoder, &alternate));
    {
      ScopedReadyPipelineState nested(first, true);
      assert(ShouldSetReadyPipeline(first, &encoder, &pipeline));
      assert(!ShouldSetReadyPipeline(first, &encoder, &pipeline));
    }
    assert(ShouldSetReadyPipeline(first, &encoder, &alternate));
    {
      ScopedReadyPipelineState disabled(first, false);
      assert(ShouldSetReadyPipeline(first, &encoder, &pipeline));
      assert(ShouldSetReadyPipeline(first, &encoder, &pipeline));
    }
    assert(ShouldSetReadyPipeline(first, &encoder, &alternate));
    try {
      ScopedReadyPipelineState throwing(second, true);
      throw std::runtime_error("scope");
    } catch (const std::runtime_error&) {}
    assert(ShouldSetReadyPipeline(first, &encoder, &alternate));
    assert(ShouldSetReadyPipeline(first, nullptr, &alternate));
    assert(ShouldSetReadyPipeline(first, &encoder, nullptr));
  }
  assert(active_ready_pipeline_state == nullptr);
}
'''
        with tempfile.TemporaryDirectory(prefix="ready-pipeline-state-") as folder:
            source = Path(folder) / "scope.cpp"
            binary = Path(folder) / "scope"
            source.write_text(prefix + HELPER + main)
            subprocess.run(["c++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
                            "-Wno-unused-function", str(source), "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)

    def test_only_redundant_pipeline_state_and_scope_change(self):
        original = SOURCE.read_text()
        selected = transform(original)
        self.assertIn("!is_profiling_", selected)
        self.assertIn("recording.graph_capture_state == GraphCaptureState::Default", selected)
        self.assertIn("if (!recording.compute_pass_encoder) ForgetReadyPipelineState(recording)", selected)
        self.assertIn("EndComputePass(recording);\n    ForgetReadyPipelineState(recording);", selected)
        self.assertEqual(selected.count("SetBindGroup(0, command.bind_group"), original.count("SetBindGroup(0, command.bind_group"))
        self.assertEqual(selected.count("DispatchWorkgroups("), original.count("DispatchWorkgroups("))
        self.assertEqual(selected.count("DispatchWorkgroupsIndirect("), original.count("DispatchWorkgroupsIndirect("))
        self.assertEqual(selected.count("WriteTimestamp(recording,"), original.count("WriteTimestamp(recording,"))
        self.assertNotIn("std::shared_ptr<ReadyPipelineState>", selected)
        self.assertNotIn("thread_local ReadyPipelineState ", selected)


if __name__ == "__main__":
    unittest.main()
