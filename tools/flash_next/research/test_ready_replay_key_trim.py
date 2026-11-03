from pathlib import Path
import subprocess
import tempfile
import unittest

from build_immediate_node_replay import SOURCE
from build_ready_replay_key_trim import COPY, FLAG, transform


HOST_SOURCE = Path(str(SOURCE).replace("/workspace", "/home/sayanshaw/repos", 1))


class ReadyReplayKeyTrimTests(unittest.TestCase):
    def test_only_ready_unprofiled_append_changes(self):
        original = HOST_SOURCE.read_text()
        changed = transform(original)
        before = original[original.index("Status WebGpuContext::AppendNodeCommands("):
                          original.index("\nnamespace {", original.index(
                              "Status WebGpuContext::AppendNodeCommands("))]
        after = changed[changed.index("Status WebGpuContext::AppendNodeCommands("):
                        changed.index("\nnamespace {", changed.index(
                            "Status WebGpuContext::AppendNodeCommands("))]
        replacement = after[after.index("    if (ReadyReplayKeyTrimEnabled()"):
                            after.index("    recording.has_unsubmitted_work")]
        self.assertEqual(before, after.replace(replacement,
            "    recording.deferred_dispatches.push_back(command);\n"))
        self.assertIn("command.compute_pipeline && command.bind_group && !command.pending_build", after)
        self.assertIn("ReadyReplayKeyTrimEnabled() && !is_profiling_", after)

    def test_original_pipeline_resolution_is_unchanged(self):
        original = HOST_SOURCE.read_text()
        changed = transform(original)
        start = "Status WebGpuContext::WaitForDeferredPipelineBuilds("
        end = "Status WebGpuContext::EncodeDeferredDispatches("
        self.assertEqual(original[original.index(start):original.index(end)],
                         changed[changed.index(start):changed.index(end)])
        self.assertIn("if (dispatch.compute_pipeline) {\n      continue;", changed)

    def test_default_off_selector_is_process_static_and_validated(self):
        changed = transform(HOST_SOURCE.read_text())
        self.assertIn(f'static const int mode = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);',
                      changed)
        self.assertIn(f'"{FLAG} must be 0 or 1"', changed)
        self.assertIn("(void)ReadyReplayKeyTrimEnabled();", changed)

    def test_actual_copy_has_zero_key_allocations_and_preserves_dispatch(self):
        source = r'''
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <optional>
#include <string>
static size_t allocations;
static bool counting;
void* operator new(size_t n) {
  if (counting) ++allocations;
  if (void* p = std::malloc(n ? n : 1)) return p;
  throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, size_t) noexcept { std::free(p); }
namespace webgpu {
struct CapturedCommandInfo {
  std::string program_key;
  std::optional<int> pending_build;
  std::optional<int> compute_pipeline;
  int bind_group;
  std::optional<uint32_t> dynamic_uniform_offset;
  std::array<uint32_t, 3> dispatch_group{1, 1, 1};
  void* indirect_buffer = nullptr;
  std::optional<int> pending_kernel_info;
};
}
''' + COPY + r'''
int main() {
  webgpu::CapturedCommandInfo command;
  command.program_key.assign(4096, 'k');
  command.compute_pipeline = 73;
  command.bind_group = 91;
  command.dynamic_uniform_offset = 256;
  command.dispatch_group = {320, 320, 2};
  command.indirect_buffer = &command;
  counting = true;
  for (int i = 0; i < 1000; ++i) {
    auto copied = command;
    assert(copied.program_key == command.program_key);
  }
  assert(allocations == 1000);
  allocations = 0;
  for (int i = 0; i < 1000; ++i) {
    auto copied = ReadyReplayWithoutKey(command);
    assert(copied.program_key.empty());
    assert(!copied.pending_build && !copied.pending_kernel_info);
    assert(copied.compute_pipeline == command.compute_pipeline);
    assert(copied.bind_group == command.bind_group);
    assert(copied.dynamic_uniform_offset == command.dynamic_uniform_offset);
    assert(copied.dispatch_group == command.dispatch_group);
    assert(copied.indirect_buffer == command.indirect_buffer);
  }
  assert(allocations == 0);
}
'''
        with tempfile.TemporaryDirectory(prefix="ready-replay-key-test-") as directory:
            path = Path(directory)
            cpp = path / "test.cpp"
            cpp.write_text(source)
            executable = path / "test"
            subprocess.run(["c++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
                            str(cpp), "-o", str(executable)], check=True)
            subprocess.run([str(executable)], check=True)


if __name__ == "__main__":
    unittest.main()
