from pathlib import Path

from build_dense_double_buffer import transform_compare
from build_immediate_node_replay import build_variant
from build_qmoe_node_replay import replace_once


FLAG = "ORT_WEBGPU_READY_REPLAY_KEY_TRIM"
OUTPUT = Path("/workspace/qwen38-int4-ready-replay-key-trim/build-v2")
COPY = '''webgpu::CapturedCommandInfo ReadyReplayWithoutKey(
    const webgpu::CapturedCommandInfo& command) {
  return {
      .program_key = {},
      .pending_build = std::nullopt,
      .compute_pipeline = command.compute_pipeline,
      .bind_group = command.bind_group,
      .dynamic_uniform_offset = command.dynamic_uniform_offset,
      .dispatch_group = command.dispatch_group,
      .indirect_buffer = command.indirect_buffer,
      .pending_kernel_info = std::nullopt,
  };
}
'''
HELPERS = f'''namespace {{
bool ReadyReplayKeyTrimEnabled() {{
  static const int mode = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_ENFORCE(mode == 0 || mode == 1, "{FLAG} must be 0 or 1");
  return mode == 1;
}}

{COPY}
struct ReadyReplayKeyStatistics {{
  uint64_t commands = 0;
  uint64_t key_bytes = 0;
  ~ReadyReplayKeyStatistics() {{
    const char* path = std::getenv("ORT_WEBGPU_READY_REPLAY_KEY_STATS_FILE");
    if (path == nullptr || commands == 0) return;
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    std::ofstream output(path, std::ios::app);
    output << "{{\\"operator\\":\\"ReadyReplayWithoutKey\\",\\"commands\\":" << commands
           << ",\\"key_bytes_avoided\\":" << key_bytes << "}}\\n";
    if (!output) {{
      LOGS_DEFAULT(ERROR) << "Cannot write ready replay key statistics to " << path;
    }}
  }}
}};
}}  // namespace

'''


def transform(source):
    source = source.replace("#include", "#include <fstream>\n#include", 1)
    signature = "Status WebGpuContext::AppendNodeCommands("
    source = replace_once(source, signature, HELPERS + signature)
    source = replace_once(source,
        "    recording.deferred_dispatches.push_back(command);",
        '''    if (ReadyReplayKeyTrimEnabled() && !is_profiling_) {
      recording.deferred_dispatches.push_back(ReadyReplayWithoutKey(command));
      static thread_local ReadyReplayKeyStatistics statistics;
      ++statistics.commands;
      statistics.key_bytes += command.program_key.size();
    } else {
      recording.deferred_dispatches.push_back(command);
    }''')
    signature = "Status WebGpuContext::Run(ComputeContextBase& context, const ProgramBase& program) {"
    return replace_once(source, signature, signature + "\n  (void)ReadyReplayKeyTrimEnabled();")


def main():
    build_variant(OUTPUT, FLAG, transform,
                  comparator_transform=lambda source: transform_compare(
                      source, flag="ORT_WEBGPU_DENSE_BLOCK32"),
                  report_extra=dict(process_static_selector=True,
                                    comparator_selector="ORT_WEBGPU_DENSE_BLOCK32",
                                    ready_unprofiled_node_replay_only=True,
                                    ready_key_trim=True))


if __name__ == "__main__":
    main()
