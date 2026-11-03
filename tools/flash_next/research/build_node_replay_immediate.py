import json
import os
from pathlib import Path
import subprocess

from build_qmoe_node_replay import replace_once, sha


ROOT = Path(__file__).parent
REPOS = Path("/workspace")
PARENT = REPOS / "qwen38-int4-qmoe-fc1-activation/build-v3"
FROZEN = REPOS / "qwen38-int4-static-node-replay/build-v8"
OUTPUT = REPOS / "qwen38-int4-node-replay-immediate/build-v1"
FLAG = "ORT_WEBGPU_NODE_REPLAY_IMMEDIATE"
APPEND = "Status WebGpuContext::AppendNodeCommands("
HELPER = f'''namespace {{
bool NodeReplayImmediateEnabled() {{
  const int mode = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_ENFORCE(mode == 0 || mode == 1, "{FLAG} must be 0 or 1");
  return mode == 1;
}}
}}  // namespace

'''
IMMEDIATE = '''  const bool immediate = NodeReplayImmediateEnabled() && !is_profiling_;
  if (immediate) {
    ORT_RETURN_IF_ERROR(EncodeDeferredDispatches(recording));
    for (const auto& command : commands) {
      ORT_RETURN_IF_NOT(command.compute_pipeline && command.bind_group && !command.pending_build,
                        "WebGPU node replay command is not ready.");
      recording.has_unsubmitted_work.store(true, std::memory_order_relaxed);
      DispatchCommand(command, recording);
      if (recording.num_pending_dispatches >= max_num_pending_dispatches_) {
        ORT_RETURN_IF_ERROR(Flush(manager, recording));
      }
    }
    return Status::OK();
  }
'''
COMPARATORS = {
    "qmoe": ("qwen38-int4-qmoe-fc1-activation/build-v3", "ORT_WEBGPU_QMOE_FC1_ACTIVATION"),
    "generic": ("qwen38-int4-generic-matmul-replay/build-v1", "ORT_WEBGPU_GENERIC_MATMUL_REPLAY"),
    "static": ("qwen38-int4-static-node-replay/build-v8", "ORT_WEBGPU_STATIC_NODE_REPLAY"),
}


def transform(source):
    source = replace_once(source, APPEND, HELPER + APPEND)
    start = source.index(APPEND)
    end = source.index("\nnamespace {", start)
    body = source[start:end]
    anchor = "  const auto& manager = ComputeContextBase::BufferManagerAccessor::Get(context);\n"
    source = source[:start] + replace_once(body, anchor, anchor + IMMEDIATE) + source[end:]
    return replace_once(source,
        "Status WebGpuContext::Run(ComputeContextBase& context, const ProgramBase& program) {\n",
        "Status WebGpuContext::Run(ComputeContextBase& context, const ProgramBase& program) {\n"
        "  (void)NodeReplayImmediateEnabled();\n")


def comparator_source(source, original_flag):
    assert f'"{original_flag}"' in source
    return source.replace(f'"{original_flag}"', f'"{FLAG}"')


def main():
    parent = json.loads((PARENT / "build-manifest.json").read_text())
    frozen = json.loads((FROZEN / "build-manifest.json").read_text())
    OUTPUT.mkdir(parents=True, exist_ok=False)
    scratch = OUTPUT / "scratch"
    scratch.mkdir()
    overlay = OUTPUT / "overlay/core/providers/webgpu"
    overlay.mkdir(parents=True)
    environment = dict(os.environ, TMPDIR=str(scratch))
    library = OUTPUT / "libonnxruntime_providers_webgpu.so"
    link = parent["link_command"].copy()
    link[link.index("-o") + 1] = str(library)
    subprocess.run(link, cwd=parent["working_directory"], env=environment, check=True)
    assert sha(library) == parent["provider_sha256"], "Baseline relink mismatch"
    library.unlink()
    original = FROZEN / "overlay/core/providers/webgpu/webgpu_context.cc"
    source = overlay / "webgpu_context.cc"
    source.write_text(transform(original.read_text()))
    commands = [c for c in frozen["compile_commands"] if str(original) in c]
    assert len(commands) == 1
    command = commands[0].copy()
    command[command.index("-c") + 1] = str(source)
    obj = OUTPUT / "webgpu_context.cc.o"
    command[command.index("-o") + 1] = str(obj)
    subprocess.run(command, cwd=parent["working_directory"], env=environment, check=True)
    original_obj = str(FROZEN / "webgpu_context.cc.o")
    assert link.count(original_obj) == 1
    link[link.index(original_obj)] = str(obj)
    subprocess.run(link, cwd=parent["working_directory"], env=environment, check=True)
    comparator_hashes = {}
    for name, (directory, flag) in COMPARATORS.items():
        path = OUTPUT / (name + "_compare.cpp")
        path.write_text(comparator_source((REPOS / directory / "native_compare.cpp").read_text(), flag))
        executable = OUTPUT / (name + "_compare")
        subprocess.run(["c++", "-O3", "-std=c++20", str(path),
            "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
            "-L/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
            "-Wl,-rpath,/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
            "-lonnxruntime", "-o", str(executable)], env=environment, check=True)
        comparator_hashes[name] = sha(executable)
    report = dict(status="built_not_qualified", provider_sha256=sha(library),
        baseline_byte_identical=True, baseline_provider_sha256=parent["provider_sha256"],
        compile_commands=[command], link_command=link, working_directory=parent["working_directory"],
        original_source_sha256=sha(original), overlay_source_sha256=sha(source),
        comparator_sha256=comparator_hashes, no_object_layout_changes=True,
        quality_qualified=False, headline_eligible=False)
    (OUTPUT / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "provider_sha256", "baseline_byte_identical")}))


if __name__ == "__main__":
    main()
