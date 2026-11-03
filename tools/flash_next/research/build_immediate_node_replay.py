import json
import os
from pathlib import Path
import subprocess

from build_dense_double_buffer import transform_compare
from build_qmoe_node_replay import replace_once, sha


FLAG = "ORT_WEBGPU_IMMEDIATE_NODE_REPLAY"
PARENT = Path("/workspace/qwen38-int4-qmoe-prefill-rows4/build-v2")
CONTEXT = Path("/workspace/qwen38-int4-static-node-replay/build-v8")
SOURCE = CONTEXT / "overlay/core/providers/webgpu/webgpu_context.cc"
COMPARATOR = Path("/workspace/qwen38-int4-dense-block32/build-v1")
OUTPUT = Path("/workspace/qwen38-int4-immediate-node-replay/build-v1")


VALIDATE = f'''
  const int immediate_node_replay = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(immediate_node_replay == 0 || immediate_node_replay == 1,
                    "{FLAG} must be 0 or 1");
'''

IMMEDIATE = VALIDATE + r'''
  if (immediate_node_replay == 1) {
    for (const auto& command : commands) {
      ORT_RETURN_IF_NOT(command.compute_pipeline && command.bind_group && !command.pending_build,
                        "WebGPU node replay command is not ready.");
      ORT_RETURN_IF_NOT(!is_profiling_ || command.pending_kernel_info.has_value(),
                        "WebGPU node replay command lacks profiling metadata.");
    }
    ORT_RETURN_IF_ERROR(EncodeDeferredDispatches(recording));
    static std::once_flag announced;
    std::call_once(announced, [] {
      LOGS_DEFAULT(WARNING) << "Experimental immediate WebGPU node replay is active.";
    });
    for (const auto& command : commands) {
      if (is_profiling_) recording.pending_kernels.emplace_back(*command.pending_kernel_info);
      DispatchCommand(command, recording);
      recording.has_unsubmitted_work.store(true, std::memory_order_relaxed);
      if (recording.num_pending_dispatches >= max_num_pending_dispatches_) {
        ORT_RETURN_IF_ERROR(Flush(manager, recording));
      }
    }
    return Status::OK();
  }
'''


def transform(source):
    signature = "Status WebGpuContext::Run(ComputeContextBase& context, const ProgramBase& program) {"
    source = replace_once(source, signature, signature + VALIDATE)
    signature = "Status WebGpuContext::AppendNodeCommands("
    start = source.index(signature)
    end = source.index("\nnamespace {", start)
    original = source[start:end]
    marker = "  const auto& manager = ComputeContextBase::BufferManagerAccessor::Get(context);\n"
    modified = replace_once(original, marker, marker + IMMEDIATE)
    return source[:start] + modified + source[end:]


def build_variant(output, flag, source_transform, comparator_transform=None, report_extra=None):
    output = Path(output)
    parent = json.loads((PARENT / "build-manifest.json").read_text())
    context = json.loads((CONTEXT / "build-manifest.json").read_text())
    for name, digest in context["overlay_source_sha256"].items():
        assert sha(CONTEXT / "overlay" / name) == digest, name
    comparator = json.loads((COMPARATOR / "build-manifest.json").read_text())
    assert sha(COMPARATOR / "native_compare.cpp") == comparator["source_sha256"]["native_compare.cpp"]
    output.mkdir(parents=True, exist_ok=False)
    source = output / "webgpu_context.cc"
    source.write_text(source_transform(SOURCE.read_text()))
    library = output / "libonnxruntime_providers_webgpu.so"
    link = parent["link_command"].copy()
    link[link.index("-o") + 1] = str(library)
    subprocess.run(link, cwd=parent["working_directory"], check=True)
    assert sha(library) == parent["provider_sha256"], "Parent relink mismatch"
    library.unlink()
    original_object = str(CONTEXT / "webgpu_context.cc.o")
    assert link.count(original_object) == 1
    command = next(item.copy() for item in context["compile_commands"]
                   if item[item.index("-c") + 1] == str(SOURCE))
    command[command.index("-c") + 1] = str(source)
    command[command.index("-o") + 1] = str(output / "webgpu_context.cc.o")
    subprocess.run(command, cwd=parent["working_directory"], check=True)
    link[link.index(original_object)] = str(output / "webgpu_context.cc.o")
    subprocess.run(link, cwd=parent["working_directory"], check=True)
    comparison = output / "native_compare.cpp"
    compare_source = (COMPARATOR / "native_compare.cpp").read_text()
    if comparator_transform is None:
        compare_source = transform_compare(replace_once(compare_source,
            'setenv("ORT_WEBGPU_DENSE_BLOCK32", value, 1)',
            f'setenv("{flag}", value, 1)'), flag=flag)
    else:
        compare_source = comparator_transform(compare_source)
    comparison.write_text(compare_source)
    comparator_command = ["c++", "-O3", "-std=c++20", str(comparison),
                          "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
                          "-L/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
                          "-Wl,-rpath,/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
                          "-lonnxruntime", "-o", str(output / "native_compare")]
    subprocess.run(comparator_command, check=True)
    source_hashes = {item.name: sha(item) for item in (source, comparison)}
    for name, digest in context["overlay_source_sha256"].items():
        item = CONTEXT / "overlay" / name
        source_hashes[os.path.relpath(item, output)] = digest
    report = dict(status="built_not_qualified", provider_sha256=sha(library),
                  baseline_provider_sha256=parent["provider_sha256"], baseline_byte_identical=True,
                  comparator_sha256=sha(output / "native_compare"), source_sha256=source_hashes,
                  compile_commands=[command], comparator_compile_command=comparator_command,
                  link_command=link, working_directory=parent["working_directory"],
                  shader_and_arithmetic_unchanged=True, no_header_or_object_layout_changes=True,
                  original_deferred_path_preserved=True, pending_dispatch_threshold_unchanged=True,
                  preceding_dispatch_order_preserved=True, default_enabled=False,
                  quality_qualified=False, headline_eligible=False)
    if report_extra is not None:
        report.update(report_extra)
    (output / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("status", "provider_sha256", "baseline_byte_identical")}))


def main():
    build_variant(OUTPUT, FLAG, transform)


if __name__ == "__main__":
    main()
