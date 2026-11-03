import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(text, old, new):
    assert text.count(old) == 1, old[:100]
    return text.replace(old, new)


def prepare_gate_templates(repository, output):
    source = output / "template-input/moe"
    source.mkdir(parents=True)
    template = repository / "onnxruntime/contrib_ops/webgpu/moe/gate_1token.wgsl.template"
    shutil.copy2(template, source / template.name)
    partial = output / "partial/wgsl_template_gen"
    subprocess.run(["python3", str(repository / "tools/python/wgsl_gen.py"),
                    "-i", str(source.parent), "-o", str(partial),
                    "-g", "static-cpp-literal", "-I", "wgsl_template_gen/"], check=True)
    original = repository / "build-int4-uniform-arena/wgsl_template_gen"
    overlay = output / "overlay/wgsl_template_gen"
    shutil.copytree(original, overlay)
    pattern = (r'template <>\nstruct TemplateParameter<"moe/gate_1token\.wgsl\.template"> \{'
               r'.*?::type params\);')
    matches = re.findall(pattern, (partial / "index.h").read_text(), flags=re.S)
    assert len(matches) == 1
    index, count = re.subn(pattern, lambda _: matches[0], (overlay / "index.h").read_text(), flags=re.S)
    assert count == 1
    (overlay / "index.h").write_text(index)
    header = Path("generated/moe/gate_1token.h")
    shutil.copy2(partial / header, overlay / header)
    changed = {str(path.relative_to(overlay)) for path in overlay.rglob("*")
               if path.is_file() and sha(path) != sha(original / path.relative_to(overlay))}
    assert changed == {"index.h", str(header)}, changed
    return sha(template)


def transform(sources, root):
    header = sources["contrib_ops/webgpu/moe/qmoe.h"]
    header = replace_once(header, "#pragma once", """#pragma once
#include <array>
#include <mutex>
#include "core/providers/webgpu/webgpu_context.h"
""")
    header = replace_once(header, "  Status ComputeInternal(ComputeContext& context) const override;",
                          "  ~QMoE() override;\n  Status ComputeInternal(ComputeContext& context) const override;")
    header = replace_once(header, " private:", """ private:
  struct ReplayEntry {
    std::vector<uint64_t> key;
    uint64_t last_used = 0;
    std::vector<onnxruntime::webgpu::CapturedCommandInfo> commands;
  };
  Status ComputeWithWorkspace(ComputeContext& context,
                             std::array<std::optional<Tensor>, 5>* scratch) const;
  mutable std::mutex replay_mutex_;
  mutable std::array<std::optional<Tensor>, 5> replay_scratch_;
  mutable std::vector<uint64_t> replay_workspace_key_;
  mutable std::vector<ReplayEntry> replay_entries_;
  mutable uint64_t replay_requests_ = 0;
  mutable uint64_t replay_hits_ = 0;
  mutable uint64_t replay_capacity_fallbacks_ = 0;
  mutable uint64_t replay_captures_ = 0;
  mutable uint64_t replay_evictions_ = 0;
  mutable uint64_t replay_shape_invalidations_ = 0;
  mutable uint64_t replay_clock_ = 0;
""")
    sources["contrib_ops/webgpu/moe/qmoe.h"] = header
    source = sources["contrib_ops/webgpu/moe/qmoe.cc"]
    source = replace_once(source, "#include <atomic>",
                          '#include <algorithm>\n#include <cstdlib>\n#include <fstream>\n#include "gsl/gsl"\n#include <atomic>')
    source = replace_once(source, "Status QMoE::ComputeInternal(ComputeContext& context) const {",
                          (root / "qmoe_node_replay.inc").read_text() + """
Status QMoE::ComputeWithWorkspace(ComputeContext& context,
                                 std::array<std::optional<Tensor>, 5>* scratch) const {""")
    source = replace_once(source, "  if (use_indirect_experts) {", """  if (use_indirect_experts) {
    std::array<std::optional<Tensor>, 5> local_scratch;
    auto temporary = [&](size_t index, MLDataType type, const TensorShape& shape) -> Tensor& {
      auto& tensor = scratch == nullptr ? local_scratch[index] : (*scratch)[index];
      if (!tensor) {
        tensor.emplace(context.CreateGPUTensor(type, shape));
      } else {
        ORT_ENFORCE(tensor->DataType() == type && tensor->Shape() == shape,
                    "QMoE replay workspace shape/type changed.");
      }
      return *tensor;
    };""")
    for index, (name, dtype, shape) in enumerate((
        ("router_values", "dtype", "gate_value_shape"),
        ("indirect_experts", "dtype_uint32", "indirect_experts_shape"),
        ("fc1_outputs", "dtype", "fc1_output_shape"),
        ("fc1_activated", "dtype", "fc1_activated_shape"),
        ("fc2_outputs", "dtype", "fc2_output_shape"),
    )):
        old = f"    Tensor {name} = context.CreateGPUTensor({dtype}, {shape});"
        assert old in source
        source = source.replace(old, f"    Tensor& {name} = temporary({index}, {dtype}, {shape});", 1)
    sources["contrib_ops/webgpu/moe/qmoe.cc"] = source

    header = sources["core/providers/webgpu/webgpu_context.h"]
    header = replace_once(header, "  Status Flush(const webgpu::BufferManager& buffer_mgr, CommandRecordingState& recording);",
                          """  bool CanCaptureNodeCommands(const ComputeContextBase& context) const;
  bool NodeReplayProfilingEnabled() const { return is_profiling_; }
  void BeginNodeCapture(const ComputeContextBase& context, std::vector<webgpu::CapturedCommandInfo>& commands);
  void EndNodeCapture(const ComputeContextBase& context);
  Status FlushNodeCommands(const ComputeContextBase& context);
  Status AppendNodeCommands(const ComputeContextBase& context,
                            const std::vector<webgpu::CapturedCommandInfo>& commands);
  Status Flush(const webgpu::BufferManager& buffer_mgr, CommandRecordingState& recording);""")
    sources["core/providers/webgpu/webgpu_context.h"] = header
    header = sources["core/providers/webgpu/compute_context.h"]
    header = replace_once(header, " protected:\n  WebGpuContext& webgpu_context_;", """
  bool CanCaptureNodeCommands() const { return webgpu_context_.CanCaptureNodeCommands(*this); }
  bool NodeReplayProfilingEnabled() const { return webgpu_context_.NodeReplayProfilingEnabled(); }
  void BeginNodeCapture(std::vector<webgpu::CapturedCommandInfo>& commands) {
    webgpu_context_.BeginNodeCapture(*this, commands);
  }
  void EndNodeCapture() { webgpu_context_.EndNodeCapture(*this); }
  Status FlushNodeCommands() { return webgpu_context_.FlushNodeCommands(*this); }
  Status AppendNodeCommands(const std::vector<webgpu::CapturedCommandInfo>& commands) {
    return webgpu_context_.AppendNodeCommands(*this, commands);
  }
 protected:
  WebGpuContext& webgpu_context_;""")
    sources["core/providers/webgpu/compute_context.h"] = header
    source = sources["core/providers/webgpu/webgpu_context.cc"]
    source = replace_once(source, "Status WebGpuContext::Run(ComputeContextBase& context, const ProgramBase& program) {",
                          (root / "node_replay_context.inc").read_text() +
                          "\nStatus WebGpuContext::Run(ComputeContextBase& context, const ProgramBase& program) {")
    source = replace_once(source, "  WGPUBuffer uniform_buffer = nullptr;",
                          "  WGPUBuffer uniform_buffer = nullptr;\n  wgpu::Buffer captured_node_uniform;")
    old = """      uniform_buffer = buffer_mgr.Create(recording, uniform_buffer_total_size,
                                         wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Uniform);"""
    source = replace_once(source, old, """      if (recording.graph_capture_state == GraphCaptureState::Capturing &&
          !context.IsGraphCaptureEnabled() &&
          ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_QMOE_COMMAND_REPLAY", 0) == 1) {
        // Captured uniforms must never return to an allocator that can overwrite them.
        wgpu::BufferDescriptor descriptor{};
        descriptor.size = uniform_buffer_total_size;
        descriptor.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Uniform;
        captured_node_uniform = device_.CreateBuffer(&descriptor);
        ORT_ENFORCE(captured_node_uniform, "Cannot allocate captured WebGPU node uniforms.");
        uniform_buffer = captured_node_uniform.Get();
      } else {
        uniform_buffer = buffer_mgr.Create(recording, uniform_buffer_total_size,
                                          wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Uniform);
      }""")
    source = replace_once(source, "  if (uniform_buffer && !use_uniform_arena) {",
                          "  if (uniform_buffer && !use_uniform_arena && !captured_node_uniform) {")
    sources["core/providers/webgpu/webgpu_context.cc"] = source
    return sources


def main():
    root = Path(__file__).parent
    repository = Path("/workspace/onnxruntime-webgpu-h200-perf")
    prior = json.loads(Path("/workspace/qwen38-int4-gdn-decode-view/build-original/build-manifest.json").read_text())
    output = Path("/workspace/qwen38-int4-qmoe-node-replay/build-v5")
    output.mkdir(parents=True, exist_ok=False)
    scratch = output / "scratch"
    scratch.mkdir()
    environment = dict(os.environ, TMPDIR=str(scratch))
    library = output / "libonnxruntime_providers_webgpu.so"
    link = prior["link_command"].copy()
    link[link.index("-o") + 1] = str(library)
    subprocess.run(link, cwd=prior["working_directory"], env=environment, check=True)
    assert sha(library) == prior["provider_sha256"], "Baseline relink mismatch"
    library.unlink()
    names = ("contrib_ops/webgpu/moe/qmoe.h", "contrib_ops/webgpu/moe/qmoe.cc",
             "core/providers/webgpu/webgpu_context.h", "core/providers/webgpu/webgpu_context.cc",
             "core/providers/webgpu/compute_context.h")
    originals = {name: (repository / "onnxruntime" / name).read_text() for name in names}
    sources = transform(dict(originals), root)
    for name, text in sources.items():
        target = output / "overlay" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    template_hash = prepare_gate_templates(repository, output)
    commands = []
    for name in ("contrib_ops/webgpu/moe/qmoe.cc", "core/providers/webgpu/webgpu_context.cc",
                 "core/providers/webgpu/wgsl_templates/wgsl_gen.cc"):
        command = prior["compile_command"].copy()
        command[command.index("-c") + 1] = str(output / "overlay" / name if name in sources
                                               else repository / "onnxruntime" / name)
        obj = output / (Path(name).name + ".o")
        command[command.index("-o") + 1] = str(obj)
        command[1:1] = ["-I" + str(output / "overlay"),
                        "-I" + str(repository / "build-int4-uniform-arena")]
        subprocess.run(command, cwd=prior["working_directory"], env=environment, check=True)
        suffix = name.replace("/", "__") + ".o"
        matches = [p for p in link if p.endswith(suffix)]
        assert len(matches) == 1, matches
        link[link.index(matches[0])] = str(obj)
        commands.append(command)
    subprocess.run(link, cwd=prior["working_directory"], env=environment, check=True)
    native = (root / "qmoe_node_replay_compare.cpp").read_text()
    native_source = output / "native_compare.cpp"
    native_source.write_text(native)
    subprocess.run(["c++", "-O3", "-std=c++20", str(native_source),
                    "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
                    "-L/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
                    "-Wl,-rpath,/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
                    "-lonnxruntime", "-o", str(output / "native_compare")], check=True)
    report = dict(status="built_not_qualified", provider_sha256=sha(library),
                  baseline_provider_sha256=prior["provider_sha256"], baseline_byte_identical=True,
                  compile_commands=commands, link_command=link, working_directory=prior["working_directory"],
                  gate_template_sha256=template_hash,
                  original_source_sha256={name: hashlib.sha256(text.encode()).hexdigest()
                                          for name, text in originals.items()},
                  overlay_source_sha256={name: hashlib.sha256(text.encode()).hexdigest()
                                         for name, text in sources.items()},
                  quality_qualified=False, headline_eligible=False)
    (output / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "provider_sha256", "baseline_byte_identical")}))


if __name__ == "__main__":
    main()
