import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from build_qmoe_node_replay import replace_once, sha


def wrap(source, start, outputs, row_elements=None):
    head, tail = source.split(start, 1)
    indent = start[:len(start) - len(start.lstrip())]
    close = indent[:-2] + "}"
    body, end = tail.split(indent + "return context.RunProgram(program);\n" + close, 1)
    extra = ", " + row_elements if row_elements else ""
    return (head + indent + "return static_replay_.Run(context, {" + outputs + "}, [&]() -> Status {\n" +
            start + body + indent + "return context.RunProgram(program);\n" +
            indent + "}" + extra + ");\n" + close + end)


def transform(sources):
    for name in ("contrib_ops/webgpu/bert/linear_attention_gates.h",
                 "contrib_ops/webgpu/bert/scaled_silu.h", "quickgelu_mul_overlay.h"):
        sources[name] = replace_once(sources[name], "#pragma once",
                                     '#pragma once\n#include "static_node_replay.h"')
    for name, cls, anchor in (
        ("contrib_ops/webgpu/bert/linear_attention_gates.h", "GatedRMSNorm",
         "  GatedRMSNormActivation activation_;\n  float epsilon_;"),
        ("contrib_ops/webgpu/bert/scaled_silu.h", "SplitScaledSiLUF16",
         'class SplitScaledSiLUF16 final : public WebGpuKernel {'),
    ):
        if cls == "GatedRMSNorm":
            sources[name] = replace_once(sources[name], anchor,
                f'  mutable onnxruntime::webgpu::StaticNodeReplay static_replay_{{"{cls}", {{0, 1, 2}}}};\n' + anchor)
        else:
            head, tail = sources[name].split(anchor, 1)
            tail = replace_once(tail, " private:", f' private:\n  mutable onnxruntime::webgpu::StaticNodeReplay static_replay_{{"{cls}", {{0}}}};')
            sources[name] = head + anchor + tail
    sources["quickgelu_mul_overlay.h"] = replace_once(
        sources["quickgelu_mul_overlay.h"],
        "  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override;",
        """  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override;
 private:
  mutable onnxruntime::webgpu::StaticNodeReplay static_replay_{"QuickGeluMulF16", {0, 1}};""")
    for name, start, outputs in (
        ("contrib_ops/webgpu/bert/linear_attention_gates.cc",
         "  GatedRMSNormProgram program{activation_};", "output"),
        ("contrib_ops/webgpu/bert/scaled_silu.cc", "  SplitScaledSiLUF16Program program;", "activated, tail"),
        ("quickgelu_mul_overlay.cc", "  QuickGeluMulF16Program program;", "y"),
    ):
        assert sources[name].count(start) == 1
        sources[name] = wrap(sources[name], start, outputs)
    header = "contrib_ops/webgpu/bert/hyper_connection_pre_mix.h"
    if header in sources:
        sources[header] = replace_once(sources[header], "#pragma once",
                                       '#pragma once\n#include "static_node_replay.h"')
        sources[header] = replace_once(sources[header], "  int64_t num_branches_;",
            '  mutable onnxruntime::webgpu::StaticNodeReplay static_replay_{"HyperConnectionPreMix", {0, 1}};\n'
            "  int64_t num_branches_;")
        name = "contrib_ops/webgpu/bert/hyper_connection_pre_mix.cc"
        sources[name] = wrap(sources[name],
            "  HyperConnectionPreMixProgram program{static_cast<int>(layout), fuse_sigmoid_};",
            "y", "params.branches * params.hidden")
        name = "hc_post_norm_overlay.cc"
        sources[name] = replace_once(sources[name],
            '#include "core/providers/webgpu/webgpu_kernel.h"',
            '#include "core/providers/webgpu/webgpu_kernel.h"\n#include "static_node_replay.h"')
        sources[name] = replace_once(sources[name],
            'epsilon_{info.GetAttrOrDefault<float>("epsilon", 1e-6f)} {}',
            'epsilon_{info.GetAttrOrDefault<float>("epsilon", 1e-6f)},\n'
            '        static_replay_{shared ? "HCSharedPostNormF16" : "HCPostNormF16",\n'
            '                       shared ? std::initializer_list<int>{0, 1, 2, 3, 4, 5} :\n'
            '                                std::initializer_list<int>{0, 1, 2, 3}} {}')
        sources[name] = replace_once(sources[name], "  float epsilon_;",
            "  float epsilon_;\n  mutable onnxruntime::webgpu::StaticNodeReplay static_replay_;")
        sources[name] = wrap(sources[name],
            "    HCPostNormProgram program{shared_, scale->Shape().Size() == params.hidden, hidden};",
            "y, normalized", "params.branches * params.hidden")
    name = "core/providers/webgpu/webgpu_context.cc"
    sources[name] = replace_once(
        sources[name],
        'ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_QMOE_COMMAND_REPLAY", 0) == 1) {',
        '(ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_QMOE_COMMAND_REPLAY", 0) == 1 ||\n'
        '           ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_STATIC_NODE_REPLAY", 0) == 1)) {')
    header = "contrib_ops/webgpu/quantization/matmul_nbits.h"
    if header in sources:
        sources[header] = replace_once(sources[header], "#pragma once",
                                       '#pragma once\n#include "static_node_replay.h"')
        sources[header] = replace_once(
            sources[header], "  int64_t K_;",
            '  mutable onnxruntime::webgpu::StaticNodeReplay static_replay_{"MatMulNBitsExact", {0, 1, 2}};\n  int64_t K_;')
        source = "contrib_ops/webgpu/quantization/matmul_nbits.cc"
        original = "  return ApplyMatMulNBits(a, b, scales, zero_points, bias, K_, N_, block_size_, accuracy_level_, bits_, context, y, 0);"
        sources[source] = replace_once(sources[source], original, """
  auto compute = [&]() {
    return ApplyMatMulNBits(a, b, scales, zero_points, bias, K_, N_, block_size_, accuracy_level_, bits_, context, y, 0);
  };
  const bool exact_route =
      context.AdapterInfo().vendor == std::string_view{"nvidia"} &&
      bits_ == 4 && block_size_ == 32 && !has_zero_points && bias == nullptr &&
      !context.EnableMatmulFp32Accumulation() &&
      a->IsDataType<MLFloat16>() && scales->IsDataType<MLFloat16>() &&
      ((K_ == 2560 && (N_ == 512 || N_ == 640) &&
        ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_EXACT_DENSE_MID", 0) != 0) ||
       (K_ == 10240 && N_ == 324 &&
        ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_HC_EXACT_DENSE", 0) != 0));
  if (!exact_route) return compute();
  return static_replay_.Run(context, {y}, compute);""")
    context_header = "core/providers/webgpu/webgpu_context.h"
    if context_header in sources:
        sources[context_header] = replace_once(
            sources[context_header], "struct CapturedCommandInfo {", """struct NodeBindingSnapshot {
  std::vector<WGPUBuffer> buffers;
  std::vector<uint32_t> segments;
  wgpu::BindGroupLayout layout;
  WGPUBuffer uniform_buffer = nullptr;
  std::string program_name;
  size_t programs = 0;
};

struct CapturedCommandInfo {""")
        sources[context_header] = replace_once(
            sources[context_header], "  bool CanCaptureNodeCommands(const ComputeContextBase& context) const;",
            """  void BeginNodeBindingCapture(const ComputeContextBase& context,
                               std::vector<webgpu::CapturedCommandInfo>& commands,
                               webgpu::NodeBindingSnapshot& snapshot);
  void EndNodeBindingCapture(const ComputeContextBase& context);
  Status PrepareNodeBinding(const webgpu::CapturedCommandInfo& prototype,
                          const webgpu::NodeBindingSnapshot& snapshot,
                          const std::vector<WGPUBuffer>& buffers,
                          webgpu::CapturedCommandInfo& command);
  bool CanCaptureNodeCommands(const ComputeContextBase& context) const;""")
        compute_header = "core/providers/webgpu/compute_context.h"
        sources[compute_header] = replace_once(
            sources[compute_header], "  bool CanCaptureNodeCommands() const",
            """  void BeginNodeBindingCapture(std::vector<CapturedCommandInfo>& commands, NodeBindingSnapshot& snapshot) {
    webgpu_context_.BeginNodeBindingCapture(*this, commands, snapshot);
  }
  void EndNodeBindingCapture() { webgpu_context_.EndNodeBindingCapture(*this); }
  Status PrepareNodeBinding(const CapturedCommandInfo& prototype, const NodeBindingSnapshot& snapshot,
                             const std::vector<WGPUBuffer>& buffers, CapturedCommandInfo& command) {
    return webgpu_context_.PrepareNodeBinding(prototype, snapshot, buffers, command);
  }
  bool CanCaptureNodeCommands() const""")
        source = "core/providers/webgpu/webgpu_context.cc"
        sources[source] = replace_once(
            sources[source], "Status WebGpuContext::Run(ComputeContextBase& context, const ProgramBase& program) {",
            Path(__file__).with_name("binding_node_context.inc").read_text() +
            "\nStatus WebGpuContext::Run(ComputeContextBase& context, const ProgramBase& program) {")
        sources[source] = replace_once(
            sources[source], "  if (use_dynamic_uniforms && uniform_buffer) {", """
  if (active_node_binding_capture.context == &context) {
    ORT_ENFORCE(!use_uniform_arena && uniform_offset == 0 &&
                    (uniform_buffer == nullptr || captured_node_uniform),
                "Node binding capture requires immutable non-arena uniforms.");
    auto& snapshot = *active_node_binding_capture.snapshot;
    ++snapshot.programs;
    snapshot.buffers = bind_buffers;
    snapshot.segments = bind_buffers_segments;
    snapshot.layout = bind_group_layout;
    snapshot.uniform_buffer = uniform_buffer;
    snapshot.program_name = program.Name();
  }
  if (use_dynamic_uniforms && uniform_buffer) {""")
    return sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", choices=("v8",), default="v8")
    revision = parser.parse_args().revision
    root = Path(__file__).parent
    repos = Path("/workspace")
    repository = repos / "onnxruntime-webgpu-h200-perf"
    replay_dir = repos / "qwen38-int4-qmoe-node-replay/build-v5"
    replay = json.loads((replay_dir / "build-manifest.json").read_text())
    base = json.loads((repos / "qwen38-int4-hc-node-replay-combined/build-v1/build-manifest.json").read_text())
    output = repos / ("qwen38-int4-static-node-replay/build-" + revision)
    output.mkdir(parents=True, exist_ok=False)
    scratch = output / "scratch"
    scratch.mkdir()
    environment = dict(os.environ, TMPDIR=str(scratch))
    library = output / "libonnxruntime_providers_webgpu.so"
    link = base["link_command"].copy()
    link[link.index("-o") + 1] = str(library)
    subprocess.run(link, cwd=base["working_directory"], env=environment, check=True)
    assert sha(library) == base["provider_sha256"], "Baseline relink mismatch"
    library.unlink()
    originals = {}
    for name in ("contrib_ops/webgpu/bert/linear_attention_gates.h",
                 "contrib_ops/webgpu/bert/linear_attention_gates.cc",
                 "contrib_ops/webgpu/bert/scaled_silu.h"):
        originals[name] = (repository / "onnxruntime" / name).read_text()
    originals["contrib_ops/webgpu/bert/scaled_silu.cc"] = (
        repository / "build-int4-split-silu-v3/scaled_silu.cc").read_text()
    for name in ("quickgelu_mul_overlay.h", "quickgelu_mul_overlay.cc"):
        originals[name] = (repos / "qwen38-int4-quickgelu-next/build" / name).read_text()
    originals["core/providers/webgpu/webgpu_context.cc"] = (
        replay_dir / "overlay/core/providers/webgpu/webgpu_context.cc").read_text()
    if revision == "v8":
        originals["contrib_ops/webgpu/quantization/matmul_nbits.h"] = (
            repository / "onnxruntime/contrib_ops/webgpu/quantization/matmul_nbits.h").read_text()
        originals["contrib_ops/webgpu/quantization/matmul_nbits.cc"] = (
            repository / "build-int4-shared-post-v4/matmul_nbits.cc").read_text()
        for name in ("core/providers/webgpu/webgpu_context.h", "core/providers/webgpu/compute_context.h"):
            originals[name] = (replay_dir / "overlay" / name).read_text()
        originals["contrib_ops/webgpu/bert/hyper_connection_pre_mix.h"] = (
            repository / "onnxruntime/contrib_ops/webgpu/bert/hyper_connection_pre_mix.h").read_text()
        originals["contrib_ops/webgpu/bert/hyper_connection_pre_mix.cc"] = (
            repository / "build-int4-hc-fusion/hyper_connection_pre_mix.cc").read_text()
        originals["hc_post_norm_overlay.cc"] = (
            repos / "qwen38-int4-hc-post-norm/build-v8/hc_post_norm_overlay.cc").read_text()
    sources = transform(dict(originals))
    for name, text in sources.items():
        target = output / "overlay" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    shutil.copy2(root / "static_node_replay.h", output / "overlay/static_node_replay.h")
    commands = []
    names = ["contrib_ops/webgpu/bert/linear_attention_gates.cc",
             "contrib_ops/webgpu/bert/scaled_silu.cc", "quickgelu_mul_overlay.cc",
             "core/providers/webgpu/webgpu_context.cc"]
    if revision == "v8":
        names.extend(["contrib_ops/webgpu/quantization/matmul_nbits.cc",
                      "contrib_ops/webgpu/bert/hyper_connection_pre_mix.cc",
                      "hc_post_norm_overlay.cc"])
    for name in names:
        command = replay["compile_commands"][0].copy()
        command[1:1] = ["-I" + str(output / "overlay")]
        command[command.index("-c") + 1] = str(output / "overlay" / name)
        obj = output / (Path(name).name + ".o")
        command[command.index("-o") + 1] = str(obj)
        subprocess.run(command, cwd=base["working_directory"], env=environment, check=True)
        suffix = ("webgpu_context.cc.o" if name.endswith("webgpu_context.cc") else
                  "linear_attention_gates.cc.o" if name.endswith("linear_attention_gates.cc") else
                  Path(name).name + ".o")
        matches = [p for p in link if p.endswith("/" + suffix) or p.endswith("__" + suffix)]
        assert len(matches) == 1, matches
        link[link.index(matches[0])] = str(obj)
        commands.append(command)
    subprocess.run(link, cwd=base["working_directory"], env=environment, check=True)
    native_source = output / "native_compare.cpp"
    shutil.copy2(root / "static_node_replay_compare.cpp", native_source)
    subprocess.run(["c++", "-O3", "-std=c++20", str(native_source),
                    "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
                    "-L/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
                    "-Wl,-rpath,/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
                    "-lonnxruntime", "-o", str(output / "native_compare")], check=True)
    report = dict(status="built_not_qualified", provider_sha256=sha(library),
                  baseline_provider_sha256=base["provider_sha256"], baseline_byte_identical=True,
                  compile_commands=commands, link_command=link, working_directory=base["working_directory"],
                  original_source_sha256={k: hashlib.sha256(v.encode()).hexdigest() for k, v in originals.items()},
                  overlay_source_sha256={k: hashlib.sha256(v.encode()).hexdigest() for k, v in sources.items()},
                  helper_sha256=sha(output / "overlay/static_node_replay.h"),
                  quality_qualified=False, headline_eligible=False)
    (output / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "provider_sha256", "baseline_byte_identical")}))


if __name__ == "__main__":
    main()
