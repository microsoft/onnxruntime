import json
import os
from pathlib import Path
import subprocess

from build_dense_double_buffer import transform_compare
from build_qmoe_node_replay import replace_once, sha


FLAG = "ORT_WEBGPU_GDN_PREP_NODE_REPLAY"
KERNEL = "GdnProjPrepReplayF16"
PARENT = Path("/workspace/qwen38-int4-qmoe-prefill-rows4/build-v2")
COMPONENT = Path("/workspace/onnxruntime-webgpu-h200-perf/build-int4-gdn-prep")
REPO = Path("/workspace/onnxruntime-webgpu-h200-perf")
SOURCE = REPO / "onnxruntime/contrib_ops/webgpu/bert/causal_conv_with_state.cc"
HEADER = SOURCE.with_suffix(".h")
REPLAY = Path("/workspace/qwen38-int4-dense-bitcast-unpack/build-v1/overlay/static_node_replay.h")
CONTEXT = Path("/workspace/qwen38-int4-static-node-replay/build-v8")
OUTPUT = Path("/workspace/qwen38-int4-gdn-prep-node-replay/build-v2")
COMPARATOR = Path("/workspace/qwen38-int4-dense-block32/build-v1/native_compare.cpp")


def transform_header(source):
    source = replace_once(source, '#include <string>',
                          '#include <string>\n#include "gdn_prep_node_replay.h"')
    source = replace_once(source, "class GdnProjPrepF16 final",
                          f"class {KERNEL} final")
    source = replace_once(source,
        "  explicit GdnProjPrepF16(const OpKernelInfo& info) : WebGpuKernel(info) {}",
        f'''  explicit {KERNEL}(const OpKernelInfo& info) : WebGpuKernel(info) {{
    const int mode = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
    ORT_ENFORCE(mode == 0 || mode == 1, "{FLAG} must be 0 or 1");
    replay_enabled_ = mode == 1;
  }}''')
    start = source.index(f"class {KERNEL} final")
    end = source.index("\n};", start)
    return source[:end] + '''

 private:
  bool replay_enabled_ = false;
  mutable onnxruntime::webgpu::GdnPrepNodeReplay replay_{
      "GdnProjPrepF16Exact", {0, 1, 2, 3, 4}};''' + source[end:]


def transform_source(source):
    source = replace_once(source, "    GdnProjPrepF16);", f"    {KERNEL});")
    source = replace_once(source, "Status GdnProjPrepF16::ComputeInternal",
                          f"Status {KERNEL}::ComputeInternal")
    start = source.index("  GdnProjPrepF16Program program;",
                         source.index(f"Status {KERNEL}::ComputeInternal"))
    end = source.index("\n}", start)
    program = source[start:end]
    return source[:start] + '''  auto compute = [&]() -> Status {
''' + "\n".join("  " + line for line in program.splitlines()) + '''
  };
  if (replay_enabled_ && batch == 1 && length == 1) {
    return replay_.Run(context, {conv, z, a, b, present}, compute);
  }
  return compute();''' + source[end:]


GPU_BANKS = r'''
      if (inputs.size() != 5) throw std::runtime_error("Expected five GDN prep inputs");
      auto gpu_memory = devices.front().GetMemoryInfo(OrtDeviceMemoryType_DEFAULT);
      Ort::Allocator gpu_allocator{session, gpu_memory};
      std::vector<std::vector<Ort::Value>> gpu_inputs(3);
      for (auto& bank : gpu_inputs) {
        for (const auto& input : inputs) {
          const auto info = input.GetTensorTypeAndShapeInfo();
          const auto shape = info.GetShape();
          if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
            throw std::runtime_error("Expected FP16 GDN prep inputs");
          bank.emplace_back(Ort::Value::CreateTensor<Ort::Float16_t>(
              gpu_allocator, shape.data(), shape.size()));
          if (bank.back().GetTensorMemoryInfo().GetDeviceType() != OrtMemoryInfoDeviceType_GPU)
            throw std::runtime_error("GDN prep fixture did not allocate GPU memory");
        }
      }
      for (size_t bank = 0; bank < gpu_inputs.size(); ++bank)
        for (size_t other = 0; other < bank; ++other)
          for (size_t input = 0; input < inputs.size(); ++input)
            if (inputs[input].GetTensorTypeAndShapeInfo().GetElementCount() > 0 &&
                gpu_inputs[bank][input].GetTensorRawData() == gpu_inputs[other][input].GetTensorRawData())
              throw std::runtime_error("Expected distinct live input buffer addresses");
      std::cout << "GDN prep distinct_live_gpu_banks=3 inputs_each=5\n";
'''


def transform_comparator(source):
    source = replace_once(source, 'setenv("ORT_WEBGPU_DENSE_BLOCK32", value, 1)',
                          f'setenv("{FLAG}", value, 1)')
    source = transform_compare(source, flag=FLAG)
    source = replace_once(source,
        "    for (int file = 2; file < argc; ++file) {",
        "    for (int file = 2; file < argc; ++file) {\n"
        "      for (int pending : {1, 64}) {")
    source = replace_once(source, 'file % 2 ? "1" : "64"', 'pending == 1 ? "1" : "64"')
    source = replace_once(source, "      Ort::Session session{env, argv[file], options};",
        '''      flag("0");
      Ort::Session session{env, argv[file], options};
      flag("1");
      Ort::Session candidate_session{env, argv[file], options};''')
    source = replace_once(source, "      for (int round = 0; round < 128; ++round) {",
                          GPU_BANKS + "      for (int round = 0; round < 128; ++round) {")
    start = source.index('        const char* exact_mode =')
    end = source.index("        for (size_t index =", start)
    source = source[:start] + source[end:]
    source = replace_once(source, '        flag("0");',
        '''        auto& reference_bank = gpu_inputs[0];
        auto& candidate_bank = gpu_inputs[1 + round % 2];
        auto copied = env.CopyTensors(inputs, reference_bank, nullptr);
        if (!copied.IsOK()) throw std::runtime_error(copied.GetErrorMessage());
        copied = env.CopyTensors(inputs, candidate_bank, nullptr);
        if (!copied.IsOK()) throw std::runtime_error(copied.GetErrorMessage());
        flag("0");''')
    source = replace_once(source,
        "auto reference = session.Run(Ort::RunOptions{}, names.data(), inputs.data(), inputs.size(),",
        "auto reference = session.Run(Ort::RunOptions{}, names.data(), reference_bank.data(), reference_bank.size(),")
    source = replace_once(source,
        "auto actual = session.Run(Ort::RunOptions{}, names.data(), inputs.data(), inputs.size(),",
        "auto actual = candidate_session.Run(Ort::RunOptions{}, names.data(), candidate_bank.data(), candidate_bank.size(),")
    source = replace_once(source,
        "        session.Run(Ort::RunOptions{}, names.data(), inputs.data(), inputs.size(), outputs.data(), outputs.size());",
        "        Ort::Session invalid_session{env, argv[file], options};")
    source = replace_once(source,
        '                << " changing_rounds=128 invalid_flag_rejected=1" << std::endl;\n    }',
        '                << " changing_rounds=128 invalid_flag_rejected=1 pending=" << pending << std::endl;\n'
        '      }\n    }')
    return source


def main():
    parent = json.loads((PARENT / "build-manifest.json").read_text())
    component = json.loads((COMPONENT / "build-manifest.json").read_text())
    for path in (SOURCE, HEADER):
        assert sha(path) == component["source_sha256"][str(path)], str(path)
    context_manifest = json.loads((CONTEXT / "build-manifest.json").read_text())
    context_headers = tuple((CONTEXT / "overlay/core/providers/webgpu" / name)
                            for name in ("compute_context.h", "webgpu_context.h"))
    for path in context_headers:
        assert sha(path) == context_manifest["overlay_source_sha256"][
            str(path.relative_to(CONTEXT / "overlay"))], str(path)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    overlay = OUTPUT / "overlay"
    header = overlay / "contrib_ops/webgpu/bert/causal_conv_with_state.h"
    header.parent.mkdir(parents=True)
    header.write_text(transform_header(HEADER.read_text()))
    replay = overlay / "gdn_prep_node_replay.h"
    replay.write_text(REPLAY.read_text().replace("StaticNodeReplay", "GdnPrepNodeReplay"))
    source = OUTPUT / "causal_conv_with_state.cc"
    source.write_text(transform_source(SOURCE.read_text()))
    library = OUTPUT / "libonnxruntime_providers_webgpu.so"
    link = parent["link_command"].copy()
    link[link.index("-o") + 1] = str(library)
    subprocess.run(link, cwd=parent["working_directory"], check=True)
    assert sha(library) == parent["provider_sha256"], "Parent relink mismatch"
    library.unlink()
    command = next(c.copy() for c in component["compile_commands"]
                   if c[c.index("-c") + 1] == str(SOURCE))
    original_object = command[command.index("-o") + 1]
    baseline = OUTPUT / "original.cc.o"
    command[command.index("-o") + 1] = str(baseline)
    subprocess.run(command, cwd=parent["working_directory"], check=True)
    assert sha(baseline) == sha(Path(original_object)), "Original GDN prep object mismatch"
    command.insert(1, "-I" + str(CONTEXT / "overlay"))
    command.insert(1, "-I" + str(overlay))
    command[command.index("-c") + 1] = str(source)
    obj = OUTPUT / "causal_conv_with_state.cc.o"
    command[command.index("-o") + 1] = str(obj)
    subprocess.run(command, cwd=parent["working_directory"], check=True)
    assert link.count(original_object) == 1
    link[link.index(original_object)] = str(obj)
    subprocess.run(link, cwd=parent["working_directory"], check=True)
    comparison = OUTPUT / "native_compare.cpp"
    comparison.write_text(transform_comparator(COMPARATOR.read_text()))
    comparator_command = ["c++", "-O3", "-std=c++20", str(comparison),
        "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
        "-L/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-Wl,-rpath,/workspace/qwen38-fp8-webgpu-runtime/sdk/lib", "-lonnxruntime",
        "-o", str(OUTPUT / "native_compare")]
    subprocess.run(comparator_command, check=True)
    sources = (SOURCE, HEADER, REPLAY, COMPARATOR, source, header, replay, comparison, *context_headers)
    report = dict(status="built_not_qualified", provider_sha256=sha(library),
        baseline_provider_sha256=parent["provider_sha256"], baseline_byte_identical=True,
        original_object_byte_identical=True, comparator_sha256=sha(OUTPUT / "native_compare"),
        source_sha256={os.path.relpath(p, OUTPUT): sha(p) for p in sources},
        compile_commands=[command], link_command=link, working_directory=parent["working_directory"],
        constructor_static_selector=True, private_kernel_type=KERNEL, private_replay_type=True,
        shader_and_arithmetic_unchanged=True, default_enabled=False,
        quality_qualified=False, headline_eligible=False)
    (OUTPUT / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "provider_sha256", "original_object_byte_identical")}))


if __name__ == "__main__":
    main()
