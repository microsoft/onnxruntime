import json
from pathlib import Path
import subprocess

from build_qmoe_node_replay import replace_once, sha
from build_scoped_webgpu_source import PARENT, PROOF


ROOT = Path(__file__).parent
FLAG = "ORT_WEBGPU_ARGMAX_PARALLEL"
SOURCE = Path("/workspace/onnxruntime-webgpu-h200-perf/onnxruntime/core/providers/webgpu/reduction/reduction_ops.cc")
OUTPUT = Path("/workspace/qwen38-int4-argmax-parallel/build-v3")
TREE = r"""
values[local_idx] = best;
indices[local_idx] = index;
workgroupBarrier();
for (var stride = 64u; stride > 0u; stride /= 2u) {
  if (local_idx < stride) {
    let other = local_idx + stride;
    if (values[other] > values[local_idx] ||
        (values[other] == values[local_idx] && indices[other] < indices[local_idx])) {
      values[local_idx] = values[other];
      indices[local_idx] = indices[other];
    }
  }
  workgroupBarrier();
}
"""
PROGRAM = r'''
class ArgMaxParallelPartialProgram final : public Program<ArgMaxParallelPartialProgram> {
 public:
  ArgMaxParallelPartialProgram() : Program{"ArgMaxParallelPartial"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"length", ProgramUniformVariableDataType::Uint32},
                                          {"parts", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& x = shader.AddInput("x", ShaderUsage::UseValueTypeAlias);
    const auto& v = shader.AddOutput("v", ShaderUsage::UseValueTypeAlias);
    const auto& ix = shader.AddOutput("ix", ShaderUsage::UseValueTypeAlias);
    shader.AdditionalImplementation() << "var<workgroup> values: array<f32, 128>;\n"
                                      << "var<workgroup> indices: array<u32, 128>;\n";
    auto& code = shader.MainFunctionBody();
    code << "let row = workgroup_idx / uniforms.parts;\n"
         << "let part = workgroup_idx % uniforms.parts;\n"
         << "let offset = row * uniforms.length;\n"
         << "var best = f32(" << x.GetByOffset("offset") << ");\n"
         << "var index = 0u;\n"
         << "let stop = min((part + 1u) * 2048u, uniforms.length);\n"
         << "for (var i = part * 2048u + local_idx; i < stop; i += 128u) {\n"
         << "  let candidate = f32(" << x.GetByOffset("offset + i") << ");\n"
         << "  if (candidate > best) { best = candidate; index = i; }\n"
         << "}\n"
         << R"(@@TREE@@)"
         << "if (local_idx == 0u) {\n"
         << v.SetByOffset("workgroup_idx", "values[0]") << "\n"
         << ix.SetByOffset("workgroup_idx", "i32(indices[0])") << "\n}\n";
    return Status::OK();
  }
};

class ArgMaxParallelFinishProgram final : public Program<ArgMaxParallelFinishProgram> {
 public:
  ArgMaxParallelFinishProgram() : Program{"ArgMaxParallelFinish"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"parts", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& v = shader.AddInput("v", ShaderUsage::UseValueTypeAlias);
    const auto& ix = shader.AddInput("ix", ShaderUsage::UseValueTypeAlias);
    const auto& y = shader.AddOutput("y", ShaderUsage::UseValueTypeAlias);
    shader.AdditionalImplementation() << "var<workgroup> values: array<f32, 128>;\n"
                                      << "var<workgroup> indices: array<u32, 128>;\n";
    auto& code = shader.MainFunctionBody();
    code << "let offset = workgroup_idx * uniforms.parts;\n"
         << "var best = " << v.GetByOffset("offset") << ";\n"
         << "var index = u32(" << ix.GetByOffset("offset") << ");\n"
         << "for (var i = local_idx; i < uniforms.parts; i += 128u) {\n"
         << "  let candidate = " << v.GetByOffset("offset + i") << ";\n"
         << "  let candidate_index = u32(" << ix.GetByOffset("offset + i") << ");\n"
         << "  if (candidate > best || (candidate == best && candidate_index < index)) {\n"
         << "    best = candidate; index = candidate_index;\n"
         << "  }\n}\n"
         << R"(@@TREE@@)"
         << "if (local_idx == 0u) {\n"
         << y.SetByOffset("workgroup_idx", "y_value_t(indices[0])") << "\n}\n";
    return Status::OK();
  }
};
'''.replace("@@TREE@@", TREE)


def transform(source):
    source = replace_once(source, '#include <memory>', '#include <memory>\n#include "core/platform/env_var_utils.h"')
    source = replace_once(source, "Status ReduceSharedProgram::GenerateShaderCode", PROGRAM + "\nStatus ReduceSharedProgram::GenerateShaderCode")
    signature = "Status ReduceKernel<allow_multi_axes>::ComputeInternal(ComputeContext& context) const {"
    source = replace_once(source, signature, signature + f'''
  static const int parallel_argmax = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(parallel_argmax == 0 || parallel_argmax == 1, "{FLAG} must be 0 or 1");
''')
    branch = r'''
  if (parallel_argmax == 1 && name_ == "ArgMax" && select_last_index_ == 0 &&
      !is_input_empty && input_axes.size() == 1 && are_axes_innermost &&
      reduce_size >= 32 && reduce_size <= 1048576 &&
      (input_tensor->IsDataType<float>() || input_tensor->IsDataType<MLFloat16>()) &&
      context.AdapterInfo().vendor == std::string_view{"nvidia"} &&
      context.DeviceLimits().maxComputeWorkgroupSizeX >= 128 &&
      context.DeviceLimits().maxComputeInvocationsPerWorkgroup >= 128 &&
      context.DeviceLimits().maxComputeWorkgroupStorageSize >= 1024 &&
      output_size <= context.DeviceLimits().maxComputeWorkgroupsPerDimension / ((reduce_size + 2047) / 2048)) {
    const uint32_t parts = static_cast<uint32_t>((reduce_size + 2047) / 2048);
    const TensorShape partial_shape{static_cast<int64_t>(output_size), static_cast<int64_t>(parts)};
    auto values = context.CreateGPUTensor(DataTypeImpl::GetType<float>(), partial_shape);
    auto indices = context.CreateGPUTensor(DataTypeImpl::GetType<int32_t>(), partial_shape);
    ArgMaxParallelPartialProgram partial;
    partial.AddInput({input_tensor, ProgramTensorMetadataDependency::TypeAndRank})
        .AddOutput({&values, ProgramTensorMetadataDependency::TypeAndRank})
        .AddOutput({&indices, ProgramTensorMetadataDependency::TypeAndRank})
        .SetWorkgroupSize(128).SetDispatchGroupSize(static_cast<uint32_t>(output_size) * parts)
        .AddUniformVariables({{static_cast<uint32_t>(reduce_size)}, {parts}});
    ORT_RETURN_IF_ERROR(context.RunProgram(partial));
    ArgMaxParallelFinishProgram finish;
    finish.AddInput({&values, ProgramTensorMetadataDependency::TypeAndRank})
        .AddInput({&indices, ProgramTensorMetadataDependency::TypeAndRank})
        .AddOutput({context.Output(0, output_shape), ProgramTensorMetadataDependency::TypeAndRank})
        .SetWorkgroupSize(128).SetDispatchGroupSize(static_cast<uint32_t>(output_size))
        .AddUniformVariable({parts});
    return context.RunProgram(finish);
  }

'''
    return replace_once(source, "  if (use_naive_reduction) {", branch + "  if (use_naive_reduction) {")


def build():
    parent_path = PARENT / "build-manifest.json"
    proof_path = PROOF / "baseline.qualification.json"
    commands_path = PROOF / "recovered-commands.json"
    parent, proof, commands = [json.loads(p.read_text()) for p in (parent_path, proof_path, commands_path)]
    if not proof["all_152_original_objects_byte_identical"] or len(commands) != 152:
        raise RuntimeError("Original-object provenance missing")
    if parent["provider_sha256"] != "0955e1d6cc30cac13044450f692100ffa5f33361aaef229e25ff0efa20a8c8d4":
        raise RuntimeError("Unexpected baseline provider")
    working = Path(parent["working_directory"])
    for i, obj in enumerate(commands):
        path = Path(obj) if Path(obj).is_absolute() else working / obj
        if sha(path) != sha(PROOF / "baseline" / f"object-{i}.o"):
            raise RuntimeError(f"Changed original object: {obj}")
    for name, digest in proof["sdk_sha256"].items():
        if sha(PROOF / name) != digest:
            raise RuntimeError(f"Changed SDK: {name}")
    original, = [obj for obj, cmd in commands.items() if cmd[cmd.index("-c") + 1] == str(SOURCE)]
    compile_command = proof["compile_commands"][list(commands).index(original)].copy()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    obj = OUTPUT / "reduction_ops.cc.o"
    compile_command[compile_command.index("-o") + 1] = str(obj)
    for option in ("-MF", "-MT"):
        if option in compile_command:
            compile_command[compile_command.index(option) + 1] = str(obj) + ".d" if option == "-MF" else str(obj)
    subprocess.run(compile_command, cwd=working, check=True)
    if sha(obj) != sha(Path(original)):
        raise RuntimeError("Original reduction TU did not rebuild identically")
    original_tu_sha = sha(obj)
    library = OUTPUT / "libonnxruntime_providers_webgpu.so"
    link = parent["link_command"].copy()
    link[link.index("-o") + 1] = str(library)
    subprocess.run(link, cwd=working, check=True)
    if sha(library) != parent["provider_sha256"]:
        raise RuntimeError("Original provider relink mismatch")
    candidate = OUTPUT / SOURCE.name
    candidate.write_text(transform(SOURCE.read_text()))
    compile_command[compile_command.index("-c") + 1] = str(candidate)
    compile_command.insert(1, f"-fmacro-prefix-map={candidate}={SOURCE}")
    subprocess.run(compile_command, cwd=working, check=True)
    if link.count(original) != 1:
        raise RuntimeError("Reduction link replacement is not unique")
    link[link.index(original)] = str(obj)
    subprocess.run(link, cwd=working, check=True)
    comparator = ROOT / "argmax_parallel_compare_v3.cpp"
    binary = OUTPUT / "argmax_compare"
    subprocess.run(["c++", "-O3", "-std=c++20", str(comparator),
                    "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
                    "-L/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
                    "-Wl,-rpath,/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
                    "-lonnxruntime", "-o", str(binary)], check=True)
    if FLAG.encode() not in library.read_bytes():
        raise RuntimeError("Selector missing from provider")
    evidence = [Path(__file__), comparator, SOURCE, candidate, binary, parent_path, proof_path, commands_path,
                *[PROOF / name for name in proof["sdk_sha256"]]]
    result = dict(status="built_not_qualified", provider_sha256=sha(library),
                  baseline_provider_sha256=parent["provider_sha256"], baseline_byte_identical=True,
                  all_152_original_source_objects_byte_identical=True,
                  original_translation_unit_sha256=original_tu_sha, no_public_layout_changes=True,
                  default_enabled=False, model_eligible=False, comparator_sha256=sha(binary),
                  flag=FLAG, process_static_selector=True, compile_command=compile_command, link_command=link,
                  source_sha256={str(path): sha(path) for path in evidence})
    (OUTPUT / "build-manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(status=result["status"], provider_sha256=result["provider_sha256"])), flush=True)


if __name__ == "__main__":
    build()
