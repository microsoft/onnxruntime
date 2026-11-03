import json
from pathlib import Path
import subprocess

from build_dense_paired_matrix import SOURCE, ROOT
from build_qmoe_node_replay import replace_once, sha
from build_scoped_webgpu_source import build


FLAG = "ORT_WEBGPU_DENSE_COLUMNS8"
CLASS = "DenseColumns8Program"
OUTPUT = Path("/workspace/qwen38-int4-dense-columns8/build-v1")
STORAGE_BYTES = 2560


def make_program():
    source = r'''
class DenseColumns8Program final : public Program<DenseColumns8Program> {
 public:
  DenseColumns8Program() : Program{"DenseColumns8"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"blocks", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& a = shader.AddInput("a", ShaderUsage::UseValueTypeAlias);
    const auto& b = shader.AddInput("b", ShaderUsage::UseValueTypeAlias);
    const auto& scales = shader.AddInput("scales", ShaderUsage::UseElementTypeAlias);
    const auto& y = shader.AddOutput("y", ShaderUsage::UseElementTypeAlias);
    shader.AdditionalImplementation() << R"(
var<workgroup> tile_a: array<vec4<f16>, 256>;
var<workgroup> partials: array<f16, 256>;
fn decode_low_nibbles(word: u32) -> vec4<f16> {
  let bottom = bitcast<vec2<f16>>((word & 0x000f000fu) | 0x64006400u) - vec2<f16>(1032.0);
  let top = bitcast<vec2<f16>>(((word >> 8u) & 0x000f000fu) | 0x64006400u) - vec2<f16>(1032.0);
  return vec4<f16>(bottom.x, top.x, bottom.y, top.y);
}
)";
    auto& code = shader.MainFunctionBody();
    code << R"(
let stream = local_idx % 32u;
let column_group = local_idx / 32u;
for (var i = local_idx; i < 256u; i += 128u) { partials[i] = f16(0); }
for (var tile = 0u; tile * 32u < uniforms.blocks; tile++) {
  for (var i = local_idx; i < 256u; i += 128u) {
    var value = vec4<f16>(0);
    let block = tile * 32u + i / 8u;
    if (block < uniforms.blocks) {
)" << "      value = " << a.GetByOffset("block * 8u + i % 8u") << ";\n"
         << R"(
    }
    tile_a[i] = value;
  }
  workgroupBarrier();
  let block = tile * 32u + stream;
  if (block < uniforms.blocks) {
)";
'''
    for j in range(2):
        source += f'''
    code << "    let col{j} = workgroup_idx * 8u + column_group + {j * 4}u;\\n"
         << "    let packed{j} = " << b.GetByOffset("col{j} * uniforms.blocks + block") << ";\\n"
         << "    let scale{j} = " << scales.GetByOffset("col{j} * uniforms.blocks + block") << ";\\n"
         << "    var sum{j} = f16(0);\\n";
'''
    source += r'''
    code << R"(
    for (var word = 0u; word < 4u; word++) {
      let x0 = tile_a[stream * 8u + word * 2u];
      let x1 = tile_a[stream * 8u + word * 2u + 1u];
)";
'''
    for j in range(2):
        source += f'''
    code << "      {{\\n"
         << "        let low = decode_low_nibbles(packed{j}[word]);\\n"
         << "        let high = decode_low_nibbles(packed{j}[word] >> 4u);\\n"
         << "        let w0 = vec4<f16>(low.x, high.x, low.y, high.y) * scale{j};\\n"
         << "        let w1 = vec4<f16>(low.z, high.z, low.w, high.w) * scale{j};\\n"
         << "        sum{j} += dot(x0, w0) + dot(x1, w1);\\n"
         << "      }}\\n";
'''
    source += '    code << "    }\\n";\n'
    for j in range(2):
        source += f'    code << "    partials[(column_group + {j * 4}u) * 32u + stream] += sum{j};\\n";\n'
    source += r'''
    code << R"(
  }
  workgroupBarrier();
}
if (local_idx < 8u) {
  var value = f16(0);
  for (var stream = 0u; stream < 32u; stream++) {
    value += partials[local_idx * 32u + stream];
  }
)" << y.SetByOffset("workgroup_idx * 8u + local_idx", "value") << "\n}\n";
    return Status::OK();
  }
};
'''
    return source


PROGRAM = make_program()


def transform(source):
    result = replace_once(source, "class GroupedPrefillRows4Program final",
                          PROGRAM + "\nclass GroupedPrefillRows4Program final")
    signature = "Status MatMulNBits::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {"
    result = replace_once(result, signature, signature + f'''
  static const int columns8_mode = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(columns8_mode == 0 || columns8_mode == 1, "{FLAG} must be 0 or 1");
''')
    result = replace_once(result, "  auto compute = [&]() {", f'''
  const bool columns8 = columns8_mode == 1 && dense_bitcast_route && block32_mode == 1 &&
      N_ >= 2560 && N_ % 8 == 0 && b->IsDataType<uint8_t>() &&
      b->Shape().Size() == N_ * (K_ / 2) && scales->Shape().Size() == N_ * (K_ / 32) &&
      context.DeviceLimits().maxComputeWorkgroupStorageSize >= {STORAGE_BYTES};
  auto compute = [&]() {{
    if (columns8) {{
      {CLASS} program;
      program.AddInput({{a, ProgramTensorMetadataDependency::TypeAndRank, 4}})
          .AddInput({{b, ProgramTensorMetadataDependency::TypeAndRank, 16}})
          .AddInput({{scales, ProgramTensorMetadataDependency::TypeAndRank, 1}})
          .AddOutput({{y, ProgramTensorMetadataDependency::TypeAndRank, 1}})
          .AddUniformVariables({{onnxruntime::narrow<uint32_t>(K_ / 32)}})
          .SetWorkgroupSize(128).SetDispatchGroupSize(N_ / 8);
      return context.RunProgram(program);
    }}''')
    return replace_once(result,
        "compute, 0, dense_bitcast_route ? (block32_mode == 1 ? 3 : 1 + dense_tile4) : 0);",
        "compute, 0, columns8 ? 6 : "
        "(dense_bitcast_route ? (block32_mode == 1 ? 3 : 1 + dense_tile4) : 0));")


if __name__ == "__main__":
    report = build(OUTPUT, SOURCE, transform, SOURCE.parent / "native_compare.cpp", FLAG,
        dict(diagnostic_only=False, model_eligible=False, shared_storage_bytes=STORAGE_BYTES,
             original_k_streams=32, output_columns=8, columns_per_thread=2,
             original_provider_full_kernel_comparison_required=True))
    cpp = OUTPUT / "paired_matrix_compare.cpp"
    cpp.write_text((ROOT / "dense_paired_matrix_compare.cpp").read_text())
    binary = OUTPUT / "paired_matrix_compare"
    subprocess.run(["c++", "-O3", "-std=c++20", str(cpp),
        "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
        "-L/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-Wl,-rpath,/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-lonnxruntime", "-o", str(binary)], check=True)
    report.update(original_compute_and_fallbacks_unchanged=False,
                  original_default_compute_and_fallbacks_unchanged=True,
                  paired_comparator_sha256=sha(binary))
    report["source_sha256"]["paired_matrix_compare.cpp"] = sha(cpp)
    (OUTPUT / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
