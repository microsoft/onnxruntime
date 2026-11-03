import json
from pathlib import Path
import subprocess

from build_qmoe_node_replay import replace_once, sha
from build_scoped_webgpu_source import build


ROOT = Path(__file__).parent
FLAG = "ORT_WEBGPU_FP16_DOT4_TREE_PROBE"
SOURCE = Path("/workspace/qwen38-int4-qmoe-prefill-rows4/build-v2/matmul_nbits.cc")
OUTPUT = Path("/workspace/qwen38-int4-fp16-dot4-tree-probe/build-v1")
PROGRAM = r'''
class Fp16Dot4TreeProbeProgram final : public Program<Fp16Dot4TreeProbeProgram> {
 public:
  Fp16Dot4TreeProbeProgram() : Program{"Fp16Dot4TreeProbe"} {}
  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& a = shader.AddInput("a", ShaderUsage::UseElementTypeAlias);
    const auto& b = shader.AddInput("b", ShaderUsage::UseValueTypeAlias);
    const auto& scale = shader.AddInput("scales", ShaderUsage::UseElementTypeAlias);
    const auto& y = shader.AddOutput("y", ShaderUsage::UseElementTypeAlias);
    shader.AdditionalImplementation() << R"(
var<workgroup> weights: array<vec4<f16>, 256>;
var<workgroup> products: array<vec4<f16>, 256>;
)";
    auto& code = shader.MainFunctionBody();
    code << "for (var i = local_idx; i < 256u; i += 32u) {\n"
         << " let row = i / 16u; let col = i % 16u;\n"
         << " let packed = " << b.GetByOffset("col") << ";\n"
         << " let nibble = (vec4<u32>(packed.x) >> vec4<u32>(0u, 4u, 8u, 12u)) & vec4<u32>(15u);\n"
         << " let wv = vec4<f16>(vec4<i32>(nibble) - vec4<i32>(8)) * "
         << scale.GetByOffset("col") << ";\n"
         << " let xv = vec4<f16>(" << a.GetByOffset("row * 4u")
         << ", " << a.GetByOffset("row * 4u + 1u")
         << ", " << a.GetByOffset("row * 4u + 2u")
         << ", " << a.GetByOffset("row * 4u + 3u") << ");\n"
         << " weights[i] = wv; products[i] = xv * wv;\n"
         << "}\nworkgroupBarrier();\n"
         << "for (var i = local_idx; i < 256u; i += 32u) {\n"
         << " let row = i / 16u; let col = i % 16u;\n"
         << " let xv = vec4<f16>(" << a.GetByOffset("row * 4u")
         << ", " << a.GetByOffset("row * 4u + 1u")
         << ", " << a.GetByOffset("row * 4u + 2u")
         << ", " << a.GetByOffset("row * 4u + 3u") << ");\n"
         << R"(
 let wv = weights[i]; let p = products[i];
 let full03 = f16(f32(xv.x) * f32(wv.x) + f32(xv.w) * f32(wv.w));
 let fused03 = fma(xv.x, wv.x, p.w);
 let rounded12 = p.y + p.z;
 let fused12 = fma(xv.z, wv.z, p.y);
)"
         << " " << y.SetByOffset("row * 80u + col", "full03 + rounded12") << "\n"
         << " " << y.SetByOffset("row * 80u + 16u + col", "full03 + fused12") << "\n"
         << " " << y.SetByOffset("row * 80u + 32u + col", "fused03 + rounded12") << "\n"
         << " " << y.SetByOffset("row * 80u + 48u + col", "fused03 + fused12") << "\n"
         << " " << y.SetByOffset("row * 80u + 64u + col", "dot(xv, wv)") << "\n}\n";
    return Status::OK();
  }
};
'''


def transform(source):
    source = replace_once(source, "class GroupedPrefillRows4Program final",
                          PROGRAM + "\nclass GroupedPrefillRows4Program final")
    start = source.index("Status ApplyMatMulNBits(")
    position = source.index("  const int grouped_prefill_rows", start)
    branch = f'''
  static const int dot4_tree_probe = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(dot4_tree_probe == 0 || dot4_tree_probe == 1, "{FLAG} must be 0 or 1");
  if (dot4_tree_probe == 1) {{
    ORT_RETURN_IF_NOT(batch_count == 1 && M == 16 && dispatch_M == M && N == 80 && K == 4 &&
        block_size == 32 && nbits == 4 && !has_bias && !has_zero_points && !has_weight_idx &&
        a->IsDataType<MLFloat16>() && scales->IsDataType<MLFloat16>(),
        "FP16 dot4 tree probe requires the diagnostic M16 K4 N80 fixture");
    Fp16Dot4TreeProbeProgram program;
    program.AddInput({{a, ProgramTensorMetadataDependency::TypeAndRank, 1}})
        .AddInput({{b, ProgramTensorMetadataDependency::TypeAndRank, 16}})
        .AddInput({{scales, ProgramTensorMetadataDependency::TypeAndRank, 1}})
        .AddOutput({{y, ProgramTensorMetadataDependency::TypeAndRank, 1}})
        .SetWorkgroupSize(32).SetDispatchGroupSize(1);
    return context.RunProgram(program);
  }}

'''
    return source[:position] + branch + source[position:]


if __name__ == "__main__":
    report = build(OUTPUT, SOURCE, transform, SOURCE.parent / "native_compare.cpp", FLAG,
                   dict(diagnostic_only=True, model_eligible=False, tree_patterns=4))
    runner = OUTPUT / "dot4_tree_compare.cpp"
    runner.write_text((ROOT / "fp16_dot4_tree_compare.cpp").read_text())
    subprocess.run(["c++", "-O3", "-std=c++20", str(runner),
        "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
        "-L/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-Wl,-rpath,/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-lonnxruntime", "-o", str(OUTPUT / "dot4_tree_compare")], check=True)
    report.update(original_compute_and_fallbacks_unchanged=False,
                  original_default_compute_and_fallbacks_unchanged=True,
                  diagnostic_runner_sha256=sha(OUTPUT / "dot4_tree_compare"))
    report["source_sha256"]["dot4_tree_compare.cpp"] = sha(runner)
    (OUTPUT / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
