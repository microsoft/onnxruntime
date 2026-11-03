import json
from pathlib import Path
import re

from build_hc_bitcast_unpack import main as build
from build_hc_staged_projection import REPLAY, stage_replay
from build_dense_double_buffer import transform_compare
from build_qmoe_node_replay import replace_once, sha


FLAG = "ORT_WEBGPU_DENSE_PREDECODE"
OUTPUT = Path("/workspace/qwen38-int4-dense-predecode/build-v2")
COMPONENT = Path("/workspace/qwen38-int4-dense-block32/build-v1")
HEADER = Path("/workspace/qwen38-int4-static-node-replay/build-v8/overlay/"
              "contrib_ops/webgpu/quantization/matmul_nbits.h")
TEMPLATE = Path("/workspace/qwen38-int4-dense-block32/block32-template-v1/dense_block32_template.inc")


DECODE = r'''
class DenseDecodeWeightsProgram final : public Program<DenseDecodeWeightsProgram> {
 public:
  DenseDecodeWeightsProgram() : Program{"DenseDecodeWeightsF16"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"blocks", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& b = shader.AddInput("b", ShaderUsage::UseValueTypeAlias);
    const auto& scales = shader.AddInput("scales");
    const auto& output = shader.AddOutput("decoded", ShaderUsage::UseValueTypeAlias);
    auto& code = shader.MainFunctionBody();
    code << "if (global_idx >= uniforms.blocks) { return; }\n"
         << "let packed = " << b.GetByOffset("global_idx") << ";\n"
         << "let scale = " << scales.GetByOffset("global_idx") << ";\n"
         << "for (var i = 0u; i < 4u; i++) {\n"
         << "  let low = vec4<f16>(unpack4xU8(packed[i] & 0x0f0f0f0fu)) - vec4<f16>(8);\n"
         << "  let high = vec4<f16>(unpack4xU8((packed[i] >> 4u) & 0x0f0f0f0fu)) - vec4<f16>(8);\n"
         << "  let w0 = vec4<f16>(low.x, high.x, low.y, high.y) * scale;\n"
         << "  let w1 = vec4<f16>(low.z, high.z, low.w, high.w) * scale;\n"
         << output.SetByOffset("global_idx * 8u + i * 2u", "w0") << "\n"
         << output.SetByOffset("global_idx * 8u + i * 2u + 1u", "w1") << "\n"
         << "}\n";
    return Status::OK();
  }
};
'''


SETUP = r'''
      const uint32_t K = onnxruntime::narrow<uint32_t>(K_);
      const uint32_t N = onnxruntime::narrow<uint32_t>(N_);
      std::call_once(decode_once_, [&]() {
        auto decoded = std::make_unique<Tensor>(
            context.CreateGPUTensor(scales->DataType(), TensorShape{N_, K_}));
        DenseDecodeWeightsProgram program;
        program.AddInput({b, ProgramTensorMetadataDependency::TypeAndRank, 16})
            .AddInput({scales, ProgramTensorMetadataDependency::TypeAndRank})
            .AddOutput({decoded.get(), ProgramTensorMetadataDependency::TypeAndRank, 4})
            .AddUniformVariables({N * (K / 32u)})
            .SetWorkgroupSize(64).SetDispatchGroupSize((N * (K / 32u) + 63u) / 64u);
        ORT_THROW_IF_ERROR(context.RunProgram(program));
        decoded_weights_ = std::move(decoded);
      });
      return decoded_replay_.Run(context, {y}, [&]() -> Status {
        DensePredecodedProgram program{4, 4, false, false, false, false, false, 32, false, false};
        program.SetWorkgroupSize(128).SetDispatchGroupSize((N + 3u) / 4u);
        program.AddInput({decoded_weights_.get(), ProgramTensorMetadataDependency::TypeAndRank, 4})
            .AddInput({a, ProgramTensorMetadataDependency::TypeAndRank, 4})
            .AddInput({scales, ProgramTensorMetadataDependency::TypeAndRank})
            .AddOutput({y, ProgramTensorMetadataDependency::TypeAndRank})
            .AddUniformVariables({1u, N, K, K / 4u, K / 32u, 32u, K / 32u,
                                  K / 32u, (N + 3u) / 4u, 1u, 0u, 1u})
            .CacheHint(K, N);
        return context.RunProgram(program);
      }, 0, 0, {decoded_weights_.get()});
'''


def predecoded_template(source):
    source = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("//")) + "\n"
    source = replace_once(source, "Status ApplyDenseBlock32Template(", "Status ApplyDensePredecodedTemplate(")
    source = replace_once(source, '''ss << "        var b_value = ";
ss << __var_b->GetByOffset("b_global * uniforms.K_of_b + k_offset + b_base_offset");
ss << ";\\n";''', "")
    source, removed = re.subn(r'^ss << " +let b_value_(?:lower|upper) = decode_low_nibbles.*\n',
                             "", source, flags=re.MULTILINE)
    assert removed == 4
    for indent, suffix in (("        ", ""), ("            ", " + i * 2u")):
        for index in (0, 1):
            pattern = (f'ss << "{indent}let b{index} = vec4<output_element_t>('
                       + ("b_value_lower[0], b_value_upper[0], b_value_lower[1], b_value_upper[1]" if index == 0
                          else "b_value_lower[2], b_value_upper[2], b_value_lower[3], b_value_upper[3]")
                       + ') * scale_b;\\n";')
            offset = f"(b_global * uniforms.K_of_b + k_offset + b_base_offset) * 8u{suffix}"
            if index == 1:
                offset += " + 1u"
            source = replace_once(source, pattern,
                                  f'ss << "{indent}let b{index} = ";\n'
                                  f'ss << __var_b->GetByOffset("{offset}");\nss << ";\\n";')
    return source


def transform(source, header):
    start = source.index("class DenseBlock32Program final")
    stop = source.index("Status ApplyDenseBlock32MatMulNBits(", start)
    program = source[start:stop].replace("DenseBlock32Program", "DensePredecodedProgram")
    program = program.replace("MatMulNBitsBlock32F16", "MatMulNBitsPredecodedF16")
    program = program.replace("ApplyDenseBlock32Template", "ApplyDensePredecodedTemplate")
    program = replace_once(program,
        '  const auto& a = shader.AddInput("input_a", ShaderUsage::UseValueTypeAlias);\n'
        '  const auto& b = shader.AddInput("input_b");',
        '  const auto& b = shader.AddInput("input_b");\n'
        '  const auto& a = shader.AddInput("input_a", ShaderUsage::UseValueTypeAlias);')
    program = replace_once(program,
        "  const uint32_t components_b = b.NumComponents() / 4;  // b is stored as uint32 which includes 4 uint8.",
        "  const uint32_t components_b = 4;")
    source = replace_once(source, '#include "dense_block32_template.inc"',
                          '#include "dense_block32_template.inc"\n#include "dense_predecoded_template.inc"')
    stop = source.index("\n}\n\nStatus MatMulNBitsProgram::")
    source = source[:stop] + "\n" + DECODE + "\n" + program + source[stop:]
    start = header.index("class MatMulNBits final")
    stop = header.index("\n};", start) + 3
    kernel = header[start:stop].replace("MatMulNBits", "PersistentDenseMatMulNBits")
    kernel = replace_once(kernel, '"PersistentDenseMatMulNBitsExact"', '"MatMulNBitsExact"')
    kernel = replace_once(kernel, "    K_ = info.GetAttr", '''
    const Tensor* constant = nullptr;
    weights_constant_ = info.TryGetConstantInput(1, &constant) && info.TryGetConstantInput(2, &constant);
    K_ = info.GetAttr''')
    kernel = replace_once(kernel, " private:\n", ''' private:
  bool weights_constant_ = false;
  mutable std::once_flag decode_once_;
  mutable std::unique_ptr<Tensor> decoded_weights_;
  mutable DensePredecodedReplay decoded_replay_{"DensePredecoded", {0, 2}};
''')
    signature = "Status MatMulNBits::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {"
    start = source.index(signature)
    stop = source.index("\n}\n", start) + 3
    method = source[start:stop].replace("MatMulNBits::", "PersistentDenseMatMulNBits::")
    method = replace_once(method, "  const int block32_mode =", f'''
  const int predecode_mode = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(predecode_mode == 0 || predecode_mode == 1, "{FLAG} must be 0 or 1");
  const int block32_mode =''')
    method = replace_once(method, "  auto compute = [&]() {", """
  if (predecode_mode == 1 && dense_bitcast_route && block32_mode == 1 && weights_constant_) {
""" + SETUP + """
  }
  auto compute = [&]() {""")
    source = source[:stop] + "\n" + method + source[stop:]
    source = replace_once(source, "ONNX_OPERATOR_KERNEL_EX(", kernel + "\n\nONNX_OPERATOR_KERNEL_EX(")
    source = replace_once(source, "    MatMulNBits);", "    PersistentDenseMatMulNBits);")
    return replace_once(source, "#include <string_view>",
                        '#include <string_view>\n#include <memory>\n#include <mutex>\n'
                        '#include "dense_predecoded_replay.h"')


def main():
    component = json.loads((COMPONENT / "build-manifest.json").read_text())
    assert sha(TEMPLATE) == component["source_sha256"]["../block32-template-v1/dense_block32_template.inc"]
    replay_manifest = json.loads((REPLAY.parent.parent / "build-manifest.json").read_text())
    assert sha(REPLAY) == replay_manifest["source_sha256"]["overlay/static_node_replay.h"]
    includes = OUTPUT.parent / "predecoded-includes-v2"
    includes.mkdir(parents=True, exist_ok=False)
    template = includes / "dense_predecoded_template.inc"
    template.write_text(predecoded_template(TEMPLATE.read_text()))
    replay = includes / "dense_predecoded_replay.h"
    replay.write_text(stage_replay(REPLAY.read_text()).replace("HCStageReplay", "DensePredecodedReplay"))
    build(output=OUTPUT, flag=FLAG,
          source_transform=lambda source: transform(source, HEADER.read_text()),
          component_dir=COMPONENT, source_name="matmul_nbits.cc",
          comparator_flag="ORT_WEBGPU_DENSE_BLOCK32",
          comparator_transform=lambda source: transform_compare(source, flag=FLAG),
          parent_dir=Path("/workspace/qwen38-int4-qsa-prefill-score16/build-v1"),
          extra_include_dirs=(includes,))
    path = OUTPUT / "build-manifest.json"
    report = json.loads(path.read_text())
    report.update(original_matmul_kernel_header_sha256=sha(HEADER),
                  constant_weights_and_scales_only=True,
                  original_kernel_class_unchanged=True,
                  private_kernel_owns_decoded_weights=True,
                  original_dot_order_activation_staging_and_serial_fold_preserved=True,
                  no_fp32_accumulation_or_tensorcore_changes=True)
    for item in (template, replay):
        report["source_sha256"]["../predecoded-includes-v2/" + item.name] = sha(item)
    path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
