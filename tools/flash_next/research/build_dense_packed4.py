import json
from pathlib import Path

from build_dense_double_buffer import transform_compare
from build_dense_lane8 import LANE_EXPRESSION
from build_dense_predecode import DECODE, HEADER, SETUP, transform as persistent_transform
from build_hc_bitcast_unpack import main as build
from build_hc_staged_projection import REPLAY, stage_replay
from build_qmoe_node_replay import replace_once, sha


FLAG = "ORT_WEBGPU_DENSE_PACKED4"
OUTPUT = Path("/workspace/qwen38-int4-dense-packed4/build-v4")
PARENT = Path("/workspace/qwen38-int4-qmoe-prefill-rows4/build-v2")
COMPARATOR = Path("/workspace/qwen38-int4-dense-block32/build-v1")
TEMPLATE = Path("/workspace/qwen38-int4-dense-block32/block32-template-v1/dense_block32_template.inc")
INCLUDES = OUTPUT.parent / "packed4-includes-v4"


PACK = r'''
class DensePack4ColumnsProgram final : public Program<DensePack4ColumnsProgram> {
 public:
  DensePack4ColumnsProgram() : Program{"DensePack4Columns"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"blocks", ProgramUniformVariableDataType::Uint32},
      {"blocks_per_column", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& b = shader.AddInput("b", ShaderUsage::UseValueTypeAlias);
    const auto& scales = shader.AddInput("scales");
    const auto& packed = shader.AddOutput("packed", ShaderUsage::UseValueTypeAlias);
    const auto& packed_scales = shader.AddOutput("packed_scales");
    auto& code = shader.MainFunctionBody();
    code << "if (global_idx >= uniforms.blocks) { return; }\n"
         << "let column = global_idx / uniforms.blocks_per_column;\n"
         << "let block = global_idx % uniforms.blocks_per_column;\n"
         << "let packed_offset = ((column / 4u) * uniforms.blocks_per_column + block) * 4u + column % 4u;\n"
         << packed.SetByOffset("packed_offset", b.GetByOffset("global_idx")) << "\n"
         << packed_scales.SetByOffset("packed_offset", scales.GetByOffset("global_idx")) << "\n";
    return Status::OK();
  }
};
'''


PACK_SETUP = r'''
      const uint32_t K = onnxruntime::narrow<uint32_t>(K_);
      const uint32_t N = onnxruntime::narrow<uint32_t>(N_);
      std::call_once(decode_once_, [&]() {
        auto weights = std::make_unique<Tensor>(context.CreateGPUTensor(b->DataType(), b->Shape()));
        auto packed_scales = std::make_unique<Tensor>(
            context.CreateGPUTensor(scales->DataType(), scales->Shape()));
        const uint32_t blocks = onnxruntime::narrow<uint32_t>(scales->Shape().Size());
        DensePack4ColumnsProgram program;
        program.AddInput({b, ProgramTensorMetadataDependency::TypeAndRank, 16})
            .AddInput({scales, ProgramTensorMetadataDependency::TypeAndRank})
            .AddOutput({weights.get(), ProgramTensorMetadataDependency::TypeAndRank, 16})
            .AddOutput({packed_scales.get(), ProgramTensorMetadataDependency::TypeAndRank})
            .AddUniformVariables({blocks, K / 32u})
            .SetWorkgroupSize(64).SetDispatchGroupSize((blocks + 63u) / 64u);
        ORT_THROW_IF_ERROR(context.RunProgram(program));
        decoded_weights_ = std::move(weights);
        packed_scales_ = std::move(packed_scales);
      });
      return decoded_replay_.Run(context, {y}, [&]() -> Status {
        DensePredecodedProgram program{4, 4, false, false, false, false, false, 32, false, false};
        program.SetWorkgroupSize(128).SetDispatchGroupSize((N + 3u) / 4u);
        program.AddInput({decoded_weights_.get(), ProgramTensorMetadataDependency::TypeAndRank, 16})
            .AddInput({a, ProgramTensorMetadataDependency::TypeAndRank, 4})
            .AddInput({packed_scales_.get(), ProgramTensorMetadataDependency::TypeAndRank})
            .AddOutput({y, ProgramTensorMetadataDependency::TypeAndRank})
            .AddUniformVariables({1u, N, K, K / 4u, K / 32u, 32u, K / 32u,
                                  K / 32u, (N + 3u) / 4u, 1u, 0u, 1u})
            .CacheHint(K, N);
        return context.RunProgram(program);
      }, 0, 0, {decoded_weights_.get(), packed_scales_.get()});
'''


def packed_template(original):
    result = replace_once(original, "Status ApplyDenseBlock32Template(", "Status ApplyDensePacked4Template(")
    result = replace_once(result, 'ss << "  let idx = local_idx % ";',
                          f'ss << "  let logical_lane = {LANE_EXPRESSION};\\n";\n'
                          'ss << "  let idx = logical_lane % ";')
    result = replace_once(result, 'ss << "  let idy = local_idx / ";',
                          'ss << "  let idy = logical_lane / ";')
    result = replace_once(result, 'GetByOffset("b_global * uniforms.K_of_b + k_offset + b_base_offset")',
                          'GetByOffset("((b_global / 4u) * uniforms.K_of_b + k_offset) * 4u + b_global % 4u")')
    return replace_once(result,
                        'GetByOffset("b_global * uniforms.blocks_per_col + block_idx + b_scale_offset")',
                        'GetByOffset("((b_global / 4u) * uniforms.blocks_per_col + block_idx) * 4u + b_global % 4u")')


def packed_replay(source):
    result = stage_replay(source).replace("HCStageReplay", "DensePacked4Replay")
    result = replace_once(result, "      for (const auto* tensor : extra_inputs) {",
                          '      ORT_ENFORCE(extra_inputs.size() == 2 && gpu_inputs_.size() == 1 && gpu_inputs_[0] == 0,\n'
                          '                  "Dense packed4 replay requires cached weights/scales and activation input0.");\n'
                          "      auto append_extra = [&](const Tensor* tensor) {")
    result = replace_once(result, "      }\n      for (int index : gpu_inputs_) {",
                          "      };\n      append_extra(*extra_inputs.begin());\n"
                          "      for (int index : gpu_inputs_) {")
    result = replace_once(result, "      for (auto* tensor : outputs) {",
                          "      append_extra(*(extra_inputs.begin() + 1));\n"
                          "      for (auto* tensor : outputs) {")
    return result.replace("HC stage replay requires default WebGPU-buffer extra inputs.",
                          "Dense packed4 replay requires default WebGPU-buffer cached inputs.")


def transform(source, header):
    result = persistent_transform(source, header)
    result = replace_once(result, DECODE, PACK)
    result = replace_once(result, SETUP, PACK_SETUP)
    result = replace_once(result, "  mutable std::unique_ptr<Tensor> decoded_weights_;",
                          "  mutable std::unique_ptr<Tensor> decoded_weights_;\n"
                          "  mutable std::unique_ptr<Tensor> packed_scales_;")
    result = replace_once(result, 'decoded_replay_{"DensePredecoded", {0, 2}}',
                          'decoded_replay_{"DensePacked4", {0}}')
    result = replace_once(result,
                          "predecode_mode == 1 && dense_bitcast_route && block32_mode == 1 && weights_constant_",
                          "predecode_mode == 1 && dense_bitcast_route && block32_mode == 1 && weights_constant_ && N_ % 4 == 0 &&\n"
                          "      b->IsDataType<uint8_t>() && b->Shape().Size() == N_ * (K_ / 2) &&\n"
                          "      scales->Shape().Size() == N_ * (K_ / 32)")
    for old, new in (("ORT_WEBGPU_DENSE_PREDECODE", FLAG),
                     ("PersistentDenseMatMulNBits", "PersistentPacked4MatMulNBits"),
                     ("DensePredecoded", "DensePacked4"),
                     ("MatMulNBitsPredecodedF16", "MatMulNBitsPacked4F16"),
                     ("dense_predecoded_replay.h", "dense_packed4_replay.h"),
                     ("dense_predecoded_template.inc", "dense_packed4_template.inc"),
                     ("predecode_mode", "packed_mode"),
                     ("decode_once_", "pack_once_"),
                     ("decoded_weights_", "packed_weights_"),
                     ("decoded_replay_", "packed_replay_")):
        result = result.replace(old, new)
    return result


def main():
    baseline = json.loads((COMPARATOR / "build-manifest.json").read_text())
    assert sha(TEMPLATE) == baseline["source_sha256"]["../block32-template-v1/dense_block32_template.inc"]
    replay_manifest = json.loads((REPLAY.parent.parent / "build-manifest.json").read_text())
    assert sha(REPLAY) == replay_manifest["source_sha256"]["overlay/static_node_replay.h"]
    INCLUDES.mkdir(parents=True, exist_ok=False)
    template = INCLUDES / "dense_packed4_template.inc"
    template.write_text(packed_template(TEMPLATE.read_text()))
    replay = INCLUDES / "dense_packed4_replay.h"
    replay.write_text(packed_replay(REPLAY.read_text()))
    build(output=OUTPUT, flag=FLAG, source_transform=lambda source: transform(source, HEADER.read_text()),
          component_dir=PARENT, source_name="matmul_nbits.cc", comparator_dir=COMPARATOR,
          comparator_flag="ORT_WEBGPU_DENSE_BLOCK32",
          comparator_transform=lambda source: transform_compare(source, flag=FLAG),
          parent_dir=PARENT, extra_include_dirs=(INCLUDES,))
    path = OUTPUT / "build-manifest.json"
    report = json.loads(path.read_text())
    report.update(original_matmul_kernel_header_sha256=sha(HEADER),
                  constant_weights_and_scales_only=True, original_kernel_class_unchanged=True,
                  packed_bytes_and_scale_bits_unchanged=True, no_expanded_fp16_weight_tensor=True,
                  private_kernel_owns_both_packed_tensors=True,
                  activation_dot_partial_and_serial_fold_order_unchanged=True,
                  default_enabled=False)
    for item in (template, replay):
        report["source_sha256"][f"../{INCLUDES.name}/{item.name}"] = sha(item)
    path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
