import json
from pathlib import Path

from build_hc_bitcast_unpack import main as build
from build_hc_staged_projection import REPLAY, stage_replay
from build_qmoe_node_replay import replace_once, sha


FLAG = "ORT_WEBGPU_HC_POST_NORM_PREDECODE"
OUTPUT = Path("/workspace/qwen38-int4-hc-post-norm-predecode/build-v2")


PREDECODE_PROGRAM = r'''
class HCPostNormDecodeWeightsProgram final : public Program<HCPostNormDecodeWeightsProgram> {
 public:
  HCPostNormDecodeWeightsProgram() : Program{"HCPostNormDecodeWeightsF16"} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& b = shader.AddInput("b", ShaderUsage::UseValueTypeAlias);
    const auto& scales = shader.AddInput("scales");
    const auto& output = shader.AddOutput("decoded", ShaderUsage::UseValueTypeAlias);
    auto& code = shader.MainFunctionBody();
    code << "if (global_idx >= 103680u) { return; }\n"
         << "let packed = " << b.GetByOffset("global_idx") << ";\n"
         << "let scale_value = " << scales.GetByOffset("global_idx") << ";\n"
         << "for (var i = 0u; i < 4u; i++) {\n"
         << "  let low = vec4<f16>(unpack4xU8(packed[i] & 0x0f0f0f0fu)) - vec4<f16>(8);\n"
         << "  let high = vec4<f16>(unpack4xU8((packed[i] >> 4u) & 0x0f0f0f0fu)) - vec4<f16>(8);\n"
         << "  let w0 = vec4<f16>(low.x, high.x, low.y, high.y) * scale_value;\n"
         << "  let w1 = vec4<f16>(low.z, high.z, low.w, high.w) * scale_value;\n"
         << output.SetByOffset("global_idx * 8u + i * 2u", "w0") << "\n"
         << output.SetByOffset("global_idx * 8u + i * 2u + 1u", "w1") << "\n"
         << "}\n";
    return Status::OK();
  }
};

'''


def decoded_program(source):
    start = source.index("class HCPostNormProjectionProgram")
    stop = source.index("class HCPostNormProjectionF16", start)
    program = source[start:stop].replace("HCPostNormProjectionProgram", "HCPostNormProjectionPredecodedProgram")
    program = replace_once(program, 'Program{"HCPostNormProjectionF16"}',
                           'Program{"HCPostNormProjectionPredecodedF16"}')
    program = replace_once(program,
        '    const auto& scales = shader.AddInput("projection_scales", ShaderUsage::UseValueTypeAlias);\n', "")
    program = replace_once(program,
        '    const auto& b = shader.AddInput("b", ShaderUsage::UseValueTypeAlias);\n', "")
    program = replace_once(program, '    const auto& streams = shader.AddInput("streams");',
        '    const auto& b = shader.AddInput("b", ShaderUsage::UseValueTypeAlias);\n'
        '    const auto& streams = shader.AddInput("streams");')
    program = replace_once(program,
        '         << "  let packed = " << b.GetByOffset("col * 320u + block") << ";\\n"\n'
        '         << "  let scale_value = " << scales.GetByOffset("col * 320u + block") << ";\\n"\n', "")
    start = program.index('         << "    let low = vec4<f16>')
    stop = program.index('         << "    let base = block * 36u', start)
    program = program[:start] + (
        '         << "    let w0 = " << b.GetByOffset("(col * 320u + block) * 8u + i * 2u") << ";\\n"\n'
        '         << "    let w1 = " << b.GetByOffset("(col * 320u + block) * 8u + i * 2u + 1u") << ";\\n"\n'
    ) + program[stop:]
    return program


def transform(source):
    source = replace_once(source, '#include "static_node_replay.h"',
                          '#include "static_node_replay.h"\n#include "hc_predecoded_replay.h"\n'
                          '#include <memory>\n#include <mutex>')
    source = replace_once(source, "class HCPostNormProjectionF16",
                          PREDECODE_PROGRAM + decoded_program(source) + "class HCPostNormProjectionF16")
    source = replace_once(source,
        "  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override {",
        f'''  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override {{
    const int predecode_mode = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
    ORT_RETURN_IF_NOT(predecode_mode == 0 || predecode_mode == 1, "{FLAG} must be 0 or 1");''')
    start = source.index("      return replay_.Run(context, {y, normalized, activated, tail}, [&]() -> Status {")
    stop = source.index("\n    }\n    HCPostNormProjectionBaselineNormProgram", start)
    block = source[start:stop]
    setup = block[block.index("        program.CacheHint"):block.index("      });")]
    setup = replace_once(setup,
        '''        program.AddInput({weights, ProgramTensorMetadataDependency::TypeAndRank, 16})
            .AddInput({scales, ProgramTensorMetadataDependency::TypeAndRank})
            .AddOutput({y, ProgramTensorMetadataDependency::Type})''',
        '''        if (!predecoded) {
          program.AddInput({weights, ProgramTensorMetadataDependency::TypeAndRank, 16})
              .AddInput({scales, ProgramTensorMetadataDependency::TypeAndRank});
        }
        program.AddOutput({y, ProgramTensorMetadataDependency::Type})''')
    replacement = (
        '''      const bool predecoded = predecode_mode == 1 && weights_constant_;
      if (predecoded) {
        std::call_once(decode_once_, [&]() {
          auto decoded = std::make_unique<Tensor>(
              context.CreateGPUTensor(scales->DataType(), TensorShape{324, 10240}));
          HCPostNormDecodeWeightsProgram program;
          program.AddInput({weights, ProgramTensorMetadataDependency::TypeAndRank, 16})
              .AddInput({scales, ProgramTensorMetadataDependency::TypeAndRank})
              .AddOutput({decoded.get(), ProgramTensorMetadataDependency::TypeAndRank, 4})
              .SetWorkgroupSize(64).SetDispatchGroupSize(1620);
          ORT_THROW_IF_ERROR(context.RunProgram(program));
          decoded_weights_ = std::move(decoded);
        });
      }
      auto run = [&]<typename P>(P& program) -> Status {
        if (predecoded) {
          program.AddInput({decoded_weights_.get(), ProgramTensorMetadataDependency::TypeAndRank, 4});
        }
''' + setup + '''      };
      if (predecoded) {
        return decoded_replay_.Run(context, {y, normalized, activated, tail}, [&]() -> Status {
          HCPostNormProjectionPredecodedProgram program{shared_, shared_scale};
          return run(program);
        }, 0, 0, {decoded_weights_.get()});
      }
      return replay_.Run(context, {y, normalized, activated, tail}, [&]() -> Status {
        HCPostNormProjectionProgram program{shared_, shared_scale};
        return run(program);
      });''')
    source = source[:start] + replacement + source[stop:]
    source = replace_once(source,
        "                       : std::initializer_list<int>{0, 1, 2, 3, 5, 6}} {",
        '''                       : std::initializer_list<int>{0, 1, 2, 3, 5, 6}},
        decoded_replay_{shared ? "HCSharedPostNormProjectionPredecodedF16" : "HCPostNormProjectionPredecodedF16",
                        shared ? std::initializer_list<int>{0, 1, 2, 3, 4, 5}
                               : std::initializer_list<int>{0, 1, 2, 3}} {''')
    source = replace_once(source,
        '                "HCPostNormProjectionF16 requires K10240/N324 symmetric block32 INT4");',
        '''                "HCPostNormProjectionF16 requires K10240/N324 symmetric block32 INT4");
    const Tensor* constant = nullptr;
    weights_constant_ = info.TryGetConstantInput(shared ? 7 : 5, &constant) &&
                        info.TryGetConstantInput(shared ? 8 : 6, &constant);''')
    return replace_once(source, "  mutable StaticNodeReplay replay_;",
        '''  mutable StaticNodeReplay replay_;
  bool weights_constant_ = false;
  mutable std::once_flag decode_once_;
  mutable std::unique_ptr<Tensor> decoded_weights_;
  mutable HCPredecodedReplay decoded_replay_;''')


if __name__ == "__main__":
    replay_manifest = json.loads((REPLAY.parent.parent / "build-manifest.json").read_text())
    assert sha(REPLAY) == replay_manifest["source_sha256"]["overlay/static_node_replay.h"]
    includes = OUTPUT.parent / "predecoded-replay-v2"
    includes.mkdir(parents=True, exist_ok=False)
    header = includes / "hc_predecoded_replay.h"
    header.write_text(stage_replay(REPLAY.read_text()).replace("HCStageReplay", "HCPredecodedReplay"))
    build(output=OUTPUT, flag=FLAG, source_transform=transform,
          parent_dir=Path("/workspace/qwen38-int4-dense-block32/build-v2"),
          extra_include_dirs=(includes,))
    path = OUTPUT / "build-manifest.json"
    report = json.loads(path.read_text())
    report.update(no_existing_kernel_object_layout_changes=False,
                  private_kernel_constant_weight_cache_added=True,
                  decoded_weight_bytes_per_eligible_kernel=324 * 10240 * 2,
                  constant_weights_and_scales_only=True,
                  frozen_extra_input_replay_parent_sha256=sha(REPLAY),
                  original_model_files_unchanged=True)
    report["source_sha256"]["../predecoded-replay-v2/hc_predecoded_replay.h"] = sha(header)
    path.write_text(json.dumps(report, indent=2) + "\n")
