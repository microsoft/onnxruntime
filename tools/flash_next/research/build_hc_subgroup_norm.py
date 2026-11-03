from pathlib import Path

from build_hc_bitcast_unpack import main as build
from build_qmoe_node_replay import replace_once


FLAG = "ORT_WEBGPU_HC_POST_NORM_SUBGROUP"


def transform(source):
    start = source.index("class HCPostNormProjectionProgram final")
    stop = source.index("class HCPostNormProjectionF16", start)
    program = source[start:stop]
    program = replace_once(program, "HCPostNormProjectionProgram(bool shared, bool shared_scale)",
                           "HCPostNormProjectionProgram(bool shared, bool shared_scale, bool subgroup_norm)")
    program = replace_once(program,
        ': Program{"HCPostNormProjectionF16"}, shared_{shared}, shared_scale_{shared_scale} {}',
        ': Program{subgroup_norm ? "HCPostNormProjectionSubgroupNormF16" : "HCPostNormProjectionF16"}, '
        'shared_{shared}, shared_scale_{shared_scale}, subgroup_norm_{subgroup_norm} {}')
    program = replace_once(program, '"for (var width = 128u; width > 0u; width >>= 1u) {\\n"',
        '(subgroup_norm_ ? "for (var width = 128u; width >= 32u; width >>= 1u) {\\n" : '
        '"for (var width = 128u; width > 0u; width >>= 1u) {\\n")')
    marker = '    code << "}\\nworkgroupBarrier();\\n}\\n";\n'
    reduction = '''    if (subgroup_norm_) {
      code << "if (subgroup_index == 0u) {\\n";
      for (int stream = 0; stream < 4; ++stream) {
        code << "var reduced" << stream << " = sums[" << stream * 256 << "u + sg_id];\\n";
      }
      code << "for (var width = 16u; width > 0u; width >>= 1u) {\\n";
      for (int stream = 0; stream < 4; ++stream) {
        code << "let other" << stream << " = subgroupShuffle(reduced" << stream
             << ", (sg_id + width) & 31u);\\n";
      }
      code << "if (sg_id < width) {\\n";
      for (int stream = 0; stream < 4; ++stream) {
        code << "reduced" << stream << " += other" << stream << ";\\n";
      }
      code << "}\\n}\\n";
      for (int stream = 0; stream < 4; ++stream) {
        code << "if (sg_id == 0u) { sums[" << stream * 256 + 32 << "u] = reduced" << stream << "; }\\n";
      }
      code << "}\\nworkgroupBarrier();\\n";
    }
'''
    program = replace_once(program, marker, marker + reduction)
    program = replace_once(program, '" = inverseSqrt(sums[" << stream * 256',
                           '" = inverseSqrt(sums[" << (stream * 256 + (subgroup_norm_ ? 32 : 0))')
    program = replace_once(program, "  bool shared_scale_;\n", "  bool shared_scale_;\n  bool subgroup_norm_;\n")
    kernel = source[stop:]
    kernel = replace_once(kernel,
        "  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override {",
        f'''  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override {{
    const int subgroup_mode = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
    ORT_RETURN_IF_NOT(subgroup_mode == 0 || subgroup_mode == 1, "{FLAG} must be 0 or 1");
    const bool subgroup_norm = subgroup_mode == 1 && context.HasFeature(wgpu::FeatureName::Subgroups) &&
        context.AdapterInfo().vendor == std::string_view{{"nvidia"}} &&
        context.AdapterInfo().subgroupMinSize == 32 && context.AdapterInfo().subgroupMaxSize == 32;''')
    kernel = replace_once(kernel, "HCPostNormProjectionProgram program{shared_, shared_scale};",
                          "HCPostNormProjectionProgram program{shared_, shared_scale, subgroup_norm};")
    kernel = replace_once(kernel, "return context.RunProgram(program);\n      });",
                          "return context.RunProgram(program);\n      }, 0, subgroup_norm);")
    return source[:start] + program + kernel


if __name__ == "__main__":
    build(output=Path("/workspace/qwen38-int4-hc-subgroup-norm/build-v2"),
          flag=FLAG, source_transform=transform)
