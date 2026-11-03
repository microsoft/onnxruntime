from pathlib import Path

from build_hc_bitcast_unpack import main as build
from build_qmoe_node_replay import replace_once


FLAG = "ORT_WEBGPU_HC_POST_NORM_FUSED_SUM"


def transform(source):
    start = source.index("class HCPostNormProjectionProgram final")
    stop = source.index("class HCPostNormProjectionF16", start)
    program = source[start:stop]
    program = replace_once(program, "HCPostNormProjectionProgram(bool shared, bool shared_scale)",
                           "HCPostNormProjectionProgram(bool shared, bool shared_scale, bool fused_sum)")
    program = replace_once(program,
        ': Program{"HCPostNormProjectionF16"}, shared_{shared}, shared_scale_{shared_scale} {}',
        ': Program{fused_sum ? "HCPostNormProjectionFusedSumF16" : "HCPostNormProjectionF16"}, '
        'shared_{shared}, shared_scale_{shared_scale}, fused_sum_{fused_sum} {}')
    declarations = '    for (int stream = 0; stream < 4; ++stream) code << "var sum" << stream << " = 0.0;\\n";\n'
    first_loop = ('    code << "for (var h = local_idx; h < uniforms.hidden; h += 256u) {\\n";\n'
                  '    if (shared_) {')
    program = replace_once(program, first_loop,
                           '    if (fused_sum_) {\n' + declarations + '    }\n' + first_loop)
    program = replace_once(program,
        '           << "post_values[address" << stream << "] = stored" << stream << ";\\n"\n'
        '           << "if (workgroup_idx == 0u) {\\n"',
        '''           << "post_values[address" << stream << "] = stored" << stream << ";\\n";
      if (fused_sum_) {
        code << "let rms_value" << stream << " = unpack2x16float(pack2x16float(vec2<f32>(f32(stored"
             << stream << "), 0.0))).x;\\n"
             << "sum" << stream << " += rms_value" << stream << " * rms_value" << stream << ";\\n";
      }
      code << "if (workgroup_idx == 0u) {\\n"''')
    rms_start = program.index('    code << "}\\nworkgroupBarrier();\\n";')
    rms_end = program.index('    for (int stream = 0; stream < 4; ++stream) {\n'
                            '      code << "sums["', rms_start)
    legacy = program[rms_start:rms_end]
    legacy = replace_once(legacy, '    code << "}\\nworkgroupBarrier();\\n";\n',
                          '    code << "workgroupBarrier();\\n";\n')
    program = (program[:rms_start] + '    code << "}\\n";\n    if (!fused_sum_) {\n' +
               legacy + '    }\n' + program[rms_end:])
    program = replace_once(program, "  bool shared_scale_;\n", "  bool shared_scale_;\n  bool fused_sum_;\n")
    kernel = source[stop:]
    kernel = replace_once(kernel,
        "  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override {",
        f'''  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override {{
    const int fused_sum = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
    ORT_RETURN_IF_NOT(fused_sum == 0 || fused_sum == 1, "{FLAG} must be 0 or 1");''')
    kernel = replace_once(kernel, "HCPostNormProjectionProgram program{shared_, shared_scale};",
                          "HCPostNormProjectionProgram program{shared_, shared_scale, fused_sum == 1};")
    kernel = replace_once(kernel, "return context.RunProgram(program);\n      });",
                          "return context.RunProgram(program);\n      }, 0, fused_sum);")
    return source[:start] + program + kernel


if __name__ == "__main__":
    build(output=Path("/workspace/qwen38-int4-hc-fused-sum/build-v3"),
          flag=FLAG, source_transform=transform)
