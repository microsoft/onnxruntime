from pathlib import Path

from build_hc_projection_inner_unroll import program_parts
from build_qmoe_node_replay import replace_once
from build_scoped_webgpu_source import build


FLAG = "ORT_WEBGPU_HC_RMS_VECTOR_LAYOUT"
OUTPUT = Path("/workspace/qwen38-int4-hc-rms-vector-layout/build-v1")
SOURCE = Path("/workspace/qwen38-int4-hc-post-norm-projection/build-v2/hc_post_norm_projection.cc")
HELPER = r'''
int HCRmsVectorLayoutMode() {
  static const int mode = []() {
    const int value = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_HC_RMS_VECTOR_LAYOUT", 0);
    ORT_ENFORCE(value == 0 || value == 1, "ORT_WEBGPU_HC_RMS_VECTOR_LAYOUT must be 0 or 1");
    return value;
  }();
  return mode;
}

void RecordHCRmsVectorLayoutShader() {
  const char* path = std::getenv("ORT_WEBGPU_HC_RMS_VECTOR_LAYOUT_STATS_FILE");
  if (path == nullptr) return;
  std::ofstream output(path, std::ios::app);
  output << "{\"operator\":\"HCRmsVectorLayoutShader\",\"mode\":"
         << HCRmsVectorLayoutMode() << ",\"components\":4}\n";
  output.flush();
  ORT_ENFORCE(output.good(), "Cannot write HC RMS vector-layout statistics to ", path);
}
'''


def transform_program(program):
    program = replace_once(program, 'Program{"HCPostNormProjectionF16"}',
                           'Program{HCRmsVectorLayoutMode() == 1 ? '
                           '"HCPostNormProjectionRmsVectorLayoutF16" : "HCPostNormProjectionF16"}')
    program = replace_once(program, "    auto& code = shader.MainFunctionBody();",
                           "    RecordHCRmsVectorLayoutShader();\n"
                           "    auto& code = shader.MainFunctionBody();")
    program = replace_once(program, '"var<workgroup> sums: array<f32, 1024>;\\n"',
                           '(HCRmsVectorLayoutMode() == 1 ? '
                           '"var<workgroup> sums: array<vec4<f32>, 256>;\\n" : '
                           '"var<workgroup> sums: array<f32, 1024>;\\n") <<')
    start = program.index('    for (int stream = 0; stream < 4; ++stream) {\n'
                          '      code << "sums["')
    end = program.index('    code << "workgroupBarrier();\\n"', start)
    original = program[start:end]
    program = program[:start] + '''    if (HCRmsVectorLayoutMode() == 0) {
''' + original + '''    } else {
      code << "sums[local_idx] = vec4<f32>(sum0, sum1, sum2, sum3);\\n";
    }
''' + program[end:]
    start = program.index('    for (int stream = 0; stream < 4; ++stream) {\n'
                          '      code << "sums["', program.index('"for (var width = 128u;'))
    end = program.index('    code << "}\\nworkgroupBarrier();\\n}\\n";', start)
    original = program[start:end]
    program = program[:start] + '''    if (HCRmsVectorLayoutMode() == 0) {
''' + original + '''    } else {
      code << "sums[local_idx] += sums[local_idx + width];\\n";
    }
''' + program[end:]
    return replace_once(program,
        '''      code << "let inv_rms" << stream << " = inverseSqrt(sums[" << stream * 256
           << "u] / f32(uniforms.hidden) + uniforms.epsilon);\\n";''',
        '''      code << "let inv_rms" << stream << " = inverseSqrt("
           << (HCRmsVectorLayoutMode() == 1
                   ? "sums[0u]." + std::string(1, "xyzw"[stream])
                   : "sums[" + std::to_string(stream * 256) + "u]")
           << " / f32(uniforms.hidden) + uniforms.epsilon);\\n";''')


def transform_source(source):
    start, end, program = program_parts(source)
    return ("#include <cstdlib>\n#include <fstream>\n" + source[:start] +
            HELPER + transform_program(program) + source[end:])


if __name__ == "__main__":
    build(OUTPUT, SOURCE, transform_source, SOURCE.parent / "native_compare.cpp", FLAG,
          dict(rms_shared_vector_components=4, rms_shared_bytes=4096,
               original_projection_and_barriers_unchanged=True))
