from pathlib import Path

from build_hc_projection_inner_unroll import program_parts
from build_qmoe_node_replay import replace_once
from build_scoped_webgpu_source import build


FLAG = "ORT_WEBGPU_HC_PROJECTION_LANE8"
OUTPUT = Path("/workspace/qwen38-int4-hc-projection-lane8/build-v1")
SOURCE = Path("/workspace/qwen38-int4-hc-post-norm-projection/build-v2/hc_post_norm_projection.cc")
HELPER = r'''
int HCProjectionLane8Mode() {
  static const int mode = []() {
    const int value = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_HC_PROJECTION_LANE8", 0);
    ORT_ENFORCE(value == 0 || value == 1, "ORT_WEBGPU_HC_PROJECTION_LANE8 must be 0 or 1");
    return value;
  }();
  return mode;
}

void RecordHCProjectionLane8Shader() {
  const char* path = std::getenv("ORT_WEBGPU_HC_PROJECTION_LANE8_STATS_FILE");
  if (path == nullptr) return;
  std::ofstream output(path, std::ios::app);
  output << "{\"operator\":\"HCProjectionLane8Shader\",\"mode\":"
         << HCProjectionLane8Mode() << ",\"columns_per_warp\":4}\n";
  output.flush();
  ORT_ENFORCE(output.good(), "Cannot write HC projection lane8 statistics to ", path);
}
'''


def transform_program(program):
    program = replace_once(program, 'Program{"HCPostNormProjectionF16"}',
                           'Program{HCProjectionLane8Mode() == 1 ? '
                           '"HCPostNormProjectionLane8F16" : "HCPostNormProjectionF16"}')
    program = replace_once(program, "    auto& code = shader.MainFunctionBody();",
                           "    RecordHCProjectionLane8Shader();\n"
                           "    auto& code = shader.MainFunctionBody();")
    program = replace_once(program,
        '''         << "let lane = local_idx % 32u;\\n"
         << "let col = workgroup_idx * 8u + local_idx / 32u;\\n"''',
        '''         << (HCProjectionLane8Mode() == 1
                 ? "let lane = ((local_idx % 128u) / 32u) * 8u + local_idx % 8u;\\n"
                   "let column = (local_idx / 128u) * 4u + (local_idx % 32u) / 8u;\\n"
                   "let col = workgroup_idx * 8u + column;\\n"
                   "let logical_idx = column * 32u + lane;\\n"
                 : "let lane = local_idx % 32u;\\n"
                   "let col = workgroup_idx * 8u + local_idx / 32u;\\n")''')
    program = replace_once(program, '"partials[local_idx] = total;\\n"',
        '(HCProjectionLane8Mode() == 1 ? "partials[logical_idx] = total;\\n" : '
        '"partials[local_idx] = total;\\n")')
    return replace_once(program,
        '"  for (var i = 0u; i < 32u; i++) { value += partials[local_idx + i]; }\\n"',
        '(HCProjectionLane8Mode() == 1 ? '
        '"  for (var i = 0u; i < 32u; i++) { value += partials[logical_idx + i]; }\\n" : '
        '"  for (var i = 0u; i < 32u; i++) { value += partials[local_idx + i]; }\\n")')


def transform_source(source):
    start, end, program = program_parts(source)
    return ("#include <cstdlib>\n#include <fstream>\n" + source[:start] +
            HELPER + transform_program(program) + source[end:])


if __name__ == "__main__":
    build(OUTPUT, SOURCE, transform_source, SOURCE.parent / "native_compare.cpp", FLAG,
          dict(projection_columns_per_warp=4, projection_k_lanes_per_column_per_warp=8,
               original_rms_and_projection_arithmetic_unchanged=True,
               original_barriers_and_workgroup_size_unchanged=True))
