from pathlib import Path

from build_hc_projection_inner_unroll import program_parts
from build_qmoe_node_replay import replace_once
from build_scoped_webgpu_source import build


FLAG = "ORT_WEBGPU_HC_PROJECTION_PADDING34"
OUTPUT = Path("/workspace/qwen38-int4-hc-projection-padding34/build-v1")
SOURCE = Path("/workspace/qwen38-int4-hc-post-norm-projection/build-v2/hc_post_norm_projection.cc")
HELPER = r'''
int HCProjectionPadding34Mode() {
  static const int mode = []() {
    const int value = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_HC_PROJECTION_PADDING34", 0);
    ORT_ENFORCE(value == 0 || value == 1, "ORT_WEBGPU_HC_PROJECTION_PADDING34 must be 0 or 1");
    return value;
  }();
  return mode;
}

uint32_t HCProjectionPaddingStride() {
  return HCProjectionPadding34Mode() == 1 ? 34u : 36u;
}

void RecordHCProjectionPaddingShader() {
  const char* path = std::getenv("ORT_WEBGPU_HC_PROJECTION_PADDING34_STATS_FILE");
  if (path == nullptr) return;
  std::ofstream output(path, std::ios::app);
  output << "{\"operator\":\"HCProjectionPaddingShader\",\"mode\":"
         << HCProjectionPadding34Mode() << ",\"stride\":" << HCProjectionPaddingStride() << "}\n";
  output.flush();
  ORT_ENFORCE(output.good(), "Cannot write HC projection padding statistics to ", path);
}
'''


def transform_program(program):
    program = replace_once(program, 'Program{"HCPostNormProjectionF16"}',
                           'Program{HCProjectionPadding34Mode() == 1 ? '
                           '"HCPostNormProjectionPadding34F16" : "HCPostNormProjectionF16"}')
    program = replace_once(program, "    auto& code = shader.MainFunctionBody();",
                           "    RecordHCProjectionPaddingShader();\n"
                           "    auto& code = shader.MainFunctionBody();")
    program = replace_once(program, '"var<workgroup> post_values: array<f16, 11520>;\\n"',
                           '"var<workgroup> post_values: array<f16, " '
                           '<< 320u * HCProjectionPaddingStride() << ">;\\n"')
    assert program.count("* 36u") == 4
    return program.replace("* 36u", '* " << HCProjectionPaddingStride() << "u')


def transform_source(source):
    start, end, program = program_parts(source)
    return ("#include <cstdlib>\n#include <fstream>\n" + source[:start] +
            HELPER + transform_program(program) + source[end:])


if __name__ == "__main__":
    build(OUTPUT, SOURCE, transform_source, SOURCE.parent / "native_compare.cpp", FLAG,
          dict(shared_block_stride_half_elements=34, shared_half_elements=10880,
               original_arithmetic_and_barriers_unchanged=True))
