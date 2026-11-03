from pathlib import Path

from build_hc_projection_inner_unroll import program_parts
from build_qmoe_node_replay import replace_once
from build_scoped_webgpu_source import build


FLAG = "ORT_WEBGPU_HC_RMS_TREE_UNROLL"
OUTPUT = Path("/workspace/qwen38-int4-hc-rms-tree-unroll/build-v1")
SOURCE = Path("/workspace/qwen38-int4-hc-post-norm-projection/build-v2/hc_post_norm_projection.cc")
HELPER = r'''
int HCRmsTreeUnrollMode() {
  static const int mode = []() {
    const int value = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_HC_RMS_TREE_UNROLL", 0);
    ORT_ENFORCE(value == 0 || value == 1, "ORT_WEBGPU_HC_RMS_TREE_UNROLL must be 0 or 1");
    return value;
  }();
  return mode;
}

void RecordHCRmsTreeUnrollShader() {
  const char* path = std::getenv("ORT_WEBGPU_HC_RMS_TREE_UNROLL_STATS_FILE");
  if (path == nullptr) return;
  std::ofstream output(path, std::ios::app);
  output << "{\"operator\":\"HCRmsTreeUnrollShader\",\"mode\":"
         << HCRmsTreeUnrollMode() << ",\"levels\":8}\n";
  output.flush();
  ORT_ENFORCE(output.good(), "Cannot write HC RMS-tree unroll statistics to ", path);
}
'''


def transform_program(program):
    program = replace_once(program, 'Program{"HCPostNormProjectionF16"}',
                           'Program{HCRmsTreeUnrollMode() == 1 ? '
                           '"HCPostNormProjectionRmsTreeUnrollF16" : "HCPostNormProjectionF16"}')
    program = replace_once(program, "    auto& code = shader.MainFunctionBody();",
                           "    RecordHCRmsTreeUnrollShader();\n"
                           "    auto& code = shader.MainFunctionBody();")
    start = program.index('    code << "workgroupBarrier();\\n"\n'
                          '         << "for (var width = 128u;')
    body_start = program.index("    for (int stream = 0;", start)
    body_end = program.index('    code << "}\\nworkgroupBarrier();\\n}\\n";', body_start)
    end = body_end + len('    code << "}\\nworkgroupBarrier();\\n}\\n";')
    original = program[start:end]
    body = program[body_start:body_end]
    replacement = '''    if (HCRmsTreeUnrollMode() == 0) {
''' + original + '''
    } else {
      code << "workgroupBarrier();\\n";
      for (uint32_t width = 128; width > 0; width >>= 1) {
        code << "{\\nlet width = " << width << "u;\\nif (local_idx < width) {\\n";
''' + body + '''        code << "}\\nworkgroupBarrier();\\n}\\n";
      }
    }'''
    return program[:start] + replacement + program[end:]


def transform_source(source):
    start, end, program = program_parts(source)
    return ("#include <cstdlib>\n#include <fstream>\n" + source[:start] +
            HELPER + transform_program(program) + source[end:])


if __name__ == "__main__":
    build(OUTPUT, SOURCE, transform_source, SOURCE.parent / "native_compare.cpp", FLAG,
          dict(ordered_reduction_levels=8, original_projection_loop_unchanged=True))
