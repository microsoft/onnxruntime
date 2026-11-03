from pathlib import Path

from build_hc_projection_inner_unroll import program_parts
from build_qmoe_node_replay import replace_once
from build_scoped_webgpu_source import build


FLAG = "ORT_WEBGPU_HC_PROJECTION_TILE_UNROLL"
OUTPUT = Path("/workspace/qwen38-int4-hc-projection-tile-unroll/build-v1")
SOURCE = Path("/workspace/qwen38-int4-hc-post-norm-projection/build-v2/hc_post_norm_projection.cc")
HELPER = r'''
int HCProjectionTileUnrollMode() {
  static const int mode = []() {
    const int value = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_HC_PROJECTION_TILE_UNROLL", 0);
    ORT_ENFORCE(value == 0 || value == 1, "ORT_WEBGPU_HC_PROJECTION_TILE_UNROLL must be 0 or 1");
    return value;
  }();
  return mode;
}

void RecordHCProjectionTileUnrollShader() {
  const char* path = std::getenv("ORT_WEBGPU_HC_PROJECTION_TILE_UNROLL_STATS_FILE");
  if (path == nullptr) return;
  std::ofstream output(path, std::ios::app);
  output << "{\"operator\":\"HCProjectionTileUnrollShader\",\"mode\":"
         << HCProjectionTileUnrollMode() << ",\"tile_steps_per_lane\":10}\n";
  output.flush();
  ORT_ENFORCE(output.good(), "Cannot write HC projection tile-unroll statistics to ", path);
}
'''


def unroll_projection(program):
    loop = '         << "for (var tile = 0u; tile < 10u; tile++) {\\n"\n'
    start = program.index(loop)
    end = program.index('         << "}\\n}\\n"', start)
    body = program[start + len(loop):end]
    steps = []
    for tile in range(10):
        step = replace_once(body, "tile * 32u + lane", f"{tile * 32}u + lane")
        steps.append('         << "{\\n"\n' + step + '         << "}\\n"\n')
    return program[:start] + "".join(steps) + '         << "}\\n"' + program[end + len('         << "}\\n}\\n"'):]


def transform_program(program):
    program = replace_once(program, 'Program{"HCPostNormProjectionF16"}',
                           'Program{HCProjectionTileUnrollMode() == 1 ? '
                           '"HCPostNormProjectionTileUnrollF16" : "HCPostNormProjectionF16"}')
    program = replace_once(program, "    auto& code = shader.MainFunctionBody();",
                           "    RecordHCProjectionTileUnrollShader();\n"
                           "    auto& code = shader.MainFunctionBody();")
    start = program.index('         << "let lane = local_idx % 32u;\\n"')
    end = program.index('    return Status::OK();', start)
    original = program[start:end]
    candidate = unroll_projection(program)[start:]
    candidate = candidate[:candidate.index('    return Status::OK();')]
    return (program[:start].rstrip() + ';\n'
            '    if (HCProjectionTileUnrollMode() == 1) {\n'
            '      code\n' + candidate +
            '    } else {\n      code\n' + original + '    }\n' + program[end:])


def transform_source(source):
    start, end, program = program_parts(source)
    return ("#include <cstdlib>\n#include <fstream>\n" + source[:start] +
            HELPER + transform_program(program) + source[end:])


if __name__ == "__main__":
    build(OUTPUT, SOURCE, transform_source, SOURCE.parent / "native_compare.cpp", FLAG,
          dict(explicit_tile_steps=10, original_component_loop_unchanged=True,
               original_dot_expressions_and_fp16_updates_unchanged=True,
               original_rms_projection_fold_and_dispatch_unchanged=True))
