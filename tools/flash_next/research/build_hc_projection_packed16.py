from pathlib import Path

from build_hc_projection_inner_unroll import program_parts
from build_qmoe_node_replay import replace_once
from build_scoped_webgpu_source import build


FLAG = "ORT_WEBGPU_HC_PROJECTION_PACKED16"
OUTPUT = Path("/workspace/qwen38-int4-hc-projection-packed16/build-v1")
SOURCE = Path("/workspace/qwen38-int4-hc-post-norm-projection/build-v2/hc_post_norm_projection.cc")
HELPER = r'''
int HCProjectionPacked16Mode() {
  static const int mode = []() {
    const int value = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_HC_PROJECTION_PACKED16", 0);
    ORT_ENFORCE(value == 0 || value == 1, "ORT_WEBGPU_HC_PROJECTION_PACKED16 must be 0 or 1");
    return value;
  }();
  return mode;
}

void RecordHCProjectionPacked16Shader() {
  const char* path = std::getenv("ORT_WEBGPU_HC_PROJECTION_PACKED16_STATS_FILE");
  if (path == nullptr) return;
  std::ofstream output(path, std::ios::app);
  output << "{\"operator\":\"HCProjectionPacked16Shader\",\"mode\":"
         << HCProjectionPacked16Mode() << ",\"logical_streams_per_thread\":2}\n";
  output.flush();
  ORT_ENFORCE(output.good(), "Cannot write HC projection packed16 statistics to ", path);
}
'''


def packed_projection(original):
    start = original.index('         << "let lane = local_idx % 32u;\\n"')
    stop = original.index('         << "  if (col < uniforms.prefix)', start)
    tile_start = original.index('         << "  let block = tile * 32u + lane;\\n"', start)
    inner_start = original.index('         << "    let low = vec4<f16>', tile_start)
    inner_end = original.index('         << "    sum += dot(x0, w0) + dot(x1, w1);\\n"', inner_start)
    tile_prefix = original[tile_start:inner_start]
    tile_prefix = tile_prefix[:tile_prefix.index('         << "  var sum =')]
    inner = original[inner_start:inner_end]

    def stream(fragment, suffix):
        import re

        return re.sub(r"\b(block|packed|scale_value|low|high|w0|w1|base|x0|x1)\b",
                      lambda match: match[0] + suffix, fragment)

    prefix = ('         << "let lane = local_idx % 16u;\\n"\n'
              '         << "let column = local_idx / 16u;\\n"\n'
              '         << "let col = workgroup_idx * 8u + column;\\n"\n'
              '         << "var total = vec2<f16>(0);\\n"\n'
              '         << "if (local_idx < 128u && col < 324u) {\\n"\n'
              '         << "for (var tile = 0u; tile < 10u; tile++) {\\n"\n')
    second = replace_once(tile_prefix, "tile * 32u + lane;", "tile * 32u + lane + 16u;")
    replacement = (prefix + stream(tile_prefix, "a") + stream(second, "b") +
                   '         << "  var sum = vec2<f16>(0);\\n"\n'
                   '         << "  for (var i = 0u; i < 4u; i++) {\\n"\n' +
                   stream(inner, "a") + stream(inner, "b") +
                   '         << "    sum += vec2<f16>(dot(x0a, w0a) + dot(x1a, w1a), dot(x0b, w0b) + dot(x1b, w1b));\\n"\n'
                   '         << "  }\\n"\n'
                   '         << "  total += sum;\\n"\n'
                   '         << "}\\n}\\n"\n'
                   '         << "if (local_idx < 128u) {\\n"\n'
                   '         << "partials[column * 32u + lane] = total.x;\\n"\n'
                   '         << "partials[column * 32u + lane + 16u] = total.y;\\n"\n'
                   '         << "}\\n"\n'
                   '         << "workgroupBarrier();\\n"\n'
                   '         << "if (local_idx < 128u && lane == 0u && col < 324u) {\\n"\n'
                   '         << "  var value = f16(0);\\n"\n'
                   '         << "  for (var i = 0u; i < 32u; i++) { value += partials[column * 32u + i]; }\\n"\n')
    return original[:start] + replacement + original[stop:]


def transform_program(program):
    program = replace_once(program, 'Program{"HCPostNormProjectionF16"}',
                           'Program{HCProjectionPacked16Mode() == 1 ? '
                           '"HCPostNormProjectionPacked16F16" : "HCPostNormProjectionF16"}')
    program = replace_once(program, "    auto& code = shader.MainFunctionBody();",
                           "    RecordHCProjectionPacked16Shader();\n"
                           "    auto& code = shader.MainFunctionBody();")
    start = program.index('         << "let lane = local_idx % 32u;\\n"')
    end = program.index('    return Status::OK();', start)
    original = program[start:end]
    candidate = packed_projection(program)[start:]
    candidate = candidate[:candidate.index('    return Status::OK();')]
    # End the preceding normalization stream before selecting the projection body.
    return (program[:start].rstrip() + ';\n'
            '    if (HCProjectionPacked16Mode() == 1) {\n'
            '      code\n' + candidate +
            '    } else {\n      code\n' + original + '    }\n' + program[end:])


def transform_source(source):
    start, end, program = program_parts(source)
    return ("#include <cstdlib>\n#include <fstream>\n" + source[:start] +
            HELPER + transform_program(program) + source[end:])


if __name__ == "__main__":
    build(OUTPUT, SOURCE, transform_source, SOURCE.parent / "native_compare.cpp", FLAG,
          dict(logical_projection_streams_per_thread=2,
               active_projection_threads=128, normalization_threads=256,
               componentwise_fp16_accumulators=True,
               original_rms_and_projection_fold_unchanged=True,
               original_barriers_and_workgroup_size_unchanged=True))
