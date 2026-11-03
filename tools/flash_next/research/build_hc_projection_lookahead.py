from pathlib import Path

from build_hc_projection_inner_unroll import program_parts
from build_hc_projection_tile_unroll import HELPER as OLD_HELPER, FLAG as OLD_FLAG, SOURCE, transform_program as old_transform_program, unroll_projection
from build_qmoe_node_replay import replace_once
from build_scoped_webgpu_source import build


FLAG = "ORT_WEBGPU_HC_PROJECTION_LOOKAHEAD"
OUTPUT = Path("/workspace/qwen38-int4-hc-projection-lookahead/build-v1")
HELPER = (OLD_HELPER.replace(OLD_FLAG, FLAG)
          .replace("HCProjectionTileUnroll", "HCProjectionLookahead")
          .replace(r'tile_steps_per_lane\":10', r'lookahead_tiles\":1'))


def prefetch_projection(program):
    loop = '         << "for (var tile = 0u; tile < 10u; tile++) {\\n"\n'
    start = program.index(loop)
    load_start = program.index('         << "  let packed = "', start)
    load_end = program.index('         << "  var sum = f16(0);\\n"', load_start)
    loads = program[load_start:load_end]
    initial = (loads.replace("  let packed =", "var packed =")
               .replace("  let scale_value =", "var scale_value =")
               .replace("col * 320u + block", "col * 320u + lane"))
    next_loads = (loads.replace("  let packed =", "    next_packed =")
                  .replace("  let scale_value =", "    next_scale_value =")
                  .replace("col * 320u + block", "col * 320u + block + 32u"))
    replacement = ('         << "  var next_packed = packed;\\n"\n'
                   '         << "  var next_scale_value = scale_value;\\n"\n'
                   '         << "  if (tile < 9u) {\\n"\n' + next_loads +
                   '         << "  }\\n"\n')
    candidate = program[:load_start] + replacement + program[load_end:]
    candidate = replace_once(candidate, loop, initial + loop)
    return replace_once(candidate, '         << "  total += sum;\\n"\n',
                        '         << "  total += sum;\\n"\n'
                        '         << "  packed = next_packed;\\n"\n'
                        '         << "  scale_value = next_scale_value;\\n"\n')


def transform_program(program):
    start = program.index('         << "let lane = local_idx % 32u;\\n"')
    unrolled = unroll_projection(program)[start:]
    unrolled = unrolled[:unrolled.index('    return Status::OK();')]
    prefetched = prefetch_projection(program)[start:]
    prefetched = prefetched[:prefetched.index('    return Status::OK();')]
    result = replace_once(old_transform_program(program), unrolled, prefetched)
    return (result.replace(OLD_FLAG, FLAG)
            .replace("HCProjectionTileUnroll", "HCProjectionLookahead")
            .replace("HCPostNormProjectionTileUnrollF16", "HCPostNormProjectionLookaheadF16"))


def transform_source(source):
    start, end, program = program_parts(source)
    return ("#include <cstdlib>\n#include <fstream>\n" + source[:start] +
            HELPER + transform_program(program) + source[end:])


if __name__ == "__main__":
    build(OUTPUT, SOURCE, transform_source, SOURCE.parent / "native_compare.cpp", FLAG,
          dict(projection_lookahead_tiles=1, original_ten_tile_loop_unchanged=True,
               original_component_loop_unchanged=True,
               original_dot_expressions_and_fp16_updates_unchanged=True,
               original_rms_projection_fold_and_geometry_unchanged=True))
