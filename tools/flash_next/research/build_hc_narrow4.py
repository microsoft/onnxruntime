from pathlib import Path

from build_hc_bitcast_unpack import main as build
from build_hc_post_norm_vec4 import private_program_transform
from build_qmoe_node_replay import replace_once


FLAG = "ORT_WEBGPU_HC_NARROW4"
PROGRAM = "HCPostNormProjectionNarrow4F16"
OUTPUT = Path("/workspace/qwen38-int4-hc-narrow4/build-v1")


def narrow_program(source):
    start = source.index("class HCPostNormProjectionProgram final")
    stop = source.index("class HCPostNormProjectionF16", start)
    program = source[start:stop].replace("HCPostNormProjectionProgram", "HCPostNormProjectionNarrow4Program")
    program = replace_once(program, 'Program{"HCPostNormProjectionF16"}', f'Program{{"{PROGRAM}"}}')
    loop = "for (var h = local_idx; h < uniforms.hidden; h += 256u)"
    assert program.count(loop) == 3
    program = program.replace(loop, "for (var h = local_idx; h < uniforms.hidden; h += 128u)")
    start = program.index('    for (int stream = 0; stream < 4; ++stream) code << "var sum"')
    stop = program.index('    code << "workgroupBarrier();\\n"', start)
    sums = program[start:stop]
    sums = replace_once(sums, "var h = local_idx; h < uniforms.hidden; h += 128u",
                        "var h = virtual_idx; h < uniforms.hidden; h += 256u")
    sums = replace_once(sums, 'u + local_idx] = sum', 'u + virtual_idx] = sum')
    # Keep each original 256-lane RMS partial and its ordered additions independent.
    sums = ('    code << "for (var virtual_idx = local_idx; virtual_idx < 256u; virtual_idx += 128u) {\\n";\n'
            + sums + '    code << "}\\n";\n')
    program = program[:start] + sums + program[stop:]
    program = replace_once(program, "workgroup_idx * 8u + local_idx / 32u",
                           "workgroup_idx * 4u + local_idx / 32u")
    return replace_once(program, "partials: array<f16, 256>", "partials: array<f16, 128>")


def transform(source):
    result = private_program_transform(source, narrow_program(source), FLAG, "narrow4",
                                       "HCPostNormProjectionNarrow4Program")
    return replace_once(result, ".SetWorkgroupSize(256).SetDispatchGroupSize(41);",
                        ".SetWorkgroupSize(narrow4 == 1 ? 128 : 256)"
                        ".SetDispatchGroupSize(narrow4 == 1 ? 81 : 41);")


if __name__ == "__main__":
    build(output=OUTPUT, flag=FLAG, source_transform=transform,
          parent_dir=Path("/workspace/qwen38-int4-qsa-prefill-score16/build-v1"))
