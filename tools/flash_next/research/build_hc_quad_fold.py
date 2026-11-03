from pathlib import Path

from build_hc_bitcast_unpack import main as build
from build_hc_post_norm_vec4 import private_program_transform
from build_qmoe_node_replay import replace_once


FLAG = "ORT_WEBGPU_HC_QUAD_FOLD"
PROGRAM = "HCPostNormProjectionQuadFoldF16"
OUTPUT = Path("/workspace/qwen38-int4-hc-quad-fold/build-v1")


def quad_program(source):
    start = source.index("class HCPostNormProjectionProgram final")
    stop = source.index("class HCPostNormProjectionF16", start)
    program = source[start:stop].replace("HCPostNormProjectionProgram", "HCPostNormProjectionQuadFoldProgram")
    program = replace_once(program, 'Program{"HCPostNormProjectionF16"}', f'Program{{"{PROGRAM}"}}')
    before = '''         << "if (lane == 0u && col < 324u) {\\n"
         << "  var value = f16(0);\\n"
         << "  for (var i = 0u; i < 32u; i++) { value += partials[local_idx + i]; }\\n"'''
    after = '''         << "if (local_idx < 2u) {\\n"
         << "  var folded = vec4<f16>(0);\\n"
         << "  for (var i = 0u; i < 32u; i++) {\\n"
         << "    let base = local_idx * 128u + i;\\n"
         << "    folded += vec4<f16>(partials[base], partials[base + 32u], partials[base + 64u], partials[base + 96u]);\\n"
         << "  }\\n"
         << "  for (var component = 0u; component < 4u; component++) {\\n"
         << "    let col = workgroup_idx * 8u + local_idx * 4u + component;\\n"
         << "    if (col < 324u) {\\n"
         << "      let value = folded[component];\\n"'''
    program = replace_once(program, before, after)
    return replace_once(program, '<< "  }\\n}\\n";', '<< "  }\\n}\\n}\\n}\\n";')


def transform(source):
    return private_program_transform(source, quad_program(source), FLAG, "quad_fold",
                                     "HCPostNormProjectionQuadFoldProgram")


if __name__ == "__main__":
    build(output=OUTPUT, flag=FLAG, source_transform=transform,
          parent_dir=Path("/workspace/qwen38-int4-qsa-prefill-score16/build-v1"))
