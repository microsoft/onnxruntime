from pathlib import Path

from build_fc2_subgroup_fold import FLAG as FOLD_FLAG, OVERLAY, transform_comparator as fold_comparator
from build_hc_bitcast_unpack import main as build
from build_qmoe_node_replay import replace_once


FLAG = "ORT_WEBGPU_QMOE_FC1_K16_TILE"
PROGRAM = "QmoeFC1K16TileBitcastF16"


def transform_program(source):
    source = replace_once(source, "QmoeFC1SwiGLUProgram(bool bias, bool bitcast)",
                          "QmoeFC1SwiGLUProgram(bool bias, bool bitcast, bool k16_tile)")
    source = replace_once(source, 'Program{bitcast ? "QmoeFC1SwiGLUBitcastF16"',
                          'Program{k16_tile ? (bitcast ? "QmoeFC1K16TileBitcastF16" : '
                          '"QmoeFC1K16TileF16") : bitcast ? "QmoeFC1SwiGLUBitcastF16"')
    source = replace_once(source, "bias_{bias}, bitcast_{bitcast} {}",
                          "bias_{bias}, bitcast_{bitcast}, k16_tile_{k16_tile} {}")
    replacements = (
        ("var<workgroup> tile_a: array<vec4<f16>, 256>;\\n",
         "var<workgroup> tile_a: array<vec4<f16>, 128>;\\n"),
        ("let lane = local_idx % 32u;\\n", "let tile_lane = local_idx % 16u;\\n"),
        ("let group = local_idx / 32u;\\n", "let group = local_idx / 16u;\\n"),
        ("for (var kidx = 0u; kidx < uniforms.K; kidx += 1024u) {\\n",
         "for (var kidx = 0u; kidx < uniforms.K; kidx += 512u) {\\n"
         "  let lane = (kidx / 32u) % 32u + tile_lane;\\n"),
        ("  for (var id = local_idx; id < 256u; id += 128u) {\\n",
         "  for (var id = local_idx; id < 128u; id += 128u) {\\n"),
        ("  for (var offset = 0u; offset < 8u; offset += 4u) {\\n",
         "  for (var offset = 0u; offset < 8u; offset += 8u) {\\n"),
        ("    let block = kidx / 32u + lane;\\n", "    let block = kidx / 32u + tile_lane;\\n"),
        ("        sum += dot(tile_a[lane * 8u + i * 2u], w0) + dot(tile_a[lane * 8u + i * 2u + 1u], w1);\\n",
         "        sum += dot(tile_a[tile_lane * 8u + i * 2u], w0) + "
         "dot(tile_a[tile_lane * 8u + i * 2u + 1u], w1);\\n"),
    )
    for before, after in replacements:
        source = replace_once(source, f'<< "{before}"', f'<< (k16_tile_ ? "{after}" : "{before}")')
    return replace_once(source, "  bool bitcast_;", "  bool bitcast_;\n  bool k16_tile_;")


def transform(source, program_source=None):
    validation = ('  ORT_RETURN_IF_NOT(prefill1024 == 0 || prefill1024 == 1, '
                  '"ORT_WEBGPU_QMOE_PREFILL1024 must be 0 or 1");')
    assert source.count(validation) == 2
    source = source.replace(validation, validation + f'''
  const int fc1_k16_tile = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(fc1_k16_tile == 0 || fc1_k16_tile == 1, "{FLAG} must be 0 or 1");''')
    source = replace_once(source, "  key.push_back(prefill1024);",
                          "  key.push_back(prefill1024);\n  key.push_back(fc1_k16_tile);")
    source = replace_once(source,
        "QmoeFC1SwiGLUProgram program{fc1_experts_bias_optional != nullptr, expert_bitcast == 1};",
        "QmoeFC1SwiGLUProgram program{fc1_experts_bias_optional != nullptr, expert_bitcast == 1, fc1_k16_tile == 1};")
    return replace_once(source, '#include "qmoe_fc1_activation.inc"',
                        transform_program(program_source if program_source is not None else
                                          (OVERLAY / "qmoe_fc1_activation.inc").read_text()))


def transform_comparator(source):
    return fold_comparator(source).replace(f'"{FOLD_FLAG}"', f'"{FLAG}"')


if __name__ == "__main__":
    build(output=Path("/workspace/qwen38-int4-fc1-k16-tile/build-v1"), flag=FLAG,
          source_transform=transform,
          component_dir=Path("/workspace/qwen38-int4-qmoe-grouped-prefill/build-v1"),
          source_name="qmoe.cc", comparator_flag="ORT_WEBGPU_QMOE_GROUPED_PREFILL",
          comparator_transform=transform_comparator,
          parent_dir=Path("/workspace/qwen38-int4-qmoe-prefill-rows4/build-v2"),
          extra_include_dirs=(OVERLAY,))
