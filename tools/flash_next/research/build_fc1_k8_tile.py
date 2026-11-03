from pathlib import Path

from build_fc1_k16_tile import FLAG as K16_FLAG, transform as transform_k16
from build_fc2_subgroup_fold import FLAG as FOLD_FLAG, OVERLAY, transform_comparator as fold_comparator
from build_hc_bitcast_unpack import main as build
from build_qmoe_node_replay import replace_once


FLAG = "ORT_WEBGPU_QMOE_FC1_K8_TILE"
PROGRAM = "QmoeFC1K8TileBitcastF16"


def transform(source, program_source=None):
    source = transform_k16(source, program_source).replace(K16_FLAG, FLAG)
    source = source.replace("k16_tile", "k8_tile").replace("QmoeFC1K16Tile", "QmoeFC1K8Tile")
    for before, after in (
        ("tile_a: array<vec4<f16>, 128>", "tile_a: array<vec4<f16>, 64>"),
        ("let tile_lane = local_idx % 16u", "let tile_lane = local_idx % 8u"),
        ("let group = local_idx / 16u", "let group = local_idx / 8u"),
        ("kidx += 512u", "kidx += 256u"),
        ("id < 128u", "id < 64u"),
        ("offset < 8u; offset += 8u", "offset < 16u; offset += 16u"),
    ):
        source = replace_once(source, before, after)
    for before, after in (
        ("var<workgroup> partials: array<array<f16, 32>, 8>;\\n",
         "var<workgroup> partials: array<array<f16, 32>, 16>;\\n"),
        ("let base_col = (workgroup_idx % uniforms.tiles) * 8u;\\n",
         "let base_col = (workgroup_idx % uniforms.tiles) * 16u;\\n"),
        ("if (local_idx < 8u && base_col + local_idx < uniforms.cols * 2u) {\\n",
         "if (local_idx < 16u && base_col + local_idx < uniforms.cols * 2u) {\\n"),
        ("if (local_idx < 4u) {\\n", "if (local_idx < 8u) {\\n"),
    ):
        source = replace_once(source, f'<< "{before}"', f'<< (k8_tile_ ? "{after}" : "{before}")')
    return replace_once(source, "const auto tiles = CeilDiv(cols * 2u, 8u);",
                        "const auto tiles = CeilDiv(cols * 2u, fc1_k8_tile == 1 ? 16u : 8u);")


def transform_comparator(source):
    return fold_comparator(source).replace(f'"{FOLD_FLAG}"', f'"{FLAG}"')


if __name__ == "__main__":
    build(output=Path("/workspace/qwen38-int4-fc1-k8-tile/build-v1"), flag=FLAG,
          source_transform=transform,
          component_dir=Path("/workspace/qwen38-int4-qmoe-grouped-prefill/build-v1"),
          source_name="qmoe.cc", comparator_flag="ORT_WEBGPU_QMOE_GROUPED_PREFILL",
          comparator_transform=transform_comparator,
          parent_dir=Path("/workspace/qwen38-int4-qmoe-prefill-rows4/build-v2"),
          extra_include_dirs=(OVERLAY,))
