from pathlib import Path

from build_dense_tile2 import COMPONENT, TEMPLATE, transform as tile2_transform
from build_hc_bitcast_unpack import main as build
from build_qmoe_node_replay import replace_once


FLAG = "ORT_WEBGPU_DENSE_WIDE8"


def transform(source):
    source = tile2_transform(source)
    for old, new in (("DenseTile2Program", "DenseWide8Program"),
                     ("ApplyDenseTile2MatMulNBits", "ApplyDenseWide8MatMulNBits"),
                     ("MatMulNBitsTile2F16", "MatMulNBitsWide8F16"),
                     ("ORT_WEBGPU_DENSE_TILE2", FLAG),
                     ("dense_tile2", "dense_wide8")):
        assert old in source
        source = source.replace(old, new)
    source = replace_once(source,
        "constexpr uint32_t workgroup_size = 64;\n  constexpr uint32_t tile_size = 2;",
        "constexpr uint32_t workgroup_size = 256;\n  constexpr uint32_t tile_size = 8;")
    validation = f'  ORT_RETURN_IF_NOT(dense_wide8 == 0 || dense_wide8 == 1, "{FLAG} must be 0 or 1");'
    source = replace_once(source, validation, validation + '''
  const bool use_dense_wide8 = dense_wide8 == 1 &&
      context.DeviceLimits().maxComputeWorkgroupSizeX >= 256 &&
      context.DeviceLimits().maxComputeInvocationsPerWorkgroup >= 256;''')
    source = replace_once(source, "if (dense_bitcast_route && dense_wide8 == 1)",
                          "if (dense_bitcast_route && use_dense_wide8)")
    return replace_once(source, "dense_wide8 == 1 ? 3 : 1 + dense_tile4",
                        "use_dense_wide8 ? 3 : 1 + dense_tile4")


def transform_compare(source):
    assert source.count(f'setenv("{FLAG}", value, 1)') == 1
    assert '" expected_bits="' in source and '" actual_bits="' in source
    return source


if __name__ == "__main__":
    build(output=Path("/workspace/qwen38-int4-dense-wide8/build-v1"), flag=FLAG,
          source_transform=transform, component_dir=COMPONENT, source_name="matmul_nbits.cc",
          comparator_flag="ORT_WEBGPU_DENSE_TILE4", comparator_transform=transform_compare,
          parent_dir=Path("/workspace/qwen38-int4-qmoe-prefill512/build-v3"),
          extra_include_dirs=(TEMPLATE,))
