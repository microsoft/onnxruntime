from pathlib import Path

from build_hc_bitcast_unpack import main as build
from build_qmoe_node_replay import replace_once
from build_qmoe_prefill_rows4 import FLAG as GROUPED_FLAG


ROOT = Path(__file__).parent
FLAG = "ORT_WEBGPU_QMOE_PREFILL_ROWS8"
PROGRAM = "MatMulNBitsGroupedRows8F16"
ROW4 = Path("/workspace/qwen38-int4-qmoe-prefill-rows4/build-v2")
GROUPED = Path("/workspace/qwen38-int4-qmoe-grouped-prefill/build-v1")
OVERLAY = Path("/workspace/qwen38-int4-qmoe-prefill256/build-v1/overlay/contrib_ops/webgpu/moe")


def transform_program(source):
    replacements = (
        ("GroupedPrefillRows4Program", "GroupedPrefillRows8Program"),
        ("MatMulNBitsGroupedRows4F16", PROGRAM),
        ("tile_a: array<vec4<f16>, 1024>", "tile_a: array<vec4<f16>, 2048>"),
        ("partials: array<array<array<f16, 32>, 8>, 4>", "partials: array<array<array<f16, 32>, 8>, 8>"),
        ("row_experts: array<u32, 4>", "row_experts: array<u32, 8>"),
        ("uniforms.num_N_tile) * 4u", "uniforms.num_N_tile) * 8u"),
        ("local_idx < 4u", "local_idx < 8u"),
        ("i < 1024u", "i < 2048u"),
        ("row < 4u", "row < 8u"),
    )
    for before, after in replacements:
        assert before in source, before
        source = source.replace(before, after)
    return source


def transform(source):
    program = transform_program((ROOT / "qmoe_prefill_rows4.inc").read_text())
    source = replace_once(source, "class GroupedPrefillRows4Program final",
                          program + "\nclass GroupedPrefillRows4Program final")
    start = source.index("Status ApplyMatMulNBits(")
    branch_start = source.index("  if (grouped_prefill_rows == 1", start)
    branch_end = source.index("\n  if (nbits == 3) {", branch_start)
    branch = source[branch_start:branch_end]
    branch = replace_once(branch, "  if (grouped_prefill_rows == 1",
                          """  if (prefill_rows8 == 1 && grouped_prefill_rows == 1 &&
      context.DeviceLimits().maxComputeWorkgroupSizeX >= 128 &&
      context.DeviceLimits().maxComputeInvocationsPerWorkgroup >= 128 &&
      context.DeviceLimits().maxComputeWorkgroupStorageSize >= 20512""")
    branch = replace_once(branch, "GroupedPrefillRows4Program", "GroupedPrefillRows8Program")
    branch = replace_once(branch, "N / 8, M / 4", "N / 8, M / 8")
    validation = f'''
  const int prefill_rows8 = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(prefill_rows8 == 0 || prefill_rows8 == 1, "{FLAG} must be 0 or 1");
'''
    return source[:branch_start] + validation + branch + source[branch_start:]


def transform_replay(source):
    validation = ('  ORT_RETURN_IF_NOT(grouped_prefill == 0 || grouped_prefill == 1, '
                  f'"{GROUPED_FLAG} must be 0 or 1");')
    assert source.count(validation) == 2
    source = source.replace(validation, validation + f'''
  const int prefill_rows8 = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(prefill_rows8 == 0 || prefill_rows8 == 1, "{FLAG} must be 0 or 1");''')
    return replace_once(source, "  key.push_back(grouped_prefill);",
                        "  key.push_back(grouped_prefill);\n  key.push_back(prefill_rows8);")


if __name__ == "__main__":
    build(output=Path("/workspace/qwen38-int4-qmoe-prefill-rows8/build-v1"), flag=FLAG,
          source_transform=transform, component_dir=ROW4, source_name="matmul_nbits.cc",
          comparator_flag=GROUPED_FLAG, comparator_dir=GROUPED,
          comparator_transform=lambda source: replace_once(
              source, "grouped-prefill-v1-native-", "grouped-prefill-rows8-v1-native-"),
          parent_dir=ROW4, additional_components=((GROUPED, "qmoe.cc", transform_replay, (OVERLAY,)),))
