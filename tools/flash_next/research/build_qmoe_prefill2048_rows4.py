import json
from pathlib import Path

from build_hc_bitcast_unpack import main as build
from build_qmoe_node_replay import replace_once


FLAG = "ORT_WEBGPU_QMOE_PREFILL2048_ROWS4"
PROGRAM = "MatMulNBitsGroupedRows4_2048F16"
PARENT = Path("/workspace/qwen38-int4-qmoe-prefill-rows4/build-v2")
GROUPED = Path("/workspace/qwen38-int4-qmoe-grouped-prefill/build-v1")
OUTPUT = Path("/workspace/qwen38-int4-qmoe-prefill2048-rows4/build-v1")
ROWS = (16, 128, 256, 512, 1023, 1024, 1025, 2047, 2048, 2049)
FALLBACK_ROWS = (1023, 1025, 2047, 2049)


def transform_qmoe(source):
    validation = ('  ORT_RETURN_IF_NOT(prefill1024 == 0 || prefill1024 == 1, '
                  '"ORT_WEBGPU_QMOE_PREFILL1024 must be 0 or 1");')
    assert source.count(validation) == 2
    source = source.replace(validation, validation + f'''
  const int prefill2048_rows4 = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(prefill2048_rows4 == 0 || prefill2048_rows4 == 1, "{FLAG} must be 0 or 1");''')
    source = replace_once(source, "  key.push_back(grouped_prefill);",
                          "  key.push_back(grouped_prefill);\n  key.push_back(prefill2048_rows4);")
    source = replace_once(source, "  const int64_t max_indirect_tokens =", '''
  const bool use_prefill2048_rows4 =
      prefill2048_rows4 == 1 && grouped_prefill == 1 && use_extended_int4_prefill &&
      moe_params.num_rows == 2048 &&
      ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_QMOE_INT4_COOPERATIVE", 0) == 0 &&
      device_limits.maxComputeWorkgroupsPerDimension >= 20480;
  const int64_t max_indirect_tokens =''')
    source = replace_once(source, "      use_prefill1024 ? 1024 :",
                          "      use_prefill2048_rows4 ? 2048 : use_prefill1024 ? 1024 :")
    source = replace_once(source, "      use_prefill1024 ? 10240 :",
                          "      use_prefill2048_rows4 ? 20480 : use_prefill1024 ? 10240 :")
    return replace_once(source, "(use_prefill512 || use_prefill1024) && use_small_batch",
                        "(use_prefill512 || use_prefill1024 || use_prefill2048_rows4) && use_small_batch")


def transform_matmul(source):
    source = replace_once(source, "explicit GroupedPrefillRows4Program(bool bias)",
                          "explicit GroupedPrefillRows4Program(bool bias, bool large = false)")
    source = replace_once(source, ': Program{"MatMulNBitsGroupedRows4F16"}, bias_{bias} {}',
                          f': Program{{large ? "{PROGRAM}" : "MatMulNBitsGroupedRows4F16"}}, bias_{{bias}} {{}}')
    anchor = '''  if (grouped_prefill_rows == 1 && batch_count == 1 && dispatch_M == M &&
      (M == 5120 || M == 10240)'''
    replacement = f'''  const int prefill2048_rows4 = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(prefill2048_rows4 == 0 || prefill2048_rows4 == 1, "{FLAG} must be 0 or 1");
  if (grouped_prefill_rows == 1 && batch_count == 1 && dispatch_M == M &&
      (M == 5120 || M == 10240 || (M == 20480 && prefill2048_rows4 == 1))'''
    source = replace_once(source, anchor, replacement)
    return replace_once(source, "GroupedPrefillRows4Program program{has_bias};",
                        "GroupedPrefillRows4Program program{has_bias, M == 20480};")


def transform_compare(source):
    source = replace_once(source, "grouped-prefill-v1-native-", "prefill2048-rows4-v1-native-")
    source = replace_once(source, "{16, 128, 256, 512, 1023, 1024, 1025}",
                          "{16, 128, 256, 512, 1023, 1024, 1025, 2047, 2048, 2049}")
    assert source.count("rows > 512 && rows != 1024") == 2
    source = source.replace("rows > 512 && rows != 1024", "rows > 512 && rows != 1024 && rows != 2048")
    return replace_once(source, "last_router.data(), 1025", "last_router.data(), 2049")


if __name__ == "__main__":
    build(output=OUTPUT, flag=FLAG, source_transform=transform_qmoe,
          component_dir=GROUPED, source_name="qmoe.cc",
          comparator_flag="ORT_WEBGPU_QMOE_GROUPED_PREFILL",
          comparator_transform=transform_compare, parent_dir=PARENT,
          additional_components=((PARENT, "matmul_nbits.cc", transform_matmul, ()),))
    path = OUTPUT / "build-manifest.json"
    manifest = json.loads(path.read_text())
    manifest.update(canonical_tokens=2048, routed_rows=20480, row_tile=4,
                    existing_rows4_shader_arithmetic_unchanged=True,
                    existing_short_and_decode_routes_preserved=True,
                    qmoe_replay_selector_keyed=True,
                    activation_logical_groups=102400,
                    existing_flat_dispatch_normalization_used=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
