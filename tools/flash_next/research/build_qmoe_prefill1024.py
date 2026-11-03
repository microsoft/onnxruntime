from pathlib import Path

from build_hc_bitcast_unpack import main as build
from build_qmoe_node_replay import replace_once


FLAG = "ORT_WEBGPU_QMOE_PREFILL1024"
ROWS = (16, 128, 256, 512, 1023, 1024, 1025)
FALLBACK_ROWS = (1023, 1025)


def transform(source):
    validation = ('  ORT_RETURN_IF_NOT(prefill512 == 0 || prefill512 == 1, '
                  '"ORT_WEBGPU_QMOE_PREFILL512 must be 0 or 1");')
    assert source.count(validation) == 2
    source = source.replace(validation, validation + f'''
  const int prefill1024 = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(prefill1024 == 0 || prefill1024 == 1, "{FLAG} must be 0 or 1");''')
    source = replace_once(source, "  key.push_back(prefill512);",
                          "  key.push_back(prefill512);\n  key.push_back(prefill1024);")
    source = replace_once(source, "  const int64_t max_indirect_tokens =",
        '''  const bool use_prefill1024 =
      prefill1024 == 1 && use_extended_int4_prefill && moe_params.num_rows == 1024 &&
      ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_QMOE_INT4_COOPERATIVE", 0) == 0 &&
      device_limits.maxComputeWorkgroupsPerDimension >= 51200;
  const int64_t max_indirect_tokens =''')
    source = replace_once(source, "      use_prefill512 ? 512 :",
                          "      use_prefill1024 ? 1024 : use_prefill512 ? 512 :")
    return replace_once(source, "      use_prefill512 ? 5120 :",
                        "      use_prefill1024 ? 10240 : use_prefill512 ? 5120 :")


def transform_compare(source):
    source = replace_once(source, "prefill512-v2-native-", "prefill1024-v1-native-")
    source = replace_once(source, "{16, 128, 256, 511, 512, 513}",
                          "{16, 128, 256, 512, 1023, 1024, 1025}")
    assert source.count("rows > 256 && rows != 512") == 2
    source = source.replace("rows > 256 && rows != 512", "rows > 512 && rows != 1024")
    return replace_once(source, "last_router.data(), 513", "last_router.data(), 1025")


if __name__ == "__main__":
    build(output=Path("/workspace/qwen38-int4-qmoe-prefill1024/build-v1"), flag=FLAG,
          source_transform=transform,
          component_dir=Path("/workspace/qwen38-int4-qmoe-prefill512/build-v2"),
          source_name="qmoe.cc", comparator_flag="ORT_WEBGPU_QMOE_PREFILL512",
          comparator_transform=transform_compare,
          parent_dir=Path("/workspace/qwen38-int4-qmoe-prefill512/build-v3"),
          extra_include_dirs=(Path("/workspace/qwen38-int4-qmoe-prefill256/build-v1/overlay/contrib_ops/webgpu/moe"),))
