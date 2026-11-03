from pathlib import Path

from build_hc_bitcast_unpack import main as build
from build_qmoe_node_replay import replace_once


FLAG = "ORT_WEBGPU_QMOE_PREFILL1280"
ROWS = (16, 128, 256, 512, 768, 1024, 1279, 1280, 1281)
FALLBACK_ROWS = (1279, 1281)
COMPONENT = Path("/workspace/qwen38-int4-qmoe-prefill1024/build-v1")
PARENT = Path("/workspace/qwen38-int4-dsa-query-cache/build-v2")
OUTPUT = Path("/workspace/qwen38-int4-qmoe-prefill1280/build-v1")


def transform(source):
    validation = ('  ORT_RETURN_IF_NOT(prefill1024 == 0 || prefill1024 == 1, '
                  '"ORT_WEBGPU_QMOE_PREFILL1024 must be 0 or 1");')
    assert source.count(validation) == 2
    source = source.replace(validation, validation + f'''
  const int prefill1280 = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(prefill1280 == 0 || prefill1280 == 1, "{FLAG} must be 0 or 1");''')
    source = replace_once(source, "  key.push_back(prefill1024);",
                          "  key.push_back(prefill1024);\n  key.push_back(prefill1280);")
    source = replace_once(source, "  const int64_t max_indirect_tokens =",
        '''  const bool use_prefill1280 =
      prefill1280 == 1 && use_extended_int4_prefill &&
      (moe_params.num_rows == 768 || moe_params.num_rows == 1280) &&
      ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_QMOE_INT4_COOPERATIVE", 0) == 0 &&
      device_limits.maxComputeWorkgroupsPerDimension >=
          (moe_params.num_rows == 1280 ? 64000u : 38400u);
  const int64_t max_indirect_tokens =''')
    source = replace_once(source, "      use_prefill1024 ? 1024 :",
                          "      use_prefill1280 ? 1280 : use_prefill1024 ? 1024 :")
    return replace_once(source, "      use_prefill1024 ? 10240 :",
                        "      use_prefill1280 ? 12800 : use_prefill1024 ? 10240 :")


def transform_compare(source):
    source = replace_once(source, "prefill1024-v1-native-", "prefill1280-v2-native-")
    source = replace_once(source, "{16, 128, 256, 512, 1023, 1024, 1025}",
                          "{16, 128, 256, 512, 768, 1024, 1279, 1280, 1281}")
    assert source.count("rows > 512 && rows != 1024") == 2
    source = source.replace("rows > 512 && rows != 1024", "rows == 1279 || rows == 1281")
    return replace_once(source, "last_router.data(), 1025", "last_router.data(), 1281")


if __name__ == "__main__":
    build(output=OUTPUT, flag=FLAG, source_transform=transform,
          component_dir=COMPONENT, source_name="qmoe.cc",
          comparator_flag="ORT_WEBGPU_QMOE_PREFILL1024", comparator_transform=transform_compare,
          parent_dir=PARENT,
          extra_include_dirs=(Path("/workspace/qwen38-int4-qmoe-prefill256/build-v1/overlay/contrib_ops/webgpu/moe"),))
