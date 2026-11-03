from pathlib import Path

from build_hc_bitcast_unpack import main as build
from build_qmoe_node_replay import replace_once


FLAG = "ORT_WEBGPU_QMOE_PREFILL2048"
ROWS = (16, 128, 256, 512, 1024, 2047, 2048, 2049)
FALLBACK_ROWS = (2047, 2049)
PARENT = Path("/workspace/qwen38-int4-dense-block32/build-v2")
OUTPUT = Path("/workspace/qwen38-int4-qmoe-prefill2048/build-v1")


def transform(source):
    validation = ('  ORT_RETURN_IF_NOT(prefill1024 == 0 || prefill1024 == 1, '
                  '"ORT_WEBGPU_QMOE_PREFILL1024 must be 0 or 1");')
    assert source.count(validation) == 2
    source = source.replace(validation, validation + f'''
  const int prefill2048 = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(prefill2048 == 0 || prefill2048 == 1, "{FLAG} must be 0 or 1");''')
    source = replace_once(source, "  key.push_back(prefill1024);",
                          "  key.push_back(prefill1024);\n  key.push_back(prefill2048);")
    source = replace_once(source, "  const int64_t max_indirect_tokens =",
        '''  const bool use_prefill2048 =
      prefill2048 == 1 && use_extended_int4_prefill && moe_params.num_rows == 2048 &&
      ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_QMOE_INT4_COOPERATIVE", 0) == 0 &&
      device_limits.maxComputeWorkgroupsPerDimension >= 51200;
  const int64_t max_indirect_tokens =''')
    source = replace_once(source, "      use_prefill1024 ? 1024 :",
                          "      use_prefill2048 ? 2048 : use_prefill1024 ? 1024 :")
    source = replace_once(source, "      use_prefill1024 ? 10240 :",
                          "      use_prefill2048 ? 20480 : use_prefill1024 ? 10240 :")
    return replace_once(source,
        ".SetDispatchGroupSize(((routed_rows * static_cast<uint32_t>(moe_params.inter_size)) + 127) / 128)",
        ".SetDispatchGroupSize(\n"
        "            use_prefill2048 ? 51200u :\n"
        "                ((routed_rows * static_cast<uint32_t>(moe_params.inter_size)) + 127) / 128,\n"
        "            use_prefill2048 ? 2u : 1u)")


def transform_compare(source):
    source = replace_once(source, "prefill1024-v1-native-", "prefill2048-v2-native-")
    source = replace_once(source, "{16, 128, 256, 512, 1023, 1024, 1025}",
                          "{16, 128, 256, 512, 1024, 2047, 2048, 2049}")
    assert source.count("rows > 512 && rows != 1024") == 2
    source = source.replace("rows > 512 && rows != 1024", "rows == 2047 || rows == 2049")
    return replace_once(source, "last_router.data(), 1025", "last_router.data(), 2049")


if __name__ == "__main__":
    build(output=OUTPUT, flag=FLAG, source_transform=transform,
          component_dir=Path("/workspace/qwen38-int4-qmoe-prefill1024/build-v1"),
          source_name="qmoe.cc", comparator_flag="ORT_WEBGPU_QMOE_PREFILL1024",
          comparator_transform=transform_compare, parent_dir=PARENT,
          extra_include_dirs=(Path("/workspace/qwen38-int4-qmoe-prefill256/build-v1/overlay/contrib_ops/webgpu/moe"),))
