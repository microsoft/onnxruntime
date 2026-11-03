import json
from pathlib import Path

from build_hc_bitcast_unpack import main as build
from build_qmoe_node_replay import replace_once, sha


FLAG = "ORT_WEBGPU_QMOE_PREFILL512"
ROWS = (16, 128, 256, 511, 512, 513)
FALLBACK_ROWS = (511, 513)


def transform(source):
    validation = ('  ORT_RETURN_IF_NOT(prefill256 == 0 || prefill256 == 1, '
                  '"ORT_WEBGPU_QMOE_PREFILL256 must be 0 or 1");')
    assert source.count(validation) == 2
    source = source.replace(validation, validation + f'''
  const int prefill512 = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(prefill512 == 0 || prefill512 == 1, "{FLAG} must be 0 or 1");''')
    source = replace_once(source, "  key.push_back(prefill256);",
                          "  key.push_back(prefill256);\n  key.push_back(prefill512);")
    source = replace_once(source, "  const int64_t max_indirect_tokens =",
        '''  const bool use_prefill512 =
      prefill512 == 1 && use_extended_int4_prefill && moe_params.num_rows == 512 &&
      ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_QMOE_INT4_COOPERATIVE", 0) == 0 &&
      device_limits.maxComputeWorkgroupsPerDimension >= 25600;
  const int64_t max_indirect_tokens =''')
    source = replace_once(source, "      use_prefill256 ? 256 :",
                          "      use_prefill512 ? 512 : use_prefill256 ? 256 :")
    return replace_once(source, "      use_prefill256 ? 2560 :",
                        "      use_prefill512 ? 5120 : use_prefill256 ? 2560 :")


def transform_compare(source):
    source = replace_once(source, "prefill256-v1-native-", "prefill512-v2-native-")
    source = replace_once(source, "{16, 128, 129, 255, 256, 257}", "{16, 128, 256, 511, 512, 513}")
    assert source.count("rows > 128 && rows != 256") == 2
    source = source.replace("rows > 128 && rows != 256", "rows > 256 && rows != 512")
    return replace_once(source, "last_router.data(), 257", "last_router.data(), 513")


if __name__ == "__main__":
    component = Path("/workspace/qwen38-int4-qmoe-prefill256/build-v1")
    manifest = json.loads((component / "build-manifest.json").read_text())
    for relative, digest in manifest["source_sha256"].items():
        assert sha(component / relative) == digest
    build(output=Path("/workspace/qwen38-int4-qmoe-prefill512/build-v2"), flag=FLAG,
          source_transform=transform,
          component_dir=component,
          source_name="overlay/contrib_ops/webgpu/moe/qmoe.cc",
          comparator_flag="ORT_WEBGPU_QMOE_PREFILL256", comparator_transform=transform_compare,
          parent_dir=Path("/workspace/qwen38-int4-dense-tile4/build-v1"),
          extra_include_dirs=(component / "overlay/contrib_ops/webgpu/moe",))
