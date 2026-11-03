from pathlib import Path

from build_gdn_value2 import main as build, transform as value2_transform, transform_compare as value2_compare
from build_gdn_value2_capture import main as build_capture
from build_qmoe_node_replay import replace_once


FLAG = "ORT_WEBGPU_GDN_VALUE8"


def transform(source, header):
    actual = value2_transform(source, header).replace("GatedDeltaNetValue2", "GatedDeltaNetValue8")
    actual = actual.replace("ORT_WEBGPU_GDN_VALUE2", FLAG).replace("value2", "value8")
    actual = replace_once(actual, "WGSL_TEMPLATE_PARAMETER(value_channels_per_workgroup, 2)",
                          "WGSL_TEMPLATE_PARAMETER(value_channels_per_workgroup, 8)")
    actual = replace_once(actual, "workgroup_size, value8 ? 2 : kValueChannelsPerWorkgroup)",
                          "workgroup_size, value8 ? 8 : kValueChannelsPerWorkgroup)")
    assert actual.count("(dv + 1) / 2") == 2
    return actual.replace("(dv + 1) / 2", "(dv + 7) / 8")


def transform_compare(source):
    return value2_compare(source).replace("ORT_WEBGPU_GDN_VALUE2", FLAG).replace(
        "ORT_TEST_TOGGLE_GDN_VALUE2", "ORT_TEST_TOGGLE_GDN_VALUE8").replace("value2", "value8")


if __name__ == "__main__":
    directory = Path("/workspace/qwen38-int4-gdn-value8/build-v1")
    build(parent_dir=Path("/workspace/qwen38-int4-dense-block32/build-v2"),
          output=directory, source_transform=transform, comparator_transform=transform_compare)
    build_capture(directory)
