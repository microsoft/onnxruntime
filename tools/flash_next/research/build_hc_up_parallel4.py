from pathlib import Path

from build_hc_bitcast_unpack import main as build
from build_qmoe_node_replay import replace_once


FLAG = "ORT_WEBGPU_HC_UP_PARALLEL4"


def packed_program(source):
    start = source.index("class HCUpPreMixPacked2Program")
    stop = source.index("class HCUpPreMixF16", start)
    program = source[start:stop].replace("HCUpPreMixPacked2Program", "HCUpPreMixPacked4Program")
    for old, new in (('Program{"HCUpPreMixPacked2F16"}', 'Program{"HCUpPreMixPacked4F16"}'),
                     ("partials: array<f16, 256>", "partials: array<f16, 512>"),
                     ("gates: array<f16, 8>", "gates: array<f16, 16>"),
                     ("workgroup_idx * 2u + column", "workgroup_idx * 4u + column")):
        program = replace_once(program, old, new)
    return program


def transform(source):
    source = replace_once(source, "class HCUpPreMixF16",
                          packed_program(source) + "class HCUpPreMixF16")
    source = replace_once(source,
        "  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override {",
        f'''  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override {{
    const int parallel4_mode = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
    ORT_RETURN_IF_NOT(parallel4_mode == 0 || parallel4_mode == 1, "{FLAG} must be 0 or 1");
    const bool parallel4 = parallel4_mode == 1 &&
        context.DeviceLimits().maxComputeWorkgroupSizeX >= 256 &&
        context.DeviceLimits().maxComputeInvocationsPerWorkgroup >= 256 &&
        context.DeviceLimits().maxComputeWorkgroupStorageSize >= 1056;''')
    source = replace_once(source,
        "program.SetWorkgroupSize(128).SetDispatchGroupSize(parallel2 == 1 ? 1280 : 2560);",
        "program.SetWorkgroupSize(parallel4 ? 256 : 128)"
        ".SetDispatchGroupSize(parallel4 ? 640 : parallel2 == 1 ? 1280 : 2560);")
    source = replace_once(source, "        if (parallel2 == 1) {",
        '''        if (parallel4) {
          HCUpPreMixPacked4Program program;
          return run(program);
        }
        if (parallel2 == 1) {''')
    return replace_once(source, "}, 0, parallel2);", "}, 0, parallel4 ? 2 : parallel2);")


def transform_compare(source):
    assert source.count(f'setenv("{FLAG}", value, 1)') == 1
    assert "expected_bits=" in source and "actual_bits=" in source
    return source


if __name__ == "__main__":
    build(output=Path("/workspace/qwen38-int4-hc-up-parallel4/build-v1"), flag=FLAG,
          source_transform=transform,
          component_dir=Path("/workspace/qwen38-int4-hc-up-parallel2/build-v1"),
          source_name="hc_up_premix.cc", comparator_flag="ORT_WEBGPU_HC_UP_PARALLEL2",
          comparator_transform=transform_compare,
          parent_dir=Path("/workspace/qwen38-int4-qmoe-prefill512/build-v3"))
