from pathlib import Path

from build_dense_paired_matrix import SOURCE
from build_qmoe_node_replay import replace_once
from build_scoped_webgpu_source import build


FLAG = "ORT_WEBGPU_NATIVE_MATRIX_LIMIT_PROBE"
OUTPUT = Path("/workspace/qwen38-int4-native-matrix-limit-probe/build-v1")


def transform(source):
    signature = "Status MatMulNBits::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {"
    return replace_once(source, signature, signature + f'''
  static const int limit_probe = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(limit_probe == 0 || limit_probe == 1, "{FLAG} must be 0 or 1");
  if (limit_probe == 1) {{
    const auto& limits = context.DeviceLimits();
    using Type = wgpu::SubgroupMatrixComponentType;
    const auto configuration = SelectSubgroupMatrixConfig(context, {{
        {{Type::F16, Type::F32, 16, 16, 16, 32, false}}}});
    LOGS_DEFAULT(WARNING) << "NATIVE_MATRIX_DEVICE_LIMITS storage=" << limits.maxComputeWorkgroupStorageSize
                         << " threads_x=" << limits.maxComputeWorkgroupSizeX
                         << " invocations=" << limits.maxComputeInvocationsPerWorkgroup
                         << " pair_configuration=" << configuration.has_value();
  }}
''')


if __name__ == "__main__":
    build(OUTPUT, SOURCE, transform, SOURCE.parent / "native_compare.cpp", FLAG,
          dict(diagnostic_only=True, model_eligible=False,
               shader_arithmetic_unchanged=True, read_only_device_limits=True,
               hardware_clock_or_power_changes=False))
