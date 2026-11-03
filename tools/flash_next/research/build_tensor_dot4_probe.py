import argparse
from pathlib import Path

from build_hc_bitcast_unpack import main as build
from build_qmoe_node_replay import replace_once


FLAG = "ORT_WEBGPU_TENSOR_DOT4_PROBE"
OUTPUT = Path("/workspace/qwen38-int4-tensor-dot4-probe/build-v1")


def transform(source, fp32_result=False):
    source = replace_once(source, '#include "core/providers/webgpu/shader_helper.h"',
                          '#include "core/providers/webgpu/shader_helper.h"\n'
                          '#include "core/providers/webgpu/math/subgroup_matrix_config.h"')
    signature = "Status MatMulNBits::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {"
    source = replace_once(source, signature, signature + f'''
  const int tensor_probe = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(tensor_probe == 0 || tensor_probe == 1, "{FLAG} must be 0 or 1");
''')
    source = replace_once(source, "  const bool dense_bitcast_route =", '''
  if (tensor_probe == 1) {
    ORT_RETURN_IF_NOT(K_ == 2560 && N_ == 2560 && bits_ == 4 && block_size_ == 32 &&
                          a->Shape().Size() == K_ && a->IsDataType<MLFloat16>() &&
                          scales->IsDataType<MLFloat16>() && !has_zero_points && bias == nullptr &&
                          !context.EnableMatmulFp32Accumulation() &&
                          context.AdapterInfo().vendor == std::string_view{"nvidia"},
                      "Tensor dot4 probe only supports the isolated FP16 K2560/N2560 fixture");
    constexpr auto f16 = wgpu::SubgroupMatrixComponentType::F16;
    auto config = onnxruntime::webgpu::SelectSubgroupMatrixConfig(
        context, {{f16, f16, 16, 16, 16, 32, true}});
    ORT_RETURN_IF_NOT(config.has_value(), "Tensor dot4 probe requires native FP16 16x16x16 matrices");
    return ApplySubgroupMatrixMatMulNBits(a, b, scales, zero_points, bias, 1, 2560, 2560,
                                         4, 80, *config, context, y, 0, nullptr);
  }
  const bool dense_bitcast_route =''')
    if fp32_result:
        source = replace_once(source, "{{f16, f16, 16, 16, 16, 32, true}}",
                              "{{f16, wgpu::SubgroupMatrixComponentType::F32, 16, 16, 16, 32, true}}")
    return source


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp32-result", action="store_true")
    args = parser.parse_args()
    build(output=OUTPUT.parent / "build-v2" if args.fp32_result else OUTPUT, flag=FLAG,
          source_transform=lambda source: transform(source, fp32_result=args.fp32_result),
          component_dir=Path("/workspace/qwen38-int4-dense-block32/build-v1"),
          source_name="matmul_nbits.cc", comparator_flag="ORT_WEBGPU_DENSE_BLOCK32",
          comparator_transform=lambda source: source,
          parent_dir=Path("/workspace/qwen38-int4-qsa-prefill-score16/build-v1"))
