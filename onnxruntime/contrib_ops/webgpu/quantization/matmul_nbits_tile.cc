// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/quantization/matmul_nbits_tile.h"

#include <string_view>

#include "core/providers/webgpu/math/subgroup_matrix_config.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_utils.h"

namespace onnxruntime::contrib::webgpu {
namespace {

using namespace onnxruntime::webgpu;

class MatMulNBitsTileProgram final : public Program<MatMulNBitsTileProgram> {
 public:
  explicit MatMulNBitsTileProgram(bool matrix)
      : Program{matrix ? "MatMulNBitsMatrixTile" : "MatMulNBitsRegisterTile"}, matrix_{matrix} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& a = shader.AddInput("a");
    const auto& b = shader.AddInput("b");
    const auto& scales = shader.AddInput("scales");
    const auto& output = shader.AddOutput("output");
    if (matrix_) {
      return WGSL_TEMPLATE_APPLY(shader, "quantization/matmul_nbits_matrix_tile.wgsl.template",
                                 WGSL_TEMPLATE_VARIABLE(a, a), WGSL_TEMPLATE_VARIABLE(b, b),
                                 WGSL_TEMPLATE_VARIABLE(output, output), WGSL_TEMPLATE_VARIABLE(scales, scales));
    }
    return WGSL_TEMPLATE_APPLY(shader, "quantization/matmul_nbits_register_tile.wgsl.template",
                               WGSL_TEMPLATE_VARIABLE(a, a), WGSL_TEMPLATE_VARIABLE(b, b),
                               WGSL_TEMPLATE_VARIABLE(output, output), WGSL_TEMPLATE_VARIABLE(scales, scales));
  }

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"M", ProgramUniformVariableDataType::Uint32}, {"N", ProgramUniformVariableDataType::Uint32},
      {"K", ProgramUniformVariableDataType::Uint32}, {"num_N_tile", ProgramUniformVariableDataType::Uint32});

 private:
  bool matrix_;
};

}  // namespace

Status TryApplyMatMulNBitsTile(const Tensor* a, const Tensor* b, const Tensor* scales, Tensor* y,
                               uint32_t m, uint32_t n, uint32_t k,
                               onnxruntime::webgpu::ComputeContext& context, bool& handled) {
  handled = false;
  const auto& adapter = context.AdapterInfo();
  if (adapter.vendor != std::string_view{"apple"} || adapter.subgroupMinSize != 32 ||
      adapter.subgroupMaxSize != 32 || !context.HasFeature(wgpu::FeatureName::Subgroups)) {
    return Status::OK();
  }
  constexpr auto f32 = wgpu::SubgroupMatrixComponentType::F32;
  const bool matrix_available = SelectSubgroupMatrixConfig(context, {{f32, f32, 8, 8, 8, 32, false}}).has_value();
  const auto tile = SelectMatMulNBitsTile(m, n, k, matrix_available, context.EnableMatmulFp32Accumulation());
  if (tile == MatMulNBitsTile::None) {
    return Status::OK();
  }
  const bool matrix = tile == MatMulNBitsTile::Matrix;
  MatMulNBitsTileProgram program{matrix};
  program.SetWorkgroupSize(matrix ? 128 : 64);
  program.SetDispatchGroupSize(CeilDiv(n, 64u), CeilDiv(m, 64u));
  program.AddInput({a, ProgramTensorMetadataDependency::TypeAndRank, 4});
  program.AddInput({b, ProgramTensorMetadataDependency::TypeAndRank, 16});
  program.AddInput({scales, ProgramTensorMetadataDependency::TypeAndRank});
  program.AddOutput({y, ProgramTensorMetadataDependency::TypeAndRank, 1});
  program.AddUniformVariables({{m}, {n}, {k}, {CeilDiv(n, 64u)}});
  ORT_RETURN_IF_ERROR(context.RunProgram(program));
  handled = true;
  return Status::OK();
}

}  // namespace onnxruntime::contrib::webgpu
