// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/math/gemm.h"

#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace webgpu {

#define WEBGPU_GEMM_VERSIONED_KERNEL(start, end)              \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(                          \
      Gemm,                                                   \
      kOnnxDomain,                                            \
      start,                                                  \
      end,                                                    \
      kWebGpuExecutionProvider,                               \
      (*KernelDefBuilder::Create())                           \
          .TypeConstraint("T", WebGpuSupportedNumberTypes()), \
      Gemm);

#define WEBGPU_GEMM_KERNEL(version)                           \
  ONNX_OPERATOR_KERNEL_EX(                                    \
      Gemm,                                                   \
      kOnnxDomain,                                            \
      version,                                                \
      kWebGpuExecutionProvider,                               \
      (*KernelDefBuilder::Create())                           \
          .TypeConstraint("T", WebGpuSupportedNumberTypes()), \
      Gemm);

WEBGPU_GEMM_VERSIONED_KERNEL(7, 8)
WEBGPU_GEMM_VERSIONED_KERNEL(9, 10)
WEBGPU_GEMM_VERSIONED_KERNEL(11, 12)
WEBGPU_GEMM_KERNEL(13)

Status GemmProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const ShaderVariableHelper& A = shader.AddInput("A", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);
  const ShaderVariableHelper& B = shader.AddInput("B", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);

  const ShaderVariableHelper& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);

  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")
                            << "  let m = global_idx / uniforms.N;\n"
                            << "  let n = global_idx % uniforms.N;\n"
                            << "  var value = A_value_t(0);\n"
                            << "\n"
                            << "  for (var k = 0u; k < uniforms.K; k = k + 1u) {\n";

  if (transA_ && transB_) {
    shader.MainFunctionBody() << "    value = value + " << A.GetByOffset("k * uniforms.M + m")
                              << " * " << B.GetByOffset("n * uniforms.K + k") << ";\n";
  } else if (transA_ && !transB_) {
    shader.MainFunctionBody() << "    value = value + " << A.GetByOffset("k * uniforms.M + m")
                              << " * " << B.GetByOffset("k * uniforms.N + n") << ";\n";
  } else if (!transA_ && transB_) {
    shader.MainFunctionBody() << "    value = value + " << A.GetByOffset("m * uniforms.K + k")
                              << " * " << B.GetByOffset("n * uniforms.K + k") << ";\n";
  } else {
    shader.MainFunctionBody() << "    value = value + " << A.GetByOffset("m * uniforms.K + k")
                              << " * " << B.GetByOffset("k * uniforms.N + n") << ";\n";
  }
  shader.MainFunctionBody() << "  }\n"
                            << "\n";
  // Calculate Alpha
  if (alpha_) {
    shader.MainFunctionBody() << "  value = value * A_value_t(uniforms.alpha);\n";
  }

  // Calculate Bias
  if (need_handle_bias_) {
    const ShaderVariableHelper& C = shader.AddInput("C", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);
    shader.MainFunctionBody() << "  value = value + A_value_t(uniforms.beta) * "
                              << C.GetByOffset(C.BroadcastedIndicesToOffset("vec2(m, n)", output)) << ";\n";
  }

  shader.MainFunctionBody() << output.SetByOffset("global_idx", "value") << "\n";

  return Status::OK();
}

Status Gemm::ComputeInternal(ComputeContext& context) const {
  const auto* A = context.Input<Tensor>(0);
  const auto* B = context.Input<Tensor>(1);
  const auto* C = context.Input<Tensor>(2);

  if (A == nullptr || B == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Gemm requires input tensors A and B.");
  }

  const auto& A_shape = A->Shape();
  const auto& B_shape = B->Shape();

  if (A_shape.NumDimensions() != 2 || B_shape.NumDimensions() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input tensors A and B must be 2 dimensional.");
  }

  int64_t M = transA_ ? A_shape[1] : A_shape[0];
  int64_t K = transA_ ? A_shape[0] : A_shape[1];
  int64_t N = transB_ ? B_shape[0] : B_shape[1];

  if ((transA_ ? A_shape[0] : A_shape[1]) != (transB_ ? B_shape[1] : B_shape[0])) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Inner dimensions of A and B must match.");
  }

  std::vector<int64_t> output_dims{M, N};
  auto* Y = context.Output(0, output_dims);
  int64_t output_size = Y->Shape().Size();

  if (output_size == 0) {
    return Status::OK();
  }

  constexpr size_t TILE_SIZE = 16;
  int64_t num_tiles_m = (M + TILE_SIZE - 1) / TILE_SIZE;
  int64_t num_tiles_n = (N + TILE_SIZE - 1) / TILE_SIZE;
  int64_t dispatch_size = num_tiles_m * num_tiles_n;

  GemmProgram program{transA_, transB_, alpha_, beta_, C && beta_};
  program.AddInputs({{A, ProgramTensorMetadataDependency::Type},
                     {B, ProgramTensorMetadataDependency::Type}});

  if (C && beta_) {
    program.AddInput({C, ProgramTensorMetadataDependency::Rank});
  }

  program.AddOutputs({Y})
      .SetDispatchGroupSize(dispatch_size)
      .SetWorkgroupSize(TILE_SIZE, TILE_SIZE, 1)
      .AddUniformVariables({
          {static_cast<uint32_t>(output_size)},  // output_size
          {static_cast<uint32_t>(M)},            // M
          {static_cast<uint32_t>(N)},            // N
          {static_cast<uint32_t>(K)},            // K
          {alpha_},                              // alpha
          {beta_}                                // beta
      });

  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace onnxruntime
