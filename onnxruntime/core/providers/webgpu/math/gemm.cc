// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/math/gemm.h"
#include "core/providers/webgpu/math/gemm_packed.h"

#include <vector>

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

Status GemmNaiveProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const ShaderVariableHelper& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);

  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")
                            << "  let m = global_idx / uniforms.N;\n"
                            << "  let n = global_idx % uniforms.N;\n"
                            << "  var value = output_value_t(0);\n"
                            << "\n";

  // When A or B is empty, we don't bind A and B. Because WebGPU doesn't support binding a zero-sized buffer.
  if (need_handle_matmul_) {
    const ShaderVariableHelper& A = shader.AddInput("A", ShaderUsage::UseUniform);
    const ShaderVariableHelper& B = shader.AddInput("B", ShaderUsage::UseUniform);

    shader.MainFunctionBody() << "  for (var k = 0u; k < uniforms.K; k = k + 1u) {\n";

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
  }

  // Calculate Alpha
  shader.MainFunctionBody() << "  value = value * output_value_t(uniforms.alpha);\n";

  // Calculate Bias
  if (need_handle_bias_) {
    const ShaderVariableHelper& C = shader.AddInput("C", ShaderUsage::UseUniform);
    shader.MainFunctionBody() << "  value = value + output_value_t(uniforms.beta) * "
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

  if ((transA_ ? A_shape[0] : A_shape[1]) != (transB_ ? B_shape[1] : B_shape[0])) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Inner dimensions of A and B must match.");
  }

  int64_t M = transA_ ? A_shape[1] : A_shape[0];
  int64_t K = transA_ ? A_shape[0] : A_shape[1];
  int64_t N = transB_ ? B_shape[0] : B_shape[1];

  std::vector<int64_t> output_dims{M, N};
  auto* Y = context.Output(0, output_dims);
  int64_t output_size = Y->Shape().Size();

  if (output_size == 0) {
    return Status::OK();
  }

  // WebGPU doesn't support binding a zero-sized buffer, so we need to check if A or B is empty.
  bool need_handle_matmul = A_shape.Size() > 0 && B_shape.Size() > 0;
  bool need_handle_bias = C && beta_;

  if (M <= 8 && N <= 8 && K <= 8) {
    // Use naive implementation for small matrices
    GemmNaiveProgram program{transA_, transB_, alpha_, need_handle_bias, need_handle_matmul};
    if (need_handle_matmul) {
      program.AddInputs({{A, ProgramTensorMetadataDependency::Type},
                         {B, ProgramTensorMetadataDependency::Type}});
    }

    if (need_handle_bias) {
      program.AddInput({C, ProgramTensorMetadataDependency::Rank});
    }

    program.CacheHint(transA_, transB_)
        .AddOutputs({{Y, ProgramTensorMetadataDependency::Type}})
        .SetDispatchGroupSize((output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
        .SetWorkgroupSize(WORKGROUP_SIZE)
        .AddUniformVariables({{static_cast<uint32_t>(output_size)},
                              {static_cast<uint32_t>(M)},
                              {static_cast<uint32_t>(N)},
                              {static_cast<uint32_t>(K)},
                              {alpha_},
                              {beta_}});
    return context.RunProgram(program);
  }

  return ApplyGemmPacked(A, B, C, transA_, transB_, alpha_, beta_, context);
}

}  // namespace webgpu
}  // namespace onnxruntime
