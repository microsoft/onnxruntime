// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/math/gemm.h"
#include "core/providers/webgpu/math/gemm_vec4.h"

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

Status GemmProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const uint32_t TILE_SIZE = 16;

  // Add shared memory arrays
  shader.AdditionalImplementation() << "var<workgroup> tile_a: array<array<output_value_t, " << TILE_SIZE << ">, " << TILE_SIZE << ">;\n"
                                    << "var<workgroup> tile_b: array<array<output_value_t, " << TILE_SIZE << ">, " << TILE_SIZE << ">;\n\n";

  const ShaderVariableHelper& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);

  shader.MainFunctionBody() << "  var value = output_value_t(0);\n\n"
                            << "  let tile_col_start = (workgroup_idx % uniforms.num_tile_n) * " << TILE_SIZE << "u;\n"
                            << "  let tile_row_start = (workgroup_idx / uniforms.num_tile_n) * " << TILE_SIZE << "u;\n";

  // When A or B is empty, we don't bind A and B. Because WebGPU doesn't support binding a zero-sized buffer.
  if (need_handle_matmul_) {
    const ShaderVariableHelper& A = shader.AddInput("A", ShaderUsage::UseUniform);
    const ShaderVariableHelper& B = shader.AddInput("B", ShaderUsage::UseUniform);

    shader.MainFunctionBody()
        << "  let num_tiles = (uniforms.K - 1u) / " << TILE_SIZE << "u + 1u;\n"
        << "  var k_start = 0u;\n"
        << "  for (var t = 0u; t < num_tiles; t = t + 1u) {\n";

    // Fill workgroup shared memory
    if (transA_ && transB_) {
      shader.MainFunctionBody() << "    var col = tile_row_start + local_id.x;\n"
                                << "    var row = k_start + local_id.y;\n"
                                << "    if (col < uniforms.M && row < uniforms.K) {\n"
                                << "      tile_a[local_id.y][local_id.x] = " << A.GetByOffset("row * uniforms.M + col") << ";\n"
                                << "    } else {\n"
                                << "      tile_a[local_id.y][local_id.x] = output_value_t(0);\n"
                                << "    }\n\n"
                                << "    col = k_start + local_id.x;\n"
                                << "    row = tile_col_start + local_id.y;\n"
                                << "    if (col < uniforms.K && row < uniforms.N) {\n"
                                << "      tile_b[local_id.y][local_id.x] = " << B.GetByOffset("row * uniforms.K + col") << ";\n"
                                << "    } else {\n"
                                << "      tile_b[local_id.y][local_id.x] = output_value_t(0);\n"
                                << "    }\n";
    } else if (transA_ && !transB_) {
      shader.MainFunctionBody() << "    var col = tile_row_start + local_id.x;\n"
                                << "    var row = k_start + local_id.y;\n"
                                << "    if (col < uniforms.M && row < uniforms.K) {\n"
                                << "      tile_a[local_id.y][local_id.x] = " << A.GetByOffset("row * uniforms.M + col") << ";\n"
                                << "    } else {\n"
                                << "      tile_a[local_id.y][local_id.x] = output_value_t(0);\n"
                                << "    }\n\n"
                                << "    col = tile_col_start + local_id.x;\n"
                                << "    row = k_start + local_id.y;\n"
                                << "    if (col < uniforms.N && row < uniforms.K) {\n"
                                << "      tile_b[local_id.y][local_id.x] = " << B.GetByOffset("row * uniforms.N + col") << ";\n"
                                << "    } else {\n"
                                << "      tile_b[local_id.y][local_id.x] = output_value_t(0);\n"
                                << "    }\n";
    } else if (!transA_ && transB_) {
      shader.MainFunctionBody() << "    var col = k_start + local_id.x;\n"
                                << "    var row = tile_row_start + local_id.y;\n"
                                << "    if (col < uniforms.K && row < uniforms.M) {\n"
                                << "      tile_a[local_id.y][local_id.x] = " << A.GetByOffset("row * uniforms.K + col") << ";\n"
                                << "    } else {\n"
                                << "      tile_a[local_id.y][local_id.x] = output_value_t(0);\n"
                                << "    }\n\n"
                                << "    col = k_start + local_id.x;\n"
                                << "    row = tile_col_start + local_id.y;\n"
                                << "    if (col < uniforms.K && row < uniforms.N) {\n"
                                << "      tile_b[local_id.y][local_id.x] = " << B.GetByOffset("row * uniforms.K + col") << ";\n"
                                << "    } else {\n"
                                << "      tile_b[local_id.y][local_id.x] = output_value_t(0);\n"
                                << "    }\n";
    } else {
      shader.MainFunctionBody() << "    var col = k_start + local_id.x;\n"
                                << "    var row = tile_row_start + local_id.y;\n"
                                << "    if (col < uniforms.K && row < uniforms.M) {\n"
                                << "      tile_a[local_id.y][local_id.x] = " << A.GetByOffset("row * uniforms.K + col") << ";\n"
                                << "    } else {\n"
                                << "      tile_a[local_id.y][local_id.x] = output_value_t(0);\n"
                                << "    }\n\n"
                                << "    col = tile_col_start + local_id.x;\n"
                                << "    row = k_start + local_id.y;\n"
                                << "    if (col < uniforms.N && row < uniforms.K) {\n"
                                << "      tile_b[local_id.y][local_id.x] = " << B.GetByOffset("row * uniforms.N + col") << ";\n"
                                << "    } else {\n"
                                << "      tile_b[local_id.y][local_id.x] = output_value_t(0);\n"
                                << "    }\n";
    }

    shader.MainFunctionBody() << "    k_start = k_start + " << TILE_SIZE << "u;\n"
                              << "    workgroupBarrier();\n\n"
                              << "    for (var k = 0u; k < " << TILE_SIZE << "u; k = k + 1u) {\n";

    if (transA_ && transB_) {
      shader.MainFunctionBody() << "      value = value + tile_a[k][local_id.y] * tile_b[local_id.x][k];\n";
    } else if (transA_ && !transB_) {
      shader.MainFunctionBody() << "      value = value + tile_a[k][local_id.y] * tile_b[k][local_id.x];\n";
    } else if (!transA_ && transB_) {
      shader.MainFunctionBody() << "      value = value + tile_a[local_id.y][k] * tile_b[local_id.x][k];\n";
    } else {
      shader.MainFunctionBody() << "      value = value + tile_a[local_id.y][k] * tile_b[k][local_id.x];\n";
    }

    shader.MainFunctionBody() << "    }\n"
                              << "    workgroupBarrier();\n"
                              << "  }\n\n";
  }

  // Calculate Alpha
  if (alpha_) {
    shader.MainFunctionBody() << "  value = value * output_value_t(uniforms.alpha);\n";
  }

  shader.MainFunctionBody() << "  let m = tile_row_start + local_id.y;\n"
                            << "  let n = tile_col_start + local_id.x;\n";

  // Calculate Bias
  if (need_handle_bias_) {
    const ShaderVariableHelper& C = shader.AddInput("C", ShaderUsage::UseUniform);
    shader.MainFunctionBody() << "  value = value + output_value_t(uniforms.beta) * "
                              << C.GetByOffset(C.BroadcastedIndicesToOffset("vec2(m, n)", output)) << ";\n";
  }

  // Write output
  shader.MainFunctionBody() << "  if (m < uniforms.M && n < uniforms.N) {\n"
                            << "    " << output.SetByOffset("m * uniforms.N + n", "value") << "\n"
                            << "  }\n";

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

  uint32_t M = onnxruntime::narrow<uint32_t>(transA_ ? A_shape[1] : A_shape[0]);
  uint32_t K = onnxruntime::narrow<uint32_t>(transA_ ? A_shape[0] : A_shape[1]);
  uint32_t N = onnxruntime::narrow<uint32_t>(transB_ ? B_shape[0] : B_shape[1]);

  if ((transA_ ? A_shape[0] : A_shape[1]) != (transB_ ? B_shape[1] : B_shape[0])) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Inner dimensions of A and B must match.");
  }

  std::vector<int64_t> output_dims{M, N};
  auto* Y = context.Output(0, output_dims);
  int64_t output_size = Y->Shape().Size();

  if (output_size == 0) {
    return Status::OK();
  }

  // First try vec4 optimization if possible
  if (CanApplyGemmVec4(A, B)) {
    return ApplyGemmVec4(A, B, C, transA_, transB_, alpha_, beta_, context, Y);
  }

  // WebGPU doesn't support binding a zero-sized buffer, so we need to check if A or B is empty.
  bool need_handle_matmul = A_shape.Size() > 0 && B_shape.Size() > 0;
  bool need_handle_bias = C && beta_;

  GemmProgram program{transA_, transB_, alpha_, need_handle_bias, need_handle_matmul};

  if (need_handle_matmul) {
    program.AddInputs({{A, ProgramTensorMetadataDependency::Type},
                       {B, ProgramTensorMetadataDependency::Type}});
  }

  if (need_handle_bias) {
    program.AddInput({C, ProgramTensorMetadataDependency::Rank});
  }

  const uint32_t TILE_SIZE = 16;
  const uint32_t num_tile_n = (N + TILE_SIZE - 1) / TILE_SIZE;
  const uint32_t num_tile_m = (M + TILE_SIZE - 1) / TILE_SIZE;

  program.CacheHint(alpha_, transA_, transB_)
      .AddOutputs({{Y, ProgramTensorMetadataDependency::Type}})
      .SetDispatchGroupSize(num_tile_n * num_tile_m)
      .SetWorkgroupSize(TILE_SIZE, TILE_SIZE)
      .AddUniformVariables({{num_tile_n},
                            {M},
                            {N},
                            {K},
                            {alpha_},
                            {beta_}});

  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace onnxruntime
