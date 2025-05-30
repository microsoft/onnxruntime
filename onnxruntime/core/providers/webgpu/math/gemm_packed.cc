// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/math/gemm_packed.h"

#include "core/providers/webgpu/webgpu_utils.h"

#include "core/providers/webgpu/math/matmul_utils.h"
#include "core/providers/webgpu/math/gemm_utils.h"

namespace onnxruntime {
namespace webgpu {

Status GemmProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const ShaderVariableHelper& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);

  // Each thread compute 4*4 elements
  InlinedVector<int64_t> elements_per_thread = InlinedVector<int64_t>({4, 4, 1});

  const std::string data_type = "output_element_t";

  if (need_handle_matmul_) {
    const auto& a = shader.AddInput("a", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
    const auto& b = shader.AddInput("b", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias);

    MatMulReadFnSource(shader, a, b, nullptr, transA_, transB_, is_vec4_);
  }
  if (is_vec4_) {
    ORT_RETURN_IF_ERROR(MakeMatMulPackedVec4Source(shader, elements_per_thread, WorkgroupSizeX(), WorkgroupSizeY(), data_type, nullptr, transA_, transB_, alpha_, need_handle_matmul_, output_components_));
  } else {
    ORT_RETURN_IF_ERROR(MakeMatMulPackedSource(shader, elements_per_thread, WorkgroupSizeX(), WorkgroupSizeY(), data_type, nullptr, transA_, transB_, alpha_, need_handle_matmul_));
  }
  MatMulWriteFnSource(shader, output, need_handle_bias_, true, c_components_, output_components_, c_is_scalar_);

  return Status::OK();
}

Status ApplyGemmPacked(const Tensor* a,
                       const Tensor* b,
                       const Tensor* c,
                       bool transA,
                       bool transB,
                       float alpha,
                       float beta,
                       ComputeContext& context) {
  const auto& a_shape = a->Shape();
  const auto& b_shape = b->Shape();

  uint32_t M = onnxruntime::narrow<uint32_t>(transA ? a_shape[1] : a_shape[0]);
  uint32_t K = onnxruntime::narrow<uint32_t>(transA ? a_shape[0] : a_shape[1]);
  uint32_t N = onnxruntime::narrow<uint32_t>(transB ? b_shape[0] : b_shape[1]);

  std::vector<int64_t> output_dims{M, N};
  auto* y = context.Output(0, output_dims);
  int64_t output_size = y->Shape().Size();

  if (output_size == 0) {
    return Status::OK();
  }

  // WebGPU doesn't support binding a zero-sized buffer, so we need to check if A or B is empty.
  bool need_handle_matmul = a_shape.Size() > 0 && b_shape.Size() > 0;
  bool need_handle_bias = c && beta;

  const bool is_vec4 = a_shape[1] % 4 == 0 && b_shape[1] % 4 == 0;

  // Components for A, B
  int components = is_vec4 ? 4 : 1;
  // Components for Y
  int output_components = (is_vec4 && N % 4 == 0) ? 4 : 1;
  // Components for C.
  int c_components = 1;

  bool c_is_scalar = false;
  if (need_handle_bias) {
    const auto& c_shape = c->Shape();
    int64_t c_last_dim = c_shape[c_shape.NumDimensions() - 1];
    // `C` in GEMM might be broadcast to the output, and broadcasting requires the components to be consistent.
    // So we use vec4 for C when its last dimension is N, and the output is also a vec4.
    c_components = (c_last_dim == N && output_components == 4) ? 4 : 1;
    c_is_scalar = c_shape.Size() == 1;
  }

  GemmProgram program{transA, transB, alpha, need_handle_bias, need_handle_matmul, c_components, c_is_scalar, output_components, is_vec4};

  if (need_handle_matmul) {
    program.AddInputs({{a, ProgramTensorMetadataDependency::TypeAndRank, components},
                       {b, ProgramTensorMetadataDependency::TypeAndRank, components}});
  }

  if (need_handle_bias) {
    program.AddInput({c, ProgramTensorMetadataDependency::TypeAndRank, c_components});
  }

  const uint32_t TILE_SIZE = 32;
  const uint32_t num_tile_n = (N + TILE_SIZE - 1) / TILE_SIZE;
  const uint32_t num_tile_m = (M + TILE_SIZE - 1) / TILE_SIZE;

  program.CacheHint(alpha, transA, transB, c_is_scalar)
      .AddOutputs({{y, ProgramTensorMetadataDependency::TypeAndRank, output_components}})
      .SetDispatchGroupSize(num_tile_n, num_tile_m, 1)
      .SetWorkgroupSize(GemmProgram::MATMUL_PACKED_WORKGROUP_SIZE_X, GemmProgram::MATMUL_PACKED_WORKGROUP_SIZE_Y, GemmProgram::MATMUL_PACKED_WORKGROUP_SIZE_Z)
      .AddUniformVariables({{alpha},
                            {beta},
                            {M}, /* dim_a_outer */
                            {N}, /* dim_b_outer */
                            {K}} /*dim_inner */
      );

  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace onnxruntime
