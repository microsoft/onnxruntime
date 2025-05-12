// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/math/gemm_vec4.h"

#include "core/providers/webgpu/webgpu_utils.h"

#include "core/providers/webgpu/math/matmul_utils.h"
#include "core/providers/webgpu/math/gemm_utils.h"

namespace onnxruntime {
namespace webgpu {

Status GemmVec4Program::GenerateShaderCode(ShaderHelper& shader) const {
  const ShaderVariableHelper& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);

  // Each thread compute 4*1 elements
  InlinedVector<int64_t> elements_per_thread = InlinedVector<int64_t>({4, 4, 1});

  const auto& batch_dims = shader.AddIndices("batch_dims", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);
  const std::string data_type = "output_element_t";

  if (need_handle_matmul_) {
    const auto& a = shader.AddInput("a", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
    const auto& b = shader.AddInput("b", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias);

    MatMulReadFnSource(shader, a, b, output, batch_dims, transA_, transB_, is_vec4_);
  }
  if (is_vec4_) {
    MakeMatMulPackedVec4Source(shader, elements_per_thread, WorkgroupSizeX(), WorkgroupSizeY(), data_type, &batch_dims, transA_, transB_, alpha_, need_handle_matmul_, output_components_);
  } else {
    MakeMatMulPackedSource(shader, elements_per_thread, WorkgroupSizeX(), WorkgroupSizeY(), data_type, &batch_dims, transA_, transB_, alpha_, need_handle_matmul_);
  }
  MatMulWriteFnSource(shader, output, need_handle_bias_, true, c_components_, output_components_, c_is_scalar_);

  return Status::OK();
}

Status ApplyGemmVec4(const Tensor* a,
                     const Tensor* b,
                     const Tensor* c,
                     bool transA,
                     bool transB,
                     float alpha,
                     float beta,
                     ComputeContext& context,
                     Tensor* y) {
  const auto& a_shape = a->Shape();
  const auto& b_shape = b->Shape();

  uint32_t M = onnxruntime::narrow<uint32_t>(transA ? a_shape[1] : a_shape[0]);
  uint32_t K = onnxruntime::narrow<uint32_t>(transA ? a_shape[0] : a_shape[1]);
  uint32_t N = onnxruntime::narrow<uint32_t>(transB ? b_shape[0] : b_shape[1]);

  // WebGPU doesn't support binding a zero-sized buffer, so we need to check if A or B is empty.
  bool need_handle_matmul = a_shape.Size() > 0 && b_shape.Size() > 0;
  bool need_handle_bias = c && beta;

  const bool is_vec4 = a_shape[1] % 4 == 0 && b_shape[1] % 4 == 0;

  int components = is_vec4 ? 4 : 1;
  int c_components = 4;
  int output_components = N % 4 == 0 ? 4 : 1;

  bool c_is_scalar = false;

  // We use vec4 for C when its last dimension equals N and N is divisible by 4.
  if (need_handle_bias) {
    const auto& c_shape = c->Shape();
    int64_t c_last_dim = c_shape[c_shape.NumDimensions() - 1];
    c_components = (c_last_dim == N && N % 4 == 0) ? 4 : 1;
    c_is_scalar = c_shape.Size() == 1;
  }

  // We use vec4 for Y when N is divisible by 4.
  c_components = is_vec4 ? c_components : 1;
  output_components = is_vec4 ? output_components : 1;

  GemmVec4Program program{transA, transB, alpha, need_handle_bias, need_handle_matmul, c_components, c_is_scalar, output_components, is_vec4};

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

  TensorShape outer_dims = TensorShape({});
  const int64_t batch_size = outer_dims.Size();

  // Get dimensions for matrix multiplication from TensorShape
  uint32_t dim_a_outer = narrow<uint32_t>(a_shape[a_shape.NumDimensions() - 2]);  // left matrix second dimension
  uint32_t dim_inner = narrow<uint32_t>(a_shape[a_shape.NumDimensions() - 1]);    // left matrix first dimension
  uint32_t dim_b_outer = narrow<uint32_t>(b_shape[b_shape.NumDimensions() - 1]);  // right matrix first dimension

  if (transA && transB) {
    dim_a_outer = narrow<uint32_t>(a_shape[a_shape.NumDimensions() - 1]);
    dim_inner = narrow<uint32_t>(a_shape[a_shape.NumDimensions() - 2]);
    dim_b_outer = narrow<uint32_t>(b_shape[b_shape.NumDimensions() - 2]);
  } else if (transA && !transB) {
    dim_a_outer = narrow<uint32_t>(a_shape[a_shape.NumDimensions() - 1]);
    dim_inner = narrow<uint32_t>(a_shape[a_shape.NumDimensions() - 2]);
    dim_b_outer = narrow<uint32_t>(b_shape[b_shape.NumDimensions() - 1]);
  } else if (!transA && transB) {
    dim_a_outer = narrow<uint32_t>(a_shape[a_shape.NumDimensions() - 2]);
    dim_inner = narrow<uint32_t>(a_shape[a_shape.NumDimensions() - 1]);
    dim_b_outer = narrow<uint32_t>(b_shape[b_shape.NumDimensions() - 2]);
  }
  program.CacheHint(alpha, transA, transB, c_is_scalar, is_vec4)
      .AddOutputs({{y, ProgramTensorMetadataDependency::TypeAndRank, output_components}})
      .SetDispatchGroupSize(num_tile_n, num_tile_m, 1)
      .SetWorkgroupSize(GemmVec4Program::MATMUL_PACKED_WORKGROUP_SIZE_X, GemmVec4Program::MATMUL_PACKED_WORKGROUP_SIZE_Y, GemmVec4Program::MATMUL_PACKED_WORKGROUP_SIZE_Z)
      .AddUniformVariables({{num_tile_n},
                            {M},
                            {N},
                            {K},
                            {M / 4},
                            {N / 4},
                            {K / 4},
                            {alpha},
                            {beta},
                            {M},
                            {N},
                            {K}})
      .AddIndices(outer_dims);

  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace onnxruntime
