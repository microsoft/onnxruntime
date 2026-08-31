// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/math/gemm_packed.h"

#include <limits>

#include "core/providers/webgpu/webgpu_utils.h"

#include "core/providers/webgpu/math/matmul.h"
#include "core/providers/webgpu/math/matmul_utils.h"
#include "core/providers/webgpu/math/gemm_utils.h"

namespace onnxruntime {
namespace webgpu {

Status GemmProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const bool need_split_k = NeedSplitK();
  const ShaderUsage output_usage = ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias;
  const ShaderVariableHelper& output = shader.AddOutput("output", output_usage);

  // Each thread compute 4*4 elements
  InlinedVector<int64_t> elements_per_thread = InlinedVector<int64_t>({4, 4, 1});

  const std::string data_type = "output_element_t";

  if (need_handle_matmul_) {
    const auto& a = shader.AddInput("a", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
    const auto& b = shader.AddInput("b", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias);

    MatMulReadFnSource(shader, a, b, nullptr, transA_, transB_);
  }
  if (is_vec4_) {
    ORT_RETURN_IF_ERROR(MakeMatMulPackedVec4Source(shader, elements_per_thread, WorkgroupSizeX(), WorkgroupSizeY(), data_type, /* batch_dims = */ nullptr, transA_, transB_, alpha_, need_handle_matmul_, output_components_, /*tile_inner*/ 32, need_split_k, split_dim_inner_));
  } else {
    ORT_RETURN_IF_ERROR(MakeMatMulPackedSource(shader, elements_per_thread, WorkgroupSizeX(), WorkgroupSizeY(), data_type, /* batch_dims = */ nullptr, transA_, transB_, alpha_, need_handle_matmul_));
  }

  const ShaderVariableHelper* c = nullptr;
  if (need_handle_bias_) {
    c = &shader.AddInput("c", ShaderUsage::UseUniform);
  }

  if (need_split_k) {
    // `beta * C` is applied by `MatMulSplitKReduceProgram` after the partials are summed, so the bias
    // input is deliberately left unbound here.
    MatMulWriteFnSourceWithSplitK(shader, output, /*is_gemm = */ true);
  } else {
    MatMulWriteFnSourceForGemm(shader, output, c, c_is_scalar_);
  }

  return Status::OK();
}

bool GemmProgram::NeedSplitK() const {
  return split_dim_inner_ > 1;
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

  ProgramOutput output(y, ProgramTensorMetadataDependency::TypeAndRank, output_components);
  uint32_t dispatch_z = 1;
  uint32_t split_dim_inner = 1;
  uint32_t splits = 1;

  // Split-K partial sums. Empty unless Split-K is used; `partials` must outlive both dispatches.
  Tensor partials;

  const SplitKConfig& split_k_config = context.GetSplitKConfig();
  // Currently we require the components for Y must also be a multiple of 4 when Split-K is used.
  const bool output_is_vec4 = output_components == 4;
  // We need to use `true` as `is_channels_last` to meet the requirement in `UseSplitK`.
  // Gemm fuses no activation.
  const bool need_split_k = split_k_config.UseSplitK(is_vec4 && output_is_vec4, Activation{}, /*batch_size*/ 1, M, N, K);
  if (need_split_k) {
    ORT_RETURN_IF_NOT(N % 4 == 0, "Split-K GEMM requires N to be a multiple of 4.");

    // With Split-K, `dim_inner` will be split into multiple parts and `dispatch_z` will be the
    // number of splits along `dim_inner`.
    split_dim_inner = split_k_config.GetSplitDimInner();
    splits = (K + split_dim_inner - 1) / split_dim_inner;
    dispatch_z = splits;

    // Each split writes to its own slot of `partials` instead of accumulating into the shared output,
    // so the summation order is fixed by index in the reduction below. `beta * C` is applied there.
    const uint64_t out_elems = static_cast<uint64_t>(M) * (N / output_components);
    const uint64_t scratch_elems = static_cast<uint64_t>(splits) * out_elems;
    const uint64_t scratch_scalars = scratch_elems * output_components;
    ORT_RETURN_IF_NOT(scratch_scalars <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                      "Split-K scratch size exceeds int64_t range: ", scratch_scalars);
    partials = context.CreateGPUTensor(y->DataType(),
                                       TensorShape{static_cast<int64_t>(scratch_scalars)});

    // Pass 1 writes whole vec4 elements at a flat offset, so bind the scratch as rank 1.
    output = ProgramOutput(&partials, ProgramTensorMetadataDependency::TypeAndRank,
                           TensorShape{static_cast<int64_t>(scratch_elems)}, output_components);
  }

  // `beta * C` is applied by the Split-K reduction rather than by `GemmProgram`.
  const bool reduce_handles_bias = need_split_k && need_handle_bias;
  if (need_split_k) {
    need_handle_bias = false;
  }

  GemmProgram program{transA, transB, alpha, need_handle_bias, need_handle_matmul, c_is_scalar, output_components, is_vec4, split_dim_inner};

  if (need_handle_matmul) {
    program.AddInputs({{a, ProgramTensorMetadataDependency::TypeAndRank, components},
                       {b, ProgramTensorMetadataDependency::TypeAndRank, components}});
  }

  if (need_handle_bias) {
    program.AddInput({c, ProgramTensorMetadataDependency::TypeAndRank, c_components});
  }

  const uint32_t TILE_SIZE = 32;
  const uint32_t dispatch_x = (N + TILE_SIZE - 1) / TILE_SIZE;
  const uint32_t dispatch_y = (M + TILE_SIZE - 1) / TILE_SIZE;

  program.CacheHint(alpha, transA, transB, c_is_scalar, split_dim_inner)
      .AddOutput(std::move(output))
      .SetDispatchGroupSize(dispatch_x, dispatch_y, dispatch_z)
      .SetWorkgroupSize(GemmProgram::MATMUL_PACKED_WORKGROUP_SIZE_X, GemmProgram::MATMUL_PACKED_WORKGROUP_SIZE_Y, GemmProgram::MATMUL_PACKED_WORKGROUP_SIZE_Z)
      .AddUniformVariables({{alpha},
                            {beta},
                            {M},          /* dim_a_outer */
                            {N},          /* dim_b_outer */
                            {K},          /*dim_inner */
                            {dispatch_x}, /* logical_dispatch_x */
                            {dispatch_y}, /* logical_dispatch_y */
                            {dispatch_z}} /* logical_dispatch_z */
      );

  ORT_RETURN_IF_ERROR(context.RunProgram(program));

  if (!need_split_k) {
    return Status::OK();
  }

  // Pass 2: sum the per-split partials in a fixed index order, apply `beta * C`, and write Y.
  const TensorShape reduce_output_shape = TensorShape{M, N / output_components};
  auto reduce_program = CreateMatMulSplitKReduceProgram(
      partials, reduce_handles_bias ? c : nullptr, y, /*is_gemm*/ true, Activation{}, beta,
      narrow<uint32_t>(output_components), narrow<uint32_t>(c_components), reduce_output_shape, splits);
  return context.RunProgram(reduce_program);
}

}  // namespace webgpu
}  // namespace onnxruntime
