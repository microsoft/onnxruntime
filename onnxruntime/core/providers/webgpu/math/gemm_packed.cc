// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/math/gemm_packed.h"

#include "core/providers/webgpu/webgpu_utils.h"

#include "core/providers/webgpu/math/matmul.h"
#include "core/providers/webgpu/math/matmul_utils.h"
#include "core/providers/webgpu/math/gemm_utils.h"

namespace onnxruntime {
namespace webgpu {

Status GemmProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const bool need_split_k = NeedSplitK();
  ShaderUsage output_usage = ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias;
  if (need_split_k) {
    // When Split-K is enabled, we will declare output as `atomic<i32>` to call atomic built-in
    // functions on it, so we need below information to correctly compute the index on the output.
    output_usage |= ShaderUsage::UseIndicesToOffset | ShaderUsage::UseShapeAndStride;
  }
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
    ORT_RETURN_IF_ERROR(MakeMatMulPackedVec4Source(shader, elements_per_thread, WorkgroupSizeX(), WorkgroupSizeY(), data_type, nullptr, transA_, transB_, alpha_, need_handle_matmul_, output_components_, /*tile_inner*/ 32, need_split_k, split_dim_inner_));
  } else {
    ORT_RETURN_IF_ERROR(MakeMatMulPackedSource(shader, elements_per_thread, WorkgroupSizeX(), WorkgroupSizeY(), data_type, nullptr, transA_, transB_, alpha_, need_handle_matmul_));
  }

  const ShaderVariableHelper* c = nullptr;
  if (need_handle_bias_) {
    c = &shader.AddInput("c", ShaderUsage::UseUniform);
  }

  const ProgramVariableDataType output_var_type = this->Outputs()[0].var_type;
  MatMulWriteFnSource(shader, output, c, /* is_gemm = */ true, c_components_, c_is_scalar_, /*activation_snippet*/ "", /*is_channels_last*/ false, need_split_k, output_var_type);

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

  // Current Split-K implementation relies on atomic operations, which are not deterministic.
  if (!context.KernelContext().GetUseDeterministicCompute()) {
    const SplitKConfig& split_k_config = context.GetSplitKConfig();
    // Currently we require the components for Y must also be a multiple of 4 when Split-K is used.
    const bool output_is_vec4 = output_components == 4;
    // The parameter `is_channel_last` is not used for GEMM.
    const bool need_split_k = split_k_config.UseSplitK(is_vec4 && output_is_vec4, ActivationKind::None, /*batch_size*/ 1, /*is_gemm*/ true, /*is_channels_last*/ true, M, N, K);
    if (need_split_k) {
      const Tensor* bias = nullptr;
      uint32_t output_components_in_fill_bias_program = 4;
      if (need_handle_bias) {
        bias = c;
        output_components_in_fill_bias_program = c_components;
      }
      const TensorShape output_shape = TensorShape{M, N / output_components_in_fill_bias_program};

      auto fill_bias_program = CreateMatMulFillBiasOrZeroBeforeSplitKProgram(
          bias, y, /*is_gemm*/ true, beta, output_components_in_fill_bias_program, c_is_scalar, output_shape);
      ORT_RETURN_IF_ERROR(context.RunProgram(fill_bias_program));

      // When Split-K is used, `bias` will be handled in `MatMulFillBiasOrZeroBeforeSplitKProgram`
      // instead of here.
      need_handle_bias = false;

      // With Split-K, `dim_inner` will be split into multiple parts and `dispatch_z` will be the
      // number of splits along `dim_inner`.
      split_dim_inner = split_k_config.GetSplitDimInner();
      dispatch_z = (K + split_dim_inner - 1) / split_dim_inner;

      // The output should be declared in atomic types in `MatMulProgram` for the use of atomic
      // built-in functions.
      output.is_atomic = true;
    }
  }

  GemmProgram program{transA, transB, alpha, need_handle_bias, need_handle_matmul, c_components, c_is_scalar, output_components, is_vec4, split_dim_inner};

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

  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace onnxruntime
