// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/vendor/intel/math/gemm_intel.h"
#include "core/providers/webgpu/vendor/intel/math/gemm_subgroup.h"
#include "core/providers/webgpu/vendor/intel/math/gemm_utils_intel.h"

namespace onnxruntime {
namespace webgpu {

Status GemmSubgroupProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const ShaderVariableHelper& output = shader.AddOutput("output", ShaderUsage::UseUniform |
                                                                      ShaderUsage::UseValueTypeAlias |
                                                                      ShaderUsage::UseElementTypeAlias);
  const std::string data_type = "output_element_t";

  if (need_handle_matmul_) {
    const auto& a = shader.AddInput("a", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias |
                                             ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
    const auto& b = shader.AddInput("b", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias |
                                             ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);

    MatMulReadFnSourceIntel(shader, a, b, nullptr, transA_, transB_, is_vec4_, true);
  }
  if (is_vec4_) {
    ORT_RETURN_IF_ERROR(MakeMatMulSubgroupVec4Source(shader, elements_per_thread_, data_type, nullptr, transA_,
                                                     transB_, alpha_, need_handle_matmul_));
  } else {
    ORT_RETURN_IF_ERROR(MakeMatMulSubgroupSource(shader, elements_per_thread_, data_type, nullptr, transA_, transB_,
                                                 alpha_, need_handle_matmul_));
  }
  MatMulWriteFnSourceIntel(shader, output, need_handle_bias_, true, c_components_, output_components_, c_is_scalar_);

  return Status::OK();
}

bool CanApplyGemmIntel(const ComputeContext& context, int64_t M, int64_t N, int64_t K, bool transA, bool transB) {
  if (context.AdapterInfo().vendor == std::string_view{"intel"}) {
    bool use_subgroup = (context.AdapterInfo().architecture == std::string_view{"xe-2lpg"} ||
                         context.AdapterInfo().architecture == std::string_view{"xe-2hpg"}) &&
                        M > 16 && N > 768 && K >= 32 && !transA && !transB;
    return use_subgroup;
  }

  return false;
}

Status ApplyGemmIntel(const Tensor* a,
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

  bool use_subgroup = M > 16 && N > 768 && K >= 32 && !transA && !transB;
  if (use_subgroup) {
    const bool is_vec4 = b_shape[1] % 4 == 0;
    // Components for A, B
    int a_components = 1;
    int b_components = is_vec4 ? 4 : 1;
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

    const int64_t elements_per_thread_y = is_vec4 ? (M <= 8 ? 1 : (M <= 16 ? 2 : (M <= 32 ? 4 : 8))) : 4;
    InlinedVector<int64_t> elements_per_thread = InlinedVector<int64_t>({4, elements_per_thread_y, 1});
    const uint32_t dispatch_x = narrow<uint32_t>((N + kSubgroupLogicalWorkGroupSizeX * elements_per_thread[0] - 1) /
                                                 (kSubgroupLogicalWorkGroupSizeX * elements_per_thread[0]));
    const uint32_t dispatch_y = narrow<uint32_t>((M + kSubgroupLogicalWorkGroupSizeY * elements_per_thread[1] - 1) /
                                                 (kSubgroupLogicalWorkGroupSizeY * elements_per_thread[1]));

    GemmSubgroupProgram program{transA, transB, alpha, need_handle_bias, need_handle_matmul, c_components, c_is_scalar,
                                output_components, is_vec4, elements_per_thread};

    if (need_handle_matmul) {
      program.AddInputs({{a, ProgramTensorMetadataDependency::TypeAndRank, a_components},
                         {b, ProgramTensorMetadataDependency::TypeAndRank, b_components}});
    }

    if (need_handle_bias) {
      program.AddInput({c, ProgramTensorMetadataDependency::TypeAndRank, c_components});
    }

    program.CacheHint(alpha, transA, transB, c_is_scalar, is_vec4, absl::StrJoin(elements_per_thread, "-"))
        .AddOutputs({{y, ProgramTensorMetadataDependency::TypeAndRank, output_components}})
        .SetDispatchGroupSize(dispatch_x, dispatch_y, 1)
        .SetWorkgroupSize(kSubgroupLogicalWorkGroupSizeX * kSubgroupLogicalWorkGroupSizeY, 1, 1)
        .AddUniformVariables({{alpha},
                              {beta},
                              {M}, /* dim_a_outer */
                              {N}, /* dim_b_outer */
                              {K}} /*dim_inner */
        );

    return context.RunProgram(program);
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, RUNTIME_EXCEPTION, "Should not reach here");
  }
}

}  // namespace webgpu
}  // namespace onnxruntime
