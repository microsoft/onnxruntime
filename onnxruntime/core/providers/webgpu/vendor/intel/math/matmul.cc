// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/webgpu/webgpu_utils.h"
#include "core/providers/webgpu/math/matmul_utils.h"
#include "core/providers/webgpu/vendor/intel/math/gemm_subgroup.h"
#include "core/providers/webgpu/math/gemm_utils.h"
#include "core/providers/webgpu/vendor/intel/math/matmul.h"

namespace onnxruntime {
namespace webgpu {
namespace intel {

Status MatMulSubgroupProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& a = shader.AddInput("a", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias |
                                           ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  const auto& b = shader.AddInput("b", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias |
                                           ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias |
                                                      ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  const auto& batch_dims = shader.AddIndices("batch_dims", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);

  const ShaderVariableHelper* bias = nullptr;
  if (has_bias_) {
    bias = &shader.AddInput("bias", ShaderUsage::UseUniform);
  }
  std::string apply_activation = GetActivationSnippet(activation_, "output_value_t", "output_element_t");
  // declare the read and write functions
  MatMulReadFnSource(shader, a, b, &batch_dims, /*transA = */ false, /*transB = */ false);
  MatMulWriteFnSource(shader, output, bias, /* is_gemm = */ false, 1,
                      false, apply_activation, /*is_channels_last = */ false);
  // generate the main function
  ORT_RETURN_IF_ERROR(MakeMatMulSubgroupSource(shader, elements_per_thread_, &batch_dims, is_vec4_));
  return Status::OK();
}

bool CanApplyMatMulIntel(const ComputeContext& context, int64_t M, int64_t N, int64_t K) {
  return CanApplySubgroup(context, M, N, K);
}

Status ApplyMatMulIntel(ComputeContext& context,
                        const Activation& activation,
                        std::vector<const Tensor*>& inputs,
                        Tensor* output) {
  const auto* a = inputs[0];
  const auto* b = inputs[1];
  bool has_bias = inputs.size() > 2;
  TensorShape a_shape = a->Shape();
  TensorShape b_shape = b->Shape();

  MatMulComputeHelper helper;
  ORT_THROW_IF_ERROR(helper.Compute(a_shape, b_shape));
  int64_t batchA = a_shape.SizeToDimension(a_shape.NumDimensions() - 2);
  int64_t batchB = b_shape.SizeToDimension(b_shape.NumDimensions() - 2);

  TensorShape output_shape = helper.OutputShape();

  const int64_t dim_output_outer = output_shape[output_shape.NumDimensions() - 2];
  // check if A is batch of vector (bach is not 1, M is 1) and B is a matrix (batch is 1)
  if (batchA != 1 && dim_output_outer == 1 && batchB == 1) {
    // optimization for batched vector matrix multiplication
    // dimensions of A: [1,`batchA`,K]
    TensorShapeVector dims_a = {1, batchA, helper.K()};
    // dimensions of B: [1,K,N]
    TensorShapeVector dims_b = {1, helper.K(), helper.N()};

    a_shape = TensorShape(dims_a);
    b_shape = TensorShape(dims_b);
    output_shape = {1, batchA, helper.N()};
  }

  // helpful dimension variables
  TensorShape outer_dims_a = a_shape.NumDimensions() > 2
                                 ? a_shape.Slice(0, a_shape.NumDimensions() - 2)
                                 : TensorShape({});

  TensorShape outer_dims_b = b_shape.NumDimensions() > 2
                                 ? b_shape.Slice(0, b_shape.NumDimensions() - 2)
                                 : TensorShape({});

  TensorShape outer_dims = output_shape.NumDimensions() > 2
                               ? output_shape.Slice(0, output_shape.NumDimensions() - 2)
                               : TensorShape({});

  const int64_t batch_size = outer_dims.Size();

  // Get dimensions for matrix multiplication from TensorShape
  const uint32_t dim_a_outer = narrow<uint32_t>(a_shape[a_shape.NumDimensions() - 2]);  // left matrix second dimension
  const uint32_t dim_inner = narrow<uint32_t>(a_shape[a_shape.NumDimensions() - 1]);    // left matrix first dimension
  const uint32_t dim_b_outer = narrow<uint32_t>(b_shape[b_shape.NumDimensions() - 1]);  // right matrix first dimension

  // Always access A with 1-component when using subgroup.
  const bool is_vec4 = dim_b_outer % 4 == 0;
  InlinedVector<int64_t> elements_per_thread = InlinedVector<int64_t>({4, intel::ElementsPerThreadY(is_vec4, dim_a_outer), 1});

  const uint32_t dispatch_x = narrow<uint32_t>((dim_b_outer + kSubgroupLogicalWorkGroupSizeX * elements_per_thread[0] - 1) /
                                               (kSubgroupLogicalWorkGroupSizeX * elements_per_thread[0]));
  const uint32_t dispatch_y = narrow<uint32_t>((dim_a_outer + kSubgroupLogicalWorkGroupSizeY * elements_per_thread[1] - 1) /
                                               (kSubgroupLogicalWorkGroupSizeY * elements_per_thread[1]));
  const uint32_t dispatch_z = narrow<uint32_t>((static_cast<uint32_t>(batch_size) +
                                                kSubgroupLogicalWorkGroupSizeZ * elements_per_thread[2] - 1) /
                                               (kSubgroupLogicalWorkGroupSizeZ * elements_per_thread[2]));

  const int components = is_vec4 ? 4 : 1;
  const int a_components = 1;
  const int b_components = components;
  const TensorShape a_shape_temp = CreateMatMulIntermediateShape(outer_dims_a, dim_a_outer, dim_inner, a_components);
  const TensorShape b_shape_temp = CreateMatMulIntermediateShape(outer_dims_b, dim_inner, dim_b_outer, b_components);
  const TensorShape output_shape_temp = TensorShape({batch_size, dim_a_outer, dim_b_outer / components});

  MatMulSubgroupProgram program{activation, has_bias, is_vec4, elements_per_thread};
  program
      .CacheHint(activation.ToString(), absl::StrJoin(elements_per_thread, "-"))
      .AddInputs({{a, ProgramTensorMetadataDependency::TypeAndRank, a_shape_temp, a_components},
                  {b, ProgramTensorMetadataDependency::TypeAndRank, b_shape_temp, b_components}})
      .AddOutputs({{output, ProgramTensorMetadataDependency::Rank, output_shape_temp, components}})
      .AddUniformVariables({{dim_a_outer}, {dim_b_outer}, {dim_inner}})
      .AddIndices(outer_dims)
      .SetDispatchGroupSize(dispatch_x, dispatch_y, dispatch_z)
      .SetWorkgroupSize(kSubgroupLogicalWorkGroupSizeX * kSubgroupLogicalWorkGroupSizeY, 1, 1);

  if (has_bias) {
    auto bias_components = 1;
    const auto* bias = inputs[2];
    TensorShape reduced_bias_shape = ReduceShapeByComponents(bias->Shape(), bias_components);
    program.AddInput({bias, ProgramTensorMetadataDependency::Rank, reduced_bias_shape, bias_components});
  }

  return context.RunProgram(program);
}

}  // namespace intel
}  // namespace webgpu
}  // namespace onnxruntime
