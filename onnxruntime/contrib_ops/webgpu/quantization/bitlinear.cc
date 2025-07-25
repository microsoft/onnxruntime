// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/quantization/bitlinear.h"

#include <string>
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/webgpu_utils.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

ONNX_OPERATOR_KERNEL_EX(
    BitLinear,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", WebGpuSupportedFloatTypes())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<uint8_t>()),
    BitLinear);

Status BitLinearQuantizeProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("input_a", ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  shader.AddOutput("output", ShaderUsage::UseValueTypeAlias);
  shader.AddOutput("output_5th", ShaderUsage::UseValueTypeAlias);
  shader.AddOutput("scales", ShaderUsage::UseElementTypeAlias);

  return WGSL_TEMPLATE_APPLY(shader, "quantization/bitlinear_quantize.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(K4, K_ / 4));
}

Status BitLinearMultiplyProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("input_a", ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  shader.AddInput("input_a5", ShaderUsage::UseValueTypeAlias);
  shader.AddInput("scales_a", ShaderUsage::UseElementTypeAlias);
  shader.AddInput("input_b", ShaderUsage::UseValueTypeAlias);
  shader.AddOutput("output", ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);

  return WGSL_TEMPLATE_APPLY(shader, "quantization/bitlinear_multiply.wgsl.template");
}

Status BitLinear::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {
  const Tensor* a = context.Input(0);
  const Tensor* b = context.Input(1);

  // Validate input shapes
  TensorShape b_shape({N_, K_});
  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b_shape, false, true));
  auto* y = context.Output(0, helper.OutputShape());

  const uint32_t data_size = onnxruntime::narrow<uint32_t>(y->Shape().Size());
  if (data_size == 0) {
    return Status::OK();
  }

  const uint32_t M = onnxruntime::narrow<uint32_t>(helper.M());
  const uint32_t N = onnxruntime::narrow<uint32_t>(helper.N());
  const uint32_t K = onnxruntime::narrow<uint32_t>(helper.K());

  // Validate input B shape more specifically
  TensorShape expected_b_shape({N, K / 5});
  ORT_ENFORCE(b->Shape() == expected_b_shape, "Input B shape must be [N, K/5], got ", b->Shape().ToString());

  // Step 1: Quantize input A using BitLinearQuantizeProgram
  const uint32_t quantize_output_size = (M * (K - K / 5) / 4);  // skipping every 5th, packed into u32
  const uint32_t quantize_5th_output_size = M * K / 20;         // every 5th element packed int u32

  TensorShape quantize_output_shape({quantize_output_size});
  TensorShape quantize_5th_output_shape({quantize_5th_output_size});
  TensorShape scales_output_shape({M});

  auto quantized_a = context.CreateGPUTensor(DataTypeImpl::GetType<uint32_t>(), quantize_output_shape);
  auto quantized_a5 = context.CreateGPUTensor(DataTypeImpl::GetType<uint32_t>(), quantize_5th_output_shape);
  auto scales_a = context.CreateGPUTensor(a->DataType(), scales_output_shape);
  constexpr uint32_t kVec4Components = 4;
  constexpr uint32_t kU32Components = 4;

  {
    BitLinearQuantizeProgram quantize_program(K);
    quantize_program
        .SetWorkgroupSize(128)
        .SetDispatchGroupSize(M)
        .AddInputs({{a, ProgramTensorMetadataDependency::TypeAndRank, static_cast<int>(kVec4Components)}})
        .AddOutputs({{&quantized_a, ProgramTensorMetadataDependency::None},
                     {&quantized_a5, ProgramTensorMetadataDependency::None},
                     {&scales_a, ProgramTensorMetadataDependency::None}});
    ORT_RETURN_IF_ERROR(context.RunProgram(quantize_program));
  }

  // Step 2: Matrix multiplication using BitLinearMultiplyProgram
  {
    BitLinearMultiplyProgram multiply_program;
    // input_a is vectorized as vec4<u32> which gives a packing of 16 elements per value.
    const uint32_t input_a_stride = (K - (K / 5)) / 16;
    constexpr uint32_t kTileSize = 64;
    multiply_program
        .SetWorkgroupSize(256)
        .SetDispatchGroupSize((M+kTileSize-1)/kTileSize, (N+kTileSize-1)/kTileSize)
        .AddInputs({{&quantized_a, ProgramTensorMetadataDependency::TypeAndRank, static_cast<int>(kVec4Components)},
                    {&quantized_a5, ProgramTensorMetadataDependency::TypeAndRank},
                    {&scales_a, ProgramTensorMetadataDependency::TypeAndRank},
                    {b, ProgramTensorMetadataDependency::TypeAndRank, static_cast<int>(kU32Components)}})
        .AddOutputs({{y, ProgramTensorMetadataDependency::TypeAndRank, static_cast<int>(kVec4Components)}})
        .AddUniformVariables({{M}, {N}, {K}, {input_a_stride}, {scale_b_}});
    ORT_RETURN_IF_ERROR(context.RunProgram(multiply_program));
  }

  return Status::OK();
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
