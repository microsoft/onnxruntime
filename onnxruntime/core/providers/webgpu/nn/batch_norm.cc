// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/inlined_containers.h"
#include "core/providers/webgpu/nn/batch_norm.h"
#include "core/providers/cpu/nn/batch_norm_helper.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace webgpu {

#define WEBGPU_BATCH_NORM_VERSIONED_KERNEL(start, end, domain, is_nhwc) \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(                                    \
      BatchNormalization,                                               \
      domain,                                                           \
      start,                                                            \
      end,                                                              \
      kWebGpuExecutionProvider,                                         \
      (*KernelDefBuilder::Create())                                     \
          .TypeConstraint("T", WebGpuSupportedFloatTypes()),            \
      BatchNormalization<is_nhwc>);

#define WEBGPU_BATCH_NORM_KERNEL(version, domain, is_nhwc)   \
  ONNX_OPERATOR_KERNEL_EX(                                   \
      BatchNormalization,                                    \
      domain,                                                \
      version,                                               \
      kWebGpuExecutionProvider,                              \
      (*KernelDefBuilder::Create())                          \
          .TypeConstraint("T", WebGpuSupportedFloatTypes()), \
      BatchNormalization<is_nhwc>);

WEBGPU_BATCH_NORM_VERSIONED_KERNEL(7, 8, kOnnxDomain, false)
WEBGPU_BATCH_NORM_VERSIONED_KERNEL(9, 13, kOnnxDomain, false)
WEBGPU_BATCH_NORM_VERSIONED_KERNEL(14, 14, kOnnxDomain, false)
WEBGPU_BATCH_NORM_KERNEL(15, kOnnxDomain, false)

WEBGPU_BATCH_NORM_VERSIONED_KERNEL(7, 8, kMSInternalNHWCDomain, true)
WEBGPU_BATCH_NORM_VERSIONED_KERNEL(9, 13, kMSInternalNHWCDomain, true)
WEBGPU_BATCH_NORM_VERSIONED_KERNEL(14, 14, kMSInternalNHWCDomain, true)
WEBGPU_BATCH_NORM_KERNEL(15, kMSInternalNHWCDomain, true)

Status BatchNormalizationProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const ShaderVariableHelper& input_tensor = shader.AddInput("input_tensor", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);
  const ShaderVariableHelper& scale = shader.AddInput("scale", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);
  const ShaderVariableHelper& B = shader.AddInput("B", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);
  const ShaderVariableHelper& input_mean = shader.AddInput("input_mean", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);
  const ShaderVariableHelper& input_var = shader.AddInput("input_var", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);
  const ShaderVariableHelper& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);

  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")
                            << "  let idx = global_idx * " << components_ << ";\n"
                            << "  var outputIndices = " << output.OffsetToIndices("idx") << ";\n";
  if (spatial_) {
    if (input_tensor.Rank() == 1) {
      shader.MainFunctionBody() << "  let cOffset = 0u;\n";
    } else {
      if (format_ == DataLayout::NHWC) {
        shader.MainFunctionBody() << "  let cOffset = outputIndices[" << input_tensor.Rank() - 1 << "] / " << components_ << ";\n";
      } else {
        shader.MainFunctionBody() << "  let cOffset = outputIndices[1];\n";
      }
    }
  } else {
    if (format_ == DataLayout::NCHW) {
      shader.MainFunctionBody() << "  " << output.IndicesSet("outputIndices", "0", "0") << "\n"
                                << "  let cOffset = " << output.IndicesToOffset("outputIndices") << ";\n";
    } else {
      // update C channel
      shader.MainFunctionBody() << "  var cIndices = scale_indices_t(0);\n"
                                << "  cIndices[0] = outputIndices[" << input_tensor.Rank() - 1 << "];\n";
      // update D1 x ... x Dn channels
      for (int i = 1; i < scale.Rank(); i++) {
        shader.MainFunctionBody() << "  cIndices[" << i << "] = outputIndices[" << i << "];\n";
      }
      shader.MainFunctionBody() << "  let cOffset = " << scale.IndicesToOffset("cIndices") << ";\n";
    }
  }

  shader.MainFunctionBody() << "  let scale = " << scale.GetByOffset("cOffset") << ";\n"
                            << "  let B = " << B.GetByOffset("cOffset") << ";\n"
                            << "  let input_mean = " << input_mean.GetByOffset("cOffset") << ";\n"
                            << "  let input_var = " << input_var.GetByOffset("cOffset") << ";\n"
                            << "  let x = " << input_tensor.GetByOffset("global_idx") << ";\n"
                            << "  let value = (x - input_mean) * inverseSqrt(input_var + " << epsilon_ << ") * scale + B;\n"
                            << "  " << output.SetByOffset("global_idx", "value") << "\n";

  return Status::OK();
}

template <bool is_nhwc>
Status BatchNormalization<is_nhwc>::ComputeInternal(ComputeContext& context) const {
  if (training_mode_) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "BatchNormalization trainingMode is not supported yet.");
  }

  if (context.InputCount() != 5) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "BatchNormalization requires 5 inputs.");
  }

  const auto* input_tensor = context.Input(0);
  const TensorShape& input_shape = input_tensor->Shape();
  size_t input_rank = input_shape.NumDimensions();
  const int components = spatial_ ? ((input_shape[input_rank - 1] % 4 == 0) ? 4 : ((input_shape[input_rank - 1] % 2 == 0) ? 2 : 1)) : 1;

  auto output_dims = input_shape.AsShapeVector();
  TensorShape output_shape(output_dims);
  auto* output_tensor = context.Output(0, output_shape);
  int64_t output_size = output_tensor->Shape().Size() / static_cast<int64_t>(components);

  if (output_size == 0) {
    return Status::OK();
  }

  const auto* scale = context.Input<Tensor>(1);
  const auto* B = context.Input<Tensor>(2);
  const auto* input_mean = context.Input<Tensor>(3);
  const auto* input_var = context.Input<Tensor>(4);

  ORT_RETURN_IF_ERROR(BatchNormHelper::ValidateInputs(input_tensor, scale, B, input_mean, input_var, spatial_ == 1, format_ == DataLayout::NHWC));

  BatchNormalizationProgram program{epsilon_, spatial_, format_, static_cast<int64_t>(components)};
  program
      .AddInputs({{input_tensor, ProgramTensorMetadataDependency::TypeAndRank},
                  {scale, ProgramTensorMetadataDependency::TypeAndRank},
                  {B, ProgramTensorMetadataDependency::TypeAndRank},
                  {input_mean, ProgramTensorMetadataDependency::TypeAndRank},
                  {input_var, ProgramTensorMetadataDependency::TypeAndRank}})
      .AddOutputs({output_tensor})
      .SetDispatchGroupSize((output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({{static_cast<uint32_t>(output_size)}});
  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace onnxruntime