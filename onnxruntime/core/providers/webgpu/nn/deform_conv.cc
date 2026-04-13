// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/nn/deform_conv.h"

#include "core/common/narrow.h"
#include "core/providers/webgpu/math/matmul.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace webgpu {
namespace {

class DeformConvIm2ColProgram final : public Program<DeformConvIm2ColProgram> {
 public:
  explicit DeformConvIm2ColProgram(bool has_mask)
      : Program{"DeformConvIm2Col"}, has_mask_{has_mask} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"output_size", ProgramUniformVariableDataType::Uint32},
      {"kernel_shape", ProgramUniformVariableDataType::Uint32},
      {"pads", ProgramUniformVariableDataType::Uint32},
      {"strides", ProgramUniformVariableDataType::Uint32},
      {"dilations", ProgramUniformVariableDataType::Uint32},
      {"x_spatial", ProgramUniformVariableDataType::Uint32},
      {"y_spatial", ProgramUniformVariableDataType::Uint32},
      {"c_per_group", ProgramUniformVariableDataType::Uint32},
      {"c_per_offset_group", ProgramUniformVariableDataType::Uint32});

 private:
  bool has_mask_;
};

class DeformConvWeightPackProgram final : public Program<DeformConvWeightPackProgram> {
 public:
  DeformConvWeightPackProgram() : Program{"DeformConvWeightPack"} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"output_size", ProgramUniformVariableDataType::Uint32},
      {"kernel_shape", ProgramUniformVariableDataType::Uint32},
      {"m_per_group", ProgramUniformVariableDataType::Uint32});
};

class DeformConvOutputProgram final : public Program<DeformConvOutputProgram> {
 public:
  explicit DeformConvOutputProgram(bool has_bias)
      : Program{"DeformConvOutput"}, has_bias_{has_bias} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"output_size", ProgramUniformVariableDataType::Uint32},
      {"out_w", ProgramUniformVariableDataType::Uint32},
      {"m_per_group", ProgramUniformVariableDataType::Uint32});

 private:
  bool has_bias_;
};

Status DeformConvIm2ColProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& x = shader.AddInput("x", ShaderUsage::UseUniform |
                                           ShaderUsage::UseIndicesTypeAlias |
                                           ShaderUsage::UseValueTypeAlias |
                                           ShaderUsage::UseElementTypeAlias);
  const auto& offset = shader.AddInput("offset", ShaderUsage::UseUniform |
                                                      ShaderUsage::UseIndicesTypeAlias);
  const auto& col = shader.AddOutput("col", ShaderUsage::UseUniform |
                                                ShaderUsage::UseIndicesTypeAlias |
                                                ShaderUsage::UseValueTypeAlias);

  shader.AdditionalImplementation()
      << "fn getX(d0 : u32, d1 : u32, d2 : u32, d3 : u32) -> x_value_t {\n"
      << "  let x_indices = x_indices_t(d0, d1, d2, d3);\n"
      << "  return " << x.GetByIndices("x_indices") << ";\n"
      << "}\n"
      << "fn getOffset(d0 : u32, d1 : u32, d2 : u32, d3 : u32) -> x_value_t {\n"
      << "  let offset_indices = offset_indices_t(d0, d1, d2, d3);\n"
      << "  return " << offset.GetByIndices("offset_indices") << ";\n"
      << "}\n";

  if (has_mask_) {
    const auto& mask = shader.AddInput("mask", ShaderUsage::UseUniform |
                                                   ShaderUsage::UseIndicesTypeAlias);
    shader.AdditionalImplementation()
        << "fn getMask(d0 : u32, d1 : u32, d2 : u32, d3 : u32) -> x_value_t {\n"
        << "  let mask_indices = mask_indices_t(d0, d1, d2, d3);\n"
        << "  return " << mask.GetByIndices("mask_indices") << ";\n"
        << "}\n";
  }

  shader.AdditionalImplementation()
      << "fn bilinearInterpolate(n : u32, c : u32, h : x_element_t, w_coord : x_element_t) -> x_element_t {\n"
      << "  let height = i32(uniforms.x_spatial[0]);\n"
      << "  let width = i32(uniforms.x_spatial[1]);\n"
      << "  if (h <= x_element_t(-1.0) || h >= x_element_t(height) || w_coord <= x_element_t(-1.0) || w_coord >= x_element_t(width)) {\n"
      << "    return x_element_t(0.0);\n"
      << "  }\n"
      << "  let h_floor = floor(h);\n"
      << "  let w_floor = floor(w_coord);\n"
      << "  let h_low = i32(h_floor);\n"
      << "  let w_low = i32(w_floor);\n"
      << "  let h_high = h_low + 1;\n"
      << "  let w_high = w_low + 1;\n"
      << "  let lh = h - h_floor;\n"
      << "  let lw = w_coord - w_floor;\n"
      << "  let hh = x_element_t(1.0) - lh;\n"
      << "  let hw = x_element_t(1.0) - lw;\n"
      << "  var result = x_element_t(0.0);\n"
      << "  if (h_low >= 0 && w_low >= 0) {\n"
      << "    result += hh * hw * x_element_t(getX(n, c, u32(h_low), u32(w_low)));\n"
      << "  }\n"
      << "  if (h_low >= 0 && w_high < width) {\n"
      << "    result += hh * lw * x_element_t(getX(n, c, u32(h_low), u32(w_high)));\n"
      << "  }\n"
      << "  if (h_high < height && w_low >= 0) {\n"
      << "    result += lh * hw * x_element_t(getX(n, c, u32(h_high), u32(w_low)));\n"
      << "  }\n"
      << "  if (h_high < height && w_high < width) {\n"
      << "    result += lh * lw * x_element_t(getX(n, c, u32(h_high), u32(w_high)));\n"
      << "  }\n"
      << "  return result;\n"
      << "}\n";

  shader.MainFunctionBody()
      << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")
      << "let col_indices = " << col.OffsetToIndices("global_idx") << ";\n"
      << "let batch = col_indices[0];\n"
      << "let group_idx = col_indices[1];\n"
      << "let spatial_idx = col_indices[2];\n"
      << "let k_idx = col_indices[3];\n"
      << "\n"
      << "let kH = uniforms.kernel_shape[0];\n"
      << "let kW = uniforms.kernel_shape[1];\n"
      << "let kernel_size = kH * kW;\n"
      << "let out_w = uniforms.y_spatial[1];\n"
      << "let C_per_group = uniforms.c_per_group;\n"
      << "let c_offset = k_idx / kernel_size;\n"
      << "let kernel_idx = k_idx % kernel_size;\n"
      << "let kh = kernel_idx / kW;\n"
      << "let kw = kernel_idx % kW;\n"
      << "let c = group_idx * C_per_group + c_offset;\n"
      << "let oh = spatial_idx / out_w;\n"
      << "let ow = spatial_idx % out_w;\n"
      << "\n"
      << "let h_in = x_element_t(oh * uniforms.strides[0] + kh * uniforms.dilations[0]) - x_element_t(uniforms.pads[0]);\n"
      << "let w_in = x_element_t(ow * uniforms.strides[1] + kw * uniforms.dilations[1]) - x_element_t(uniforms.pads[1]);\n"
      << "let offset_group_idx = c / uniforms.c_per_offset_group;\n"
      << "let offset_idx = (offset_group_idx * kernel_size + kernel_idx) * 2u;\n"
      << "let offset_h = x_element_t(getOffset(batch, offset_idx, oh, ow));\n"
      << "let offset_w = x_element_t(getOffset(batch, offset_idx + 1u, oh, ow));\n"
      << "var value = bilinearInterpolate(batch, c, h_in + offset_h, w_in + offset_w);\n";

  if (has_mask_) {
    shader.MainFunctionBody()
        << "let mask_idx = offset_group_idx * kernel_size + kernel_idx;\n"
        << "value = value * x_element_t(getMask(batch, mask_idx, oh, ow));\n";
  }

  shader.MainFunctionBody() << col.SetByOffset("global_idx", "x_value_t(value)");

  return Status::OK();
}

Status DeformConvWeightPackProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& w = shader.AddInput("w", ShaderUsage::UseUniform |
                                           ShaderUsage::UseIndicesTypeAlias |
                                           ShaderUsage::UseValueTypeAlias);
  const auto& packed_w = shader.AddOutput("packed_w", ShaderUsage::UseUniform |
                                                      ShaderUsage::UseIndicesTypeAlias |
                                                      ShaderUsage::UseValueTypeAlias);

  shader.MainFunctionBody()
      << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")
      << "let packed_indices = " << packed_w.OffsetToIndices("global_idx") << ";\n"
      << "let group_idx = packed_indices[0];\n"
      << "let k_idx = packed_indices[1];\n"
      << "let m_offset = packed_indices[2];\n"
      << "\n"
      << "let kH = uniforms.kernel_shape[0];\n"
      << "let kW = uniforms.kernel_shape[1];\n"
      << "let kernel_size = kH * kW;\n"
      << "let c_offset = k_idx / kernel_size;\n"
      << "let kernel_idx = k_idx % kernel_size;\n"
      << "let kh = kernel_idx / kW;\n"
      << "let kw = kernel_idx % kW;\n"
      << "let out_channel = group_idx * uniforms.m_per_group + m_offset;\n"
      << "let w_indices = w_indices_t(out_channel, c_offset, kh, kw);\n"
      << packed_w.SetByOffset("global_idx", w.GetByIndices("w_indices"));

  return Status::OK();
}

Status DeformConvOutputProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& mm_output = shader.AddInput("mm_output", ShaderUsage::UseUniform |
                                                       ShaderUsage::UseIndicesTypeAlias |
                                                       ShaderUsage::UseValueTypeAlias);
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform |
                                                   ShaderUsage::UseIndicesTypeAlias |
                                                   ShaderUsage::UseValueTypeAlias);
  const ShaderVariableHelper* bias = nullptr;

  if (has_bias_) {
    bias = &shader.AddInput("bias", ShaderUsage::UseUniform);
  }

  shader.MainFunctionBody()
      << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")
      << "let output_indices = " << output.OffsetToIndices("global_idx") << ";\n"
      << "let batch = output_indices[0];\n"
      << "let out_channel = output_indices[1];\n"
      << "let oh = output_indices[2];\n"
      << "let ow = output_indices[3];\n"
      << "let m_per_group = uniforms.m_per_group;\n"
      << "let group_idx = out_channel / m_per_group;\n"
      << "let m_offset = out_channel % m_per_group;\n"
      << "let spatial_idx = oh * uniforms.out_w + ow;\n"
      << "let mm_indices = mm_output_indices_t(batch, group_idx, spatial_idx, m_offset);\n"
      << "var value = " << mm_output.GetByIndices("mm_indices") << ";\n";

  if (has_bias_) {
    shader.MainFunctionBody()
        << "value = value + " << bias->GetByOffset("out_channel") << ";\n";
  }

  shader.MainFunctionBody() << output.SetByOffset("global_idx", "value");

  return Status::OK();
}

Status RunDeformConvWeightPack(ComputeContextBase& context,
                               const Tensor* weight,
                               const DeformConvParams& params,
                               const DeformConvCommonDims& common_dims,
                               Tensor& packed_weight) {
  const uint32_t packed_weight_size = narrow<uint32_t>(packed_weight.Shape().Size());
  DeformConvWeightPackProgram program;
  program.CacheHint(params.group, common_dims.kernel_dim, params.M / params.group)
      .AddInput({weight, ProgramTensorMetadataDependency::TypeAndRank, weight->Shape(), 1})
      .AddOutput({&packed_weight, ProgramTensorMetadataDependency::TypeAndRank, packed_weight.Shape(), 1})
      .AddUniformVariables({{packed_weight_size},
                            {std::vector<uint32_t>{narrow<uint32_t>(params.kH), narrow<uint32_t>(params.kW)}},
                            {narrow<uint32_t>(params.M / params.group)}})
      .SetDispatchGroupSize((packed_weight_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE);

  return context.RunProgram(program);
}

Status RunDeformConvIm2Col(ComputeContext& context,
                           const Tensor* x,
                           const Tensor* offset,
                           const Tensor* mask,
                           const DeformConvParams& params,
                           const DeformConvCommonDims& common_dims,
                           Tensor& col_buffer) {
  const bool has_mask = mask != nullptr;
  const uint32_t col_size = narrow<uint32_t>(col_buffer.Shape().Size());
  DeformConvIm2ColProgram program(has_mask);
  program.CacheHint(has_mask, params.group, params.offset_group, common_dims.kernel_dim)
      .AddInput({x, ProgramTensorMetadataDependency::TypeAndRank, x->Shape(), 1})
      .AddInput({offset, ProgramTensorMetadataDependency::TypeAndRank, offset->Shape(), 1})
      .AddOutput({&col_buffer, ProgramTensorMetadataDependency::TypeAndRank, col_buffer.Shape(), 1})
      .AddUniformVariables({{col_size},
                            {std::vector<uint32_t>{narrow<uint32_t>(params.kH), narrow<uint32_t>(params.kW)}},
                            {std::vector<uint32_t>{narrow<uint32_t>(params.pad_h), narrow<uint32_t>(params.pad_w)}},
                            {std::vector<uint32_t>{narrow<uint32_t>(params.stride_h), narrow<uint32_t>(params.stride_w)}},
                            {std::vector<uint32_t>{narrow<uint32_t>(params.dilation_h), narrow<uint32_t>(params.dilation_w)}},
                            {std::vector<uint32_t>{narrow<uint32_t>(params.H), narrow<uint32_t>(params.W_in)}},
                            {std::vector<uint32_t>{narrow<uint32_t>(params.out_h), narrow<uint32_t>(params.out_w)}},
                            {narrow<uint32_t>(params.C / params.group)},
                            {narrow<uint32_t>(params.C / params.offset_group)}})
      .SetDispatchGroupSize((col_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE);

  if (has_mask) {
    program.AddInput({mask, ProgramTensorMetadataDependency::TypeAndRank, mask->Shape(), 1});
  }

  return context.RunProgram(program);
}

Status RunDeformConvOutput(ComputeContext& context,
                           const Tensor* mm_output,
                           const Tensor* bias,
                           const DeformConvParams& params,
                           Tensor* output) {
  const bool has_bias = bias != nullptr;
  const uint32_t output_size = narrow<uint32_t>(output->Shape().Size());
  DeformConvOutputProgram program(has_bias);
  program.CacheHint(has_bias, params.group, params.M / params.group)
      .AddInput({mm_output, ProgramTensorMetadataDependency::TypeAndRank, mm_output->Shape(), 1})
      .AddOutput({output, ProgramTensorMetadataDependency::TypeAndRank, output->Shape(), 1})
      .AddUniformVariables({{output_size},
                            {narrow<uint32_t>(params.out_w)},
                            {narrow<uint32_t>(params.M / params.group)}})
      .SetDispatchGroupSize((output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE);

  if (has_bias) {
    program.AddInput({bias, ProgramTensorMetadataDependency::TypeAndRank, bias->Shape(), 1});
  }

  return context.RunProgram(program);
}

}  // namespace

Status DeformConv::ComputeInternal(ComputeContext& context) const {
  const auto* x = context.Input<Tensor>(0);
  const auto* weight = context.Input<Tensor>(1);
  const auto* offset = context.Input<Tensor>(2);
  const auto* bias = context.Input<Tensor>(3);
  const auto* mask = context.Input<Tensor>(4);

  DeformConvParams params;
  ORT_RETURN_IF_ERROR(DeformConvValidateAndParse(
      attrs_,
      x->Shape(), weight->Shape(), offset->Shape(),
      bias ? &bias->Shape() : nullptr,
      mask ? &mask->Shape() : nullptr,
      params));

  DeformConvCommonDims common_dims;
  TensorShape output_shape{params.N, params.M, params.out_h, params.out_w};
  auto* output = context.Output(0, output_shape);
  if (output_shape.Size() == 0) {
    return Status::OK();
  }
  ORT_RETURN_IF_ERROR(DeformConvValidateAndComputeCommonDims(params, common_dims));

  const int64_t m_per_group = params.M / params.group;
  Tensor packed_weight = context.CreateGPUTensor(weight->DataType(), TensorShape{params.group, common_dims.kernel_dim, m_per_group});
  ORT_RETURN_IF_ERROR(RunDeformConvWeightPack(context, weight, params, common_dims, packed_weight));

  Tensor col_buffer = context.CreateGPUTensor(x->DataType(), TensorShape{params.N, params.group, common_dims.output_image_size, common_dims.kernel_dim});
  ORT_RETURN_IF_ERROR(RunDeformConvIm2Col(context, x, offset, mask, params, common_dims, col_buffer));

  Tensor mm_output = context.CreateGPUTensor(output->DataType(), TensorShape{params.N, params.group, common_dims.output_image_size, m_per_group});
  std::vector<const Tensor*> matmul_inputs{&col_buffer, &packed_weight};
  ORT_RETURN_IF_ERROR(ComputeMatMul(&context,
                                    Activation{},
                                    matmul_inputs,
                                    &mm_output,
                                    true /* is_channels_last */));

  return RunDeformConvOutput(context, &mm_output, bias, params, output);
}

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    DeformConv,
    kOnnxDomain,
    19, 21,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes()),
    DeformConv);

ONNX_OPERATOR_KERNEL_EX(
    DeformConv,
    kOnnxDomain,
    22,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes()),
    DeformConv);

}  // namespace webgpu
}  // namespace onnxruntime
