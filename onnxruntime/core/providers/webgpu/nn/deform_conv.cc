// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/nn/deform_conv.h"

#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/shader_variable.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace webgpu {

Status DeformConvProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& x = shader.AddInput("x", ShaderUsage::UseUniform |
                                           ShaderUsage::UseIndicesTypeAlias |
                                           ShaderUsage::UseValueTypeAlias |
                                           ShaderUsage::UseElementTypeAlias);
  const auto& w = shader.AddInput("w", ShaderUsage::UseUniform |
                                           ShaderUsage::UseIndicesTypeAlias);
  const auto& offset = shader.AddInput("offset", ShaderUsage::UseUniform |
                                                      ShaderUsage::UseIndicesTypeAlias);
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform |
                                                       ShaderUsage::UseIndicesTypeAlias |
                                                       ShaderUsage::UseValueTypeAlias);

  // Helper functions to access tensors by 4D indices
  shader.AdditionalImplementation()
      << "fn getX(d0 : u32, d1 : u32, d2 : u32, d3 : u32) -> x_value_t {\n"
      << "  let aIndices = x_indices_t(d0, d1, d2, d3);\n"
      << "  return " << x.GetByIndices("aIndices") << ";\n"
      << "}\n"
      << "fn getW(d0 : u32, d1 : u32, d2 : u32, d3 : u32) -> x_value_t {\n"
      << "  let aIndices = w_indices_t(d0, d1, d2, d3);\n"
      << "  return " << w.GetByIndices("aIndices") << ";\n"
      << "}\n"
      << "fn getOffset(d0 : u32, d1 : u32, d2 : u32, d3 : u32) -> x_value_t {\n"
      << "  let aIndices = offset_indices_t(d0, d1, d2, d3);\n"
      << "  return " << offset.GetByIndices("aIndices") << ";\n"
      << "}\n";

  if (has_mask_) {
    const auto& mask = shader.AddInput("mask", ShaderUsage::UseUniform |
                                                   ShaderUsage::UseIndicesTypeAlias);
    shader.AdditionalImplementation()
        << "fn getMask(d0 : u32, d1 : u32, d2 : u32, d3 : u32) -> x_value_t {\n"
        << "  let aIndices = mask_indices_t(d0, d1, d2, d3);\n"
        << "  return " << mask.GetByIndices("aIndices") << ";\n"
        << "}\n";
  }

  // Bilinear interpolation function -- matches CPU BilinearPlanOneSample logic.
  // Each of the 4 corners is checked individually; out-of-bounds corners contribute 0.
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

  // Main function body
  shader.MainFunctionBody()
      << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")
      << "let output_indices = " << output.OffsetToIndices("global_idx") << ";\n"
      << "let batch = output_indices[0];\n"
      << "let out_channel = output_indices[1];\n"
      << "let oh = output_indices[2];\n"
      << "let ow = output_indices[3];\n"
      << "\n"
      << "let C_per_group = uniforms.channels[2];\n"
      << "let M_per_group = uniforms.channels[3];\n"
      << "let kH = uniforms.kernel_shape[0];\n"
      << "let kW = uniforms.kernel_shape[1];\n"
      << "let stride_h = uniforms.strides[0];\n"
      << "let stride_w = uniforms.strides[1];\n"
      << "let pad_h = uniforms.pads[0];\n"
      << "let pad_w = uniforms.pads[1];\n"
      << "let dil_h = uniforms.dilations[0];\n"
      << "let dil_w = uniforms.dilations[1];\n"
      << "let c_per_offset_group = uniforms.c_per_offset_group;\n"
      << "\n"
      << "let g = out_channel / M_per_group;\n"
      << "\n"
      << "var value = x_element_t(0.0);\n"
      << "\n"
      << "for (var c_offset = 0u; c_offset < C_per_group; c_offset++) {\n"
      << "  let c = g * C_per_group + c_offset;\n"
      << "  let offset_grp = c / c_per_offset_group;\n"
      << "  for (var kh = 0u; kh < kH; kh++) {\n"
      << "    for (var kw = 0u; kw < kW; kw++) {\n"
      << "      let h_in = x_element_t(oh * stride_h + kh * dil_h) - x_element_t(pad_h);\n"
      << "      let w_in = x_element_t(ow * stride_w + kw * dil_w) - x_element_t(pad_w);\n"
      << "      let offset_idx = (offset_grp * kH * kW + kh * kW + kw) * 2u;\n"
      << "      let offset_h = x_element_t(getOffset(batch, offset_idx, oh, ow));\n"
      << "      let offset_w = x_element_t(getOffset(batch, offset_idx + 1u, oh, ow));\n"
      << "      let h_im = h_in + offset_h;\n"
      << "      let w_im = w_in + offset_w;\n"
      << "      var val = bilinearInterpolate(batch, c, h_im, w_im);\n";

  if (has_mask_) {
    shader.MainFunctionBody()
        << "      let mask_idx = offset_grp * kH * kW + kh * kW + kw;\n"
        << "      val = val * x_element_t(getMask(batch, mask_idx, oh, ow));\n";
  }

  shader.MainFunctionBody()
      << "      let w_val = x_element_t(getW(out_channel, c_offset, kh, kw));\n"
      << "      value = value + val * w_val;\n"
      << "    }\n"
      << "  }\n"
      << "}\n";

  if (has_bias_) {
    const auto& bias = shader.AddInput("bias", ShaderUsage::UseUniform);
    shader.MainFunctionBody()
        << "value = value + x_element_t(" << bias.GetByOffset("out_channel") << ");\n";
  }

  shader.MainFunctionBody() << output.SetByOffset("global_idx", "x_value_t(value)");

  return Status::OK();
}

Status DeformConv::ComputeInternal(ComputeContext& context) const {
  const auto* X = context.Input<Tensor>(0);
  const auto* W = context.Input<Tensor>(1);
  const auto* offset = context.Input<Tensor>(2);
  const auto* B = context.Input<Tensor>(3);
  const auto* mask = context.Input<Tensor>(4);

  DeformConvParams params;
  ORT_RETURN_IF_ERROR(DeformConvValidateAndParse(
      attrs_,
      X->Shape(), W->Shape(), offset->Shape(),
      B ? &B->Shape() : nullptr,
      mask ? &mask->Shape() : nullptr,
      params));

  TensorShape output_shape{params.N, params.M, params.out_h, params.out_w};
  auto* output = context.Output(0, output_shape);

  const uint32_t output_size = static_cast<uint32_t>(output_shape.Size());
  if (output_size == 0) {
    return Status::OK();
  }

  const bool has_bias = (B != nullptr);
  const bool has_mask = (mask != nullptr);

  DeformConvProgram program(has_bias, has_mask);
  program.CacheHint(std::to_string(has_bias), std::to_string(has_mask))
      .AddInput({X, ProgramTensorMetadataDependency::TypeAndRank, X->Shape(), 1})
      .AddInput({W, ProgramTensorMetadataDependency::TypeAndRank, W->Shape(), 1})
      .AddInput({offset, ProgramTensorMetadataDependency::TypeAndRank, offset->Shape(), 1})
      .AddOutput({output, ProgramTensorMetadataDependency::TypeAndRank, output_shape, 1})
      .AddUniformVariables({{output_size},
                            {std::vector<uint32_t>{static_cast<uint32_t>(params.kH),
                                                   static_cast<uint32_t>(params.kW)}},
                            {std::vector<uint32_t>{static_cast<uint32_t>(params.pad_h),
                                                   static_cast<uint32_t>(params.pad_w)}},
                            {std::vector<uint32_t>{static_cast<uint32_t>(params.stride_h),
                                                   static_cast<uint32_t>(params.stride_w)}},
                            {std::vector<uint32_t>{static_cast<uint32_t>(params.dilation_h),
                                                   static_cast<uint32_t>(params.dilation_w)}},
                            {std::vector<uint32_t>{static_cast<uint32_t>(params.H),
                                                   static_cast<uint32_t>(params.W_in)}},
                            {std::vector<uint32_t>{static_cast<uint32_t>(params.C),
                                                   static_cast<uint32_t>(params.M),
                                                   static_cast<uint32_t>(params.C / params.group),
                                                   static_cast<uint32_t>(params.M / params.group)}},
                            {std::vector<uint32_t>{static_cast<uint32_t>(params.group),
                                                   static_cast<uint32_t>(params.offset_group)}},
                            {static_cast<uint32_t>(params.C / params.offset_group)}})
      .SetDispatchGroupSize((output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE);

  if (has_mask) {
    program.AddInput({mask, ProgramTensorMetadataDependency::TypeAndRank, mask->Shape(), 1});
  }
  if (has_bias) {
    program.AddInput({B, ProgramTensorMetadataDependency::TypeAndRank, B->Shape(), 1});
  }

  return context.RunProgram(program);
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
