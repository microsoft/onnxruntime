// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/vendor/intel/contrib/format_transform.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/string_macros.h"
#include "core/common/narrow.h"

namespace onnxruntime {
namespace webgpu {
namespace intel {

namespace {
std::string GetFormatName(BlockedFormat format) {
  switch (format) {
    case BlockedFormat::Plain:
      return "Plain";
    case BlockedFormat::nChw4c:
      return "nChw4c";
    case BlockedFormat::ABcd16a4b:
      return "ABcd16a4b";
    default:
      return "Unknown";
  }
}

BlockedFormat ParseFormat(const std::string& format_str) {
  if (format_str == "Plain" || format_str == "NCHW") {
    return BlockedFormat::Plain;
  } else if (format_str == "nChw4c") {
    return BlockedFormat::nChw4c;
  } else if (format_str == "ABcd16a4b") {
    return BlockedFormat::ABcd16a4b;
  } else {
    ORT_THROW("Unsupported format: ", format_str);
  }
}
}  // namespace

FormatTransformProgram::FormatTransformProgram(BlockedFormat src_format, BlockedFormat dst_format,
                                               const TensorShape& input_shape)
    : Program{"FormatTransform"},
      src_format_(src_format),
      dst_format_(dst_format),
      input_shape_(input_shape) {
}

Status FormatTransformProgram::GenerateShaderCode(ShaderHelper& sh) const {
  const auto& input = sh.AddInput("input", ShaderUsage::UseUniform | ShaderUsage::UseShapeAndStride);
  const auto& output = sh.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseShapeAndStride | ShaderUsage::UseValueTypeAlias);

  auto rank = input_shape_.NumDimensions();
  ORT_RETURN_IF_NOT(rank == 4, "FormatTransform currently only supports 4D tensors (NCHW)");

  const int src_format_val = static_cast<int>(src_format_);
  const int dst_format_val = static_cast<int>(dst_format_);

  return WGSL_TEMPLATE_APPLY(sh, "vendor/intel/contrib/format_transform.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(dst_format, dst_format_val),
                             WGSL_TEMPLATE_PARAMETER(src_format, src_format_val),
                             WGSL_TEMPLATE_VARIABLE(input, input),
                             WGSL_TEMPLATE_VARIABLE(output, output));
}

FormatTransform::FormatTransform(const OpKernelInfo& info)
    : WebGpuKernel(info) {
  std::string src_format_str = info.GetAttrOrDefault<std::string>("src_format", "Plain");
  std::string dst_format_str = info.GetAttrOrDefault<std::string>("dst_format", "Plain");

  src_format_ = ParseFormat(src_format_str);
  dst_format_ = ParseFormat(dst_format_str);
}

Status FormatTransform::ComputeInternal(ComputeContext& context) const {
  const auto* input = context.Input<Tensor>(0);
  const auto& input_shape = input->Shape();

  ORT_RETURN_IF_NOT(input_shape.NumDimensions() == 4, "FormatTransform only supports 4D tensors");

  // Calculate output shape with padding if needed for blocked formats
  TensorShape output_shape = input_shape;

  if (dst_format_ == BlockedFormat::nChw4c) {
    // For nChw4c, pad channels to multiple of 4
    int64_t C = input_shape[1];
    int64_t padded_C = ((C + 3) / 4) * 4;  // Round up to multiple of 4
    output_shape = TensorShape({input_shape[0], padded_C, input_shape[2], input_shape[3]});
  } else if (dst_format_ == BlockedFormat::ABcd16a4b) {
    // For ABcd16a4b, pad N to multiple of 16 and C to multiple of 4
    int64_t N = input_shape[0];
    int64_t C = input_shape[1];
    int64_t padded_N = ((N + 15) / 16) * 16;  // Round up to multiple of 16
    int64_t padded_C = ((C + 3) / 4) * 4;     // Round up to multiple of 4
    output_shape = TensorShape({padded_N, padded_C, input_shape[2], input_shape[3]});
  }
  // For Plain output format, no padding needed (output_shape remains input_shape)

  auto* output = context.Output(0, output_shape);

  FormatTransformProgram program{src_format_, dst_format_, input_shape};
  program
      .AddInput({input, ProgramTensorMetadataDependency::TypeAndRank})
      .AddOutput({output, ProgramTensorMetadataDependency::None})
      .SetDispatchGroupSize((static_cast<uint32_t>(output_shape.Size()) + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .CacheHint(static_cast<int>(src_format_), static_cast<int>(dst_format_))
      .AddUniformVariables({{static_cast<uint32_t>(output_shape.Size())}});

  return context.RunProgram(program);
}

}  // namespace intel
}  // namespace webgpu
}  // namespace onnxruntime
