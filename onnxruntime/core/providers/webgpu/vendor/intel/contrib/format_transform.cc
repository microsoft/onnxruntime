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

Status GetOutputShapeForFormat(const TensorShape& input_shape,
                               BlockedFormat dst_format,
                               TensorShape& output_shape) {
  ORT_RETURN_IF_NOT(input_shape.NumDimensions() == 4, "FormatTransform only supports 4D tensors");

  const auto& dims = input_shape.GetDims();
  const int64_t N = dims[0];
  const int64_t C = dims[1];
  const int64_t H = dims[2];
  const int64_t W = dims[3];

  switch (dst_format) {
    case BlockedFormat::Plain:
      output_shape = input_shape;
      break;
    case BlockedFormat::nChw4c: {
      const int64_t padded_C = ((C + 3) / 4) * 4;
      output_shape = TensorShape({N, padded_C, H, W});
      break;
    }
    case BlockedFormat::ABcd16a4b: {
      const int64_t padded_N = ((N + 15) / 16) * 16;
      const int64_t padded_C = ((C + 3) / 4) * 4;
      output_shape = TensorShape({padded_N, padded_C, H, W});
      break;
    }
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Unsupported destination format: ", static_cast<int>(dst_format));
  }

  return Status::OK();
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

  TensorShape output_shape;
  ORT_RETURN_IF_ERROR(GetOutputShapeForFormat(input_shape, dst_format_, output_shape));

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

/*static*/ Status FormatTransform::TransformPlainToBlocked(ComputeContextBase& context,
                                                           const Tensor& input,
                                                           BlockedFormat dst_format,
                                                           AllocatorPtr alloc,
                                                           std::unique_ptr<Tensor>& output) {
  ORT_RETURN_IF(dst_format == BlockedFormat::Plain,
                "TransformPlainToBlocked should only be used for blocked destinations.");
  ORT_RETURN_IF(alloc == nullptr, "Allocator must not be null.");

  TensorShape output_shape;
  ORT_RETURN_IF_ERROR(GetOutputShapeForFormat(input.Shape(), dst_format, output_shape));

  auto transformed = std::make_unique<Tensor>(input.DataType(), output_shape, alloc);

  FormatTransformProgram program{BlockedFormat::Plain, dst_format, input.Shape()};
  program
      .AddInput({&input, ProgramTensorMetadataDependency::TypeAndRank})
      .AddOutput({transformed.get(), ProgramTensorMetadataDependency::None})
      .SetDispatchGroupSize((static_cast<uint32_t>(output_shape.Size()) + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .CacheHint(static_cast<int>(BlockedFormat::Plain), static_cast<int>(dst_format))
      .AddUniformVariables({{static_cast<uint32_t>(output_shape.Size())}});

  ORT_RETURN_IF_ERROR(context.RunProgram(program));

  output = std::move(transformed);
  return Status::OK();
}

}  // namespace intel
}  // namespace webgpu
}  // namespace onnxruntime
