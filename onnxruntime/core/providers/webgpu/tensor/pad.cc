// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/tensor/pad.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_common.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace webgpu {

template <typename T>
Status PadProgram<T>::GenerateShaderCode(ShaderHelper& shader) const {
  if (!dim_value_zero_) {
    shader.AddInput("data", ShaderUsage::UseUniform | ShaderUsage::UseShapeAndStride);
  }
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseShapeAndStride);

  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size");
  if (dim_value_zero_) {
    // Only Constant mode needs fill output if the one dim value or mores dims' values of input are zero.
    shader.MainFunctionBody() << "output[global_idx] = uniforms.constant_value;\n";
    return Status::OK();
  }

  shader.MainFunctionBody() << "  let output_indices = " << output.OffsetToIndices("global_idx") << ";\n"
                            << "  var input_index = u32(0);\n"
                            << "  var use_pad_value = false;\n"
                            << "  var in_coord = i32(0);\n";

  std::string shapeDimStr = output.Rank() == 1 ? "" : "[dim]";
  std::string strideDimStr = output.Rank() < 3 ? "" : "[dim]";
  std::string begin_axis_statement, end_axis_statement;
  std::string in_axis_statement = "in_coord = i32(output_indices" + shapeDimStr + ") - uniforms.lower_pads" +
                                  shapeDimStr + ";\n";
  switch (mode_) {
    case Mode::Constant:
      begin_axis_statement = "use_pad_value = true;\n";
      end_axis_statement = "use_pad_value = true;\n";
      break;
    case Mode::Edge:
      begin_axis_statement = "in_coord = 0;\n";
      end_axis_statement = "in_coord = i32(uniforms.data_shape" + shapeDimStr + ") - 1;\n";
      break;
    case Mode::Reflect:
      begin_axis_statement = "in_coord = uniforms.lower_pads" + shapeDimStr + " - i32(output_indices" +
                             shapeDimStr + ");\n";
      end_axis_statement = "in_coord = i32(uniforms.data_shape" + shapeDimStr + ") - 2 - (i32(output_indices" +
                           shapeDimStr + ") - (uniforms.lower_pads" + shapeDimStr + " + i32(uniforms.data_shape" +
                           shapeDimStr + ")));\n";
      break;
    case Mode::Wrap:
      begin_axis_statement = "in_coord = i32(uniforms.data_shape" + shapeDimStr + " + output_indices" +
                             shapeDimStr + ") - uniforms.lower_pads" + shapeDimStr + ";\n";
      end_axis_statement = "in_coord = i32(output_indices" + shapeDimStr + ") - uniforms.lower_pads" +
                           shapeDimStr + " - i32(uniforms.data_shape" + shapeDimStr + ");\n";
      break;
    default:
      break;
  }

  std::string input_index_statement = output.Rank() < 2 ? "" : "    if (dim + 1 < " + std::to_string(output.Rank()) + ") {\n" + "      input_index += uniforms.data_stride" + strideDimStr + " * u32(in_coord);\n" + "    }\n";
  shader.MainFunctionBody() << "  for (var dim = 0; dim < " << output.Rank() << " && !use_pad_value; dim++) {\n"
                            << "    if (i32(output_indices" << shapeDimStr << ") < uniforms.lower_pads" << shapeDimStr << ") {\n"
                            << "      " << begin_axis_statement << "    }\n"
                            << "    else if (i32(output_indices" << shapeDimStr << ") >= uniforms.lower_pads"
                            << shapeDimStr << " + i32(uniforms.data_shape" << shapeDimStr << ")) {\n"
                            << "      " << end_axis_statement << "    }\n"
                            << "    else {\n"
                            << "      " << in_axis_statement << "    }\n"
                            << input_index_statement
                            << "  }\n"
                            << "  input_index += u32(in_coord);\n"
                            << "  output[global_idx] = select(data[input_index], uniforms.constant_value, use_pad_value);\n";

  return Status::OK();
}

template <typename T>
typename ToWebGpuType<T>::MappedType ToWebGpuValue(const T& value) {
  return value;
}

template <>
typename ToWebGpuType<MLFloat16>::MappedType ToWebGpuValue<MLFloat16>(const MLFloat16& value) {
  return *reinterpret_cast<const typename ToWebGpuType<MLFloat16>::MappedType*>(&value.val);
}

template <typename T>
Status Pad<T>::ComputeInternal(ComputeContext& context) const {
  typedef typename ToWebGpuType<T>::MappedType WebGpuT;
  const Tensor* input_tensor = context.Input<Tensor>(0);
  auto const& input_shape = input_tensor->Shape();
  int32_t dimension_count = static_cast<int32_t>(input_shape.NumDimensions());

  const PadsVector* p_pads = &pads_;
  const PadsVector* p_slices = &slices_;
  WebGpuT value = ToWebGpuType<T>::FromFloat(value_);

  PadsVector pads;
  PadsVector slices;
  // kOnnxDomain Pad opset >= 11 (Or) kMsDomain opset == 1
  if (is_dynamic_) {
    size_t data_rank = input_tensor->Shape().NumDimensions();

    const Tensor* pads_tensor = context.Input<Tensor>(1);
    auto pads_tensor_dims = pads_tensor->Shape().GetDims();
    ORT_ENFORCE(pads_tensor_dims.size() == 1 || (pads_tensor_dims.size() == 2 && pads_tensor_dims[0] == 1),
                "Pads tensor should be a 1D tensor of shape [2 * num_axes] "
                "or a 2D tensor of shape [1, 2 * num_axes]");

    const auto pads_data = pads_tensor->DataAsSpan<int64_t>();

    // Compute Pads by applying axes if specified otherwise copy the supplied pads.
    PadBase::ComputePads(context.KernelContext(), data_rank, pads_data, pads);

    // Separate out any negative pads into the slices array
    PadBase::SeparateNegativeToSlices(pads, slices);

    T raw_value{};
    const Tensor* value_tensor = context.Input<Tensor>(2);
    if (nullptr != value_tensor) {
      ORT_ENFORCE(utils::IsPrimitiveDataType<T>(value_tensor->DataType()) &&
                      value_tensor->Shape().Size() == 1,
                  "Value tensor should be a 1D tensor of size 1 with the same type as that of the input tensor");
      raw_value = value_tensor->Data<T>()[0];
      value = ToWebGpuValue<T>(raw_value);
    }
    p_pads = &pads;
    p_slices = &slices;
  }

  auto output_dims(input_shape.AsShapeVector());
  ORT_ENFORCE(static_cast<size_t>(dimension_count) * 2 == p_pads->size(), "'pads' attribute has wrong number of values");

  // Calculate output dimensions, and handle any negative padding
  std::vector<int32_t> lower_pads(dimension_count);
  for (auto i = 0; i < dimension_count; i++) {
    int64_t lower_pad = (*p_pads)[i] + (*p_slices)[i];
    int64_t upper_pad = (*p_pads)[i + dimension_count] + (*p_slices)[i + dimension_count];
    lower_pads[i] = static_cast<int32_t>(lower_pad);
    output_dims[i] += lower_pad + upper_pad;
  }
  TensorShape output_shape(output_dims);

  // special case when there is a dim value of 0 in the shape. behavior depends on mode
  bool dim_value_zero = input_shape.Size() == 0;
  if (dim_value_zero) {
    ORT_RETURN_IF_ERROR(PadBase::HandleDimValueZero(mode_, input_shape, output_shape));
  }

  auto* output_tensor = context.Output(0, output_shape);
  uint32_t output_size = gsl::narrow<uint32_t>(output_shape.Size());
  if (output_size == 0) {
    // Do not need to fill output, return
    return Status::OK();
  }

  PadProgram<T> program{mode_, dim_value_zero};
  if (!dim_value_zero) {
    program.AddInput({input_tensor, ProgramTensorMetadataDependency::TypeAndRank});
  }
  program.AddOutput({output_tensor, ProgramTensorMetadataDependency::Rank})
      .SetDispatchGroupSize((output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .CacheHint(std::to_string(static_cast<int>(mode_)), dim_value_zero)
      .AddUniformVariables({{gsl::span<const int32_t>(lower_pads.data(), lower_pads.size())}, {output_size}, {value}});

  return context.RunProgram(program);
}

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                        \
      Pad,                                                        \
      kOnnxDomain,                                                \
      2, 10,                                                      \
      T,                                                          \
      kWebGpuExecutionProvider,                                   \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Pad<T>);                                                    \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                        \
      Pad,                                                        \
      kOnnxDomain,                                                \
      11, 12,                                                     \
      T,                                                          \
      kWebGpuExecutionProvider,                                   \
      (*KernelDefBuilder::Create())                               \
          .InputMemoryType(OrtMemTypeCPUInput, 1)                 \
          .InputMemoryType(OrtMemTypeCPUInput, 2)                 \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Pad<T>);                                                    \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                        \
      Pad,                                                        \
      kOnnxDomain,                                                \
      13, 17,                                                     \
      T,                                                          \
      kWebGpuExecutionProvider,                                   \
      (*KernelDefBuilder::Create())                               \
          .InputMemoryType(OrtMemTypeCPUInput, 1)                 \
          .InputMemoryType(OrtMemTypeCPUInput, 2)                 \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Pad<T>);                                                    \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                        \
      Pad,                                                        \
      kOnnxDomain,                                                \
      18, 18,                                                     \
      T,                                                          \
      kWebGpuExecutionProvider,                                   \
      (*KernelDefBuilder::Create())                               \
          .InputMemoryType(OrtMemTypeCPUInput, 1)                 \
          .InputMemoryType(OrtMemTypeCPUInput, 2)                 \
          .InputMemoryType(OrtMemTypeCPUInput, 3)                 \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Pad<T>);                                                    \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                        \
      Pad,                                                        \
      kOnnxDomain,                                                \
      19, 20,                                                     \
      T,                                                          \
      kWebGpuExecutionProvider,                                   \
      (*KernelDefBuilder::Create())                               \
          .InputMemoryType(OrtMemTypeCPUInput, 1)                 \
          .InputMemoryType(OrtMemTypeCPUInput, 2)                 \
          .InputMemoryType(OrtMemTypeCPUInput, 3)                 \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Pad<T>);                                                    \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                        \
      Pad,                                                        \
      kOnnxDomain,                                                \
      21, 22,                                                     \
      T,                                                          \
      kWebGpuExecutionProvider,                                   \
      (*KernelDefBuilder::Create())                               \
          .InputMemoryType(OrtMemTypeCPUInput, 1)                 \
          .InputMemoryType(OrtMemTypeCPUInput, 2)                 \
          .InputMemoryType(OrtMemTypeCPUInput, 3)                 \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Pad<T>);                                                    \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      Pad,                                                        \
      kOnnxDomain,                                                \
      23,                                                         \
      T,                                                          \
      kWebGpuExecutionProvider,                                   \
      (*KernelDefBuilder::Create())                               \
          .InputMemoryType(OrtMemTypeCPUInput, 1)                 \
          .InputMemoryType(OrtMemTypeCPUInput, 2)                 \
          .InputMemoryType(OrtMemTypeCPUInput, 3)                 \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Pad<T>);

#define SPECIALIZED_COMPUTE(T) \
  REGISTER_KERNEL_TYPED(T)     \
  template Status Pad<T>::ComputeInternal(ComputeContext& context) const;

SPECIALIZED_COMPUTE(float)
SPECIALIZED_COMPUTE(MLFloat16)
SPECIALIZED_COMPUTE(uint32_t)
SPECIALIZED_COMPUTE(int32_t)

}  // namespace webgpu
}  // namespace onnxruntime
