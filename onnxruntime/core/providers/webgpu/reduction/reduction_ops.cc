// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/reduction/reduction_ops.h"
#include <sstream>
#include "core/framework/data_transfer_manager.h"
#include "core/providers/webgpu/data_transfer.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace webgpu {

#define REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ReduceOp, begin, end)              \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(                                                   \
      ReduceOp,                                                                        \
      kOnnxDomain,                                                                     \
      begin, end,                                                                      \
      kWebGpuExecutionProvider,                                                        \
      (*KernelDefBuilder::Create()).TypeConstraint("T", WebGpuSupportedNumberTypes()), \
      ReduceOp);

#define REGISTER_UNARY_ELEMENTWISE_KERNEL(ReduceOp, version)                                                                  \
  ONNX_OPERATOR_KERNEL_EX(                                                                                                    \
      ReduceOp,                                                                                                               \
      kOnnxDomain,                                                                                                            \
      version,                                                                                                                \
      kWebGpuExecutionProvider,                                                                                               \
      (*KernelDefBuilder::Create()).TypeConstraint("T", WebGpuSupportedNumberTypes()).InputMemoryType(OrtMemTypeCPUInput, 1), \
      ReduceOp);

REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ReduceMean, 1, 10);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ReduceMean, 11, 12);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ReduceMean, 13, 17);
REGISTER_UNARY_ELEMENTWISE_KERNEL(ReduceMean, 18);

REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ReduceMax, 1, 10);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ReduceMax, 11, 11);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ReduceMax, 12, 12);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ReduceMax, 13, 17);
REGISTER_UNARY_ELEMENTWISE_KERNEL(ReduceMax, 18);

REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ReduceSum, 1, 10);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ReduceSum, 11, 12);
REGISTER_UNARY_ELEMENTWISE_KERNEL(ReduceSum, 13);

Status ReduceKernelProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias);
  if (is_input_empty_) {
    shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")
                              << code_[0]
                              << code_[2]
                              << output.SetByOffset("global_idx", "output_value");
    return Status::OK();
  }
  const auto& input = shader.AddInput("input", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias);
  bool reduce_on_all_axes = no_op_with_empty_axes_ == false && axes_.empty();
  std::string loop_header = code_[0].find("first_element") == std::string::npos ? code_[0] : "let first_element = " + input.GetByIndices("input_indices") + ";\n" + code_[0] + "\n";
  std::string loop_body = "let current_element: input_value_t = " + input.GetByIndices("input_indices") + ";\n" + code_[1];
  std::string loop_footer = code_[2];
  const auto input_rank = input.Rank();
  for (int i = 0, l = 0; i < input_rank; ++i) {
    if (reduce_on_all_axes || std::find(axes_.begin(), axes_.end(), i) != axes_.end()) {
      if (keepdims_) {
        l++;
      }
      std::stringstream ss;
      std::string index = "i" + std::to_string(i);
      ss << "for (var " << index << " : u32 = 0; " << index << " < " << input.IndicesGet("uniforms.input_shape", i) << "; " << index << "++) {\n";
      ss << input.IndicesSet("input_indices", i, index) << ";\n";
      ss << loop_body << "\n";
      ss << "}\n";
      loop_body = ss.str();
    } else {
      std::stringstream ss;
      std::string index = "i" + std::to_string(i);
      ss << "let " << index << " = " << output.IndicesGet("output_indices", l) << ";\n";
      ss << input.IndicesSet("input_indices", i, index) << ";\n";
      ss << loop_header << "\n";
      loop_header = ss.str();
      l++;
    }
  }
  std::stringstream input_indices_init_value;
  for (int i = 0; i < input_rank - 1; ++i) {
    input_indices_init_value << "0, ";
  }
  input_indices_init_value << "0";
  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")
                            << "let output_indices: output_indices_t = " << output.OffsetToIndices("global_idx") << ";\n"
                            << "var input_indices: input_indices_t = input_indices_t(" << input_indices_init_value.str() << ");\n"
                            << loop_header << loop_body << loop_footer;
  shader.MainFunctionBody() << output.SetByOffset("global_idx", "output_value");
  return Status::OK();
}

template <bool allow_multi_axes>
Status ReduceKernel<allow_multi_axes>::ComputeInternal(ComputeContext& context) const {
  const auto* input_tensor = context.Input(0);
  ORT_RETURN_IF_ERROR(CheckInput(input_tensor));
  InlinedVector<uint32_t> input_axes;
  auto rank = input_tensor->Shape().NumDimensions();
  auto transform_axis = [rank](int64_t axis) {
    if (axis < 0) {
      axis += rank;
    }
    if (axis < 0 || static_cast<size_t>(axis) >= rank) {
      ORT_THROW("Axes values must be in the range [-rank, rank-1]. Got: ", axis);
    }
    return static_cast<uint32_t>(axis);
  };
  // Check if axes input is provided and copy the axes values to input_axes
  if (context.InputCount() > 1) {
    ORT_ENFORCE(axes_.empty(), "Axes attribute may not be specified when axes input is also provided.");
    const Tensor* axes_tensor = context.Input<Tensor>(1);
    if (nullptr != axes_tensor) {
      auto size = static_cast<size_t>(axes_tensor->Shape()[0]);
      const auto* data = axes_tensor->Data<int64_t>();
      input_axes.reserve(size);
      std::transform(data, data + size, std::back_inserter(input_axes), transform_axis);
    }
  } else {
    input_axes.reserve(axes_.size());
    std::transform(axes_.begin(), axes_.end(), std::back_inserter(input_axes), transform_axis);
  }
  if (input_axes.empty()) {
    if (noop_with_empty_axes_ || rank == 0) {
      // If axes is empty and noop_with_empty_axes_ is true, it is a no-op according to the spec
      // If input tensor is a scalar, return the input tensor as is.
      // This is not correct for ReduceLogSum and ReduceSumSquare
      // TODO handle these cases separately.
      auto output = context.Output(0, input_tensor->Shape());
      if (output->DataRaw() != input_tensor->DataRaw()) {
        ORT_RETURN_IF_ERROR(Info().GetDataTransferManager().CopyTensor(*input_tensor, *output));
      }
      return Status::OK();
    } else {
      // If axes is empty and noop_with_empty_axes_ is false, it is a reduction over all axes
      input_axes.resize(rank);
      std::iota(input_axes.begin(), input_axes.end(), 0);
    }
  }
  const auto code = GetOpSpecificCode(input_tensor);
  // Compute output shape
  std::vector<int64_t> output_shape;
  bool is_input_empty = false;
  for (size_t i = 0; i < input_tensor->Shape().NumDimensions(); ++i) {
    is_input_empty |= input_tensor->Shape()[i] == 0;
    if (std::find(input_axes.begin(), input_axes.end(), i) != input_axes.end()) {
      if (keepdims_) {
        output_shape.push_back(1);
      }
    } else {
      output_shape.push_back(input_tensor->Shape()[i]);
    }
  }
  TensorShape output_tensor_shape(output_shape);
  int64_t output_size = output_tensor_shape.Size();
  if (output_size == 0) {
    ORT_IGNORE_RETURN_VALUE(context.Output(0, output_tensor_shape));
    return Status::OK();
  }

  auto input_rank = input_tensor->Shape().NumDimensions();
  // reduce_axes element is either 1 or 0 depending on whether the axis is reduced or not
  std::vector<uint32_t> reduce_axes;
  reduce_axes.resize(input_rank, 0);
  for (auto axis : input_axes) {
    reduce_axes[axis] = 1;
  }

  ReduceKernelProgram program(name_, keepdims_, noop_with_empty_axes_, input_axes, code, is_input_empty);
  if (!is_input_empty) {
    program.AddInput({input_tensor, ProgramTensorMetadataDependency::TypeAndRank});
  }

  program.CacheHint(is_input_empty)
      .AddOutput({context.Output(0, output_shape), ProgramTensorMetadataDependency::TypeAndRank})
      .SetDispatchGroupSize((output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({{static_cast<uint32_t>(output_size)},
                            {static_cast<uint32_t>(noop_with_empty_axes_ ? 1 : 0)},
                            {reduce_axes}});

  return context.RunProgram(program);
}

ReduceOpSpecificCode ReduceMean::GetOpSpecificCode(const Tensor* input_tensor) const {
  const TensorShape& input_shape = input_tensor->Shape();
  size_t input_rank = input_shape.NumDimensions();
  std::string loop_header = "var sum = f32(0);";
  std::string loop_body = "sum += f32(current_element);";
  std::stringstream ss;
  ss << "var size: u32 = 1;\n"
     << "for (var i: u32 = 0; i < " << input_rank << "; i += 1) { \n"
     << "  let index_reduced_or_not = " << GetElementAt("uniforms.reduce_axes", "i", input_rank) << ";\n"
     << "  if (index_reduced_or_not == 1) { \n"
     << "    size = size * " << GetElementAt("uniforms.input_shape", "i", input_rank) << ";\n"
     << "  }\n"
     << "}\n"
     << "let output_value = output_value_t(sum / f32(size));";
  std::string loop_footer = ss.str();
  ReduceOpSpecificCode code({loop_header, loop_body, loop_footer});
  return code;
}

ReduceOpSpecificCode ReduceMax::GetOpSpecificCode(const Tensor* input_tensor) const {
  ORT_UNUSED_PARAMETER(input_tensor);
  std::string loop_header = "var max_element = first_element;";
  std::string loop_body = "max_element = max(max_element, current_element);";
  std::string loop_footer = "let output_value = output_value_t(max_element);";
  ReduceOpSpecificCode code({loop_header, loop_body, loop_footer});
  return code;
}
ReduceOpSpecificCode ReduceSum::GetOpSpecificCode(const Tensor* input_tensor) const {
  ORT_UNUSED_PARAMETER(input_tensor);
  std::string loop_header = "var sum = f32(0);";
  std::string loop_body = "sum += f32(current_element);";
  std::string loop_footer = "let output_value = output_value_t(sum);";
  ReduceOpSpecificCode code({loop_header, loop_body, loop_footer});
  return code;
}

}  // namespace webgpu
}  // namespace onnxruntime
