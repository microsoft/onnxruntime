// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/reduction/reduction_ops.h"
#include <sstream>
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

Status ReduceKernelProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& input = shader.AddInput("input", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias);
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias);
  bool reduce_on_all_axes = no_op_with_empty_axes_ == false && axes_.empty();
  std::string loop_header = code_[0];
  std::string loop_body = "let current_element: input_value_t = " + input.GetByIndices("input_indices") + ";\n" + code_[1];
  std::string loop_footer = code_[2];
  const auto input_rank = input.Rank();
  for (size_t i = 0, l = 0; i < input_rank; ++i) {
    if (reduce_on_all_axes || std::find(axes_.begin(), axes_.end(), i) != axes_.end()) {
      if (keepdims_) {
        l++;
      }
      std::stringstream ss;
      std::string index = "i" + std::to_string(i);
      ss << "for (var " << index << " : u32 = 0; " << index << " < " << input.IndicesGet("uniforms.input_shape", i)  << "; " << index << "++) {\n";
      ss << input.IndicesSet("input_indices", i, index) << ";\n";
      ss << loop_body << "\n";
      ss << "}\n";
      loop_body = ss.str();
    } else {
      std::stringstream ss;
      ss << loop_header << "\n";
      std::string index = "i" + std::to_string(i);
      ss << "let " << index << " = " << output.IndicesGet("output_indices", l) << ";\n";
      ss << input.IndicesSet("input_indices", i, index) << ";\n";
      loop_header = ss.str();
      l++;
    }
  }
  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")
                            << "let output_indices: output_indices_t = " << output.OffsetToIndices("global_idx") << ";\n"
                            << "var input_indices: input_indices_t = input_indices_t(0);\n"
                            << loop_header << loop_body << loop_footer;
  shader.MainFunctionBody() << output.SetByOffset("global_idx", "output_value");
  return Status::OK();
}

template <bool allow_multi_axes>
Status ReduceKernel<allow_multi_axes>::ComputeInternal(ComputeContext& context) const {
  const auto* input_tensor = context.Input(0);
  std::vector<int64_t> input_axes;
  // Check if axes input is provided and copy the axes values to input_axes
  if (context.InputCount() > 1) {
    ORT_ENFORCE(axes_.empty(), "Axes attribute may not be specified when axes input is also provided.");
    const Tensor* axes_tensor = context.Input<Tensor>(1);
    auto size = static_cast<size_t>(axes_tensor->Shape()[0]);
    const auto* data = axes_tensor->Data<int64_t>();
    input_axes.resize(size);
    std::copy(data, data + size, input_axes.begin());
  } else {
    input_axes.resize(axes_.size());
    std::copy(axes_.begin(), axes_.end(), input_axes.begin());
  }
  const auto code = GetOpSpecificCode(input_tensor, input_axes);
  // Compute output shape
  std::vector<int64_t> output_shape;
  for (int i = 0; i < input_tensor->Shape().NumDimensions(); ++i) {
    if ((input_axes.empty() && !noop_with_empty_axes_) || std::find(input_axes.begin(), input_axes.end(), i) != input_axes.end()) {
      if (keepdims_) {
        output_shape.push_back(1);
      }
    } else {
      output_shape.push_back(input_tensor->Shape()[i]);
    }
  }
  TensorShape output_tensor_shape(output_shape);
  int64_t output_size = output_tensor_shape.Size();
  ReduceKernelProgram program("ReduceMean", keepdims_, noop_with_empty_axes_, input_axes, code);
  program.AddInput({input_tensor, ProgramTensorMetadataDependency::TypeAndRank})
      .AddOutput({context.Output(0, output_shape), ProgramTensorMetadataDependency::TypeAndRank})
      .SetDispatchGroupSize((output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({{static_cast<uint32_t>(output_size)}});
  return context.RunProgram(program);
}

ReduceOpSpecificCode ReduceMean::GetOpSpecificCode(const Tensor* input_tensor, const std::vector<int64_t>& axes) const {
  const TensorShape& input_shape = input_tensor->Shape();
  size_t input_rank = input_shape.NumDimensions();
  size_t size = 1;
  for (size_t i = 0; i < input_rank; ++i) {
    if ((axes.empty() && !noop_with_empty_axes_) || std::find(axes.begin(), axes.end(), i) != axes.end()) {
      size *= input_shape[i];
    }
  }
  std::stringstream ss;
  ss << "let output_value = output_value_t(sum / f32(" << size << "));";
  ReduceOpSpecificCode code({"var sum = f32(0);", "sum += f32(current_element);", ss.str()});
  return code;
}

Status ReduceMean::ComputeInternal(ComputeContext& ctx) const {
  return ReduceKernel<true>::ComputeInternal(ctx);
}

}  // namespace webgpu
}  // namespace onnxruntime
