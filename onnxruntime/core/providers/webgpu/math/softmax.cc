// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>

#include "core/common/inlined_containers.h"
#include "core/providers/common.h"
#include "core/providers/webgpu/math/softmax.h"
#include "core/providers/webgpu/tensor/transpose.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/webgpu/shader_variable.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/webgpu_utils.h"
namespace onnxruntime {
namespace webgpu {

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Softmax,
    kOnnxDomain,
    1, 10,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedNumberTypes()),
    Softmax);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Softmax,
    kOnnxDomain,
    11, 12,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedNumberTypes()),
    Softmax);

ONNX_OPERATOR_KERNEL_EX(
    Softmax,
    kOnnxDomain,
    13,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedNumberTypes()),
    Softmax);

static std::string MaxVector(const std::string& name, int components) {
  switch (components) {
    case 1:
      return name;
    case 2:
      return "max(" + name + ".x, " + name + ".y)";
    case 3:
      return "max(max(" + name + ".x, " + name + ".y), " + name + ".z)";
    case 4:
      return "max(max(" + name + ".x, " + name + ".y), max(" + name + ".z, " + name + ".w))";
    default:
      ORT_THROW("Unsupported number of components: ", components);
  }
}

Status SoftmaxProgram::GenerateShaderCode(ShaderHelper& shader) const {
  // Add input and output variables
  const auto& input = shader.AddInput("x", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias);
  shader.AddOutput("result", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);
  int components = input.NumComponents();

  const std::string thread_max_decl = is_fp32_
                                          ? "var thread_max = x_value_t(-3.402823e+38f);\n"
                                          : "var thread_max = x_value_t(-65504.0h);\n";

  // Define shared memory for row max and row sum
  shader.AdditionalImplementation()
      << "var<workgroup> row_max_shared : x_value_t;\n"
      << "var<workgroup> row_sum_shared : x_value_t;\n"
      << "var<workgroup> thread_shared : array<x_value_t, " << wg_ << ">;\n";

  // Define helper functions to get and set values
  shader.AdditionalImplementation()
      << "fn getValue(row: i32, col: i32, row_stride: i32) -> x_value_t {\n"
      << "  let index = row * row_stride + col;\n"
      << "  return x[index];\n"
      << "}\n"
      << "fn setValue(row: i32, col: i32, row_stride: i32, value: x_value_t) {\n"
      << "  let index = row * row_stride + col;\n"
      << "  result[index] = value;\n"
      << "}\n";

  // Main function body
  shader.MainFunctionBody()
      << "  let gindex = i32(global_idx);\n"
      << "  let lindex = i32(local_idx);\n"
      << "  const wg = " << wg_ << ";\n"
      << "  let row = gindex / wg;\n"
      << "  let cols = uniforms.packedCols;\n"
      << "  let row_stride : i32 = uniforms.packedCols;\n"

      // Find the row's max value
      << thread_max_decl
      << "  for (var col = lindex; col < cols; col += wg) {\n"
      << "    let value = getValue(row, col, row_stride);\n"
      << "    thread_max = max(thread_max, value);\n"
      << "  }\n"
      << "  if (lindex < cols) {\n"
      << "    thread_shared[lindex] = thread_max;\n"
      << "  }\n"
      << "  workgroupBarrier();\n"

      // Reduce to find the max value
      << "  var reduce_size = min(cols, wg);\n"
      << "  for (var curr_size = reduce_size >> 1; curr_size > 0; curr_size = reduce_size >> 1) {\n"
      << "    reduce_size = curr_size + (reduce_size & 1);\n"
      << "    if (lindex < curr_size) {\n"
      << "      thread_shared[lindex] = max(thread_shared[lindex], thread_shared[lindex + reduce_size]);\n"
      << "    }\n"
      << "    workgroupBarrier();\n"
      << "  }\n"
      << "  if (lindex == 0) {\n"
      << "    row_max_shared = x_value_t(" << MaxVector("thread_shared[0]", components) << ");\n"
      << "  }\n"
      << "  workgroupBarrier();\n"

      // Find the row's sum of exponentials
      << "  var thread_sum = x_value_t(0.0);\n"
      << "  for (var col = lindex; col < cols; col += wg) {\n"
      << "    let sub_exp = exp(getValue(row, col, row_stride) - row_max_shared);\n"
      << "    thread_sum += sub_exp;\n"
      << "  }\n"
      << "  thread_shared[lindex] = thread_sum;\n"
      << "  workgroupBarrier();\n"

      // Reduce to find the sum of exponentials
      << "  for (var curr_size = wg >> 1; curr_size > 0; curr_size = curr_size >> 1) {\n"
      << "    if (lindex < curr_size) {\n"
      << "      thread_shared[lindex] = thread_shared[lindex] + thread_shared[lindex + curr_size];\n"
      << "    }\n"
      << "    workgroupBarrier();\n"
      << "  }\n"
      << "  if (lindex == 0) {\n"
      << "    row_sum_shared = x_value_t(" << SumVector("thread_shared[0]", components) << ");\n"
      << "  }\n"
      << "  workgroupBarrier();\n"

      // Calculate the final value for each element in the row
      << "  for (var col = lindex; col < cols; col += wg) {\n"
      << "    let value = exp(getValue(row, col, row_stride) - row_max_shared) / row_sum_shared;\n"
      << "    setValue(row, col, row_stride, value);\n"
      << "  }\n";

  return Status::OK();
}

Status Softmax::ComputeInternal(ComputeContext& context) const {
  const auto* input_tensor = context.Input(0);
  const TensorShape& input_shape = input_tensor->Shape();
  size_t input_rank = input_shape.NumDimensions();
  auto* output_tensor = context.Output(0, input_shape);

  // normalize axis
  size_t axis = static_cast<size_t>(HandleNegativeAxis(axis_, input_rank));
  // The `axis` attribute of the opset lower than version 13 describes the axis of the inputs when coerced to 2D,
  // the 0th axis most likely describes the batch_size, so transpose is not required on old opset versions.
  bool is_transpose_required = axis < input_rank - 1 && opset_ >= 13;

  TensorShape transposed_input_shape;
  Tensor transposed_input_tensor;
  Tensor intermediate_output;
  InlinedVector<size_t> perm(input_rank);

  if (is_transpose_required) {
    std::iota(std::begin(perm), std::end(perm), 0);
    perm[axis] = input_rank - 1;
    perm[input_rank - 1] = axis;

    TensorShapeVector transposed_input_dims;
    for (auto e : perm) {
      transposed_input_dims.push_back(input_shape[e]);
    }

    transposed_input_shape = TensorShape(transposed_input_dims);
    transposed_input_tensor = context.CreateGPUTensor(input_tensor->DataType(), transposed_input_shape);
    ORT_RETURN_IF_ERROR(Transpose::DoTranspose(context, perm, *input_tensor, transposed_input_tensor));
    intermediate_output = context.CreateGPUTensor(output_tensor->DataType(), transposed_input_shape);
  }

  // The `axis` attribute of the opset lower than version 13 separates input tensor's dimensions into two parts,
  // one part is treated as batch size, and the other part is performed by Softmax.
  const int64_t cols = is_transpose_required ? transposed_input_shape[input_rank - 1] : (opset_ >= 13 ? input_shape[input_rank - 1] : input_shape.SizeFromDimension(axis));
  const int64_t rows = input_shape.Size() / cols;
  const int64_t components = GetMaxComponents(cols);
  const auto packed_cols = cols / components;
  uint32_t workgroup_size = rows == 1 ? 256 : 64;
  // check input tensor element type is float
  const bool is_fp32 = input_tensor->GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;

  SoftmaxProgram program{workgroup_size, is_fp32};
  if (is_transpose_required) {
    program
        .AddInputs({{&transposed_input_tensor, ProgramTensorMetadataDependency::TypeAndRank, static_cast<int>(components)}})
        .AddOutputs({{&intermediate_output, ProgramTensorMetadataDependency::TypeAndRank, static_cast<int>(components)}});
  } else {
    program
        .AddInputs({{input_tensor, ProgramTensorMetadataDependency::TypeAndRank, static_cast<int>(components)}})
        .AddOutputs({{output_tensor, ProgramTensorMetadataDependency::TypeAndRank, static_cast<int>(components)}});
  }

  program
      .CacheHint(std::to_string(components), std::to_string(workgroup_size))
      .SetWorkgroupSize(workgroup_size)
      .SetDispatchGroupSize(static_cast<uint32_t>(rows))
      .AddUniformVariables({{static_cast<int32_t>(packed_cols)}});

  ORT_RETURN_IF_ERROR(context.RunProgram(program));

  // If transpose was required, transpose the result back
  if (is_transpose_required) {
    ORT_RETURN_IF_ERROR(Transpose::DoTranspose(context, perm, intermediate_output, *output_tensor));
  }

  return Status::OK();
}
}  // namespace webgpu
}  // namespace onnxruntime
