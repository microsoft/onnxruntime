// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/inlined_containers.h"
#include "core/providers/webgpu/tensor/softmax.h"
#include "core/providers/webgpu/tensor/transpose.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/webgpu/shader_variable.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

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

static std::string MaxVector(std::string name, int components) {
  switch (components) {
    case 1:
      return name;
    case 2:
      return "max(" + name + ".x, " + name + ".y)";
    case 4:
      return "max(max(" + name + ".x, " + name + ".y), max(" + name + ".z, " + name + ".w))";
    default:
      ORT_THROW("Unsupported number of components: ", components);
  }
}

static std::string SumVector(std::string x, int components) {
  switch (components) {
    case 1:
      return x;
    case 2:
      return "(" + x + ".x + " + x + ".y" + ")";
    case 4:
      return "(" + x + ".x + " + x + ".y + " + x + ".w + " + x + ".z" + ")";
    default:
      ORT_THROW("Unsupported number of components: ", components);
  }
}

static int GetMaxComponents(int64_t size) {
  if (size % 4 == 0) {
    return 4;
  } else if (size % 2 == 0) {
    return 2;
  }
  return 1;
}

Status SoftmaxProgram::GenerateShaderCode(ShaderHelper& shader) const {
  // Add input and output variables
  const auto& input = shader.AddInput("x", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);
  const auto& output = shader.AddOutput("result", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);
  int components = input.NumComponents();

  std::string threadMaxDecl =  input.StorageType() == "f32" ?
                                "val threadMax = x_value_t(-3.402823e+38f);\n" :
                                "val threadMax = x_value_t(-65504.0h));\n";


  // Define shared memory for row max and row sum
  shader.AdditionalImplementation()
      << "var<workgroup> rowMaxShared : x_value_t;\n"
      << "var<workgroup> rowSumShared : x_value_t;\n"
      << "var<workgroup> threadShared : array<x_value_t, " << WG << ">;\n";

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
      << "  const wg = " << WG << ";\n"
      << "  let row = gindex / wg;\n"
      << "  let cols = uniforms.packedCols;\n"
      << "  let row_stride : i32 = uniforms.packedCols;\n"

      // Find the row's max value
      << threadMaxDecl
      << "  for (var col = lindex; col < cols; col += wg) {\n"
      << "    let value = getValue(row, col, row_stride);\n"
      << "    threadMax = max(threadMax, value);\n"
      << "  }\n"
      << "  if (lindex < cols) {\n"
      << "    threadShared[lindex] = threadMax;\n"
      << "  }\n"
      << "  workgroupBarrier();\n"

      // Reduce to find the max value
      << "  var reduceSize = min(cols, wg);\n"
      << "  for (var currSize = reduceSize >> 1; currSize > 0; currSize = reduceSize >> 1) {\n"
      << "    reduceSize = currSize + (reduceSize & 1);\n"
      << "    if (lindex < currSize) {\n"
      << "      threadShared[lindex] = max(threadShared[lindex], threadShared[lindex + reduceSize]);\n"
      << "    }\n"
      << "    workgroupBarrier();\n"
      << "  }\n"
      << "  if (lindex == 0) {\n"
      << "    rowMaxShared = x_value_t(" <<  MaxVector('threadShared[0]', components)   << ");\n"
      << "  }\n"
      << "  workgroupBarrier();\n"

      // Find the row's sum of exponentials
      << "  var threadSum = x_value_t(0.0);\n"
      << "  for (var col = lindex; col < cols; col += wg) {\n"
      << "    let subExp = exp(getValue(row, col, row_stride) - rowMaxShared);\n"
      << "    threadSum += subExp;\n"
      << "  }\n"
      << "  threadShared[lindex] = threadSum;\n"
      << "  workgroupBarrier();\n"

      // Reduce to find the sum of exponentials
      << "  for (var currSize = wg >> 1; currSize > 0; currSize = currSize >> 1) {\n"
      << "    if (lindex < currSize) {\n"
      << "      threadShared[lindex] = threadShared[lindex] + threadShared[lindex + currSize];\n"
      << "    }\n"
      << "    workgroupBarrier();\n"
      << "  }\n"
      << "  if (lindex == 0) {\n"
      << "    rowSumShared = x_value_t(" << SumVector("threadShared[0]", components)  << ");\n"
      << "  }\n"
      << "  workgroupBarrier();\n"

      // Calculate the final value for each element in the row
      << "  for (var col = lindex; col < cols; col += wg) {\n"
      << "    let value = exp(getValue(row, col, row_stride) - rowMaxShared) / rowSumShared;\n"
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
  int64_t axis = axis < 0 ? axis_ + input_rank : axis_;

  bool is_transpose_required = axis < input_rank - 1;
  TensorShape transposed_input_shape = input_shape;
  Tensor transposed_input_tensor;
  Tensor intermediate_output;
  InlinedVector<size_t> perm;

  if (is_transpose_required) {
    AllocatorPtr alloc;
    perm.reserve(input_rank);
    for (size_t i = 0; i < input_rank; ++i) {
      perm[i] = i;
    }
    perm[axis] = input_rank - 1;
    perm[input_rank - 1] = axis;

    // allocate a temporary tensor to hold transposed input
    Tensor temp_input(input_tensor->DataType(), TensorShape(transposed_input_shape), alloc);

    ORT_RETURN_IF_ERROR(Transpose::DoTranspose( perm, *input_tensor, temp_input));
    transposed_input_tensor = std::move(temp_input);
    transposed_input_shape = transposed_input_tensor.Shape();

    // Allocate memory for the intermediate output
    Tensor temp_output(output_tensor->DataType(), TensorShape(transposed_input_shape), alloc);
    intermediate_output = std::move(temp_output);
  } else {
    transposed_input_tensor = *input_tensor;
  }


  const size_t cols = transposed_input_shape[input_rank - 1];
  const size_t rows = input_shape.Size() / cols;
  const size_t components = GetMaxComponents(cols);
  const auto packedCols = cols / components;

  size_t WG = rows == 1 ? 256: 64;

  SoftmaxProgram program{WG};


  program
      .CacheHint(std::to_string(components), std::to_string(WG))
      .AddInputs({*transposed_input_tensor, ProgramTensorMetadataDependency::TypeAndRank}})
      .AddOutputs({ is_transpose_required ? *intermediate_output : output_tensor})
      .SetWorkgroupSize(WG)
      .SetDispatchGroupSize(rows)
      .AddUniformVariables({
        {static_cast<int32_t>(packedCols)}
      });


  ORT_RETURN_IF_ERROR(context.RunProgram(program));

  // If transpose was required, transpose the result back
  if (is_transpose_required) {
    Tensor transposed_output_tensor;
    ORT_RETURN_IF_ERROR(Transpose::DoTranspose(perm, intermediate_output, *output_tensor));
  }

  return Status::OK();
}
}  // namespace webgpu
}  // namespace onnxruntime
