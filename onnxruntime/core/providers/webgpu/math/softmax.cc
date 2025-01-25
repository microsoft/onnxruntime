// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/inlined_containers.h"
#include "core/providers/webgpu/math/softmax.h"
#include "core/providers/webgpu/tensor/transpose.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/webgpu/shader_variable.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

#include "core/common/logging/logging.h"

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
    case 3:
      return "max(max(" + name + ".x, " + name + ".y), " + name + ".z)";
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
  const auto& input = shader.AddInput("x", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias);
  shader.AddOutput("result", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);
  int components = input.NumComponents();

  LOGS_DEFAULT(VERBOSE) << "Input StorageType: " << input.StorageType() << "\n";
  LOGS_DEFAULT(VERBOSE) << "Input ElementType: " << input.ElementType() << "\n";
  LOGS_DEFAULT(VERBOSE) << "Input ValueType: " << input.ValueType() << "\n";



  std::string threadMaxDecl =  input.ElementType() == "f32" ?
                                "var threadMax = x_value_t(-3.402823e+38f);\n" :
                                "var threadMax = x_value_t(-65504.0h);\n";


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
      << "    rowMaxShared = x_value_t(" <<  MaxVector("threadShared[0]", components)   << ");\n"
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
  int64_t  axis = axis_ < 0 ? axis_ + input_rank : axis_;

  bool is_transpose_required = axis < input_rank - 1;
  LOGS_DEFAULT(VERBOSE) <<"axis_: " << axis_ << " axis: " << axis << "\n";
  LOGS_DEFAULT(VERBOSE) << "Transpose required: " << (is_transpose_required ? "true" : "false") << "\n";
  LOGS_DEFAULT(VERBOSE) << "Input shape: " << input_shape.ToString() << "\n";
  LOGS_DEFAULT(VERBOSE) << "Output shape: " << output_tensor->Shape().ToString() << "\n";
  LOGS_DEFAULT(VERBOSE) << "Input rank: " << input_rank << "\n";

  TensorShape transposed_input_shape = input_shape;
  Tensor transposed_input_tensor;
  Tensor intermediate_output;
  InlinedVector<size_t> perm;

  if (is_transpose_required) {
    AllocatorPtr alloc;
    perm.resize(input_rank);
    for (size_t i = 0; i < perm.size(); ++i) {
      perm[i] = i;
    }
    perm[axis] = input_rank - 1;
    perm[input_rank - 1] = axis;

    LOGS_DEFAULT(VERBOSE) << "Allocating temporary tensors for transpose\n";

    // allocate a temporary tensor to hold transposed input
    Tensor temp_input(input_tensor->DataType(), TensorShape(transposed_input_shape), alloc);

    LOGS_DEFAULT(VERBOSE) << "Performing transpose\n";

    ORT_RETURN_IF_ERROR(Transpose::DoTranspose(perm, *input_tensor, temp_input));

    LOGS_DEFAULT(VERBOSE) << "Transpose done\n";

    LOGS_DEFAULT(VERBOSE) << "Allocating memory for intermediate output\n";
    transposed_input_tensor = std::move(temp_input);
    transposed_input_shape = transposed_input_tensor.Shape();

    LOGS_DEFAULT(VERBOSE) << "Transposed input shape: " << transposed_input_shape.ToString() << "\n";

    // Allocate memory for the intermediate output
    LOGS_DEFAULT(VERBOSE) << "Allocating memory for intermediate output\n";
    Tensor temp_output(output_tensor->DataType(), TensorShape(transposed_input_shape), alloc);
    intermediate_output = std::move(temp_output);
  }


  const size_t cols = transposed_input_shape[input_rank - 1];
  const size_t rows = input_shape.Size() / cols;
  const size_t components = GetMaxComponents(cols);
  const auto packedCols = cols / components;

  LOGS_DEFAULT(VERBOSE) << "Cols: " << cols << " Rows: " << rows << " Components: " << components << " PackedCols: " << packedCols << "\n";

  size_t WG = rows == 1 ? 256: 64;

  SoftmaxProgram program{WG};
  if  (is_transpose_required) {
    program
        .AddInputs({{&transposed_input_tensor, ProgramTensorMetadataDependency::TypeAndRank,  static_cast<int>(components)}})
        .AddOutputs({{&intermediate_output, ProgramTensorMetadataDependency::TypeAndRank, static_cast<int>(components)}});
  } else {
    program
        .AddInputs({{input_tensor, ProgramTensorMetadataDependency::TypeAndRank, static_cast<int>(components)}})
        .AddOutputs({{output_tensor, ProgramTensorMetadataDependency::TypeAndRank, static_cast<int>(components)}});
  }


  program
      .CacheHint(std::to_string(components), std::to_string(WG))
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
