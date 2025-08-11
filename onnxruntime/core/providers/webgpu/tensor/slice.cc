// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/inlined_containers.h"
#include "core/providers/webgpu/tensor/slice.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace webgpu {

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Slice,
    kOnnxDomain,
    1, 9,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes()),
    Slice);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Slice,
    kOnnxDomain,
    10, 10,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes())
        .InputMemoryType(OrtMemTypeCPU, 1)
        .InputMemoryType(OrtMemTypeCPU, 2)
        .InputMemoryType(OrtMemTypeCPU, 3)
        .InputMemoryType(OrtMemTypeCPU, 4),
    Slice);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Slice,
    kOnnxDomain,
    11, 12,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes())
        .InputMemoryType(OrtMemTypeCPU, 1)
        .InputMemoryType(OrtMemTypeCPU, 2)
        .InputMemoryType(OrtMemTypeCPU, 3)
        .InputMemoryType(OrtMemTypeCPU, 4),
    Slice);

ONNX_OPERATOR_KERNEL_EX(
    Slice,
    kOnnxDomain,
    13,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes())
        .InputMemoryType(OrtMemTypeCPU, 1)
        .InputMemoryType(OrtMemTypeCPU, 2)
        .InputMemoryType(OrtMemTypeCPU, 3)
        .InputMemoryType(OrtMemTypeCPU, 4),
    Slice);

Status SliceProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const ShaderVariableHelper& input = shader.AddInput("input", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);
  const ShaderVariableHelper& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);

  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")
                            << "let output_indices = " << output.OffsetToIndices("global_idx") << ";\n"
                            << "var input_indices: input_indices_t;\n"
                            << "var carry = 0u;\n";

  for (int i = input.Rank() - 1; i >= 0; i--) {
    std::string input_shape_i = absl::StrCat("input_shape_", i);
    std::string steps_i = absl::StrCat("steps_", i);
    std::string starts_i = absl::StrCat("starts_", i);
    std::string output_index_i = absl::StrCat("output_index_", i);
    std::string input_index_i = absl::StrCat("input_index_", i);

    shader.MainFunctionBody() << "let " << input_shape_i << " = " << input.IndicesGet("uniforms.input_shape", i) << ";\n"
                              << "let " << steps_i << " = " << input.IndicesGet("uniforms.steps", i) << ";\n"
                              << "let " << starts_i << " = " << input.IndicesGet("uniforms.starts", i) << ";\n"
                              << "var " << output_index_i << " = " << output.IndicesGet("output_indices", i) << ";\n"
                              << "var " << input_index_i << " = " << output_index_i << " * " << steps_i << " + " << starts_i << " + carry;\n"
                              << "carry = " << input_index_i << " / " << input_shape_i << ";\n"
                              << input_index_i << " = " << input_index_i << " % " << input_shape_i << ";\n"
                              << "if (" << input.IndicesGet("uniforms.signs", i) << " < 0) {\n"
                              << "  " << input_index_i << " = " << input_shape_i << " - " << input_index_i << " - 1u + " << starts_i << ";\n"
                              << "}\n"
                              << input.IndicesSet("input_indices", i, input_index_i) << ";\n";
  }

  shader.MainFunctionBody() << output.SetByOffset("global_idx", input.GetByIndices("input_indices"));

  return Status::OK();
}

Status Slice::ComputeInternal(ComputeContext& context) const {
  // READ INPUTS
  const Tensor* input_tensor = context.Input(0);
  const TensorShape& input_shape = input_tensor->Shape();
  auto input_rank = input_shape.NumDimensions();

  auto starts_raw = attr_starts_.empty() ? context.Input(1)->DataAsSpan<int64_t>() : gsl::make_span(attr_starts_);
  auto ends_raw = attr_ends_.empty() ? context.Input(2)->DataAsSpan<int64_t>() : gsl::make_span(attr_ends_);

  ORT_ENFORCE(starts_raw.size() == ends_raw.size(), "starts and ends must have the same size");

  int input_count = context.InputCount();

  const Tensor* axes_tensor = nullptr;
  const Tensor* steps_tensor = nullptr;

  if (input_count >= 4) {
    // axes provided as input
    axes_tensor = context.Input(3);
  }

  if (input_count == 5) {
    // steps provided as input
    steps_tensor = context.Input(4);
  }

  // Inject defaults if axes or steps not provided
  std::vector<int64_t> axes_default;
  if (axes_tensor == nullptr) {
    // if axes not provided, set to [0, ..., len(starts)-1]
    for (size_t i = 0; i < starts_raw.size(); i++) {
      axes_default.push_back(i);
    }
  }
  auto axes_raw = attr_axes_.empty() ? (axes_tensor == nullptr ? gsl::make_span(axes_default) : axes_tensor->DataAsSpan<int64_t>()) : gsl::make_span(attr_axes_);

  std::vector<int64_t> steps_default;
  if (steps_tensor == nullptr) {
    // if steps not provided, set to [1, ..., 1] of len(starts)
    for (size_t i = 0; i < starts_raw.size(); i++) {
      steps_default.push_back(1);
    }
  }
  auto steps_raw = steps_tensor == nullptr ? gsl::make_span(steps_default) : steps_tensor->DataAsSpan<int64_t>();

  // get final axes
  std::vector<uint32_t> axes, axes_fixed;
  for (unsigned int i = 0; i < axes_raw.size(); i++) {
    int64_t val = axes_raw[i];
    if (val < 0) {
      val += input_rank;
    }
    axes_fixed.push_back(static_cast<int32_t>(val));
    axes.push_back(static_cast<int32_t>(val));
  }

  std::vector<uint32_t> starts;
  std::vector<uint32_t> ends;
  std::vector<int32_t> signs;
  std::vector<uint32_t> steps;
  std::vector<int64_t> output_dims;
  output_dims.resize(input_rank, 0);

  // main loop over axes that will setup
  // starts, ends, steps, signs and output_dims
  for (unsigned int i = 0; i < starts_raw.size(); i++) {
    int64_t start = starts_raw[i];
    int64_t end = ends_raw[i];
    int64_t step = steps_raw[i];
    int64_t dim_value = input_shape[axes[i]];
    if (start < 0) {
      start += dim_value;
    }
    if (end == std::numeric_limits<int32_t>::max() || end == std::numeric_limits<int64_t>::max()) {
      end = step < 0 ? -1 : dim_value;
    } else if (end < 0) {
      end += dim_value;
    }
    if (step < 0) {
      // we are slicing in reverse
      start = dim_value > 0 ? std::clamp(start, int64_t{0}, dim_value - 1) : 0;
      end = dim_value > 0 ? std::clamp(end, int64_t{-1}, dim_value - 1) : -1;
      // note that we are flipping start and end to switch to forward step
      signs.push_back(-1);
      steps.push_back(static_cast<uint32_t>(-step));
      starts.push_back(static_cast<uint32_t>((end < 0) ? 0 : end));
      ends.push_back(static_cast<uint32_t>(start));
    } else {
      // we are slicing in forward direction
      start = std::clamp(start, int64_t{0}, dim_value);
      end = std::clamp(end, int64_t{0}, dim_value);
      signs.push_back(1);
      steps.push_back(static_cast<uint32_t>(step));
      starts.push_back(static_cast<uint32_t>(start));
      ends.push_back(static_cast<uint32_t>(end));
    }
    auto temp = static_cast<int64_t>(ceil(1.0 * (end - start) / static_cast<float>(step)));
    output_dims[axes[i]] = (temp > 0 && dim_value != 0) ? temp : 0;
  }

  // insert missing dimensions
  if (axes.size() != input_rank) {
    for (uint32_t i = 0; i < input_rank; i++) {
      int idx = -1;
      for (unsigned int j = 0; j < axes_fixed.size(); j++) {
        if (axes_fixed[j] == i) {
          idx = j;
          break;
        }
      }
      if (idx == -1) {
        uint32_t dim_value = static_cast<uint32_t>(input_shape[i]);
        axes.insert(axes.begin() + i, i);
        starts.insert(starts.begin() + i, 0);
        ends.insert(ends.begin() + i, dim_value);
        signs.insert(signs.begin() + i, 1);
        steps.insert(steps.begin() + i, 1);
        output_dims[i] = dim_value;
      }
    }
  }

  // Reorder inputs in order of axis
  std::vector<int32_t> signs_reordered;
  std::vector<uint32_t> steps_reordered, starts_reordered, ends_reordered;
  signs_reordered.resize(input_rank, 0);
  steps_reordered.resize(input_rank, 1);
  starts_reordered.resize(input_rank, 0);
  ends_reordered.resize(input_rank, 0);
  for (unsigned int i = 0; i < input_rank; i++) {
    int32_t dim = axes[i];
    signs_reordered[dim] = signs[i];
    steps_reordered[dim] = steps[i];
    starts_reordered[dim] = starts[i];
    ends_reordered[dim] = ends[i];
  }

  TensorShape output_shape(output_dims);

  auto* output_tensor = context.Output(0, output_shape);
  uint32_t output_size = static_cast<uint32_t>(output_shape.Size());

  if (output_size == 0) {
    return Status::OK();
  }

  SliceProgram program{};
  program
      .AddInputs({{input_tensor, ProgramTensorMetadataDependency::TypeAndRank}})
      .AddOutputs({output_tensor})
      .SetDispatchGroupSize((output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({{output_size}, {starts_reordered}, {steps_reordered}, {signs_reordered}});
  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace onnxruntime
