// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/inlined_containers.h"
#include "core/providers/webgpu/tensor/slice.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/cpu/tensor/slice_helper.h"
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
                            << "var input_indices: input_indices_t;\n";

  for (int i = 0; i < input.Rank(); ++i) {
    const auto start = input.IndicesGet("uniforms.starts", i);
    const auto displacement = output.IndicesGet("output_indices", i) + " * " + input.IndicesGet("uniforms.steps", i);
    const auto index = "select(" + start + " + " + displacement + ", " + start + " - " + displacement +
                       ", " + input.IndicesGet("uniforms.signs", i) + " < 0)";
    shader.MainFunctionBody() << input.IndicesSet("input_indices", i, index) << ";\n";
  }

  shader.MainFunctionBody() << output.SetByOffset("global_idx", input.GetByIndices("input_indices"));

  return Status::OK();
}

static std::vector<int64_t> getInt64Input(const Tensor* tensor) {
  if (tensor->IsDataType<int64_t>()) {
    return std::vector<int64_t>(tensor->DataAsSpan<int64_t>().begin(), tensor->DataAsSpan<int64_t>().end());
  }
  ORT_ENFORCE(tensor->IsDataType<int32_t>(), "Expected tensor of type int32 or int64");
  std::vector<int64_t> result;
  auto span = tensor->DataAsSpan<int32_t>();
  result.reserve(span.size());
  for (auto v : span) {
    result.push_back(static_cast<int64_t>(v));
  }
  return result;
}

Status Slice::ComputeInternal(ComputeContext& context) const {
  // READ INPUTS
  const Tensor* input_tensor = context.Input(0);
  const TensorShape& input_shape = input_tensor->Shape();
  auto input_rank = input_shape.NumDimensions();

  const bool uses_attributes = Node().SinceVersion() < 10;
  auto starts_raw = uses_attributes ? attr_starts_ : getInt64Input(context.Input(1));
  auto ends_raw = uses_attributes ? attr_ends_ : getInt64Input(context.Input(2));

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
  auto axes_raw = attr_axes_.empty() ? (axes_tensor == nullptr ? axes_default : getInt64Input(axes_tensor)) : attr_axes_;

  std::vector<int64_t> steps_default;
  if (steps_tensor == nullptr) {
    // if steps not provided, set to [1, ..., 1] of len(starts)
    for (size_t i = 0; i < starts_raw.size(); i++) {
      steps_default.push_back(1);
    }
  }
  auto steps_raw = steps_tensor == nullptr ? steps_default : getInt64Input(steps_tensor);

  ORT_RETURN_IF_NOT(axes_raw.size() == starts_raw.size(), "axes and starts must have the same size");
  ORT_RETURN_IF_NOT(steps_raw.size() == starts_raw.size(), "steps and starts must have the same size");
  ORT_RETURN_IF_NOT(ends_raw.size() == starts_raw.size(), "ends and starts must have the same size");

  SliceOp::PrepareForComputeMetadata metadata(input_shape.GetDims());
  if (uses_attributes) {
    ORT_RETURN_IF_ERROR(SliceOp::PrepareForComputeHelper(starts_raw, ends_raw, axes_raw, metadata));
  } else {
    ORT_RETURN_IF_ERROR(SliceOp::PrepareForComputeHelper(starts_raw, ends_raw, axes_raw, steps_raw, metadata));
  }
  const auto& output_dims = metadata.output_dims_;
  InlinedVector<uint32_t> starts_reordered, steps_reordered;
  InlinedVector<int32_t> signs_reordered;
  starts_reordered.reserve(input_rank);
  steps_reordered.reserve(input_rank);
  signs_reordered.reserve(input_rank);
  for (size_t i = 0; i < input_rank; ++i) {
    const int64_t step = metadata.steps_[i];
    // A larger step can only select one element; avoid narrowing INT64_MIN or negating it.
    const uint64_t magnitude = step < 0 ? uint64_t(-(step + 1)) + 1 : uint64_t(step);
    starts_reordered.push_back(output_dims[i] == 0 ? 0u : narrow<uint32_t>(metadata.starts_[i]));
    steps_reordered.push_back(static_cast<uint32_t>(std::min(magnitude, uint64_t{UINT32_MAX})));
    signs_reordered.push_back(step < 0 ? -1 : 1);
  }
  TensorShape output_shape(output_dims);

  auto* output_tensor = context.Output(0, output_shape);
  uint32_t output_size = static_cast<uint32_t>(output_shape.Size());

  if (output_size == 0) {
    return Status::OK();
  }

  // The shader decomposes a full index per output element, one div and one mod per rank, so a
  // scalar thread does a lot of arithmetic to move two bytes. When the innermost axis is taken
  // as a contiguous, four-aligned run, four output elements come from four consecutive input
  // elements and the whole thing can run in units of four - the index arithmetic is unchanged,
  // it just operates on the reduced shapes. int64 stays scalar (it is stored as vec2<u32>).
  const auto& in_shape = input_tensor->Shape();
  const size_t last = input_rank - 1;
  int components = 1;
  if (input_rank > 0 && !input_tensor->IsDataType<int64_t>() &&
      steps_reordered[last] == 1 && signs_reordered[last] > 0 &&
      starts_reordered[last] % 4 == 0 && in_shape[last] % 4 == 0 && output_dims[last] % 4 == 0) {
    components = 4;
  }
  if (components > 1) {
    starts_reordered[last] /= components;
    output_size /= components;
  }

  SliceProgram program{};
  program
      .AddInputs({{input_tensor, ProgramTensorMetadataDependency::TypeAndRank, components}})
      .AddOutputs({{output_tensor, ProgramTensorMetadataDependency::TypeAndRank, components}})
      .SetDispatchGroupSize((output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({{output_size}, {gsl::make_span(starts_reordered)}, {gsl::make_span(steps_reordered)}, {gsl::make_span(signs_reordered)}});
  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace onnxruntime
