// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/bert/conv_state_replay.h"

#include <algorithm>
#include <array>
#include <limits>

#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime::contrib::webgpu {

using namespace onnxruntime::webgpu;

ONNX_OPERATOR_KERNEL_EX(
    ConvStateReplay,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes())
        .TypeConstraint("TI", DataTypeImpl::GetTensorType<int64_t>())
        .InputMemoryType(OrtMemTypeCPUInput, 3)
        .MayInplace(2, 0),
    ConvStateReplay);

namespace {

constexpr uint32_t kWorkgroupSize = 128;
constexpr uint64_t kMaxIndex = std::numeric_limits<uint32_t>::max();

class ConvStateReplayProgram final : public Program<ConvStateReplayProgram> {
 public:
  ConvStateReplayProgram(bool half, bool has_capture)
      : Program{"ConvStateReplay"}, half_{half}, has_capture_{has_capture} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& source = shader.AddInput("source", ShaderUsage::UseUniform);
    const ShaderVariableHelper* capture = nullptr;
    if (has_capture_) {
      capture = &shader.AddInput("capture", ShaderUsage::UseUniform);
    }
    const auto& destination = shader.AddOutput("destination", ShaderUsage::UseUniform);
    return WGSL_TEMPLATE_APPLY(shader, "bert/conv_state_replay.wgsl.template",
                               WGSL_TEMPLATE_PARAMETER(half, half_),
                               WGSL_TEMPLATE_PARAMETER(has_capture, has_capture_),
                               WGSL_TEMPLATE_OPTIONAL_VARIABLE(capture, capture),
                               WGSL_TEMPLATE_VARIABLE(destination, destination),
                               WGSL_TEMPLATE_VARIABLE(source, source));
  }

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"state_elements", ProgramUniformVariableDataType::Uint32},
      {"output_words", ProgramUniformVariableDataType::Uint32},
      {"channels", ProgramUniformVariableDataType::Uint32},
      {"history_width", ProgramUniformVariableDataType::Uint32},
      {"capacity", ProgramUniformVariableDataType::Uint32},
      {"kept_count", ProgramUniformVariableDataType::Uint32},
      {"source_lane", ProgramUniformVariableDataType::Uint32},
      {"destination_lane", ProgramUniformVariableDataType::Uint32},
      {"capture_lane", ProgramUniformVariableDataType::Uint32});

 private:
  const bool half_;
  const bool has_capture_;
};

Status EnqueueReplay(onnxruntime::webgpu::ComputeContext& context, const Tensor& source, const Tensor& capture,
                     Tensor& destination, const std::array<uint32_t, 8>& desc,
                     uint32_t state_elements, uint32_t capture_elements,
                     uint32_t output_words, uint32_t dispatch_x, uint32_t dispatch_y) {
  const bool half = source.IsDataType<MLFloat16>();
  const uint32_t element_bytes = half ? 2 : 4;
  const uint32_t elements_per_word = 4 / element_bytes;
  const bool has_capture = desc[7] > 0;
  // WebGPU allocations are rounded to four bytes. Reinterpret the entire allocation, never
  // do pointer arithmetic on an opaque buffer handle or load/store floating-point values.
  const auto word_shape = [](const Tensor& tensor) {
    return TensorShape{static_cast<int64_t>((tensor.SizeInBytes() + 3) / 4)};
  };
  Tensor source_words(DataTypeImpl::GetType<uint32_t>(), word_shape(source),
                      const_cast<void*>(source.DataRaw()), source.Location());
  Tensor capture_words(DataTypeImpl::GetType<uint32_t>(), word_shape(capture),
                       const_cast<void*>(capture.DataRaw()), capture.Location());
  Tensor destination_words(DataTypeImpl::GetType<uint32_t>(), word_shape(destination),
                           destination.MutableDataRaw(), destination.Location());
  const auto view_words = [elements_per_word, element_bytes](uint32_t offset, uint32_t count) {
    return TensorShape{static_cast<int64_t>(
        (static_cast<uint64_t>(offset % 4 / element_bytes) + count + elements_per_word - 1) / elements_per_word)};
  };
  ConvStateReplayProgram program{half, has_capture};
  program.AddInput(ProgramInput::BufferView(&source_words, ProgramTensorMetadataDependency::Type,
                                            view_words(desc[0], state_elements), desc[0] / 4));
  if (has_capture) {
    program.AddInput(ProgramInput::BufferView(&source == &capture ? &source_words : &capture_words,
                                              ProgramTensorMetadataDependency::Type,
                                              view_words(desc[2], capture_elements), desc[2] / 4));
  }
  program.AddOutput(ProgramOutput::BufferView(&destination_words, ProgramTensorMetadataDependency::Type,
                                              TensorShape{output_words}, desc[1] / 4))
      .CacheHint(half, has_capture)
      .SetDispatchGroupSize(dispatch_x, dispatch_y)
      .SetWorkgroupSize(kWorkgroupSize)
      .AddUniformVariables({{state_elements},
                            {output_words},
                            {desc[4]},
                            {desc[5]},
                            {desc[6]},
                            {desc[7]},
                            {desc[0] % 4 / element_bytes},
                            {desc[1] % 4 / element_bytes},
                            {desc[2] % 4 / element_bytes}});

  const auto& limits = context.DeviceLimits();
  const auto segments = [&limits](const Tensor& tensor) {
    return (static_cast<uint64_t>(tensor.SizeInBytes()) - 1) / limits.maxStorageBufferBindingSize + 1;
  };
  uint64_t bindings = segments(destination_words);
  for (size_t index = 0; index < program.Inputs().size(); ++index) {
    if (program.InputBufferOwner(index) == index) {
      bindings += segments(*program.Inputs()[index].tensor);
    }
  }
  ORT_RETURN_IF_NOT(bindings <= limits.maxStorageBuffersPerShaderStage &&
                        bindings + 1 <= limits.maxBindingsPerBindGroup,
                    "ConvStateReplay backing tensors exceed WebGPU binding limits.");
  return context.RunProgram(program);
}

}  // namespace

Status ConvStateReplay::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {
  ORT_RETURN_IF(context.IsGraphCaptureEnabled(), "ConvStateReplay does not support graph capture.");
  const auto* source = context.Input(0);
  const auto* capture = context.Input(1);
  const auto* destination = context.Input(2);
  const auto* metadata = context.Input(3);
  ORT_RETURN_IF_NOT(source && capture && destination && metadata, "ConvStateReplay requires all four inputs.");
  for (const auto* backing : {source, capture, destination}) {
    ORT_RETURN_IF_NOT((backing->IsDataType<float>() || backing->IsDataType<MLFloat16>()) &&
                          backing->GetElementType() == source->GetElementType() &&
                          backing->Shape().NumDimensions() == 1,
                      "ConvStateReplay backing tensors must be rank-1 with the same FP16 or FP32 type.");
    ORT_RETURN_IF_NOT(static_cast<uint64_t>(backing->SizeInBytes()) <= kMaxIndex,
                      "ConvStateReplay backing extent exceeds the uint32 indexing range.");
  }
  ORT_RETURN_IF_NOT(metadata->IsDataType<int64_t>() && metadata->Shape() == TensorShape{8},
                    "ConvStateReplay metadata must be CPU int64 with shape [8].");
  std::array<uint32_t, 8> desc{};
  for (size_t index = 0; index < desc.size(); ++index) {
    const auto value = metadata->Data<int64_t>()[index];
    ORT_RETURN_IF_NOT(value >= 0 && static_cast<uint64_t>(value) <= kMaxIndex,
                      "ConvStateReplay metadata field ", index, " is outside the uint32 indexing range.");
    desc[index] = static_cast<uint32_t>(value);
  }
  const uint64_t batch = desc[3], channels = desc[4], width = desc[5], capacity = desc[6];
  ORT_RETURN_IF_NOT(channels > 0, "ConvStateReplay channels must be positive.");
  ORT_RETURN_IF_NOT(desc[7] <= capacity, "ConvStateReplay kept_count must be in [0, capacity].");
  const uint32_t element_bytes = source->IsDataType<MLFloat16>() ? 2 : 4;
  ORT_RETURN_IF_NOT(batch * channels <= kMaxIndex,
                    "ConvStateReplay geometry exceeds the uint32 indexing range.");
  const uint64_t state_elements = batch * channels * width;
  const uint64_t capture_elements = batch * channels * capacity;
  ORT_RETURN_IF_NOT(state_elements <= kMaxIndex / element_bytes &&
                        capture_elements <= kMaxIndex / element_bytes,
                    "ConvStateReplay geometry exceeds the uint32 indexing range.");
  const auto check_view = [element_bytes](const Tensor& backing, uint32_t offset, uint64_t count) -> Status {
    ORT_RETURN_IF_NOT(offset % element_bytes == 0, "ConvStateReplay byte offsets must be element-aligned.");
    ORT_RETURN_IF_NOT(static_cast<uint64_t>(offset) + count * element_bytes <= backing.SizeInBytes(),
                      "ConvStateReplay view exceeds its backing tensor.");
    return Status::OK();
  };
  ORT_RETURN_IF_ERROR(check_view(*source, desc[0], state_elements));
  ORT_RETURN_IF_ERROR(check_view(*destination, desc[1], state_elements));
  ORT_RETURN_IF_ERROR(check_view(*capture, desc[2], capture_elements));
  ORT_RETURN_IF(destination->DataRaw() &&
                    (destination->DataRaw() == source->DataRaw() || destination->DataRaw() == capture->DataRaw()),
                "ConvStateReplay destination backing must be distinct from source and capture.");
  auto* output = context.Output(0, destination->Shape());
  ORT_RETURN_IF_NOT(output && output->MutableDataRaw() == destination->DataRaw(),
                    "ConvStateReplay output must be prebound to destination_backing.");
  if (state_elements == 0) {
    return Status::OK();
  }

  const uint64_t output_words = (desc[1] % 4 + state_elements * element_bytes + 3) / 4;
  const auto& limits = context.DeviceLimits();
  const uint64_t groups = (output_words + kWorkgroupSize - 1) / kWorkgroupSize;
  const uint64_t dispatch_x = std::min(groups, static_cast<uint64_t>(limits.maxComputeWorkgroupsPerDimension));
  const uint64_t dispatch_y = (groups + dispatch_x - 1) / dispatch_x;
  ORT_RETURN_IF_NOT(dispatch_y <= limits.maxComputeWorkgroupsPerDimension &&
                        dispatch_x * dispatch_y * kWorkgroupSize <= kMaxIndex + 1,
                    "ConvStateReplay dispatch exceeds the supported indexing range.");
  return context.RunAndWaitChecked([&]() {
    return EnqueueReplay(context, *source, *capture, *output, desc,
                         static_cast<uint32_t>(state_elements), static_cast<uint32_t>(capture_elements),
                         static_cast<uint32_t>(output_words), static_cast<uint32_t>(dispatch_x),
                         static_cast<uint32_t>(dispatch_y));
  });
}

}  // namespace onnxruntime::contrib::webgpu
