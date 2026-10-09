// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/bert/gated_delta_net_state_replay.h"

#include <algorithm>
#include <array>
#include <limits>

#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "core/providers/webgpu/shader_helper.h"

namespace onnxruntime::contrib::webgpu {

using namespace onnxruntime::webgpu;

ONNX_OPERATOR_KERNEL_EX(
    GatedDeltaNetStateReplay,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("TI", DataTypeImpl::GetTensorType<int64_t>())
        .InputMemoryType(OrtMemTypeCPUInput, 3)
        .MayInplace(2, 0),
    GatedDeltaNetStateReplay);

namespace {

constexpr uint32_t kWorkgroupSize = 128;
constexpr uint64_t kMaxIndex = std::numeric_limits<uint32_t>::max();

class GatedDeltaNetStateReplayProgram final : public Program<GatedDeltaNetStateReplayProgram> {
 public:
  GatedDeltaNetStateReplayProgram() : Program{"GatedDeltaNetStateReplay"} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& source = shader.AddInput("source", ShaderUsage::UseUniform);
    const auto& decay = shader.AddInput("decay", ShaderUsage::UseUniform);
    const auto& key = shader.AddInput("key", ShaderUsage::UseUniform);
    const auto& delta = shader.AddInput("delta", ShaderUsage::UseUniform);
    const auto& destination = shader.AddOutput("destination", ShaderUsage::UseUniform);
    return WGSL_TEMPLATE_APPLY(shader, "bert/gated_delta_net_state_replay.wgsl.template",
                               WGSL_TEMPLATE_VARIABLE(decay, decay),
                               WGSL_TEMPLATE_VARIABLE(delta, delta),
                               WGSL_TEMPLATE_VARIABLE(destination, destination),
                               WGSL_TEMPLATE_VARIABLE(key, key),
                               WGSL_TEMPLATE_VARIABLE(source, source));
  }

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"state_elements", ProgramUniformVariableDataType::Uint32},
      {"num_heads_v", ProgramUniformVariableDataType::Uint32},
      {"head_size_v", ProgramUniformVariableDataType::Uint32},
      {"head_size_k", ProgramUniformVariableDataType::Uint32},
      {"num_heads_k", ProgramUniformVariableDataType::Uint32},
      {"kept_count", ProgramUniformVariableDataType::Uint32});
};

// Enqueue only. The operator's completion boundary is deliberately separate from the dispatch.
Status EnqueueReplay(onnxruntime::webgpu::ComputeContext& context, const Tensor& source, const Tensor& capsule,
                     Tensor& destination, const std::array<uint32_t, 11>& desc,
                     uint32_t state_elements, uint32_t decay_elements,
                     uint32_t key_elements, uint32_t delta_elements,
                     uint32_t dispatch_x, uint32_t dispatch_y) {
  GatedDeltaNetStateReplayProgram program;
  program
      .AddInputs({ProgramInput::BufferView(&source, ProgramTensorMetadataDependency::Type,
                                           TensorShape{state_elements}, desc[0]),
                  ProgramInput::BufferView(&capsule, ProgramTensorMetadataDependency::Type,
                                           TensorShape{decay_elements}, desc[2]),
                  ProgramInput::BufferView(&capsule, ProgramTensorMetadataDependency::Type,
                                           TensorShape{key_elements}, desc[3]),
                  ProgramInput::BufferView(&capsule, ProgramTensorMetadataDependency::Type,
                                           TensorShape{delta_elements}, desc[4])})
      .AddOutput(ProgramOutput::BufferView(&destination, ProgramTensorMetadataDependency::Type,
                                           TensorShape{state_elements}, desc[1]))
      .SetDispatchGroupSize(dispatch_x, dispatch_y)
      .SetWorkgroupSize(kWorkgroupSize)
      .AddUniformVariables({{state_elements}, {desc[5]}, {desc[6]}, {desc[7]}, {desc[8]}, {desc[10]}});
  const auto& limits = context.DeviceLimits();
  const auto segment_count = [&limits](const Tensor& backing) {
    return (static_cast<uint64_t>(backing.SizeInBytes()) - 1) / limits.maxStorageBufferBindingSize + 1;
  };
  uint64_t binding_count = 0;
  // Use the program's Tensor-owner identity, not raw buffer equality (wrappers may differ).
  for (size_t index = 0; index < program.Inputs().size(); ++index) {
    if (program.InputBufferOwner(index) == index) {
      binding_count += segment_count(*program.Inputs()[index].tensor);
    }
  }
  for (size_t index = 0; index < program.Outputs().size(); ++index) {
    if (program.OutputBufferOwner(index) == index) {
      binding_count += segment_count(*program.Outputs()[index].tensor);
    }
  }
  ORT_RETURN_IF_NOT(binding_count <= limits.maxStorageBuffersPerShaderStage &&
                        binding_count + 1 <= limits.maxBindingsPerBindGroup,
                    "GatedDeltaNetStateReplay backing tensors exceed WebGPU binding limits.");
  return context.RunProgram(program);
}

}  // namespace

Status GatedDeltaNetStateReplay::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {
  ORT_RETURN_IF(context.IsGraphCaptureEnabled(), "GatedDeltaNetStateReplay does not support graph capture.");
  const auto* source = context.Input(0);
  const auto* capsule = context.Input(1);
  const auto* destination = context.Input(2);
  const auto* metadata = context.Input(3);
  ORT_RETURN_IF_NOT(source && capsule && destination && metadata,
                    "GatedDeltaNetStateReplay requires all four inputs.");
  for (const auto* backing : {source, capsule, destination}) {
    ORT_RETURN_IF_NOT(backing->IsDataType<float>() && backing->Shape().NumDimensions() == 1,
                      "GatedDeltaNetStateReplay backing tensors must be rank-1 FP32.");
    ORT_RETURN_IF_NOT(backing->Shape().Size() > 0 &&
                          static_cast<uint64_t>(backing->Shape().Size()) <= kMaxIndex,
                      "GatedDeltaNetStateReplay backing extent exceeds the supported indexing range.");
  }
  ORT_RETURN_IF_NOT(metadata->IsDataType<int64_t>() && metadata->Shape() == TensorShape{11},
                    "GatedDeltaNetStateReplay metadata must be CPU int64 with shape [11].");

  std::array<uint32_t, 11> desc{};
  const auto* values = metadata->Data<int64_t>();
  for (size_t index = 0; index < desc.size(); ++index) {
    ORT_RETURN_IF_NOT(values[index] >= 0 && static_cast<uint64_t>(values[index]) <= kMaxIndex,
                      "GatedDeltaNetStateReplay metadata field ", index, " is outside the uint32 indexing range.");
    desc[index] = static_cast<uint32_t>(values[index]);
  }
  const uint64_t hv = desc[5], dv = desc[6], dk = desc[7], hk = desc[8], capacity = desc[9];
  ORT_RETURN_IF_NOT(hv > 0 && dv > 0 && dk > 0 && hk > 0 && capacity > 0,
                    "GatedDeltaNetStateReplay dimensions and capacity must be positive.");
  ORT_RETURN_IF_NOT(desc[10] > 0 && desc[10] <= capacity,
                    "GatedDeltaNetStateReplay kept_count must be in [1, capacity].");
  ORT_RETURN_IF_NOT(hv * dv <= kMaxIndex && hk * dk <= kMaxIndex && (hv - 1) * hk <= kMaxIndex,
                    "GatedDeltaNetStateReplay head geometry exceeds the uint32 indexing range.");
  const uint64_t state_elements = hv * dv * dk;
  const uint64_t decay_elements = capacity * hv;
  const uint64_t key_elements = capacity * (hk * dk);
  const uint64_t delta_elements = capacity * (hv * dv);
  ORT_RETURN_IF_NOT(state_elements <= kMaxIndex &&
                        decay_elements <= kMaxIndex && key_elements <= kMaxIndex && delta_elements <= kMaxIndex,
                    "GatedDeltaNetStateReplay tensor geometry exceeds the uint32 indexing range.");
  const auto check_view = [](const Tensor& backing, uint32_t offset, uint64_t length) -> Status {
    ORT_RETURN_IF_NOT(static_cast<uint64_t>(offset) + length <= static_cast<uint64_t>(backing.Shape().Size()),
                      "GatedDeltaNetStateReplay view exceeds its backing tensor.");
    return Status::OK();
  };
  ORT_RETURN_IF_ERROR(check_view(*source, desc[0], state_elements));
  ORT_RETURN_IF_ERROR(check_view(*destination, desc[1], state_elements));
  ORT_RETURN_IF_ERROR(check_view(*capsule, desc[2], decay_elements));
  ORT_RETURN_IF_ERROR(check_view(*capsule, desc[3], key_elements));
  ORT_RETURN_IF_ERROR(check_view(*capsule, desc[4], delta_elements));
  ORT_RETURN_IF(destination->DataRaw() == source->DataRaw() || destination->DataRaw() == capsule->DataRaw(),
                "GatedDeltaNetStateReplay destination backing must be distinct from source and capsule.");

  const auto& limits = context.DeviceLimits();
  const uint64_t groups = (state_elements - 1) / kWorkgroupSize + 1;
  const uint64_t dispatch_x = std::min(groups, static_cast<uint64_t>(limits.maxComputeWorkgroupsPerDimension));
  const uint64_t dispatch_y = (groups - 1) / dispatch_x + 1;
  // Include 2D dispatch padding so the framework's flattened global index cannot wrap.
  ORT_RETURN_IF_NOT(dispatch_y <= limits.maxComputeWorkgroupsPerDimension &&
                        dispatch_x * dispatch_y * kWorkgroupSize <= kMaxIndex + 1,
                    "GatedDeltaNetStateReplay dispatch exceeds the supported indexing range.");

  auto* output = context.Output(0, destination->Shape());
  ORT_RETURN_IF_NOT(output && output->MutableDataRaw() == destination->DataRaw(),
                    "GatedDeltaNetStateReplay output must be prebound to destination_backing.");
  // Do not bind destination_backing as a read-only shader input alongside its writable output.
  ORT_RETURN_IF_ERROR(EnqueueReplay(context, *source, *capsule, *output, desc,
                                    static_cast<uint32_t>(state_elements), static_cast<uint32_t>(decay_elements),
                                    static_cast<uint32_t>(key_elements), static_cast<uint32_t>(delta_elements),
                                    static_cast<uint32_t>(dispatch_x), static_cast<uint32_t>(dispatch_y)));
  return context.FlushAndWaitChecked();
}

}  // namespace onnxruntime::contrib::webgpu
