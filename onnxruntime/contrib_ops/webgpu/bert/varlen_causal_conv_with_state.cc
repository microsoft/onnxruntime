// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/bert/varlen_causal_conv_with_state.h"

#include <limits>

#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "contrib_ops/cpu/bert/causal_conv_with_state_helper.h"

using namespace onnxruntime::webgpu;

namespace onnxruntime {
namespace contrib {
namespace webgpu {

ONNX_OPERATOR_KERNEL_EX(
    VarlenCausalConvWithState,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .MayInplace(4, 1)
        .TypeConstraint("T", WebGpuSupportedFloatTypes())
        .TypeConstraint("M", DataTypeImpl::GetTensorType<int32_t>()),
    VarlenCausalConvWithState);

VarlenCausalConvWithState::VarlenCausalConvWithState(const OpKernelInfo& info)
    : WebGpuKernel(info) {
  std::string activation_str = info.GetAttrOrDefault<std::string>("activation", "none");
  activation_ = ParseCausalConvActivation(activation_str);
  ORT_ENFORCE(activation_ != CausalConvActivation::Invalid, "Invalid activation type");
  ORT_THROW_IF_ERROR(causal_conv_with_state_helper::ParseDilation(info, dilation_));
  const int64_t capacity = info.GetAttrOrDefault<int64_t>("state_update_capacity", 0);
  ORT_ENFORCE(capacity >= 0 && capacity <= 8, "state_update_capacity must be in [0, 8]");
  state_update_capacity_ = static_cast<int>(capacity);
}

Status VarlenCausalConvWithStateProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& input = shader.AddInput("input", ShaderUsage::UseElementTypeAlias);
  const auto& weight = shader.AddInput("weight", ShaderUsage::UseUniform);
  const auto& cumulative_sequence_length =
      shader.AddInput("cumulative_sequence_length", ShaderUsage::UseUniform);
  const ShaderVariableHelper* bias = &input;
  if (has_bias_) {
    bias = &shader.AddInput("bias", ShaderUsage::UseUniform);
  }
  const ShaderVariableHelper* initial_state = &input;
  if (has_state_ && !state_in_final_state_) {
    initial_state = &shader.AddInput("initial_state", ShaderUsage::UseUniform);
  }
  const ShaderVariableHelper* capture_count = &cumulative_sequence_length;
  if (has_capture_count_) {
    capture_count = &shader.AddInput("capture_count", ShaderUsage::UseUniform);
  }

  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform);
  const ShaderVariableHelper* final_state = &output;
  if (has_state_) {
    final_state = &shader.AddOutput(
        "final_state", ShaderUsage::UseUniform | ShaderUsage::UseGetByOffsetSegments);
  }
  const ShaderVariableHelper* state_update = &output;
  if (has_state_update_ && has_capture_count_) {
    state_update = &shader.AddOutput("state_update", ShaderUsage::UseUniform);
  }

  return WGSL_TEMPLATE_APPLY(shader, "bert/varlen_causal_conv_with_state.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(has_bias, has_bias_),
                             WGSL_TEMPLATE_PARAMETER(has_state, has_state_),
                             WGSL_TEMPLATE_PARAMETER(has_state_update, has_state_update_ && has_capture_count_),
                             WGSL_TEMPLATE_PARAMETER(state_in_final_state, state_in_final_state_),
                             WGSL_TEMPLATE_PARAMETER(use_silu, use_silu_),
                             WGSL_TEMPLATE_VARIABLE(bias, *bias),
                             WGSL_TEMPLATE_VARIABLE(capture_count, *capture_count),
                             WGSL_TEMPLATE_VARIABLE(cumulative_sequence_length, cumulative_sequence_length),
                             WGSL_TEMPLATE_VARIABLE(final_state, *final_state),
                             WGSL_TEMPLATE_VARIABLE(initial_state, *initial_state),
                             WGSL_TEMPLATE_VARIABLE(input, input),
                             WGSL_TEMPLATE_VARIABLE(output, output),
                             WGSL_TEMPLATE_VARIABLE(state_update, *state_update),
                             WGSL_TEMPLATE_VARIABLE(weight, weight));
}

Status VarlenCausalConvWithState::ComputeInternal(ComputeContext& context) const {
  const Tensor* input = context.Input(0);          // (total_tokens, channels)
  const Tensor* weight = context.Input(1);         // (channels, 1, kernel_size)
  const Tensor* cu_seqlens = context.Input(2);     // (batch_size + 1) int32
  const Tensor* bias = context.Input(3);           // optional (channels,)
  const Tensor* initial_state = context.Input(4);  // required (batch_size, channels, (K-1)*dilation)
  const Tensor* capture_count = context.Input(5);  // optional (batch_size) int32

  ORT_RETURN_IF(input == nullptr, "input is required");
  ORT_RETURN_IF(weight == nullptr, "weight is required");
  ORT_RETURN_IF(cu_seqlens == nullptr, "cumulative_sequence_length is required");
  ORT_RETURN_IF(initial_state == nullptr, "initial_state is required");

  const auto& input_shape = input->Shape();
  const auto& weight_shape = weight->Shape();
  ORT_RETURN_IF(input_shape.NumDimensions() != 2,
                "input must be rank 2 (total_tokens, channels)");
  ORT_RETURN_IF(weight_shape.NumDimensions() != 3,
                "weight must be rank 3 (channels, 1, kernel_size)");

  const int64_t total_tokens = input_shape[0];
  const int64_t channels = input_shape[1];
  const int64_t kernel_size = weight_shape[2];

  ORT_RETURN_IF(total_tokens < 0 || total_tokens > std::numeric_limits<int32_t>::max(),
                "total_tokens is too large for WebGPU");
  ORT_RETURN_IF(channels <= 0 || channels > std::numeric_limits<uint32_t>::max(),
                "channels is invalid for WebGPU");
  ORT_RETURN_IF(kernel_size <= 0 || kernel_size > std::numeric_limits<uint32_t>::max(),
                "kernel_size is invalid for WebGPU");
  ORT_RETURN_IF(weight_shape[0] != channels, "weight first dim must match input channels");
  ORT_RETURN_IF(weight_shape[1] != 1, "weight second dim must be 1 for depthwise convolution");
  const int64_t pad = (kernel_size - 1) * dilation_;
  ORT_RETURN_IF(pad > std::numeric_limits<int32_t>::max(),
                "pad is too large for WebGPU");

  const auto& cu_seqlens_shape = cu_seqlens->Shape();
  ORT_RETURN_IF(cu_seqlens_shape.NumDimensions() != 1 || cu_seqlens_shape[0] < 2,
                "cumulative_sequence_length must be rank 1 with at least 2 elements");
  const int64_t batch_size = cu_seqlens_shape[0] - 1;
  ORT_RETURN_IF(total_tokens < batch_size,
                "total_tokens must be at least batch_size because every sequence must contain a token");
  ORT_RETURN_IF(batch_size > std::numeric_limits<uint32_t>::max(),
                "batch size is too large for WebGPU");

  ORT_RETURN_IF((state_update_capacity_ > 0) != (capture_count != nullptr),
                "capture_count must be present exactly when state_update_capacity is positive");
  if (capture_count != nullptr) {
    ORT_RETURN_IF(capture_count->Shape().NumDimensions() != 1 ||
                      capture_count->Shape()[0] != batch_size,
                  "capture_count must have shape (", batch_size, ")");
  }

  if (bias != nullptr) {
    ORT_RETURN_IF(bias->Shape().NumDimensions() != 1 || bias->Shape()[0] != channels,
                  "bias must be rank 1 with size equal to channels");
  }

  const TensorShape state_shape({batch_size, channels, pad});
  ORT_RETURN_IF(initial_state->Shape() != state_shape,
                "initial_state must have shape ", state_shape.ToString(),
                ", got ", initial_state->Shape().ToString());

  const int capacity = (capture_count != nullptr) ? state_update_capacity_ : 0;

  const auto validate_flattened_size = [](const char* name,
                                          std::initializer_list<int64_t> dimensions) -> Status {
    uint64_t size = 1;
    for (const int64_t dimension : dimensions) {
      const uint64_t unsigned_dimension = static_cast<uint64_t>(dimension);
      ORT_RETURN_IF(unsigned_dimension != 0 &&
                        size > std::numeric_limits<uint64_t>::max() / unsigned_dimension,
                    name, " overflows uint64_t");
      size *= unsigned_dimension;
    }
    ORT_RETURN_IF(size > std::numeric_limits<uint32_t>::max(),
                  name, " is too large for WebGPU");
    return Status::OK();
  };

  ORT_RETURN_IF_ERROR(validate_flattened_size("total_tokens * channels", {total_tokens, channels}));
  ORT_RETURN_IF_ERROR(validate_flattened_size("channels * kernel_size", {channels, kernel_size}));
  ORT_RETURN_IF_ERROR(validate_flattened_size("batch_size * channels * pad", {batch_size, channels, pad}));
  ORT_RETURN_IF_ERROR(validate_flattened_size("batch_size * state_update_capacity * channels",
                                              {batch_size, state_update_capacity_, channels}));

  Tensor* output = context.Output(0, input_shape);
  Tensor* final_state = context.Output(1, state_shape);
  const TensorShape state_update_shape({batch_size, static_cast<int64_t>(state_update_capacity_), channels});
  Tensor* state_update = context.Output(2, state_update_shape);

  if (input_shape.Size() == 0 || batch_size == 0) {
    return Status::OK();
  }

  const bool has_bias = (bias != nullptr);
  const bool has_state = (pad > 0);
  const bool state_in_final_state = has_state && initial_state->DataRaw() == final_state->DataRaw();
  const bool has_capture_count = (capacity > 0);
  const bool has_state_update = (state_update != nullptr);

  VarlenCausalConvWithStateProgram program{has_bias, has_state, state_in_final_state,
                                           has_state_update, has_capture_count,
                                           activation_ == CausalConvActivation::Silu};
  program.CacheHint(has_bias, has_state, state_in_final_state, has_state_update, has_capture_count,
                    activation_ == CausalConvActivation::Silu);

  const uint64_t num_invocations_64 =
      static_cast<uint64_t>(batch_size) * static_cast<uint64_t>(channels);
  ORT_RETURN_IF(num_invocations_64 > std::numeric_limits<uint32_t>::max(),
                "batch_size * channels is too large for WebGPU");
  const uint32_t num_invocations = static_cast<uint32_t>(num_invocations_64);
  const uint32_t dispatch_groups =
      num_invocations / WORKGROUP_SIZE + static_cast<uint32_t>(num_invocations % WORKGROUP_SIZE != 0);

  program.AddInput({input, ProgramTensorMetadataDependency::Type})
      .AddInput({weight, ProgramTensorMetadataDependency::None})
      .AddInput({cu_seqlens, ProgramTensorMetadataDependency::None});
  if (has_bias) {
    program.AddInput({bias, ProgramTensorMetadataDependency::None});
  }
  if (has_state && !state_in_final_state) {
    program.AddInput({initial_state, ProgramTensorMetadataDependency::None});
  }
  if (has_capture_count) {
    program.AddInput({capture_count, ProgramTensorMetadataDependency::None});
  }

  program.AddOutput({output, ProgramTensorMetadataDependency::None});
  if (has_state) {
    program.AddOutput({final_state, ProgramTensorMetadataDependency::None});
  }
  if (has_state_update && has_capture_count) {
    program.AddOutput({state_update, ProgramTensorMetadataDependency::None});
  }

  program.SetDispatchGroupSize(dispatch_groups)
      .AddUniformVariable({static_cast<uint32_t>(channels)})
      .AddUniformVariable({static_cast<uint32_t>(kernel_size)})
      .AddUniformVariable({static_cast<uint32_t>(dilation_)})
      .AddUniformVariable({static_cast<uint32_t>(pad)})
      .AddUniformVariable({static_cast<uint32_t>(state_update_capacity_)})
      .AddUniformVariable({static_cast<uint32_t>(batch_size)})
      .AddUniformVariable({static_cast<uint32_t>(total_tokens)})
      .AddUniformVariable({num_invocations});

  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
