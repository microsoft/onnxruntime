// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/bert/causal_conv1d_with_state.h"

#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"

using namespace onnxruntime::webgpu;

namespace onnxruntime {
namespace contrib {
namespace webgpu {

CausalConv1DActivation ParseCausalConv1DActivation(const std::string& activation_str) {
  if (activation_str == "silu" || activation_str == "swish") {
    return CausalConv1DActivation::Silu;
  } else if (activation_str == "none" || activation_str.empty()) {
    return CausalConv1DActivation::None;
  }
  ORT_THROW("Unknown activation for CausalConv1DWithState: ", activation_str);
}

// =============================================================================
// CausalConv1DWithState Implementation
// =============================================================================

ONNX_OPERATOR_KERNEL_EX(
    CausalConv1DWithState,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes()),
    CausalConv1DWithState);

CausalConv1DWithState::CausalConv1DWithState(const OpKernelInfo& info)
    : WebGpuKernel(info) {
  std::string activation_str = info.GetAttrOrDefault<std::string>("activation", "silu");
  activation_ = ParseCausalConv1DActivation(activation_str);
}

Status CausalConv1DWithStateProgram::GenerateShaderCode(ShaderHelper& shader) const {
  // Input tensors
  const auto& input = shader.AddInput("input", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  const auto& weight = shader.AddInput("weight", ShaderUsage::UseUniform);

  // Optional inputs
  const ShaderVariableHelper* bias_ptr = nullptr;
  const ShaderVariableHelper* conv_state_ptr = nullptr;
  if (has_bias_) {
    bias_ptr = &shader.AddInput("bias", ShaderUsage::UseUniform);
  }
  if (has_conv_state_) {
    conv_state_ptr = &shader.AddInput("conv_state", ShaderUsage::UseUniform);
  }

  // Output tensors
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform);
  const auto& present_state = shader.AddOutput("present_state", ShaderUsage::UseUniform);

  // Activation function implementation
  if (activation_ == CausalConv1DActivation::Silu) {
    shader.AdditionalImplementation() << R"SHADER(
fn silu(x: input_element_t) -> input_element_t {
  return x / (1.0 + exp(-x));
}
)SHADER";
  }

  // Flatten to 1D dispatch: each thread handles one (batch, channel, pos) triple.
  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")
                            << R"SHADER(
  let batch_size = uniforms.batch_size;
  let channels = uniforms.channels;
  let input_length = uniforms.input_length;
  let kernel_size = uniforms.kernel_size;
  let state_length = uniforms.state_length;  // = kernel_size - 1

  let pos = global_idx % input_length;
  let bc_idx = global_idx / input_length;
  let batch_idx = bc_idx / channels;
  let channel_idx = bc_idx % channels;

  // Perform depthwise causal convolution for this (batch, channel, pos).
  // The convolution window looks back kernel_size-1 positions.
  // With conv_state providing the history before position 0, the
  // "virtual" input is: [conv_state[0..state_length-1], input[0..L-1]]
  //
  // For output position pos:
  //   output[pos] = sum_{j=0}^{kernel_size-1} weight[j] * virtual_input[pos + j]
  // where virtual_input is state_length positions of conv_state
  // followed by input_length positions of input.

  var acc: input_element_t = 0.0;

  // Weight layout: (D, 1, K) -> channel_idx * kernel_size + j
  let weight_base = channel_idx * kernel_size;

  for (var j: u32 = 0; j < kernel_size; j = j + 1) {
    // virtual_pos is the position in the concatenated [conv_state, input]
    let virtual_pos = pos + j;

    var val: input_element_t = 0.0;
)SHADER";

  if (has_conv_state_) {
    shader.MainFunctionBody() << R"SHADER(
    if (virtual_pos < state_length) {
      // Read from conv_state: (B, D, state_length)
      let state_idx = (batch_idx * channels + channel_idx) * state_length + virtual_pos;
      val = )SHADER"
                               << conv_state_ptr->GetByOffset("state_idx") << R"SHADER(;
    } else {
      // Read from input: (B, D, L)
      let input_pos = virtual_pos - state_length;
      let input_idx = (batch_idx * channels + channel_idx) * input_length + input_pos;
      val = )SHADER"
                               << input.GetByOffset("input_idx") << R"SHADER(;
    }
)SHADER";
  } else {
    // No conv_state: pad with zeros for positions before the input
    shader.MainFunctionBody() << R"SHADER(
    if (virtual_pos >= state_length) {
      let input_pos = virtual_pos - state_length;
      let input_idx = (batch_idx * channels + channel_idx) * input_length + input_pos;
      val = )SHADER"
                               << input.GetByOffset("input_idx") << R"SHADER(;
    }
)SHADER";
  }

  shader.MainFunctionBody() << R"SHADER(
    let w = )SHADER"
                             << weight.GetByOffset("weight_base + j") << R"SHADER(;
    acc = acc + val * w;
  }
)SHADER";

  // Add bias if present
  if (has_bias_) {
    shader.MainFunctionBody() << "  acc = acc + " << bias_ptr->GetByOffset("channel_idx") << ";\n";
  }

  // Apply activation
  if (activation_ == CausalConv1DActivation::Silu) {
    shader.MainFunctionBody() << "  acc = silu(acc);\n";
  }

  // Write output: (B, D, L)
  shader.MainFunctionBody() << R"SHADER(
  let out_idx = (batch_idx * channels + channel_idx) * input_length + pos;
  )SHADER" << output.SetByOffset("out_idx", "acc")
                             << "\n";

  // Write present_state: the last (kernel_size - 1) elements from the
  // virtual input [conv_state, input]. The virtual input has total length
  // state_length + input_length. We want positions from
  // (state_length + input_length - state_length) to (state_length + input_length - 1),
  // i.e. the last state_length positions of the virtual input, which are the
  // last state_length positions of input (when input_length >= state_length).
  //
  // We only write present_state once per (batch, channel), using the thread
  // at pos == 0 to write all state_length values.
  shader.MainFunctionBody() << R"SHADER(
  if (pos == 0u) {
    for (var s: u32 = 0; s < state_length; s = s + 1) {
      var state_val: input_element_t = 0.0;
      // total_len = state_length + input_length
      // We want virtual_input[total_len - state_length + s] = virtual_input[input_length + s]
      let vp = input_length + s;
)SHADER";

  if (has_conv_state_) {
    shader.MainFunctionBody() << R"SHADER(
      if (vp < state_length) {
        let si = (batch_idx * channels + channel_idx) * state_length + vp;
        state_val = )SHADER"
                               << conv_state_ptr->GetByOffset("si") << R"SHADER(;
      } else {
        let ip = vp - state_length;
        let ii = (batch_idx * channels + channel_idx) * input_length + ip;
        state_val = )SHADER"
                               << input.GetByOffset("ii") << R"SHADER(;
      }
)SHADER";
  } else {
    shader.MainFunctionBody() << R"SHADER(
      if (vp >= state_length) {
        let ip = vp - state_length;
        let ii = (batch_idx * channels + channel_idx) * input_length + ip;
        state_val = )SHADER"
                               << input.GetByOffset("ii") << R"SHADER(;
      }
)SHADER";
  }

  shader.MainFunctionBody() << R"SHADER(
      let ps_idx = (batch_idx * channels + channel_idx) * state_length + s;
      )SHADER" << present_state.SetByOffset("ps_idx", "state_val")
                             << R"SHADER(
    }
  }
)SHADER";

  return Status::OK();
}

Status CausalConv1DWithState::ComputeInternal(ComputeContext& context) const {
  const Tensor* input = context.Input(0);   // (B, D, L)
  const Tensor* weight = context.Input(1);  // (D, 1, K)
  const Tensor* bias = context.Input(2);    // optional (D,)
  const Tensor* conv_state = context.Input(3);  // optional (B, D, K-1)

  ORT_RETURN_IF(input == nullptr, "Input tensor must not be null");
  ORT_RETURN_IF(weight == nullptr, "Weight tensor must not be null");

  const auto& input_shape = input->Shape();
  const auto& weight_shape = weight->Shape();

  ORT_RETURN_IF(input_shape.NumDimensions() != 3,
                "Input must be 3D (batch_size, channels, length)");
  ORT_RETURN_IF(weight_shape.NumDimensions() != 3,
                "Weight must be 3D (channels, 1, kernel_size)");

  const int batch_size = static_cast<int>(input_shape[0]);
  const int channels = static_cast<int>(input_shape[1]);
  const int input_length = static_cast<int>(input_shape[2]);
  const int kernel_size = static_cast<int>(weight_shape[2]);
  const int state_length = kernel_size - 1;

  ORT_RETURN_IF(static_cast<int>(weight_shape[0]) != channels,
                "Weight first dim must match input channels");
  ORT_RETURN_IF(static_cast<int>(weight_shape[1]) != 1,
                "Weight second dim must be 1 for depthwise convolution");

  if (bias != nullptr) {
    ORT_RETURN_IF(bias->Shape().NumDimensions() != 1,
                  "Bias must be 1D");
    ORT_RETURN_IF(static_cast<int>(bias->Shape()[0]) != channels,
                  "Bias size must match channels");
  }

  if (conv_state != nullptr) {
    ORT_RETURN_IF(conv_state->Shape().NumDimensions() != 3,
                  "conv_state must be 3D (batch_size, channels, kernel_size - 1)");
    ORT_RETURN_IF(static_cast<int>(conv_state->Shape()[0]) != batch_size,
                  "conv_state batch_size must match input");
    ORT_RETURN_IF(static_cast<int>(conv_state->Shape()[1]) != channels,
                  "conv_state channels must match input");
    ORT_RETURN_IF(static_cast<int>(conv_state->Shape()[2]) != state_length,
                  "conv_state last dim must be kernel_size - 1");
  }

  const bool has_bias = (bias != nullptr);
  const bool has_conv_state = (conv_state != nullptr);

  // Allocate outputs
  // Output 0: (B, D, L)
  Tensor* output = context.Output(0, input_shape);

  // Output 1: present_state (B, D, K-1)
  std::vector<int64_t> state_dims{batch_size, channels, state_length};
  Tensor* present_state = context.Output(1, TensorShape(state_dims));

  if (input_length == 0) {
    return Status::OK();
  }

  // Create and run the shader program
  CausalConv1DWithStateProgram program{activation_, has_bias, has_conv_state, kernel_size};

  uint32_t output_size = static_cast<uint32_t>(batch_size * channels * input_length);

  program.AddInput({input, ProgramTensorMetadataDependency::Type})
      .AddInput({weight, ProgramTensorMetadataDependency::None});

  if (has_bias) {
    program.AddInput({bias, ProgramTensorMetadataDependency::None});
  }
  if (has_conv_state) {
    program.AddInput({conv_state, ProgramTensorMetadataDependency::None});
  }

  program.AddOutput({output, ProgramTensorMetadataDependency::None})
      .AddOutput({present_state, ProgramTensorMetadataDependency::None})
      .SetDispatchGroupSize((output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariable({static_cast<uint32_t>(batch_size)})
      .AddUniformVariable({static_cast<uint32_t>(channels)})
      .AddUniformVariable({static_cast<uint32_t>(input_length)})
      .AddUniformVariable({static_cast<uint32_t>(kernel_size)})
      .AddUniformVariable({static_cast<uint32_t>(state_length)})
      .AddUniformVariable({output_size});

  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
