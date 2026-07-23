// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/rnn/gru.h"

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace webgpu {

namespace {

std::string ActivationToWgslFn(const std::string& activation) {
  std::string lower;
  lower.reserve(activation.size());
  for (char c : activation) {
    lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
  }
  if (lower == "sigmoid") return "sigmoid_f";
  if (lower == "tanh") return "tanh_f";
  if (lower == "relu") return "relu_f";
  ORT_THROW("Unsupported GRU activation for WebGPU: ", activation);
}

// WGSL helpers shared by both compute passes.
void AppendActivationImpl(ShaderHelper& shader) {
  shader.AdditionalImplementation()
      << "fn sigmoid_f(v: f32) -> f32 {\n"
      << "  let clamped = clamp(v, -20.0, 20.0);\n"
      << "  return 1.0 / (1.0 + exp(-clamped));\n"
      << "}\n"
      << "fn tanh_f(v: f32) -> f32 {\n"
      << "  let clamped = clamp(v, -10.0, 10.0);\n"
      << "  return tanh(clamped);\n"
      << "}\n"
      << "fn relu_f(v: f32) -> f32 { return max(0.0, v); }\n\n";
}

}  // namespace

// ===========================================================================
// Gru constructor
// ===========================================================================
Gru::Gru(const OpKernelInfo& info) : WebGpuKernel(info) {
  std::string direction;
  info.GetAttrOrDefault("direction", &direction, std::string("forward"));
  direction_ = direction;
  ORT_ENFORCE(info.GetAttr("hidden_size", &hidden_size_).IsOK());
  float clip = 0.0f;
  if (info.GetAttr("clip", &clip).IsOK()) {
    clip_ = clip;
  } else {
    clip_ = 0.0f;
  }
  info.GetAttrOrDefault("linear_before_reset", &linear_before_reset_, static_cast<int64_t>(0));
  info.GetAttrOrDefault("layout", &layout_, static_cast<int64_t>(0));
  int num_directions = (direction_ == "bidirectional") ? 2 : 1;
  // GRU uses two activations per direction: f for the update/reset gates, g for the hidden gate.
  if (!info.GetAttrs("activations", activations_).IsOK() || activations_.empty()) {
    activations_.clear();
    for (int d = 0; d < num_directions; d++) {
      activations_.push_back("Sigmoid");
      activations_.push_back("Tanh");
    }
  }
  ORT_ENFORCE(activations_.size() == static_cast<size_t>(num_directions) * 2);
}

// ===========================================================================
// GruStateCopyProgram
// ===========================================================================
Status GruStateCopyProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("src", ShaderUsage::UseElementTypeAlias);
  if (has_seq_lens_) shader.AddInput("seq_lens", ShaderUsage::UseElementTypeAlias);
  shader.AddOutput("dst", ShaderUsage::UseElementTypeAlias);
  auto& body = shader.MainFunctionBody();
  body << "  let H = uniforms.hidden_size;\n"
       << "  let B = uniforms.batch_size;\n"
       << "  let dir = uniforms.direction;\n"
       << "  let num_dir = uniforms.num_directions;\n"
       << "  if (global_idx >= B * H) { return; }\n"
       << "  let batch_idx = global_idx / H;\n"
       << "  let j = global_idx % H;\n"
       << "  let flat_idx = batch_idx * H + j;\n";
  if (layout_ == 0) {
    body << "  let state_idx = (dir * B + batch_idx) * H + j;\n";
  } else {
    body << "  let state_idx = (batch_idx * num_dir + dir) * H + j;\n";
  }
  if (to_state_) {
    if (has_seq_lens_) {
      body << "  if (u32(seq_lens[batch_idx]) == 0u) {\n"
           << "    dst[state_idx] = dst_element_t(0.0);\n"
           << "    return;\n"
           << "  }\n";
    }
    body << "  dst[state_idx] = src[flat_idx];\n";
  } else {
    body << "  dst[flat_idx] = src[state_idx];\n";
  }
  return Status::OK();
}

// ===========================================================================
// GruGateProgram - compute update (z) and reset (r) gates for the whole [batch, H].
// ===========================================================================
Status GruGateProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("x", ShaderUsage::UseElementTypeAlias);
  shader.AddInput("w", ShaderUsage::UseElementTypeAlias);
  shader.AddInput("r", ShaderUsage::UseElementTypeAlias);
  shader.AddInput("h_prev", ShaderUsage::UseElementTypeAlias);
  if (has_bias_) shader.AddInput("b", ShaderUsage::UseElementTypeAlias);

  shader.AddOutput("z_out", ShaderUsage::UseElementTypeAlias);
  shader.AddOutput("reset_out", ShaderUsage::UseElementTypeAlias);

  AppendActivationImpl(shader);

  auto& body = shader.MainFunctionBody();
  body << "  let H = uniforms.hidden_size;\n"
       << "  let I = uniforms.input_size;\n"
       << "  let B = uniforms.batch_size;\n"
       << "  let dir = uniforms.direction;\n"
       << "  if (global_idx >= B * H) { return; }\n"
       << "  let batch_idx = global_idx / H;\n"
       << "  let j = global_idx % H;\n\n";

  // Gate accumulators. GRU W/R/B gate order is z, r, h.
  body << "  var gate_z: f32 = 0.0;\n"
       << "  var gate_r: f32 = 0.0;\n\n";

  // X * W^T
  body << "  let w_base = dir * 3u * H * I;\n";
  if (layout_ == 0) {
    body << "  let x_base = (uniforms.timestep * B + batch_idx) * I;\n";
  } else {
    body << "  let x_base = (batch_idx * uniforms.seq_length + uniforms.timestep) * I;\n";
  }
  body << "  for (var k: u32 = 0u; k < I; k++) {\n"
       << "    let xv = f32(x[x_base + k]);\n"
       << "    gate_z += xv * f32(w[w_base + j * I + k]);\n"
       << "    gate_r += xv * f32(w[w_base + (H + j) * I + k]);\n"
       << "  }\n\n";

  // H_prev * R^T  (h_prev is flat [batch, H])
  body << "  let r_base = dir * 3u * H * H;\n"
       << "  let h_base = batch_idx * H;\n"
       << "  for (var k: u32 = 0u; k < H; k++) {\n"
       << "    let hv = f32(h_prev[h_base + k]);\n"
       << "    gate_z += hv * f32(r[r_base + j * H + k]);\n"
       << "    gate_r += hv * f32(r[r_base + (H + j) * H + k]);\n"
       << "  }\n\n";

  // Bias: B = [Wbz, Wbr, Wbh, Rbz, Rbr, Rbh] per direction.
  if (has_bias_) {
    body << "  let bb = dir * 6u * H;\n"
         << "  gate_z += f32(b[bb + j]) + f32(b[bb + 3u * H + j]);\n"
         << "  gate_r += f32(b[bb + H + j]) + f32(b[bb + 4u * H + j]);\n\n";
  }

  if (has_clip_) {
    body << "  gate_z = clamp(gate_z, -uniforms.clip_value, uniforms.clip_value);\n"
         << "  gate_r = clamp(gate_r, -uniforms.clip_value, uniforms.clip_value);\n\n";
  }

  body << "  gate_z = " << f_activation_fn_ << "(gate_z);\n"
       << "  gate_r = " << f_activation_fn_ << "(gate_r);\n\n";

  body << "  let oi = batch_idx * H + j;\n"
       << "  z_out[oi] = z_out_element_t(gate_z);\n";
  if (linear_before_reset_) {
    // The hidden pass applies the reset gate after the recurrent matmul, so pass r[j] through.
    body << "  reset_out[oi] = reset_out_element_t(gate_r);\n";
  } else {
    // The hidden pass consumes (r (.) H_prev) directly, so fold the multiply in here.
    body << "  reset_out[oi] = reset_out_element_t(gate_r * f32(h_prev[oi]));\n";
  }
  return Status::OK();
}

// ===========================================================================
// GruHiddenProgram - compute hidden gate (h) and new hidden state.
// ===========================================================================
Status GruHiddenProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("x", ShaderUsage::UseElementTypeAlias);
  shader.AddInput("w", ShaderUsage::UseElementTypeAlias);
  shader.AddInput("r", ShaderUsage::UseElementTypeAlias);
  shader.AddInput("h_prev", ShaderUsage::UseElementTypeAlias);
  shader.AddInput("z", ShaderUsage::UseElementTypeAlias);
  shader.AddInput("reset", ShaderUsage::UseElementTypeAlias);
  if (has_bias_) shader.AddInput("b", ShaderUsage::UseElementTypeAlias);
  if (has_seq_lens_) shader.AddInput("seq_lens", ShaderUsage::UseElementTypeAlias);

  shader.AddOutput("h_new", ShaderUsage::UseElementTypeAlias);
  if (has_Y_) shader.AddOutput("y_out", ShaderUsage::UseElementTypeAlias);

  AppendActivationImpl(shader);

  auto& body = shader.MainFunctionBody();
  body << "  let H = uniforms.hidden_size;\n"
       << "  let I = uniforms.input_size;\n"
       << "  let B = uniforms.batch_size;\n"
       << "  let dir = uniforms.direction;\n"
       << "  let num_dir = uniforms.num_directions;\n"
       << "  if (global_idx >= B * H) { return; }\n"
       << "  let batch_idx = global_idx / H;\n"
       << "  let j = global_idx % H;\n"
       << "  let flat_idx = batch_idx * H + j;\n\n";

  // Sequence length masking - carry the previous hidden state forward past the batch's seq length.
  // Use timestep (not the processing step) so reverse direction masks the correct positions.
  if (has_seq_lens_) {
    body << "  let batch_seq_len = u32(seq_lens[batch_idx]);\n"
         << "  if (uniforms.timestep >= batch_seq_len) {\n"
         << "    h_new[flat_idx] = h_prev[flat_idx];\n";
    if (has_Y_) {
      if (layout_ == 0) {
        body << "    y_out[((uniforms.timestep * num_dir + dir) * B + batch_idx) * H + j] = y_out_element_t(0.0);\n";
      } else {
        body << "    y_out[((batch_idx * uniforms.seq_length + uniforms.timestep) * num_dir + dir) * H + j] = y_out_element_t(0.0);\n";
      }
    }
    body << "    return;\n"
         << "  }\n\n";
  }

  // X * Wh^T
  body << "  var gate_h: f32 = 0.0;\n"
       << "  let w_base = dir * 3u * H * I;\n";
  if (layout_ == 0) {
    body << "  let x_base = (uniforms.timestep * B + batch_idx) * I;\n";
  } else {
    body << "  let x_base = (batch_idx * uniforms.seq_length + uniforms.timestep) * I;\n";
  }
  body << "  for (var k: u32 = 0u; k < I; k++) {\n"
       << "    gate_h += f32(x[x_base + k]) * f32(w[w_base + (2u * H + j) * I + k]);\n"
       << "  }\n\n";

  // Recurrent term depends on linear_before_reset.
  body << "  let r_base = dir * 3u * H * H;\n"
       << "  let h_base = batch_idx * H;\n";
  if (linear_before_reset_) {
    // ht = g(Xt*Wh + r (.) (H_prev*Rh + Rbh) + Wbh)
    body << "  var rec: f32 = 0.0;\n"
         << "  for (var k: u32 = 0u; k < H; k++) {\n"
         << "    rec += f32(h_prev[h_base + k]) * f32(r[r_base + (2u * H + j) * H + k]);\n"
         << "  }\n";
    if (has_bias_) {
      body << "  rec += f32(b[dir * 6u * H + 5u * H + j]);\n";  // Rbh inside the reset gate
    }
    body << "  gate_h += f32(reset[h_base + j]) * rec;\n\n";
  } else {
    // ht = g(Xt*Wh + (r (.) H_prev)*Rh + Rbh + Wbh); reset[] already holds (r (.) H_prev).
    body << "  for (var k: u32 = 0u; k < H; k++) {\n"
         << "    gate_h += f32(reset[h_base + k]) * f32(r[r_base + (2u * H + j) * H + k]);\n"
         << "  }\n";
    if (has_bias_) {
      body << "  gate_h += f32(b[dir * 6u * H + 5u * H + j]);\n";  // Rbh
    }
    body << "\n";
  }

  if (has_bias_) {
    body << "  gate_h += f32(b[dir * 6u * H + 2u * H + j]);\n\n";  // Wbh
  }

  if (has_clip_) {
    body << "  gate_h = clamp(gate_h, -uniforms.clip_value, uniforms.clip_value);\n";
  }
  body << "  gate_h = " << g_activation_fn_ << "(gate_h);\n\n";

  // Ht = (1 - z) (.) h + z (.) H_prev
  body << "  let zv = f32(z[flat_idx]);\n"
       << "  let hu = (1.0 - zv) * gate_h + zv * f32(h_prev[flat_idx]);\n"
       << "  h_new[flat_idx] = h_new_element_t(hu);\n";
  if (has_Y_) {
    if (layout_ == 0) {
      body << "  y_out[((uniforms.timestep * num_dir + dir) * B + batch_idx) * H + j] = y_out_element_t(hu);\n";
    } else {
      body << "  y_out[((batch_idx * uniforms.seq_length + uniforms.timestep) * num_dir + dir) * H + j] = y_out_element_t(hu);\n";
    }
  }
  return Status::OK();
}

// ===========================================================================
// ComputeInternal
// ===========================================================================
Status Gru::ComputeInternal(ComputeContext& context) const {
  const auto* X = context.Input(0);
  const auto* W = context.Input(1);
  const auto* R = context.Input(2);
  const auto* B = context.Input(3);
  const auto* sequence_lens = context.Input(4);
  const auto* initial_h = context.Input(5);

  ORT_RETURN_IF(X == nullptr || W == nullptr || R == nullptr,
                "GRU: X, W, R are required inputs.");

  const auto& X_shape = X->Shape();
  int num_directions = (direction_ == "bidirectional") ? 2 : 1;
  int64_t seq_length, batch_size, input_size;
  if (layout_ == 0) {
    seq_length = X_shape[0];
    batch_size = X_shape[1];
    input_size = X_shape[2];
  } else {
    batch_size = X_shape[0];
    seq_length = X_shape[1];
    input_size = X_shape[2];
  }

  // Validate that the W and R weight shapes are consistent with the hidden_size attribute. ONNX shape
  // inference does not verify this relationship, so an inconsistent model (e.g. a bogus hidden_size)
  // would otherwise be used to drive the cell computation and read out of bounds from the W/R buffers.
  const auto& W_shape = W->Shape();
  const auto& R_shape = R->Shape();
  ORT_RETURN_IF(W_shape.NumDimensions() != 3 || W_shape[0] != num_directions ||
                    W_shape[1] != 3 * hidden_size_ || W_shape[2] != input_size,
                "GRU: Input W must have shape {", num_directions, ", 3*", hidden_size_, ", ", input_size,
                "}. Actual: ", W_shape);
  ORT_RETURN_IF(R_shape.NumDimensions() != 3 || R_shape[0] != num_directions ||
                    R_shape[1] != 3 * hidden_size_ || R_shape[2] != hidden_size_,
                "GRU: Input R must have shape {", num_directions, ", 3*", hidden_size_, ", ", hidden_size_,
                "}. Actual: ", R_shape);

  uint32_t H = static_cast<uint32_t>(hidden_size_);

  // Output shapes
  TensorShape Y_shape, Y_h_shape;
  if (layout_ == 0) {
    Y_shape = TensorShape({seq_length, static_cast<int64_t>(num_directions), batch_size, hidden_size_});
    Y_h_shape = TensorShape({static_cast<int64_t>(num_directions), batch_size, hidden_size_});
  } else {
    Y_shape = TensorShape({batch_size, seq_length, static_cast<int64_t>(num_directions), hidden_size_});
    Y_h_shape = TensorShape({batch_size, static_cast<int64_t>(num_directions), hidden_size_});
  }

  auto* Y = context.Output(0, Y_shape);
  auto* Y_h = context.Output(1, Y_h_shape);
  bool has_Y = (Y != nullptr);
  bool has_Y_h = (Y_h != nullptr);
  if (!has_Y && !has_Y_h) {
    return Status::OK();
  }

  TensorShape state_shape({batch_size, hidden_size_});
  auto dtype = X->DataType();
  uint32_t total_threads = static_cast<uint32_t>(batch_size) * H;
  uint32_t wg_size = std::min(total_threads, 256u);
  if (wg_size == 0) wg_size = 1;
  uint32_t num_groups = (total_threads + wg_size - 1) / wg_size;

  // sequence_lens is on GPU (no InputMemoryType CPU override)
  bool has_seq_lens = (sequence_lens != nullptr);
  bool lbr = (linear_before_reset_ != 0);
  bool has_clip = (clip_ > 0.0f);

  if (seq_length == 0) {
    if (has_Y_h) {
      context.FillZero(*Y_h);
    }
    return Status::OK();
  }

  auto copy_from_state = [&](const Tensor* src, Tensor* dst, int dir) -> Status {
    GruStateCopyProgram prog{/*to_state=*/false, static_cast<int>(layout_)};
    prog.CacheHint("from", std::to_string(layout_));
    prog.SetWorkgroupSize(wg_size).SetDispatchGroupSize(num_groups);
    prog.AddInputs({{src, ProgramTensorMetadataDependency::Type}});
    prog.AddOutputs({{dst, ProgramTensorMetadataDependency::None}});
    prog.AddUniformVariables({
        {static_cast<uint32_t>(batch_size)},
        {H},
        {static_cast<uint32_t>(dir)},
        {static_cast<uint32_t>(num_directions)},
    });
    return context.RunProgram(prog);
  };

  auto copy_to_state = [&](Tensor* src, Tensor* dst, int dir) -> Status {
    GruStateCopyProgram prog{/*to_state=*/true, static_cast<int>(layout_), has_seq_lens};
    prog.CacheHint("to", std::to_string(layout_), std::to_string(has_seq_lens));
    prog.SetWorkgroupSize(wg_size).SetDispatchGroupSize(num_groups);
    prog.AddInputs({{src, ProgramTensorMetadataDependency::Type}});
    if (has_seq_lens) prog.AddInputs({{sequence_lens, ProgramTensorMetadataDependency::Type}});
    prog.AddOutputs({{dst, ProgramTensorMetadataDependency::None}});
    prog.AddUniformVariables({
        {static_cast<uint32_t>(batch_size)},
        {H},
        {static_cast<uint32_t>(dir)},
        {static_cast<uint32_t>(num_directions)},
    });
    return context.RunProgram(prog);
  };

  for (int dir = 0; dir < num_directions; dir++) {
    std::string fa = ActivationToWgslFn(activations_[dir * 2 + 0]);
    std::string ga = ActivationToWgslFn(activations_[dir * 2 + 1]);

    bool is_reverse = (direction_ == "reverse") || (direction_ == "bidirectional" && dir == 1);

    // Ping-pong hidden state buffers [batch, H]; z/reset scratch reused each timestep.
    Tensor H_a = context.CreateGPUTensor(dtype, state_shape);
    Tensor H_b = context.CreateGPUTensor(dtype, state_shape);
    Tensor Z = context.CreateGPUTensor(dtype, state_shape);
    Tensor Reset = context.CreateGPUTensor(dtype, state_shape);

    if (initial_h != nullptr) {
      ORT_RETURN_IF_ERROR(copy_from_state(initial_h, &H_a, dir));
    } else {
      context.FillZero(H_a);
    }

    for (int64_t t = 0; t < seq_length; t++) {
      int64_t timestep = is_reverse ? (seq_length - 1 - t) : t;

      const Tensor* h_read = (t % 2 == 0) ? &H_a : &H_b;
      Tensor* h_write = (t % 2 == 0) ? &H_b : &H_a;

      // Pass 1: update and reset gates.
      GruGateProgram gate_prog{B != nullptr, lbr, has_clip, static_cast<int>(layout_), fa};
      gate_prog.CacheHint(std::to_string(B != nullptr), std::to_string(lbr),
                          std::to_string(has_clip), std::to_string(layout_), fa);
      gate_prog.SetWorkgroupSize(wg_size).SetDispatchGroupSize(num_groups);
      gate_prog.AddInputs({{X, ProgramTensorMetadataDependency::Type}})
          .AddInputs({{W, ProgramTensorMetadataDependency::Type}})
          .AddInputs({{R, ProgramTensorMetadataDependency::Type}})
          .AddInputs({{h_read, ProgramTensorMetadataDependency::Type}});
      if (B != nullptr) gate_prog.AddInputs({{B, ProgramTensorMetadataDependency::Type}});
      gate_prog.AddOutputs({{&Z, ProgramTensorMetadataDependency::None}})
          .AddOutputs({{&Reset, ProgramTensorMetadataDependency::None}});
      gate_prog.AddUniformVariables({
          {static_cast<uint32_t>(batch_size)},
          {static_cast<uint32_t>(input_size)},
          {H},
          {static_cast<uint32_t>(dir)},
          {static_cast<uint32_t>(num_directions)},
          {static_cast<uint32_t>(timestep)},
          {static_cast<uint32_t>(seq_length)},
          {clip_},
      });
      ORT_RETURN_IF_ERROR(context.RunProgram(gate_prog));

      // Pass 2: hidden gate and new state.
      GruHiddenProgram hidden_prog{B != nullptr, has_Y, has_seq_lens, lbr, has_clip,
                                   static_cast<int>(layout_), ga};
      hidden_prog.CacheHint(std::to_string(B != nullptr), std::to_string(has_Y),
                            std::to_string(has_seq_lens), std::to_string(lbr),
                            std::to_string(has_clip), std::to_string(layout_), ga);
      hidden_prog.SetWorkgroupSize(wg_size).SetDispatchGroupSize(num_groups);
      hidden_prog.AddInputs({{X, ProgramTensorMetadataDependency::Type}})
          .AddInputs({{W, ProgramTensorMetadataDependency::Type}})
          .AddInputs({{R, ProgramTensorMetadataDependency::Type}})
          .AddInputs({{h_read, ProgramTensorMetadataDependency::Type}})
          .AddInputs({{&Z, ProgramTensorMetadataDependency::Type}})
          .AddInputs({{&Reset, ProgramTensorMetadataDependency::Type}});
      if (B != nullptr) hidden_prog.AddInputs({{B, ProgramTensorMetadataDependency::Type}});
      if (has_seq_lens) hidden_prog.AddInputs({{sequence_lens, ProgramTensorMetadataDependency::Type}});
      hidden_prog.AddOutputs({{h_write, ProgramTensorMetadataDependency::None}});
      if (has_Y) hidden_prog.AddOutputs({{Y, ProgramTensorMetadataDependency::None}});
      hidden_prog.AddUniformVariables({
          {static_cast<uint32_t>(batch_size)},
          {static_cast<uint32_t>(input_size)},
          {H},
          {static_cast<uint32_t>(dir)},
          {static_cast<uint32_t>(num_directions)},
          {static_cast<uint32_t>(timestep)},
          {static_cast<uint32_t>(seq_length)},
          {clip_},
      });
      ORT_RETURN_IF_ERROR(context.RunProgram(hidden_prog));
    }

    Tensor* final_h = (seq_length % 2 == 1) ? &H_b : &H_a;
    if (has_Y_h) {
      ORT_RETURN_IF_ERROR(copy_to_state(final_h, Y_h, dir));
    }
  }

  return Status::OK();
}

// ===========================================================================
// Kernel registrations
// ===========================================================================
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    GRU,
    kOnnxDomain,
    7, 13,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int32_t>()),
    Gru);

ONNX_OPERATOR_KERNEL_EX(
    GRU,
    kOnnxDomain,
    14,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int32_t>()),
    Gru);

}  // namespace webgpu
}  // namespace onnxruntime
