// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/rnn/gru.h"

#include <algorithm>
#include <cctype>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/webgpu_utils.h"

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
  // Validate activations while loading the model rather than waiting until the first Run().
  for (const auto& activation : activations_) {
    static_cast<void>(ActivationToWgslFn(activation));
  }
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
    body << "  dst[state_idx] = dst_element_t(src[flat_idx]);\n";
  } else {
    body << "  dst[flat_idx] = dst_element_t(src[state_idx]);\n";
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
// Input validation shared by both paths
// ===========================================================================
Status Gru::ValidateInputs(ComputeContext& context, int64_t& seq_length, int64_t& batch_size,
                           int64_t& input_size) const {
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
  ORT_RETURN_IF(hidden_size_ <= 0 ||
                    hidden_size_ > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
                "GRU: hidden_size must be in the range [1, ", std::numeric_limits<uint32_t>::max(),
                "]; actual: ", hidden_size_);
  ORT_RETURN_IF(X_shape.NumDimensions() != 3,
                "GRU: Input X must have 3 dimensions. Actual: ", X_shape);
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

  if (B != nullptr) {
    const auto& B_shape = B->Shape();
    ORT_RETURN_IF(B_shape.NumDimensions() != 2 || B_shape[0] != num_directions ||
                      B_shape[1] != 6 * hidden_size_,
                  "GRU: Input B must have shape {", num_directions, ", 6*", hidden_size_,
                  "}. Actual: ", B_shape);
  }
  if (sequence_lens != nullptr) {
    const auto& sequence_lens_shape = sequence_lens->Shape();
    ORT_RETURN_IF(sequence_lens_shape.NumDimensions() != 1 || sequence_lens_shape[0] != batch_size,
                  "GRU: Input sequence_lens must have shape {", batch_size,
                  "}. Actual: ", sequence_lens_shape);
  }
  if (initial_h != nullptr) {
    const auto& initial_h_shape = initial_h->Shape();
    const bool initial_h_shape_valid =
        (layout_ == 0 && initial_h_shape.NumDimensions() == 3 &&
         initial_h_shape[0] == num_directions && initial_h_shape[1] == batch_size &&
         initial_h_shape[2] == hidden_size_) ||
        (layout_ == 1 && initial_h_shape.NumDimensions() == 3 &&
         initial_h_shape[0] == batch_size && initial_h_shape[1] == num_directions &&
         initial_h_shape[2] == hidden_size_);
    ORT_RETURN_IF(!initial_h_shape_valid,
                  "GRU: Input initial_h has an invalid shape for layout ", layout_,
                  ". Actual: ", initial_h_shape);
  }

  return Status::OK();
}

// ===========================================================================
// Stepwise fallback: two kernels per timestep and direction
// ===========================================================================
Status Gru::ComputeStepwise(ComputeContext& context) const {
  int64_t seq_length = 0, batch_size = 0, input_size = 0;
  ORT_RETURN_IF_ERROR(ValidateInputs(context, seq_length, batch_size, input_size));
  const auto* X = context.Input(0);
  const auto* W = context.Input(1);
  const auto* R = context.Input(2);
  const auto* B = context.Input(3);
  const auto* sequence_lens = context.Input(4);
  const auto* initial_h = context.Input(5);
  int num_directions = (direction_ == "bidirectional") ? 2 : 1;

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
  // The recurrent state stays in f32 so float16 models don't round it at every timestep.
  const auto state_dtype = DataTypeImpl::GetType<float>();
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
    prog.AddOutputs({{dst, ProgramTensorMetadataDependency::Type}});
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
    prog.AddOutputs({{dst, ProgramTensorMetadataDependency::Type}});
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
    Tensor H_a = context.CreateGPUTensor(state_dtype, state_shape);
    Tensor H_b = context.CreateGPUTensor(state_dtype, state_shape);
    Tensor Z = context.CreateGPUTensor(state_dtype, state_shape);
    Tensor Reset = context.CreateGPUTensor(state_dtype, state_shape);

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
// GEMM path: one GEMM for the input projection of every timestep, then per timestep batched GEMMs for
// the recurrent projections of every direction and fused element-wise gate kernels.
// ===========================================================================
namespace {

uint32_t Groups(uint32_t size) { return (size + 255) / 256; }

// Index of (direction d, batch b, unit j) in an ONNX Y_h / initial_h tensor.
std::string StateTensorIndex(int layout) {
  return layout == 0 ? "(d * B + b) * H + j" : "(b * uniforms.num_directions + d) * H + j";
}

}  // namespace

Status GruPackWeightsProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& src = shader.AddInput("src", ShaderUsage::UseUniform);
  const auto& dst = shader.AddOutput("dst", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.size")
                            << "let per_dir = uniforms.rows * uniforms.cols;\n"
                            << "let d = global_idx / per_dir;\n"
                            << "let rem = global_idx % per_dir;\n"
                            << "let k = rem / uniforms.rows;\n"
                            << "let n = rem % uniforms.rows;\n"
                            << dst.SetByOffset("global_idx", "dst_element_t(" + src.GetByOffset("(d * uniforms.src_rows + uniforms.row_offset + n) * uniforms.cols + k") + ")")
                            << "\n";
  return Status::OK();
}

Status GruStateTransferProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& src = shader.AddInput("src", ShaderUsage::UseUniform);
  const ShaderVariableHelper* seq_lens = has_seq_lens_ ? &shader.AddInput("seq_lens", ShaderUsage::UseUniform) : nullptr;
  const auto& dst = shader.AddOutput("dst", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  auto& body = shader.MainFunctionBody();
  body << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.size")
       << "let H = uniforms.hidden_size;\n"
       << "let B = uniforms.batch_size;\n"
       << "let d = global_idx / (B * H);\n"
       << "let b = (global_idx / H) % B;\n"
       << "let j = global_idx % H;\n"
       << "let ext = " << StateTensorIndex(layout_) << ";\n";
  if (to_output_) {
    body << "var v = f32(" << src.GetByOffset("global_idx") << ");\n";
    if (seq_lens != nullptr) {
      body << "if (u32(" << seq_lens->GetByOffset("b") << ") == 0u) { v = 0.0; }\n";
    }
    body << dst.SetByOffset("ext", "dst_element_t(v)") << "\n";
  } else {
    body << dst.SetByOffset("global_idx", "dst_element_t(" + src.GetByOffset("ext") + ")") << "\n";
  }
  return Status::OK();
}

Status GruGatesProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const bool final_stage = stage_ == Stage::kHidden;
  const auto& xw = shader.AddInput("xw", ShaderUsage::UseUniform);
  const auto& g = shader.AddInput("g", ShaderUsage::UseUniform);
  const auto& h_prev = shader.AddInput("h_prev", ShaderUsage::UseUniform);
  const ShaderVariableHelper* z_in = stage_ == Stage::kHidden ? &shader.AddInput("z_in", ShaderUsage::UseUniform) : nullptr;
  const ShaderVariableHelper* bias = has_bias_ ? &shader.AddInput("bias", ShaderUsage::UseUniform) : nullptr;
  const ShaderVariableHelper* seq_lens =
      final_stage && has_seq_lens_ ? &shader.AddInput("seq_lens", ShaderUsage::UseUniform) : nullptr;
  const ShaderVariableHelper* h_new = nullptr;
  const ShaderVariableHelper* y = nullptr;
  const ShaderVariableHelper* z_out = nullptr;
  const ShaderVariableHelper* rh_out = nullptr;
  if (final_stage) {
    h_new = &shader.AddOutput("h_new", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
    if (has_Y_) y = &shader.AddOutput("y", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  } else {
    z_out = &shader.AddOutput("z_out", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
    rh_out = &shader.AddOutput("rh_out", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  }

  auto& impl = shader.AdditionalImplementation();
  AppendActivationImpl(shader);
  // Per-direction activations f (update and reset gates) and g (hidden gate).
  const size_t num_directions = activations_.size() / 2;
  for (size_t role = 0; role < 2; ++role) {
    impl << "fn act" << role << "(d: u32, v: f32) -> f32 {\n";
    for (size_t d = 1; d < num_directions; ++d) {
      impl << "  if (d == " << d << "u) { return " << activations_[d * 2 + role] << "(v); }\n";
    }
    impl << "  return " << activations_[role] << "(v);\n}\n";
  }
  auto clip = [&](const std::string& v) {
    return has_clip_ ? v + " = clamp(" + v + ", -uniforms.clip_value, uniforms.clip_value);\n" : std::string{};
  };
  // Bias B = [Wbz, Wbr, Wbh, Rbz, Rbr, Rbh] per direction; k selects the block.
  auto bias_at = [&](uint32_t k) { return "f32(" + bias->GetByOffset("bb + " + std::to_string(k) + "u * H") + ")"; };

  auto& body = shader.MainFunctionBody();
  body << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.size")
       << "let H = uniforms.hidden_size;\n"
       << "let B = uniforms.batch_size;\n"
       << "let S = uniforms.seq_length;\n"
       << "let D = uniforms.num_directions;\n"
       << "let d = global_idx / (B * H);\n"
       << "let b = (global_idx / H) % B;\n"
       << "let j = global_idx % H;\n"
       << "let reverse = ((uniforms.reverse_mask >> d) & 1u) == 1u;\n"
       << "let t = select(uniforms.step, S - 1u - uniforms.step, reverse);\n"
       << "let row = " << (layout_ == 0 ? "t * B + b" : "b * S + t") << ";\n"
       << "let xb = (d * S * B + row) * 3u * H + j;\n"
       << "let hpv = f32(" << h_prev.GetByOffset("global_idx") << ");\n";
  if (bias != nullptr) {
    body << "let bb = d * 6u * H + j;\n";
  }
  if (y != nullptr) {
    body << "let yi = " << (layout_ == 0 ? "((t * D + d) * B + b) * H + j" : "((b * S + t) * D + d) * H + j") << ";\n";
  }
  // Past a batch entry's sequence length the state carries through and Y is zero.
  if (seq_lens != nullptr) {
    body << "if (t >= u32(" << seq_lens->GetByOffset("b") << ")) {\n"
         << "  " << h_new->SetByOffset("global_idx", "h_new_element_t(hpv)") << "\n";
    if (y != nullptr) {
      body << "  " << y->SetByOffset("yi", "y_element_t(0.0)") << "\n";
    }
    body << "  return;\n}\n";
  }

  if (stage_ == Stage::kReset) {
    body << "let gb = (d * B + b) * 2u * H + j;\n"
         << "var gate_z = f32(" << xw.GetByOffset("xb") << ") + f32(" << g.GetByOffset("gb") << ");\n"
         << "var gate_r = f32(" << xw.GetByOffset("xb + H") << ") + f32(" << g.GetByOffset("gb + H") << ");\n";
    if (bias != nullptr) {
      body << "gate_z += " << bias_at(0) << " + " << bias_at(3) << ";\n"
           << "gate_r += " << bias_at(1) << " + " << bias_at(4) << ";\n";
    }
    body << clip("gate_z") << clip("gate_r")
         << "gate_z = act0(d, gate_z);\n"
         << "gate_r = act0(d, gate_r);\n";
    body << z_out->SetByOffset("global_idx", "z_out_element_t(gate_z)") << "\n"
         << rh_out->SetByOffset("global_idx", "rh_out_element_t(gate_r * hpv)") << "\n";
    return Status::OK();
  }

  body << "var gate_h = f32(" << xw.GetByOffset("xb + 2u * H") << ");\n";
  if (bias != nullptr) {
    body << "gate_h += " << bias_at(2) << ";\n";
  }
  // ht = g(Xt * Wh + (r (.) H_prev) * Rh + Rbh + Wbh)
  body << "let gate_z = f32(" << z_in->GetByOffset("global_idx") << ");\n"
       << "gate_h += f32(" << g.GetByOffset("global_idx") << ");\n";
  if (bias != nullptr) {
    body << "gate_h += " << bias_at(5) << ";\n";
  }
  body << clip("gate_h")
       << "gate_h = act1(d, gate_h);\n"
       << "let hu = (1.0 - gate_z) * gate_h + gate_z * hpv;\n"
       << h_new->SetByOffset("global_idx", "h_new_element_t(hu)") << "\n";
  if (y != nullptr) {
    body << y->SetByOffset("yi", "y_element_t(hu)") << "\n";
  }
  return Status::OK();
}

Status GruFusedStepProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& xw = shader.AddInput("xw", ShaderUsage::UseUniform);
  const auto& r_t = shader.AddInput("r_t", ShaderUsage::UseUniform);
  const auto& h_prev = shader.AddInput("h_prev", ShaderUsage::UseUniform);
  const ShaderVariableHelper* bias = has_bias_ ? &shader.AddInput("bias", ShaderUsage::UseUniform) : nullptr;
  const ShaderVariableHelper* seq_lens = has_seq_lens_ ? &shader.AddInput("seq_lens", ShaderUsage::UseUniform) : nullptr;
  const auto& h_new = shader.AddOutput("h_new", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  const ShaderVariableHelper* y =
      has_Y_ ? &shader.AddOutput("y", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias) : nullptr;

  constexpr uint32_t kTileK = 64;
  const uint32_t tile_batch = 256 / tile_units_ * rows_per_thread_;  // hidden-state values staged in workgroup memory per pass
  auto& impl = shader.AdditionalImplementation();
  AppendActivationImpl(shader);
  const size_t num_directions = activations_.size() / 2;
  for (size_t role = 0; role < 2; ++role) {
    impl << "fn act" << role << "(d: u32, v: f32) -> f32 {\n";
    for (size_t d = 1; d < num_directions; ++d) {
      impl << "  if (d == " << d << "u) { return " << activations_[d * 2 + role] << "(v); }\n";
    }
    impl << "  return " << activations_[role] << "(v);\n}\n";
  }
  impl << "var<workgroup> h_tile: array<array<f32, " << kTileK << ">, " << tile_batch << ">;\n";
  auto clip = [&](const std::string& v) {
    return has_clip_ ? "    " + v + " = clamp(" + v + ", -uniforms.clip_value, uniforms.clip_value);\n" : std::string{};
  };
  auto bias_at = [&](uint32_t k) { return "f32(" + bias->GetByOffset("bb + " + std::to_string(k) + "u * H") + ")"; };

  auto& body = shader.MainFunctionBody();
  body << "if (workgroup_idx >= uniforms.num_tiles) { return; }\n"
       << "let H = uniforms.hidden_size;\n"
       << "let B = uniforms.batch_size;\n"
       << "let S = uniforms.seq_length;\n"
       << "let D = uniforms.num_directions;\n"
       << "let unit_tile = workgroup_idx % uniforms.unit_tiles;\n"
       << "let batch_tile = (workgroup_idx / uniforms.unit_tiles) % uniforms.batch_tiles;\n"
       << "let d = workgroup_idx / (uniforms.unit_tiles * uniforms.batch_tiles);\n"
       << "let j = unit_tile * " << tile_units_ << "u + local_idx % " << tile_units_ << "u;\n"
       << "let row0 = (local_idx / " << tile_units_ << "u) * " << rows_per_thread_ << "u;\n"
       << "let b0 = batch_tile * " << tile_batch << "u;\n"
       << "var acc_z: array<f32, " << rows_per_thread_ << ">;\n"
       << "var acc_r: array<f32, " << rows_per_thread_ << ">;\n"
       << "var acc_h: array<f32, " << rows_per_thread_ << ">;\n"
       << "for (var k0 = 0u; k0 < H; k0 += " << kTileK << "u) {\n"
       << "  for (var i = local_idx; i < " << tile_batch * kTileK << "u; i += 256u) {\n"
       << "    let tb = i / " << kTileK << "u;\n"
       << "    let k = k0 + i % " << kTileK << "u;\n"
       << "    let b = b0 + tb;\n"
       << "    var v = 0.0;\n"
       << "    if (b < B && k < H) { v = f32(" << h_prev.GetByOffset("(d * B + b) * H + k") << "); }\n"
       << "    h_tile[tb][i % " << kTileK << "u] = v;\n"
       << "  }\n"
       << "  workgroupBarrier();\n"
       << "  if (j < H) {\n"
       << "    let kn = min(" << kTileK << "u, H - k0);\n"
       << "    for (var kk = 0u; kk < kn; kk++) {\n"
       << "      let rb = (d * H + k0 + kk) * 3u * H + j;\n"
       << "      let rz = f32(" << r_t.GetByOffset("rb") << ");\n"
       << "      let rr = f32(" << r_t.GetByOffset("rb + H") << ");\n"
       << "      let rh = f32(" << r_t.GetByOffset("rb + 2u * H") << ");\n"
       << "      for (var i = 0u; i < " << rows_per_thread_ << "u; i++) {\n"
       << "        let hv = h_tile[row0 + i][kk];\n"
       << "        acc_z[i] = fma(hv, rz, acc_z[i]);\n"
       << "        acc_r[i] = fma(hv, rr, acc_r[i]);\n"
       << "        acc_h[i] = fma(hv, rh, acc_h[i]);\n"
       << "      }\n"
       << "    }\n"
       << "  }\n"
       << "  workgroupBarrier();\n"
       << "}\n"
       << "if (j >= H) { return; }\n"
       << "let reverse = ((uniforms.reverse_mask >> d) & 1u) == 1u;\n"
       << "let t = select(uniforms.step, S - 1u - uniforms.step, reverse);\n";
  if (bias != nullptr) {
    body << "let bb = d * 6u * H + j;\n";
  }
  body << "for (var i = 0u; i < " << rows_per_thread_ << "u; i++) {\n"
       << "  let b = b0 + row0 + i;\n"
       << "  if (b >= B) { break; }\n"
       << "  let si = (d * B + b) * H + j;\n"
       << "  let hpv = f32(" << h_prev.GetByOffset("si") << ");\n";
  if (y != nullptr) {
    body << "  let yi = " << (layout_ == 0 ? "((t * D + d) * B + b) * H + j" : "((b * S + t) * D + d) * H + j") << ";\n";
  }
  // Past a batch entry's sequence length the state carries through and Y is zero.
  if (seq_lens != nullptr) {
    body << "  if (t >= u32(" << seq_lens->GetByOffset("b") << ")) {\n"
         << "    " << h_new.SetByOffset("si", "h_new_element_t(hpv)") << "\n";
    if (y != nullptr) {
      body << "    " << y->SetByOffset("yi", "y_element_t(0.0)") << "\n";
    }
    body << "    continue;\n  }\n";
  }
  body << "  let row = " << (layout_ == 0 ? "t * B + b" : "b * S + t") << ";\n"
       << "  let xb = (d * S * B + row) * 3u * H + j;\n"
       << "  var gate_z = f32(" << xw.GetByOffset("xb") << ") + acc_z[i];\n"
       << "  var gate_r = f32(" << xw.GetByOffset("xb + H") << ") + acc_r[i];\n"
       << "  var gate_h = f32(" << xw.GetByOffset("xb + 2u * H") << ");\n"
       << "  var rec = acc_h[i];\n";
  if (bias != nullptr) {
    body << "  gate_z += " << bias_at(0) << " + " << bias_at(3) << ";\n"
         << "  gate_r += " << bias_at(1) << " + " << bias_at(4) << ";\n"
         << "  gate_h += " << bias_at(2) << ";\n"
         << "  rec += " << bias_at(5) << ";\n";
  }
  body << clip("gate_z") << clip("gate_r")
       << "  gate_z = act0(d, gate_z);\n"
       << "  gate_r = act0(d, gate_r);\n"
       // ht = g(Xt * Wh + r (.) (H_prev * Rh + Rbh) + Wbh)
       << "  gate_h += gate_r * rec;\n"
       << clip("gate_h")
       << "  gate_h = act1(d, gate_h);\n"
       << "  let hu = (1.0 - gate_z) * gate_h + gate_z * hpv;\n"
       << "  " << h_new.SetByOffset("si", "h_new_element_t(hu)") << "\n";
  if (y != nullptr) {
    body << "  " << y->SetByOffset("yi", "y_element_t(hu)") << "\n";
  }
  body << "}\n";
  return Status::OK();
}

namespace {

// Packs rows [row_offset, row_offset + rows) of W [D, 3H, I] or R [D, 3H, H] into `packed` [D, K, rows].
Status PackGruWeights(ComputeContextBase& context, const Tensor& weights, uint32_t row_offset, uint32_t rows,
                      Tensor& packed) {
  const TensorShape& shape = weights.Shape();
  const uint32_t cols = narrow<uint32_t>(shape[2]);
  const uint32_t size = narrow<uint32_t>(shape[0]) * rows * cols;
  GruPackWeightsProgram program;
  program.AddInput({&weights, ProgramTensorMetadataDependency::Type})
      .AddOutput({&packed, ProgramTensorMetadataDependency::Type})
      .SetDispatchGroupSize(Groups(size))
      .SetWorkgroupSize(256)
      .AddUniformVariables({{size}, {narrow<uint32_t>(shape[1])}, {row_offset}, {rows}, {cols}});
  return context.RunProgram(program);
}

// Element-wise copy of `src` into the f32 tensor `dst`: the pack program on a [1, size, 1] view is a cast.
Status ConvertGruInputToF32(ComputeContextBase& context, const Tensor& src, Tensor& dst) {
  const int64_t size = src.Shape().Size();
  const Tensor src_view = CreateTensorView(src, TensorShape({1, size, 1}));
  Tensor dst_view = CreateTensorView(dst, TensorShape({1, 1, size}));
  return PackGruWeights(context, src_view, 0, narrow<uint32_t>(size), dst_view);
}

// The packed GEMM operands for W or R: W^T, and R^T or ([Rz; Rr]^T, Rh^T) depending on linear_before_reset.
Status PackGruOperand(ComputeContextBase& context, const Tensor& weights, bool is_r, bool linear_before_reset,
                      const std::function<Tensor(const TensorShape&)>& create, Tensor& first, Tensor* second) {
  const TensorShape& shape = weights.Shape();
  const int64_t d = shape[0];
  const int64_t k = shape[2];
  const auto three_h = narrow<uint32_t>(shape[1]);
  const uint32_t h = three_h / 3;
  if (!is_r || linear_before_reset) {
    first = create(TensorShape({d, k, three_h}));
    return PackGruWeights(context, weights, 0, three_h, first);
  }
  first = create(TensorShape({d, k, 2 * h}));
  ORT_RETURN_IF_ERROR(PackGruWeights(context, weights, 0, 2 * h, first));
  *second = create(TensorShape({d, k, h}));
  return PackGruWeights(context, weights, 2 * h, h, *second);
}

// Whether a weight of `elements` values fits one buffer and one binding once packed to f32.
bool FitsF32Buffer(const wgpu::Limits& limits, uint64_t elements) {
  return elements * sizeof(float) <= std::min<uint64_t>(limits.maxBufferSize, limits.maxStorageBufferBindingSize);
}

}  // namespace

Status Gru::PrePackInternal(ComputeContextBase& context, const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                            /*out*/ bool& is_packed) {
  // The original weights stay alive: the stepwise fallback and shape validation read them.
  is_packed = false;
  const TensorShape& shape = tensor.Shape();
  if ((input_idx != 1 && input_idx != 2) || shape.NumDimensions() != 3 || shape.Size() == 0 || shape[1] % 3 != 0 ||
      !FitsF32Buffer(context.DeviceLimits(), static_cast<uint64_t>(shape.Size()))) {
    return Status::OK();
  }
  // Both projections run in f32, like the stepwise kernel, so float16 models keep f32 gate preactivations.
  const auto f32 = DataTypeImpl::GetType<float>();
  auto create = [&](const TensorShape& s) { return Tensor(f32, s, alloc); };
  Tensor first, second;
  ORT_RETURN_IF_ERROR(PackGruOperand(context, tensor, input_idx == 2, linear_before_reset_ != 0, create, first, &second));
  if (input_idx == 1) {
    packed_w_ = std::make_unique<Tensor>(std::move(first));
  } else {
    packed_r_ = std::make_unique<Tensor>(std::move(first));
    if (linear_before_reset_ == 0) {
      packed_rh_ = std::make_unique<Tensor>(std::move(second));
    }
  }
  return Status::OK();
}

Status Gru::ComputeInternal(ComputeContext& context) const {
  const auto* X = context.Input(0);
  const auto* W = context.Input(1);
  const auto* R = context.Input(2);
  if (X == nullptr || W == nullptr || R == nullptr || X->Shape().NumDimensions() != 3) {
    return ComputeStepwise(context);  // reports the error
  }
  const int64_t num_directions = direction_ == "bidirectional" ? 2 : 1;
  const int64_t rows = X->Shape()[0] * X->Shape()[1];  // seq_length * batch_size in either layout
  // The f32 input projection for every timestep and direction (and, for float16, the f32 copy of X) must
  // fit in one storage binding.
  uint64_t max_bytes = static_cast<uint64_t>(num_directions) * rows * 3 * hidden_size_ * sizeof(float);
  if (X->DataType() != DataTypeImpl::GetType<float>()) {
    max_bytes = std::max<uint64_t>(max_bytes, static_cast<uint64_t>(X->Shape().Size()) * sizeof(float));
  }
  const auto& limits = context.DeviceLimits();
  if (rows == 0 || hidden_size_ <= 0 || max_bytes > limits.maxStorageBufferBindingSize ||
      max_bytes > limits.maxBufferSize) {
    return ComputeStepwise(context);
  }
  // The f32 packed weights must fit too, whether prepacked or packed for this run (the recurrent split
  // layouts are no larger than R).
  if (!FitsF32Buffer(limits, static_cast<uint64_t>(W->Shape().Size())) ||
      !FitsF32Buffer(limits, static_cast<uint64_t>(R->Shape().Size()))) {
    return ComputeStepwise(context);
  }
  return ComputeWithGemm(context);
}

Status Gru::ComputeWithGemm(ComputeContext& context) const {
  int64_t seq_length = 0, batch_size = 0, input_size = 0;
  ORT_RETURN_IF_ERROR(ValidateInputs(context, seq_length, batch_size, input_size));
  const auto* X = context.Input(0);
  const auto* W = context.Input(1);
  const auto* R = context.Input(2);
  const auto* B = context.Input(3);
  const auto* sequence_lens = context.Input(4);
  const auto* initial_h = context.Input(5);
  const int64_t num_directions = direction_ == "bidirectional" ? 2 : 1;
  const int64_t hidden = hidden_size_;
  const bool lbr = linear_before_reset_ != 0;

  TensorShape Y_shape, Y_h_shape;
  if (layout_ == 0) {
    Y_shape = TensorShape({seq_length, num_directions, batch_size, hidden});
    Y_h_shape = TensorShape({num_directions, batch_size, hidden});
  } else {
    Y_shape = TensorShape({batch_size, seq_length, num_directions, hidden});
    Y_h_shape = TensorShape({batch_size, num_directions, hidden});
  }
  auto* Y = context.Output(0, Y_shape);
  auto* Y_h = context.Output(1, Y_h_shape);
  if (Y == nullptr && Y_h == nullptr) {
    return Status::OK();
  }

  const auto f32 = DataTypeImpl::GetType<float>();
  auto create = [&](const TensorShape& s) { return context.CreateGPUTensor(f32, s); };
  Tensor w_for_run, r_for_run, rh_for_run;
  const Tensor* w_t = packed_w_.get();
  const Tensor* r_t = packed_r_.get();
  const Tensor* rh_t = packed_rh_.get();
  if (w_t == nullptr) {
    ORT_RETURN_IF_ERROR(PackGruOperand(context, *W, false, lbr, create, w_for_run, nullptr));
    w_t = &w_for_run;
  }
  if (r_t == nullptr) {
    ORT_RETURN_IF_ERROR(PackGruOperand(context, *R, true, lbr, create, r_for_run, &rh_for_run));
    r_t = &r_for_run;
    rh_t = lbr ? nullptr : &rh_for_run;
  }

  // XW[d, row, :] = X[row, :] * W[d]^T for every timestep at once (rows follow X's layout), in f32.
  const int64_t rows = seq_length * batch_size;
  const Activation no_activation{};
  Tensor xw = create(TensorShape({num_directions, rows, 3 * hidden}));
  Tensor x_f32;
  const Tensor* x_src = X;
  if (X->DataType() != f32) {
    x_f32 = create(X->Shape());
    ORT_RETURN_IF_ERROR(ConvertGruInputToF32(context, *X, x_f32));
    x_src = &x_f32;
  }
  const Tensor x_rows = CreateTensorView(*x_src, TensorShape({1, rows, input_size}));
  ORT_RETURN_IF_ERROR(matmul_.Compute(context, no_activation, {&x_rows, w_t}, &xw, /*is_channels_last=*/true,
                                      packed_w_ != nullptr));

  const TensorShape state_shape({num_directions, batch_size, hidden});
  const uint32_t state_size = narrow<uint32_t>(state_shape.Size());
  const bool has_seq_lens = sequence_lens != nullptr;
  Tensor state[2] = {create(state_shape), create(state_shape)};
  auto transfer = [&](const Tensor* src, Tensor* dst, bool to_output) -> Status {
    GruStateTransferProgram program{to_output, static_cast<int>(layout_), to_output && has_seq_lens};
    program.AddInput({src, ProgramTensorMetadataDependency::Type});
    if (to_output && has_seq_lens) {
      program.AddInput({sequence_lens, ProgramTensorMetadataDependency::Type});
    }
    program.AddOutput({dst, ProgramTensorMetadataDependency::Type})
        .CacheHint(to_output, layout_, to_output && has_seq_lens)
        .SetWorkgroupSize(256)
        .SetDispatchGroupSize(Groups(state_size))
        .AddUniformVariables({{state_size}, {narrow<uint32_t>(batch_size)}, {narrow<uint32_t>(hidden)}, {narrow<uint32_t>(num_directions)}});
    return context.RunProgram(program);
  };
  if (initial_h != nullptr) {
    ORT_RETURN_IF_ERROR(transfer(initial_h, &state[0], /*to_output=*/false));
  } else {
    context.FillZero(state[0]);
  }

  std::vector<std::string> activation_fns;
  for (const auto& a : activations_) activation_fns.push_back(ActivationToWgslFn(a));
  std::string activation_key;
  for (const auto& a : activation_fns) activation_key += a + ";";
  const uint32_t reverse_mask = direction_ == "bidirectional" ? 2u : (direction_ == "reverse" ? 1u : 0u);
  const bool has_clip = clip_ > 0.0f;

  Tensor g, z, rh, gh;
  if (!lbr) {
    g = create(TensorShape({num_directions, batch_size, 2 * hidden}));
    z = create(state_shape);
    rh = create(state_shape);
    gh = create(state_shape);
  }

  auto run_gates = [&](GruGatesProgram::Stage stage, const Tensor& g_in, const Tensor& h_read, Tensor* h_write,
                       int64_t step) -> Status {
    const bool final_stage = stage == GruGatesProgram::Stage::kHidden;
    GruGatesProgram program{stage, B != nullptr, Y != nullptr, has_seq_lens, has_clip, static_cast<int>(layout_),
                            activation_fns};
    program.AddInputs({{&xw, ProgramTensorMetadataDependency::Type},
                       {&g_in, ProgramTensorMetadataDependency::Type},
                       {&h_read, ProgramTensorMetadataDependency::Type}});
    if (stage == GruGatesProgram::Stage::kHidden) program.AddInput({&z, ProgramTensorMetadataDependency::Type});
    if (B != nullptr) program.AddInput({B, ProgramTensorMetadataDependency::Type});
    if (final_stage && has_seq_lens) program.AddInput({sequence_lens, ProgramTensorMetadataDependency::Type});
    if (final_stage) {
      program.AddOutput({h_write, ProgramTensorMetadataDependency::Type});
      if (Y != nullptr) program.AddOutput({Y, ProgramTensorMetadataDependency::Type});
    } else {
      program.AddOutputs({{&z, ProgramTensorMetadataDependency::Type}, {&rh, ProgramTensorMetadataDependency::Type}});
    }
    program
        .CacheHint(static_cast<int>(stage), B != nullptr, Y != nullptr, has_seq_lens, has_clip, layout_,
                   activation_key)
        .SetWorkgroupSize(256)
        .SetDispatchGroupSize(Groups(state_size))
        .AddUniformVariables({{state_size}, {narrow<uint32_t>(batch_size)}, {narrow<uint32_t>(hidden)}, {narrow<uint32_t>(num_directions)}, {narrow<uint32_t>(seq_length)}, {narrow<uint32_t>(step)}, {reverse_mask}, {clip_}});
    return context.RunProgram(program);
  };

  // Small tiles: per timestep the work is tiny, so filling the GPU matters more than reusing R^T values
  // across batch rows (one row per thread measured fastest; L1 still shares R^T between rows).
  constexpr uint32_t tile_units = 32;
  constexpr uint32_t rows_per_thread = 1;
  const uint32_t tile_batch = 256 / tile_units * rows_per_thread;
  const uint32_t unit_tiles = (narrow<uint32_t>(hidden) + tile_units - 1) / tile_units;
  const uint32_t batch_tiles = (narrow<uint32_t>(batch_size) + tile_batch - 1) / tile_batch;
  const uint32_t num_tiles = narrow<uint32_t>(num_directions) * unit_tiles * batch_tiles;
  auto run_fused_step = [&](const Tensor& h_read, Tensor* h_write, int64_t step) -> Status {
    GruFusedStepProgram program{tile_units, rows_per_thread, B != nullptr, Y != nullptr, has_seq_lens, has_clip,
                                static_cast<int>(layout_), activation_fns};
    program.AddInputs({{&xw, ProgramTensorMetadataDependency::Type},
                       {r_t, ProgramTensorMetadataDependency::Type},
                       {&h_read, ProgramTensorMetadataDependency::Type}});
    if (B != nullptr) program.AddInput({B, ProgramTensorMetadataDependency::Type});
    if (has_seq_lens) program.AddInput({sequence_lens, ProgramTensorMetadataDependency::Type});
    program.AddOutput({h_write, ProgramTensorMetadataDependency::Type});
    if (Y != nullptr) program.AddOutput({Y, ProgramTensorMetadataDependency::Type});
    program.CacheHint(tile_units, rows_per_thread, B != nullptr, Y != nullptr, has_seq_lens, has_clip, layout_,
                      activation_key)
        .SetWorkgroupSize(256)
        .SetDispatchGroupSize(num_tiles)
        .AddUniformVariables({{narrow<uint32_t>(batch_size)}, {narrow<uint32_t>(hidden)}, {narrow<uint32_t>(num_directions)}, {narrow<uint32_t>(seq_length)}, {narrow<uint32_t>(step)}, {reverse_mask}, {unit_tiles}, {batch_tiles}, {num_tiles}, {clip_}});
    return context.RunProgram(program);
  };

  for (int64_t step = 0; step < seq_length; ++step) {
    const Tensor& h_read = state[step % 2];
    Tensor* h_write = &state[(step + 1) % 2];
    if (lbr) {
      ORT_RETURN_IF_ERROR(run_fused_step(h_read, h_write, step));
    } else {
      // G[d] = H_prev[d] * [Rz; Rr][d]^T for every direction in one batched GEMM.
      ORT_RETURN_IF_ERROR(matmul_.Compute(context, no_activation, {&h_read, r_t}, &g, /*is_channels_last=*/true,
                                          packed_r_ != nullptr));
      ORT_RETURN_IF_ERROR(run_gates(GruGatesProgram::Stage::kReset, g, h_read, nullptr, step));
      ORT_RETURN_IF_ERROR(matmul_.Compute(context, no_activation, {&rh, rh_t}, &gh, /*is_channels_last=*/true,
                                          packed_rh_ != nullptr));
      ORT_RETURN_IF_ERROR(run_gates(GruGatesProgram::Stage::kHidden, gh, h_read, h_write, step));
    }
  }

  if (Y_h != nullptr) {
    ORT_RETURN_IF_ERROR(transfer(&state[seq_length % 2], Y_h, /*to_output=*/true));
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
        .TypeConstraint("T", WebGpuSupportedFloatTypes())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int32_t>()),
    Gru);

ONNX_OPERATOR_KERNEL_EX(
    GRU,
    kOnnxDomain,
    14,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int32_t>()),
    Gru);

}  // namespace webgpu
}  // namespace onnxruntime
