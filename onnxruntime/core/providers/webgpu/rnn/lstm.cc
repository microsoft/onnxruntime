// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/rnn/lstm.h"

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

#include "core/common/inlined_containers.h"
#include "core/providers/webgpu/nn/fuse_utils.h"
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
  ORT_THROW("Unsupported LSTM activation for WebGPU: ", activation);
}

}  // namespace

// ===========================================================================
// Lstm constructor
// ===========================================================================
Lstm::Lstm(const OpKernelInfo& info) : WebGpuKernel(info) {
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
  info.GetAttrOrDefault("input_forget", &input_forget_, static_cast<int64_t>(0));
  info.GetAttrOrDefault("layout", &layout_, static_cast<int64_t>(0));
  int num_directions = (direction_ == "bidirectional") ? 2 : 1;
  if (!info.GetAttrs("activations", activations_).IsOK() || activations_.empty()) {
    activations_.clear();
    for (int d = 0; d < num_directions; d++) {
      activations_.push_back("Sigmoid");
      activations_.push_back("Tanh");
      activations_.push_back("Tanh");
    }
  }
  ORT_ENFORCE(activations_.size() == static_cast<size_t>(num_directions) * 3);
}

// ===========================================================================
// LstmStateCopyProgram
// ===========================================================================
Status LstmStateCopyProgram::GenerateShaderCode(ShaderHelper& shader) const {
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
// LstmCellProgram - one cell step, always flat [batch, H] for h_prev/c_prev
// ===========================================================================
Status LstmCellProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("x", ShaderUsage::UseElementTypeAlias);
  shader.AddInput("w", ShaderUsage::UseElementTypeAlias);
  shader.AddInput("r", ShaderUsage::UseElementTypeAlias);
  shader.AddInput("h_prev", ShaderUsage::UseElementTypeAlias);
  shader.AddInput("c_prev", ShaderUsage::UseElementTypeAlias);
  if (has_bias_) shader.AddInput("b", ShaderUsage::UseElementTypeAlias);
  if (has_peephole_) shader.AddInput("p", ShaderUsage::UseElementTypeAlias);
  if (has_seq_lens_) shader.AddInput("seq_lens", ShaderUsage::UseElementTypeAlias);

  shader.AddOutput("h_new", ShaderUsage::UseElementTypeAlias);
  shader.AddOutput("c_new", ShaderUsage::UseElementTypeAlias);
  if (has_Y_) shader.AddOutput("y_out", ShaderUsage::UseElementTypeAlias);

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

  auto& body = shader.MainFunctionBody();

  body << "  let H = uniforms.hidden_size;\n"
       << "  let I = uniforms.input_size;\n"
       << "  let B = uniforms.batch_size;\n"
       << "  let dir = uniforms.direction;\n"
       << "  let num_dir = uniforms.num_directions;\n"
       << "  if (global_idx >= B * H) { return; }\n"
       << "  let batch_idx = global_idx / H;\n"
       << "  let j = global_idx % H;\n\n";

  // Sequence length masking - early exit if past this batch's seq length.
  // Use timestep (not processing_step) so that reverse direction masks correctly:
  // for forward, timestep == processing_step; for reverse, timestep counts down.
  if (has_seq_lens_) {
    body << "  let batch_seq_len = u32(seq_lens[batch_idx]);\n"
         << "  if (uniforms.timestep >= batch_seq_len) {\n"
         << "    let flat_idx = batch_idx * H + j;\n"
         << "    h_new[flat_idx] = h_prev[flat_idx];\n"
         << "    c_new[flat_idx] = c_prev[flat_idx];\n";
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

  // Gate accumulators (ONNX gate order: i, o, f, c)
  body << "  var gate_i: f32 = 0.0;\n"
       << "  var gate_o: f32 = 0.0;\n"
       << "  var gate_f: f32 = 0.0;\n"
       << "  var gate_c: f32 = 0.0;\n\n";

  // X * W^T
  body << "  let w_base = dir * 4u * H * I;\n";
  if (layout_ == 0) {
    body << "  let x_base = (uniforms.timestep * B + batch_idx) * I;\n";
  } else {
    body << "  let x_base = (batch_idx * uniforms.seq_length + uniforms.timestep) * I;\n";
  }
  body << "  for (var k: u32 = 0u; k < I; k++) {\n"
       << "    let xv = f32(x[x_base + k]);\n"
       << "    gate_i += xv * f32(w[w_base + j * I + k]);\n"
       << "    gate_o += xv * f32(w[w_base + (H + j) * I + k]);\n"
       << "    gate_f += xv * f32(w[w_base + (2u * H + j) * I + k]);\n"
       << "    gate_c += xv * f32(w[w_base + (3u * H + j) * I + k]);\n"
       << "  }\n\n";

  // H_prev * R^T  (h_prev always [batch, H] - flat indexing)
  body << "  let r_base = dir * 4u * H * H;\n"
       << "  let h_base = batch_idx * H;\n"
       << "  for (var k: u32 = 0u; k < H; k++) {\n"
       << "    let hv = f32(h_prev[h_base + k]);\n"
       << "    gate_i += hv * f32(r[r_base + j * H + k]);\n"
       << "    gate_o += hv * f32(r[r_base + (H + j) * H + k]);\n"
       << "    gate_f += hv * f32(r[r_base + (2u * H + j) * H + k]);\n"
       << "    gate_c += hv * f32(r[r_base + (3u * H + j) * H + k]);\n"
       << "  }\n\n";

  // Bias
  if (has_bias_) {
    body << "  let bb = dir * 8u * H;\n"
         << "  gate_i += f32(b[bb + j]) + f32(b[bb + 4u * H + j]);\n"
         << "  gate_o += f32(b[bb + H + j]) + f32(b[bb + 5u * H + j]);\n"
         << "  gate_f += f32(b[bb + 2u * H + j]) + f32(b[bb + 6u * H + j]);\n"
         << "  gate_c += f32(b[bb + 3u * H + j]) + f32(b[bb + 7u * H + j]);\n\n";
  }

  // c_prev (flat [batch, H])
  body << "  let c_base = batch_idx * H;\n"
       << "  let cpv = f32(c_prev[c_base + j]);\n\n";

  // Peephole for i, f gates
  if (has_peephole_) {
    body << "  let pb = dir * 3u * H;\n"
         << "  gate_i += f32(p[pb + j]) * cpv;\n"
         << "  gate_f += f32(p[pb + 2u * H + j]) * cpv;\n\n";
  }

  // Clip
  if (has_clip_) {
    body << "  gate_i = clamp(gate_i, -uniforms.clip_value, uniforms.clip_value);\n"
         << "  gate_o = clamp(gate_o, -uniforms.clip_value, uniforms.clip_value);\n"
         << "  gate_f = clamp(gate_f, -uniforms.clip_value, uniforms.clip_value);\n"
         << "  gate_c = clamp(gate_c, -uniforms.clip_value, uniforms.clip_value);\n\n";
  }

  // Gate activations
  body << "  gate_i = " << f_activation_fn_ << "(gate_i);\n"
       << "  gate_f = " << f_activation_fn_ << "(gate_f);\n"
       << "  gate_c = " << g_activation_fn_ << "(gate_c);\n\n";

  if (input_forget_) {
    body << "  gate_f = 1.0 - gate_i;\n\n";
  }

  // Cell state update
  body << "  let cu = gate_f * cpv + gate_i * gate_c;\n\n";

  // Peephole for output gate (uses C_t, not C_{t-1})
  if (has_peephole_) {
    body << "  gate_o += f32(p[pb + H + j]) * cu;\n";
    if (has_clip_) {
      body << "  gate_o = clamp(gate_o, -uniforms.clip_value, uniforms.clip_value);\n";
    }
    body << "\n";
  }

  // Output gate and hidden state
  body << "  gate_o = " << f_activation_fn_ << "(gate_o);\n"
       << "  let hu = gate_o * " << h_activation_fn_ << "(cu);\n\n";

  // Write h_new, c_new to temp buffers [batch, H]
  body << "  let oi = batch_idx * H + j;\n"
       << "  h_new[oi] = h_new_element_t(hu);\n"
       << "  c_new[oi] = c_new_element_t(cu);\n\n";

  // Write Y output
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
// LstmWriteYProgram - copies h_new to Y with optional seq_lens masking.
// Used when the cell program cannot include Y output due to storage buffer limits.
// ===========================================================================
Status LstmWriteYProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("h_new", ShaderUsage::UseElementTypeAlias);
  if (has_seq_lens_) shader.AddInput("seq_lens", ShaderUsage::UseElementTypeAlias);
  shader.AddOutput("y_out", ShaderUsage::UseElementTypeAlias);
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
    body << "  let y_offset = ((uniforms.timestep * num_dir + dir) * B + batch_idx) * H + j;\n";
  } else {
    body << "  let y_offset = ((batch_idx * uniforms.seq_length + uniforms.timestep) * num_dir + dir) * H + j;\n";
  }
  if (has_seq_lens_) {
    body << "  if (uniforms.timestep >= u32(seq_lens[batch_idx])) {\n"
         << "    y_out[y_offset] = y_out_element_t(0.0);\n"
         << "  } else {\n"
         << "    y_out[y_offset] = y_out_element_t(h_new[flat_idx]);\n"
         << "  }\n";
  } else {
    body << "  y_out[y_offset] = y_out_element_t(h_new[flat_idx]);\n";
  }
  return Status::OK();
}

// ===========================================================================
// ComputeInternal
// ===========================================================================
Status Lstm::ComputeStepwise(ComputeContext& context) const {
  const auto* X = context.Input(0);
  const auto* W = context.Input(1);
  const auto* R = context.Input(2);
  const auto* B = context.Input(3);
  const auto* sequence_lens = context.Input(4);
  const auto* initial_h = context.Input(5);
  const auto* initial_c = context.Input(6);
  const auto* P = context.Input(7);

  ORT_RETURN_IF(X == nullptr || W == nullptr || R == nullptr,
                "LSTM: X, W, R are required inputs.");

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
                    W_shape[1] != 4 * hidden_size_ || W_shape[2] != input_size,
                "LSTM: Input W must have shape {", num_directions, ", 4*", hidden_size_, ", ", input_size,
                "}. Actual: ", W_shape);
  ORT_RETURN_IF(R_shape.NumDimensions() != 3 || R_shape[0] != num_directions ||
                    R_shape[1] != 4 * hidden_size_ || R_shape[2] != hidden_size_,
                "LSTM: Input R must have shape {", num_directions, ", 4*", hidden_size_, ", ", hidden_size_,
                "}. Actual: ", R_shape);

  uint32_t H = static_cast<uint32_t>(hidden_size_);

  // Output shapes
  TensorShape Y_shape, Y_h_shape, Y_c_shape;
  if (layout_ == 0) {
    Y_shape = TensorShape({seq_length, static_cast<int64_t>(num_directions), batch_size, hidden_size_});
    Y_h_shape = TensorShape({static_cast<int64_t>(num_directions), batch_size, hidden_size_});
    Y_c_shape = TensorShape({static_cast<int64_t>(num_directions), batch_size, hidden_size_});
  } else {
    Y_shape = TensorShape({batch_size, seq_length, static_cast<int64_t>(num_directions), hidden_size_});
    Y_h_shape = TensorShape({batch_size, static_cast<int64_t>(num_directions), hidden_size_});
    Y_c_shape = TensorShape({batch_size, static_cast<int64_t>(num_directions), hidden_size_});
  }

  auto* Y = context.Output(0, Y_shape);
  auto* Y_h = context.Output(1, Y_h_shape);
  auto* Y_c = context.Output(2, Y_c_shape);
  bool has_Y = (Y != nullptr);
  bool has_Y_h = (Y_h != nullptr);
  bool has_Y_c = (Y_c != nullptr);
  if (!has_Y && !has_Y_h && !has_Y_c) {
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

  if (seq_length == 0) {
    if (has_Y_h) {
      context.FillZero(*Y_h);
    }
    if (has_Y_c) {
      context.FillZero(*Y_c);
    }
    return Status::OK();
  }

  auto copy_from_state = [&](const Tensor* src, Tensor* dst, int dir) -> Status {
    LstmStateCopyProgram prog{/*to_state=*/false, static_cast<int>(layout_)};
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
    LstmStateCopyProgram prog{/*to_state=*/true, static_cast<int>(layout_), has_seq_lens};
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

  // Check if the cell program would exceed storage buffer limits (max 10).
  // Base bindings: x, w, r, h_prev, c_prev (5 inputs) + h_new, c_new (2 outputs) = 7.
  int cell_bindings = 7 + (B != nullptr ? 1 : 0) + (P != nullptr ? 1 : 0) + (has_seq_lens ? 1 : 0) + (has_Y ? 1 : 0);
  bool split_y = (cell_bindings > 10) && has_Y;
  bool cell_has_Y = has_Y && !split_y;

  for (int dir = 0; dir < num_directions; dir++) {
    std::string fa = ActivationToWgslFn(activations_[dir * 3 + 0]);
    std::string ga = ActivationToWgslFn(activations_[dir * 3 + 1]);
    std::string ha = ActivationToWgslFn(activations_[dir * 3 + 2]);

    bool is_reverse = (direction_ == "reverse") || (direction_ == "bidirectional" && dir == 1);

    // Ping-pong state buffers [batch, H]
    Tensor H_a = context.CreateGPUTensor(state_dtype, state_shape);
    Tensor C_a = context.CreateGPUTensor(state_dtype, state_shape);
    Tensor H_b = context.CreateGPUTensor(state_dtype, state_shape);
    Tensor C_b = context.CreateGPUTensor(state_dtype, state_shape);

    // Initialize H_a / C_a with zeros or initial state
    if (initial_h != nullptr) {
      ORT_RETURN_IF_ERROR(copy_from_state(initial_h, &H_a, dir));
    } else {
      context.FillZero(H_a);
    }
    if (initial_c != nullptr) {
      ORT_RETURN_IF_ERROR(copy_from_state(initial_c, &C_a, dir));
    } else {
      context.FillZero(C_a);
    }

    // Per-timestep loop
    for (int64_t t = 0; t < seq_length; t++) {
      int64_t timestep = is_reverse ? (seq_length - 1 - t) : t;

      const Tensor* h_read;
      const Tensor* c_read;
      Tensor* h_write;
      Tensor* c_write;

      if (t % 2 == 0) {
        h_read = &H_a;
        c_read = &C_a;
        h_write = &H_b;
        c_write = &C_b;
      } else {
        h_read = &H_b;
        c_read = &C_b;
        h_write = &H_a;
        c_write = &C_a;
      }

      LstmCellProgram program{B != nullptr, P != nullptr, cell_has_Y, has_seq_lens,
                              input_forget_ != 0, clip_ > 0.0f,
                              static_cast<int>(layout_), fa, ga, ha};

      program.CacheHint(std::to_string(B != nullptr), std::to_string(P != nullptr),
                        std::to_string(cell_has_Y), std::to_string(has_seq_lens),
                        std::to_string(input_forget_ != 0), std::to_string(clip_ > 0.0f),
                        std::to_string(layout_), fa, ga, ha);

      program.SetWorkgroupSize(wg_size).SetDispatchGroupSize(num_groups);

      program.AddInputs({{X, ProgramTensorMetadataDependency::Type}})
          .AddInputs({{W, ProgramTensorMetadataDependency::Type}})
          .AddInputs({{R, ProgramTensorMetadataDependency::Type}})
          .AddInputs({{h_read, ProgramTensorMetadataDependency::Type}})
          .AddInputs({{c_read, ProgramTensorMetadataDependency::Type}});
      if (B != nullptr) program.AddInputs({{B, ProgramTensorMetadataDependency::Type}});
      if (P != nullptr) program.AddInputs({{P, ProgramTensorMetadataDependency::Type}});
      if (has_seq_lens) program.AddInputs({{sequence_lens, ProgramTensorMetadataDependency::Type}});

      program.AddOutputs({{h_write, ProgramTensorMetadataDependency::None}})
          .AddOutputs({{c_write, ProgramTensorMetadataDependency::None}});
      if (cell_has_Y) program.AddOutputs({{Y, ProgramTensorMetadataDependency::None}});

      program.AddUniformVariables({
          {static_cast<uint32_t>(batch_size)},
          {static_cast<uint32_t>(input_size)},
          {H},
          {static_cast<uint32_t>(dir)},
          {static_cast<uint32_t>(num_directions)},
          {static_cast<uint32_t>(timestep)},
          {static_cast<uint32_t>(seq_length)},
          {clip_},
      });

      ORT_RETURN_IF_ERROR(context.RunProgram(program));

      // When Y is split out of the cell (due to binding limits), write it separately.
      if (split_y) {
        LstmWriteYProgram y_prog{has_seq_lens, static_cast<int>(layout_)};
        y_prog.CacheHint(std::to_string(has_seq_lens), std::to_string(layout_));
        y_prog.SetWorkgroupSize(wg_size).SetDispatchGroupSize(num_groups);
        y_prog.AddInputs({{h_write, ProgramTensorMetadataDependency::Type}});
        if (has_seq_lens) y_prog.AddInputs({{sequence_lens, ProgramTensorMetadataDependency::Type}});
        y_prog.AddOutputs({{Y, ProgramTensorMetadataDependency::Type}});
        y_prog.AddUniformVariables({
            {static_cast<uint32_t>(batch_size)},
            {H},
            {static_cast<uint32_t>(dir)},
            {static_cast<uint32_t>(num_directions)},
            {static_cast<uint32_t>(timestep)},
            {static_cast<uint32_t>(seq_length)},
        });
        ORT_RETURN_IF_ERROR(context.RunProgram(y_prog));
      }
    }

    // Final state is in the last-written buffer
    Tensor* final_h = (seq_length % 2 == 1) ? &H_b : &H_a;
    Tensor* final_c = (seq_length % 2 == 1) ? &C_b : &C_a;

    // Copy to Y_h / Y_c
    if (has_Y_h) {
      ORT_RETURN_IF_ERROR(copy_to_state(final_h, Y_h, dir));
    }
    if (has_Y_c) {
      ORT_RETURN_IF_ERROR(copy_to_state(final_c, Y_c, dir));
    }
  }

  return Status::OK();
}

// ===========================================================================
// GEMM path: one GEMM for the input projection of every timestep, then per timestep a batched GEMM
// for the recurrent projection of every direction and one fused gate kernel.
// ===========================================================================
namespace {

void AppendActivations(OStringStream& impl) {
  impl << "fn sigmoid_f(v: f32) -> f32 {\n"
       << "  let clamped = clamp(v, -20.0, 20.0);\n"
       << "  return 1.0 / (1.0 + exp(-clamped));\n"
       << "}\n"
       << "fn tanh_f(v: f32) -> f32 {\n"
       << "  let clamped = clamp(v, -10.0, 10.0);\n"
       << "  return tanh(clamped);\n"
       << "}\n"
       << "fn relu_f(v: f32) -> f32 { return max(0.0, v); }\n";
}

uint32_t Groups(uint32_t size) { return (size + 255) / 256; }

// Index of (direction d, batch b, unit j) in an ONNX Y_h / Y_c / initial_h / initial_c tensor.
std::string StateTensorIndex(int layout) {
  return layout == 0 ? "(d * B + b) * H + j" : "(b * uniforms.num_directions + d) * H + j";
}

}  // namespace

Status LstmPackWeightsProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& src = shader.AddInput("src", ShaderUsage::UseUniform);
  const auto& dst = shader.AddOutput("dst", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.size")
                            << "let per_dir = uniforms.rows * uniforms.cols;\n"
                            << "let d = global_idx / per_dir;\n"
                            << "let rem = global_idx % per_dir;\n"
                            << "let k = rem / uniforms.rows;\n"
                            << "let n = rem % uniforms.rows;\n"
                            << dst.SetByOffset("global_idx", "dst_element_t(" + src.GetByOffset("(d * uniforms.rows + n) * uniforms.cols + k") + ")")
                            << "\n";
  return Status::OK();
}

Status LstmStateTransferProgram::GenerateShaderCode(ShaderHelper& shader) const {
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

Status LstmGatesProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& xw = shader.AddInput("xw", ShaderUsage::UseUniform);
  const auto& g = shader.AddInput("g", ShaderUsage::UseUniform);
  const auto& h_prev = shader.AddInput("h_prev", ShaderUsage::UseUniform);
  const auto& c_prev = shader.AddInput("c_prev", ShaderUsage::UseUniform);
  const ShaderVariableHelper* bias = has_bias_ ? &shader.AddInput("bias", ShaderUsage::UseUniform) : nullptr;
  const ShaderVariableHelper* p = has_peephole_ ? &shader.AddInput("p", ShaderUsage::UseUniform) : nullptr;
  const ShaderVariableHelper* seq_lens = has_seq_lens_ ? &shader.AddInput("seq_lens", ShaderUsage::UseUniform) : nullptr;
  const auto& h_new = shader.AddOutput("h_new", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  const auto& c_new = shader.AddOutput("c_new", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  const ShaderVariableHelper* y =
      has_Y_ ? &shader.AddOutput("y", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias) : nullptr;

  auto& impl = shader.AdditionalImplementation();
  AppendActivations(impl);
  // Per-direction activations f (gates), g (cell input) and h (output).
  const size_t num_directions = activations_.size() / 3;
  for (size_t role = 0; role < 3; ++role) {
    impl << "fn act" << role << "(d: u32, v: f32) -> f32 {\n";
    for (size_t d = 1; d < num_directions; ++d) {
      impl << "  if (d == " << d << "u) { return " << activations_[d * 3 + role] << "(v); }\n";
    }
    impl << "  return " << activations_[role] << "(v);\n}\n";
  }

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
       << "let t = select(uniforms.step, S - 1u - uniforms.step, reverse);\n";
  if (y != nullptr) {
    body << "let yi = " << (layout_ == 0 ? "((t * D + d) * B + b) * H + j" : "((b * S + t) * D + d) * H + j") << ";\n";
  }
  // Past a batch entry's sequence length the state carries through and Y is zero. The reverse
  // direction therefore starts from the initial state at its last valid timestep.
  if (seq_lens != nullptr) {
    body << "if (t >= u32(" << seq_lens->GetByOffset("b") << ")) {\n"
         << "  " << h_new.SetByOffset("global_idx", "h_new_element_t(" + h_prev.GetByOffset("global_idx") + ")") << "\n"
         << "  " << c_new.SetByOffset("global_idx", "c_new_element_t(" + c_prev.GetByOffset("global_idx") + ")") << "\n";
    if (y != nullptr) {
      body << "  " << y->SetByOffset("yi", "y_element_t(0.0)") << "\n";
    }
    body << "  return;\n}\n";
  }
  body << "let row = " << (layout_ == 0 ? "t * B + b" : "b * S + t") << ";\n"
       << "let xb = (d * S * B + row) * 4u * H + j;\n"
       << "let gb = (d * B + b) * 4u * H + j;\n"
       // ONNX gate order: i, o, f, c.
       << "var gate_i = f32(" << xw.GetByOffset("xb") << ") + f32(" << g.GetByOffset("gb") << ");\n"
       << "var gate_o = f32(" << xw.GetByOffset("xb + H") << ") + f32(" << g.GetByOffset("gb + H") << ");\n"
       << "var gate_f = f32(" << xw.GetByOffset("xb + 2u * H") << ") + f32(" << g.GetByOffset("gb + 2u * H") << ");\n"
       << "var gate_c = f32(" << xw.GetByOffset("xb + 3u * H") << ") + f32(" << g.GetByOffset("gb + 3u * H") << ");\n";
  if (bias != nullptr) {
    auto gate_bias = [&](const std::string& k) {
      return "f32(" + bias->GetByOffset("bb + " + k + " * H") + ") + f32(" + bias->GetByOffset("bb + (" + k + " + 4u) * H") + ")";
    };
    body << "let bb = d * 8u * H + j;\n"
         << "gate_i += " << gate_bias("0u") << ";\n"
         << "gate_o += " << gate_bias("1u") << ";\n"
         << "gate_f += " << gate_bias("2u") << ";\n"
         << "gate_c += " << gate_bias("3u") << ";\n";
  }
  body << "let cpv = f32(" << c_prev.GetByOffset("global_idx") << ");\n";
  if (p != nullptr) {
    body << "let pb = d * 3u * H + j;\n"
         << "gate_i += f32(" << p->GetByOffset("pb") << ") * cpv;\n"
         << "gate_f += f32(" << p->GetByOffset("pb + 2u * H") << ") * cpv;\n";
  }
  if (has_clip_) {
    body << "gate_i = clamp(gate_i, -uniforms.clip_value, uniforms.clip_value);\n"
         << "gate_o = clamp(gate_o, -uniforms.clip_value, uniforms.clip_value);\n"
         << "gate_f = clamp(gate_f, -uniforms.clip_value, uniforms.clip_value);\n"
         << "gate_c = clamp(gate_c, -uniforms.clip_value, uniforms.clip_value);\n";
  }
  body << "gate_i = act0(d, gate_i);\n"
       << "gate_f = act0(d, gate_f);\n"
       << "gate_c = act1(d, gate_c);\n";
  if (input_forget_) {
    body << "gate_f = 1.0 - gate_i;\n";
  }
  body << "let cu = gate_f * cpv + gate_i * gate_c;\n";
  // The output gate's peephole uses the new cell state.
  if (p != nullptr) {
    body << "gate_o += f32(" << p->GetByOffset("pb + H") << ") * cu;\n";
    if (has_clip_) {
      body << "gate_o = clamp(gate_o, -uniforms.clip_value, uniforms.clip_value);\n";
    }
  }
  body << "let hu = act0(d, gate_o) * act2(d, cu);\n"
       << h_new.SetByOffset("global_idx", "h_new_element_t(hu)") << "\n"
       << c_new.SetByOffset("global_idx", "c_new_element_t(cu)") << "\n";
  if (y != nullptr) {
    body << y->SetByOffset("yi", "y_element_t(hu)") << "\n";
  }
  return Status::OK();
}

namespace {

// Transposes W [D, 4H, I] or R [D, 4H, H] into `packed` [D, K, 4H].
Status PackLstmWeights(ComputeContextBase& context, const Tensor& weights, Tensor& packed) {
  const TensorShape& shape = weights.Shape();
  const uint32_t size = narrow<uint32_t>(shape.Size());
  LstmPackWeightsProgram program;
  program.AddInput({&weights, ProgramTensorMetadataDependency::Type})
      .AddOutput({&packed, ProgramTensorMetadataDependency::Type})
      .SetDispatchGroupSize(Groups(size))
      .SetWorkgroupSize(256)
      .AddUniformVariables({{size}, {narrow<uint32_t>(shape[1])}, {narrow<uint32_t>(shape[2])}});
  return context.RunProgram(program);
}

TensorShape PackedLstmWeightShape(const TensorShape& shape) { return TensorShape({shape[0], shape[2], shape[1]}); }

// Element-wise copy of `src` into the f32 tensor `dst`: the pack program on a [1, size, 1] view is a cast.
Status ConvertToF32(ComputeContextBase& context, const Tensor& src, Tensor& dst) {
  const int64_t size = src.Shape().Size();
  const Tensor src_view = CreateTensorView(src, TensorShape({1, size, 1}));
  Tensor dst_view = CreateTensorView(dst, TensorShape({1, 1, size}));
  return PackLstmWeights(context, src_view, dst_view);
}

}  // namespace

Status Lstm::PrePackInternal(ComputeContextBase& context, const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                             /*out*/ bool& is_packed) {
  // The original weights stay alive: the stepwise fallback and shape validation read them.
  is_packed = false;
  if ((input_idx != 1 && input_idx != 2) || tensor.Shape().NumDimensions() != 3 || tensor.Shape().Size() == 0) {
    return Status::OK();
  }
  // Both projections run in f32, like the stepwise kernel, so float16 models keep f32 gate preactivations.
  auto packed = std::make_unique<Tensor>(DataTypeImpl::GetType<float>(), PackedLstmWeightShape(tensor.Shape()), alloc);
  ORT_RETURN_IF_ERROR(PackLstmWeights(context, tensor, *packed));
  (input_idx == 1 ? packed_w_ : packed_r_) = std::move(packed);
  return Status::OK();
}

Status Lstm::ComputeInternal(ComputeContext& context) const {
  const auto* X = context.Input(0);
  const auto* W = context.Input(1);
  if (X == nullptr || W == nullptr || X->Shape().NumDimensions() != 3) {
    return ComputeStepwise(context);  // reports the error
  }
  const int64_t num_directions = direction_ == "bidirectional" ? 2 : 1;
  const int64_t rows = X->Shape()[0] * X->Shape()[1];  // seq_length * batch_size in either layout
  // The f32 input projection for every timestep and direction (and, for float16, the f32 copy of X) must
  // fit in one storage binding.
  uint64_t max_bytes = static_cast<uint64_t>(num_directions) * rows * 4 * hidden_size_ * sizeof(float);
  if (X->DataType() != DataTypeImpl::GetType<float>()) {
    max_bytes = std::max<uint64_t>(max_bytes, static_cast<uint64_t>(X->Shape().Size()) * sizeof(float));
  }
  const auto& limits = context.DeviceLimits();
  if (rows == 0 || hidden_size_ <= 0 || max_bytes > limits.maxStorageBufferBindingSize ||
      max_bytes > limits.maxBufferSize) {
    return ComputeStepwise(context);
  }
  return ComputeWithGemm(context);
}

Status Lstm::ComputeWithGemm(ComputeContext& context) const {
  const auto* X = context.Input(0);
  const auto* W = context.Input(1);
  const auto* R = context.Input(2);
  const auto* B = context.Input(3);
  const auto* sequence_lens = context.Input(4);
  const auto* initial_h = context.Input(5);
  const auto* initial_c = context.Input(6);
  const auto* P = context.Input(7);
  ORT_RETURN_IF(R == nullptr, "LSTM: X, W, R are required inputs.");

  const auto& X_shape = X->Shape();
  const int64_t num_directions = direction_ == "bidirectional" ? 2 : 1;
  const int64_t seq_length = layout_ == 0 ? X_shape[0] : X_shape[1];
  const int64_t batch_size = layout_ == 0 ? X_shape[1] : X_shape[0];
  const int64_t input_size = X_shape[2];
  const int64_t hidden = hidden_size_;
  const auto& W_shape = W->Shape();
  const auto& R_shape = R->Shape();
  ORT_RETURN_IF(W_shape.NumDimensions() != 3 || W_shape[0] != num_directions ||
                    W_shape[1] != 4 * hidden || W_shape[2] != input_size,
                "LSTM: Input W must have shape {", num_directions, ", 4*", hidden, ", ", input_size,
                "}. Actual: ", W_shape);
  ORT_RETURN_IF(R_shape.NumDimensions() != 3 || R_shape[0] != num_directions ||
                    R_shape[1] != 4 * hidden || R_shape[2] != hidden,
                "LSTM: Input R must have shape {", num_directions, ", 4*", hidden, ", ", hidden,
                "}. Actual: ", R_shape);

  TensorShape Y_shape, state_out_shape;
  if (layout_ == 0) {
    Y_shape = TensorShape({seq_length, num_directions, batch_size, hidden});
    state_out_shape = TensorShape({num_directions, batch_size, hidden});
  } else {
    Y_shape = TensorShape({batch_size, seq_length, num_directions, hidden});
    state_out_shape = TensorShape({batch_size, num_directions, hidden});
  }
  auto* Y = context.Output(0, Y_shape);
  auto* Y_h = context.Output(1, state_out_shape);
  auto* Y_c = context.Output(2, state_out_shape);
  if (Y == nullptr && Y_h == nullptr && Y_c == nullptr) {
    return Status::OK();
  }

  const auto f32 = DataTypeImpl::GetType<float>();
  Tensor w_for_run, r_for_run;
  const Tensor* w_t = packed_w_.get();
  const Tensor* r_t = packed_r_.get();
  if (w_t == nullptr) {
    w_for_run = context.CreateGPUTensor(f32, PackedLstmWeightShape(W_shape));
    ORT_RETURN_IF_ERROR(PackLstmWeights(context, *W, w_for_run));
    w_t = &w_for_run;
  }
  if (r_t == nullptr) {
    r_for_run = context.CreateGPUTensor(f32, PackedLstmWeightShape(R_shape));
    ORT_RETURN_IF_ERROR(PackLstmWeights(context, *R, r_for_run));
    r_t = &r_for_run;
  }

  // XW[d, row, :] = X[row, :] * W[d]^T for every timestep at once (rows follow X's layout).
  const int64_t rows = seq_length * batch_size;
  const Activation no_activation{};
  // Computed and stored in f32: rounding XW to float16 would lose what the bias and recurrent term cancel.
  Tensor xw = context.CreateGPUTensor(f32, TensorShape({num_directions, rows, 4 * hidden}));
  Tensor x_f32;
  const Tensor* x_src = X;
  if (X->DataType() != f32) {
    x_f32 = context.CreateGPUTensor(f32, X_shape);
    ORT_RETURN_IF_ERROR(ConvertToF32(context, *X, x_f32));
    x_src = &x_f32;
  }
  const Tensor x_rows = CreateTensorView(*x_src, TensorShape({1, rows, input_size}));
  ORT_RETURN_IF_ERROR(matmul_.Compute(context, no_activation, {&x_rows, w_t}, &xw, /*is_channels_last=*/true,
                                      packed_w_ != nullptr));

  const TensorShape state_shape({num_directions, batch_size, hidden});
  const uint32_t state_size = narrow<uint32_t>(state_shape.Size());
  const bool has_seq_lens = sequence_lens != nullptr;
  Tensor state[2][2] = {{context.CreateGPUTensor(f32, state_shape), context.CreateGPUTensor(f32, state_shape)},
                        {context.CreateGPUTensor(f32, state_shape), context.CreateGPUTensor(f32, state_shape)}};
  auto transfer = [&](const Tensor* src, Tensor* dst, bool to_output) -> Status {
    LstmStateTransferProgram program{to_output, static_cast<int>(layout_), to_output && has_seq_lens};
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
  for (int s = 0; s < 2; ++s) {
    const Tensor* initial = s == 0 ? initial_h : initial_c;
    if (initial != nullptr) {
      ORT_RETURN_IF_ERROR(transfer(initial, &state[0][s], /*to_output=*/false));
    } else {
      context.FillZero(state[0][s]);
    }
  }

  InlinedVector<std::string> activation_fns;
  for (const auto& a : activations_) activation_fns.push_back(ActivationToWgslFn(a));
  std::string activation_key;
  for (const auto& a : activation_fns) activation_key += a + ";";
  const uint32_t reverse_mask = direction_ == "bidirectional" ? 2u : (direction_ == "reverse" ? 1u : 0u);
  Tensor g = context.CreateGPUTensor(f32, TensorShape({num_directions, batch_size, 4 * hidden}));

  for (int64_t step = 0; step < seq_length; ++step) {
    Tensor* h_read = &state[step % 2][0];
    Tensor* c_read = &state[step % 2][1];
    Tensor* h_write = &state[(step + 1) % 2][0];
    Tensor* c_write = &state[(step + 1) % 2][1];

    // G[d] = H_prev[d] * R[d]^T for every direction in one batched GEMM.
    ORT_RETURN_IF_ERROR(matmul_.Compute(context, no_activation, {h_read, r_t}, &g, /*is_channels_last=*/true,
                                        packed_r_ != nullptr));

    LstmGatesProgram program{B != nullptr, P != nullptr, Y != nullptr, has_seq_lens, input_forget_ != 0,
                             clip_ > 0.0f, static_cast<int>(layout_),
                             std::vector<std::string>(activation_fns.begin(), activation_fns.end())};
    program.AddInputs({{&xw, ProgramTensorMetadataDependency::Type},
                       {&g, ProgramTensorMetadataDependency::Type},
                       {h_read, ProgramTensorMetadataDependency::Type},
                       {c_read, ProgramTensorMetadataDependency::Type}});
    if (B != nullptr) program.AddInput({B, ProgramTensorMetadataDependency::Type});
    if (P != nullptr) program.AddInput({P, ProgramTensorMetadataDependency::Type});
    if (has_seq_lens) program.AddInput({sequence_lens, ProgramTensorMetadataDependency::Type});
    program.AddOutputs({{h_write, ProgramTensorMetadataDependency::Type},
                        {c_write, ProgramTensorMetadataDependency::Type}});
    if (Y != nullptr) program.AddOutput({Y, ProgramTensorMetadataDependency::Type});
    program
        .CacheHint(B != nullptr, P != nullptr, Y != nullptr, has_seq_lens, input_forget_ != 0, clip_ > 0.0f, layout_,
                   activation_key)
        .SetWorkgroupSize(256)
        .SetDispatchGroupSize(Groups(state_size))
        .AddUniformVariables({{state_size}, {narrow<uint32_t>(batch_size)}, {narrow<uint32_t>(hidden)}, {narrow<uint32_t>(num_directions)}, {narrow<uint32_t>(seq_length)}, {narrow<uint32_t>(step)}, {reverse_mask}, {clip_}});
    ORT_RETURN_IF_ERROR(context.RunProgram(program));
  }

  const int last = static_cast<int>(seq_length % 2);
  if (Y_h != nullptr) ORT_RETURN_IF_ERROR(transfer(&state[last][0], Y_h, /*to_output=*/true));
  if (Y_c != nullptr) ORT_RETURN_IF_ERROR(transfer(&state[last][1], Y_c, /*to_output=*/true));
  return Status::OK();
}

// ===========================================================================
// Kernel registrations
// ===========================================================================
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    LSTM,
    kOnnxDomain,
    7, 13,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int32_t>()),
    Lstm);

ONNX_OPERATOR_KERNEL_EX(
    LSTM,
    kOnnxDomain,
    14,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int32_t>()),
    Lstm);

}  // namespace webgpu
}  // namespace onnxruntime
