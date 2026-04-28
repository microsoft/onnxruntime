// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/rnn/lstm.h"

#include <algorithm>
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
    lower.push_back(static_cast<char>(std::tolower(c)));
  }
  if (lower == "sigmoid") return "sigmoid_f";
  if (lower == "tanh") return "tanh_f";
  if (lower == "relu") return "relu_f";
  return "sigmoid_f";
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
}

// ===========================================================================
// LstmZeroFillProgram
// ===========================================================================
Status LstmZeroFillProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddOutput("dst", ShaderUsage::UseElementTypeAlias);
  auto& body = shader.MainFunctionBody();
  body << "  if (global_idx < uniforms.size) {\n"
       << "    dst[global_idx] = dst_element_t(0.0);\n"
       << "  }\n";
  return Status::OK();
}

// ===========================================================================
// LstmStateCopyProgram
// ===========================================================================
Status LstmStateCopyProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("src", ShaderUsage::UseElementTypeAlias);
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
    body << "  dst[state_idx] = src[flat_idx];\n";
  } else {
    body << "  dst[flat_idx] = src[state_idx];\n";
  }
  return Status::OK();
}

// ===========================================================================
// LstmCellProgram — one cell step, always flat [batch, H] for h_prev/c_prev
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

  // Sequence length masking — early exit if past this batch's seq length.
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

  // H_prev * R^T  (h_prev always [batch, H] — flat indexing)
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
// LstmWriteYProgram — copies h_new to Y with optional seq_lens masking.
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
Status Lstm::ComputeInternal(ComputeContext& context) const {
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
  auto dtype = X->DataType();
  uint32_t total_threads = static_cast<uint32_t>(batch_size) * H;
  uint32_t wg_size = std::min(total_threads, 256u);
  if (wg_size == 0) wg_size = 1;
  uint32_t num_groups = (total_threads + wg_size - 1) / wg_size;

  // Helper lambdas for dispatching utility programs
  auto zero_fill = [&](Tensor* buf) -> Status {
    LstmZeroFillProgram prog;
    prog.CacheHint("zfill");
    prog.SetWorkgroupSize(wg_size).SetDispatchGroupSize(num_groups);
    prog.AddOutputs({{buf, ProgramTensorMetadataDependency::None}});
    prog.AddUniformVariables({{total_threads}});
    return context.RunProgram(prog);
  };

  auto copy_from_state = [&](const Tensor* src, Tensor* dst, int dir) -> Status {
    LstmStateCopyProgram prog{/*to_state=*/false, static_cast<int>(layout_)};
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
    LstmStateCopyProgram prog{/*to_state=*/true, static_cast<int>(layout_)};
    prog.CacheHint("to", std::to_string(layout_));
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

  // sequence_lens is on GPU (no InputMemoryType CPU override)
  bool has_seq_lens = (sequence_lens != nullptr);

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
    Tensor H_a = context.CreateGPUTensor(dtype, state_shape);
    Tensor C_a = context.CreateGPUTensor(dtype, state_shape);
    Tensor H_b = context.CreateGPUTensor(dtype, state_shape);
    Tensor C_b = context.CreateGPUTensor(dtype, state_shape);

    // Initialize H_a / C_a with zeros or initial state
    if (initial_h != nullptr) {
      ORT_RETURN_IF_ERROR(copy_from_state(initial_h, &H_a, dir));
    } else {
      ORT_RETURN_IF_ERROR(zero_fill(&H_a));
    }
    if (initial_c != nullptr) {
      ORT_RETURN_IF_ERROR(copy_from_state(initial_c, &C_a, dir));
    } else {
      ORT_RETURN_IF_ERROR(zero_fill(&C_a));
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
        h_write = &H_a; c_write = &C_a;
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
        y_prog.AddOutputs({{Y, ProgramTensorMetadataDependency::None}});
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
    if (seq_length > 0 && has_Y_h) {
      ORT_RETURN_IF_ERROR(copy_to_state(final_h, Y_h, dir));
    }
    if (seq_length > 0 && has_Y_c) {
      ORT_RETURN_IF_ERROR(copy_to_state(final_c, Y_c, dir));
    }
  }

  return Status::OK();
}

// ===========================================================================
// Kernel registrations
// ===========================================================================
ONNX_OPERATOR_VERSIONED_KERNEL_EX(LSTM, kOnnxDomain, 7, 13, kWebGpuExecutionProvider,

ONNX_OPERATOR_KERNEL_EX(LSTM, kOnnxDomain, 14, kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int32_t>()),
    Lstm);

}  // namespace webgpu
}  // namespace onnxruntime
