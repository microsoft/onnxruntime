// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "contrib_ops/webgpu/bert/rotary_embedding.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

ONNX_OPERATOR_KERNEL_EX(
    RotaryEmbedding,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes())
        .TypeConstraint("M", DataTypeImpl::GetTensorType<int64_t>()),
    RotaryEmbedding);

Status RotaryEmbeddingProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& input = shader.AddInput("input", ShaderUsage::UseUniform);
  // The second input is either seqlens (use_seqlens_for_position_) or position_ids (legacy path).
  // Declared here so the input order matches the caller's AddInputs order:
  // [input, seqlens|position_ids, cos_cache, sin_cache].
  const auto& position_ids_or_seqlens = use_seqlens_for_position_
                                            ? shader.AddInput("seqlens", ShaderUsage::UseUniform)
                                            : shader.AddInput("position_ids", ShaderUsage::UseUniform);
  const auto& cos_cache = shader.AddInput("cos_cache", ShaderUsage::UseUniform);
  const auto& sin_cache = shader.AddInput("sin_cache", ShaderUsage::UseUniform);
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform);
  const auto interleaved_str = interleaved_ ? "true" : "false";
  if (use_seqlens_for_position_) {
    // Seqlens path (GQA): inputs are [input, seqlens, cos_cache, sin_cache].
    // Compute per-batch past_seqlen from seqlens[batch_idx] = total_seqlen - 1.
    shader.MainFunctionBody() << "  let half_rotary_emb_dim = uniforms.cos_cache_shape[1];\n"
                                 "  let bsnh = global_idx / uniforms.global_stride % uniforms.global_shape;\n"
                                 "  let size = uniforms.global_shape[0] * uniforms.global_stride[0];\n"
                                 "  if (global_idx >= size) { return; }\n"
                                 "  if (bsnh[3] < half_rotary_emb_dim) {\n"
                                 "    let batch_idx = bsnh[0];\n"
                              << "    let seqlen_i = " << position_ids_or_seqlens.GetByOffset("batch_idx") << ";\n"
                              << "    let seqlen = u32(seqlen_i);\n"
                                 "    let total_seqlen = seqlen + 1u;\n"
                                 "    // Right-padded batches with prompt shorter than global_shape[1] would underflow u32; clamp to 0.\n"
                                 "    let past_seqlen = select(total_seqlen - uniforms.global_shape[1], 0u, total_seqlen <= uniforms.global_shape[1]);\n"
                                 "    let position_id = past_seqlen + bsnh[1];\n"
                              << "    let i = dot(bsnh, uniforms.input_output_stride) + select(0u, bsnh[3], " << interleaved_str << ");\n"
                              << "    let j = i + select(half_rotary_emb_dim, 1u, " << interleaved_str << ");\n"
                                                                                                          "    let max_position = uniforms.cos_cache_shape[0];\n"
                                                                                                          "    if (position_id >= max_position) {\n"
                              << "      " << output.SetByOffset("i", input.GetByOffset("i")) << "\n"
                              << "      " << output.SetByOffset("j", input.GetByOffset("j")) << "\n"
                                                                                                "    } else {\n"
                              << "      let re = " << input.GetByOffset("i") << " * " << cos_cache.GetByIndices("vec2<u32>(position_id, bsnh[3])") << " - " << input.GetByOffset("j") << " * " << sin_cache.GetByIndices("vec2<u32>(position_id, bsnh[3])") << ";\n"
                              << "      " << output.SetByOffset("i", "re") << "\n"
                              << "      let im = " << input.GetByOffset("i") << " * " << sin_cache.GetByIndices("vec2<u32>(position_id, bsnh[3])") << " + " << input.GetByOffset("j") << " * " << cos_cache.GetByIndices("vec2<u32>(position_id, bsnh[3])") << ";\n"
                              << "      " << output.SetByOffset("j", "im") << "\n"
                                                                              "    }\n"
                              << "  } else {\n"
                                 "    let k = dot(bsnh, uniforms.input_output_stride) + half_rotary_emb_dim;\n"
                              << "    " << output.SetByOffset("k", input.GetByOffset("k")) << "\n"
                              << "  }";
  } else {
    // Original path: inputs are [input, position_ids, cos_cache, sin_cache].
    const auto& position_ids = position_ids_or_seqlens;
    // TODO: remove output_indices.
    const auto& output_indices = shader.AddIndices("output_indices", ShaderUsage::None);
    shader.MainFunctionBody() << "  let half_rotary_emb_dim = uniforms.cos_cache_shape[1];\n"
                                 "  let bsnh = global_idx / uniforms.global_stride % uniforms.global_shape;\n"
                                 "  let size = uniforms.global_shape[0] * uniforms.global_stride[0];\n"
                                 "  if (global_idx >= size) { return; }\n"
                                 "  if (bsnh[3] < half_rotary_emb_dim) {\n"
                              << "    let position_ids_idx = " << position_ids.BroadcastedIndicesToOffset("bsnh.xy", output_indices) << ";\n"
                              << "    let raw_pos = " << position_ids.GetByOffset("position_ids_idx") << ";\n"
                              << "    let i = dot(bsnh, uniforms.input_output_stride) + select(0, bsnh[3], " << interleaved_str << ");\n"
                              << "    let j = i + select(half_rotary_emb_dim, 1, " << interleaved_str << ");\n"
                                                                                                         "    let max_position = uniforms.cos_cache_shape[0];\n"
                                                                                                         // Bounds check: raw_pos < 0 catches negative position_ids (i32 from truncated int64).
                                                                                                         // After u32 conversion + offset, check >= max_position catches too-large values.
                                                                                                         // On OOB, pass through input unchanged (same as CUDA kernel behavior).
                                                                                                         "    if (raw_pos < 0) {\n"
                              << "      " << output.SetByOffset("i", input.GetByOffset("i")) << "\n"
                              << "      " << output.SetByOffset("j", input.GetByOffset("j")) << "\n"
                                                                                                "    } else {\n"
                                                                                                "      let position_id = u32(raw_pos) + select(0, bsnh[1], position_ids_idx == 0);\n"
                                                                                                "      if (position_id >= max_position) {\n"
                              << "        " << output.SetByOffset("i", input.GetByOffset("i")) << "\n"
                              << "        " << output.SetByOffset("j", input.GetByOffset("j")) << "\n"
                                                                                                  "      } else {\n"
                              << "        let re = " << input.GetByOffset("i") << " * " << cos_cache.GetByIndices("vec2<u32>(position_id, bsnh[3])") << " - " << input.GetByOffset("j") << " * " << sin_cache.GetByIndices("vec2<u32>(position_id, bsnh[3])") << ";\n"
                              << "        " << output.SetByOffset("i", "re") << "\n"
                              << "        let im = " << input.GetByOffset("i") << " * " << sin_cache.GetByIndices("vec2<u32>(position_id, bsnh[3])") << " + " << input.GetByOffset("j") << " * " << cos_cache.GetByIndices("vec2<u32>(position_id, bsnh[3])") << ";\n"
                              << "        " << output.SetByOffset("j", "im") << "\n"
                                                                                "      }\n"
                                                                                "    }\n"
                              << "  } else { \n"
                                 "    let k = dot(bsnh, uniforms.input_output_stride) + half_rotary_emb_dim;\n"
                              << "    " << output.SetByOffset("k", input.GetByOffset("k")) << "\n"
                              << "  }";
  }

  return Status::OK();
}

Status FusedQKRotaryEmbeddingProgram::GenerateShaderCode(ShaderHelper& shader) const {
  // Inputs. q_input/k_input use the element-type alias when has_qk_norm_ is true so we can
  // mix in the f32-computed inverse-RMS scale at element-type precision.
  const ShaderUsage qk_input_usage = has_qk_norm_
                                         ? (ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias)
                                         : ShaderUsage::UseUniform;
  const auto& q_input = shader.AddInput("q_input", qk_input_usage);
  const auto& k_input = shader.AddInput("k_input", qk_input_usage);
  const auto& seqlens = shader.AddInput("seqlens", ShaderUsage::UseUniform);
  const auto& cos_cache = shader.AddInput("cos_cache", ShaderUsage::UseUniform);
  const auto& sin_cache = shader.AddInput("sin_cache", ShaderUsage::UseUniform);

  // Optional per-head RMS norm weights (1D tensors of length head_size). When present,
  // a fused per-head normalization is applied to Q/K before the rotary rotation:
  //     x_norm[c] = x[c] * inverseSqrt(mean(x[..]^2) + epsilon) * weight[c]
  // Decode-only fast path: each thread re-reads its own head's head_size channels to
  // compute the sum-of-squares (no reductions, no shared memory). The redundant L1
  // traffic is sub-microsecond on Qwen3-1.7B decode geometry.
  if (has_qk_norm_) {
    shader.AddInput("q_norm_weight", ShaderUsage::UseUniform);
    shader.AddInput("k_norm_weight", ShaderUsage::UseUniform);
  }

  // Outputs
  const auto& q_output = shader.AddOutput("q_output", ShaderUsage::UseUniform);
  const auto& k_output = shader.AddOutput("k_output", ShaderUsage::UseUniform);

  const auto interleaved_str = interleaved_ ? "true" : "false";

  auto& body = shader.MainFunctionBody();
  body
      << "  if (global_idx >= uniforms.q_domain_size) { return; }\n"
      << "  let half_rotary_dim = uniforms.cos_cache_shape[1];\n"
      << "  let bsnh = global_idx / uniforms.q_global_stride % uniforms.q_global_shape;\n"
      << "  let needs_k = bsnh[2] < uniforms.k_global_shape[2];\n";

  // Per-head RMS computation (Approach A, no reductions). For non-interleaved layouts the
  // bsnh[3] coordinate is the lower channel of a rotary pair, so the head base offset is
  // dot(bsnh, stride) - bsnh[3] (i.e. drop the channel contribution). q_input_output_stride[3]
  // is always 1 (channel stride), so subtracting bsnh[3] gives the head's channel-0 offset
  // for both interleaved and non-interleaved layouts in the rotated branch. In the
  // passthrough else-branch we recompute from bsnh[0..2] explicitly.
  if (has_qk_norm_) {
    body
        << "  let q_head_base = bsnh[0] * uniforms.q_input_output_stride[0]\n"
        << "                  + bsnh[1] * uniforms.q_input_output_stride[1]\n"
        << "                  + bsnh[2] * uniforms.q_input_output_stride[2];\n"
        << "  var q_sumsq: f32 = 0.0;\n"
        << "  for (var c: u32 = 0u; c < uniforms.head_size; c = c + 1u) {\n"
        << "    let q_v = f32(" << q_input.GetByOffset("q_head_base + c") << ");\n"
        << "    q_sumsq = q_sumsq + q_v * q_v;\n"
        << "  }\n"
        << "  let q_inv_rms = q_input_element_t(inverseSqrt(q_sumsq / f32(uniforms.head_size) + uniforms.qk_norm_epsilon));\n"
        << "  let k_head_base = bsnh[0] * uniforms.k_input_output_stride[0]\n"
        << "                  + bsnh[1] * uniforms.k_input_output_stride[1]\n"
        << "                  + bsnh[2] * uniforms.k_input_output_stride[2];\n"
        << "  var k_inv_rms = k_input_element_t(0);\n"
        << "  if (needs_k) {\n"
        << "    var k_sumsq: f32 = 0.0;\n"
        << "    for (var c: u32 = 0u; c < uniforms.head_size; c = c + 1u) {\n"
        << "      let k_v = f32(" << k_input.GetByOffset("k_head_base + c") << ");\n"
        << "      k_sumsq = k_sumsq + k_v * k_v;\n"
        << "    }\n"
        << "    k_inv_rms = k_input_element_t(inverseSqrt(k_sumsq / f32(uniforms.head_size) + uniforms.qk_norm_epsilon));\n"
        << "  }\n";
  }

  // Helpers that load Q/K and (when has_qk_norm_) apply the fused per-channel norm scale.
  // The channel index expressions match the qi/qj/ki/kj/qk/kk computations used below.
  auto load_q = [&](const std::string& off, const std::string& chan) {
    if (!has_qk_norm_) {
      return q_input.GetByOffset(off);
    }
    return std::string("(") + q_input.GetByOffset(off) + " * q_inv_rms * q_norm_weight[" + chan + "])";
  };
  auto load_k = [&](const std::string& off, const std::string& chan) {
    if (!has_qk_norm_) {
      return k_input.GetByOffset(off);
    }
    return std::string("(") + k_input.GetByOffset(off) + " * k_inv_rms * k_norm_weight[" + chan + "])";
  };

  // Channel index expressions for the rotated branch. For interleaved layout the pair is
  // (2*bsnh[3], 2*bsnh[3]+1); otherwise it is (bsnh[3], bsnh[3]+half_rotary_dim).
  const std::string c_i = interleaved_ ? "(2u * bsnh[3])" : "bsnh[3]";
  const std::string c_j = interleaved_ ? "(2u * bsnh[3] + 1u)" : "(bsnh[3] + half_rotary_dim)";
  // Channel index for the passthrough else-branch (only fires when head_size > 2 * half_rotary_dim).
  const std::string c_k = "(bsnh[3] + half_rotary_dim)";

  body
      << "  if (bsnh[3] < half_rotary_dim) {\n"
      << "    let batch_idx = bsnh[0];\n"
      << "    let sequence_idx = bsnh[1];\n"
      << "    let seqlen_i = " << seqlens.GetByOffset("batch_idx") << ";\n"
      << "    let seqlen = u32(seqlen_i);\n"
      << "    let total_seqlen = seqlen + 1u;\n"
      << "    // Right-padded batches with prompt shorter than q_global_shape[1] would underflow u32; clamp to 0.\n"
      << "    let past_seqlen = select(total_seqlen - uniforms.q_global_shape[1], 0u, total_seqlen <= uniforms.q_global_shape[1]);\n"
      << "    let position_id = past_seqlen + sequence_idx;\n"
      << "    let qi = dot(bsnh, uniforms.q_input_output_stride) + select(0u, bsnh[3], " << interleaved_str << ");\n"
      << "    let qj = qi + select(half_rotary_dim, 1u, " << interleaved_str << ");\n"
      << "    let q_at_qi = " << load_q("qi", c_i) << ";\n"
      << "    let q_at_qj = " << load_q("qj", c_j) << ";\n"
      << "    let max_position = uniforms.cos_cache_shape[0];\n"
      << "    if (position_id >= max_position) {\n"
      // Bounds check: position_id must be within cos/sin cache range.
      // On OOB, pass through input (norm-applied if has_qk_norm_) unchanged.
      << "      " << q_output.SetByOffset("qi", "q_at_qi") << "\n"
      << "      " << q_output.SetByOffset("qj", "q_at_qj") << "\n"
      << "      if (needs_k) {\n"
      << "        let ki = dot(bsnh, uniforms.k_input_output_stride) + select(0u, bsnh[3], " << interleaved_str << ");\n"
      << "        let kj = ki + select(half_rotary_dim, 1u, " << interleaved_str << ");\n"
      << "        let k_at_ki = " << load_k("ki", c_i) << ";\n"
      << "        let k_at_kj = " << load_k("kj", c_j) << ";\n"
      << "        " << k_output.SetByOffset("ki", "k_at_ki") << "\n"
      << "        " << k_output.SetByOffset("kj", "k_at_kj") << "\n"
      << "      }\n"
      << "    } else {\n"
      << "      let cos_v = " << cos_cache.GetByIndices("vec2<u32>(position_id, bsnh[3])") << ";\n"
      << "      let sin_v = " << sin_cache.GetByIndices("vec2<u32>(position_id, bsnh[3])") << ";\n"
      << "      let q_re = q_at_qi * cos_v - q_at_qj * sin_v;\n"
      << "      " << q_output.SetByOffset("qi", "q_re") << "\n"
      << "      let q_im = q_at_qi * sin_v + q_at_qj * cos_v;\n"
      << "      " << q_output.SetByOffset("qj", "q_im") << "\n"
      << "      if (needs_k) {\n"
      << "        let ki = dot(bsnh, uniforms.k_input_output_stride) + select(0u, bsnh[3], " << interleaved_str << ");\n"
      << "        let kj = ki + select(half_rotary_dim, 1u, " << interleaved_str << ");\n"
      << "        let k_at_ki = " << load_k("ki", c_i) << ";\n"
      << "        let k_at_kj = " << load_k("kj", c_j) << ";\n"
      << "        let k_re = k_at_ki * cos_v - k_at_kj * sin_v;\n"
      << "        " << k_output.SetByOffset("ki", "k_re") << "\n"
      << "        let k_im = k_at_ki * sin_v + k_at_kj * cos_v;\n"
      << "        " << k_output.SetByOffset("kj", "k_im") << "\n"
      << "      }\n"
      << "    }\n"
      << "  } else {\n"
      << "    let qk = dot(bsnh, uniforms.q_input_output_stride) + half_rotary_dim;\n"
      << "    let q_at_qk = " << load_q("qk", c_k) << ";\n"
      << "    " << q_output.SetByOffset("qk", "q_at_qk") << "\n"
      << "    if (needs_k) {\n"
      << "      let kk = dot(bsnh, uniforms.k_input_output_stride) + half_rotary_dim;\n"
      << "      let k_at_kk = " << load_k("kk", c_k) << ";\n"
      << "      " << k_output.SetByOffset("kk", "k_at_kk") << "\n"
      << "    }\n"
      << "  }\n";
  return Status::OK();
}

RotaryEmbedding::RotaryEmbedding(const OpKernelInfo& info) : WebGpuKernel(info) {
  scale_ = info.GetAttrOrDefault<float>("scale", 1.0);
  rotary_embedding_dim_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("rotary_embedding_dim", 0));
  num_heads_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("num_heads", 0));
  interleaved_ = (info.GetAttrOrDefault<int64_t>("interleaved", 0) == 1);
  is_packed_batching_ = (info.GetAttrOrDefault<int64_t>("is_packed_batching", 0) == 1);
}

Status RunRotaryEmbedding(onnxruntime::webgpu::ComputeContext& context,
                          const Tensor* input,
                          const Tensor* position_ids_or_seqlens,
                          const Tensor* cos_cache,
                          const Tensor* sin_cache,
                          Tensor* output,
                          int batch_size,
                          int sequence_length,
                          int hidden_size,
                          int head_size,
                          float scale,
                          bool rotary_interleaved,
                          bool use_seqlens_for_position,
                          const std::vector<uint32_t>& input_output_strides) {
  const auto half_rotary_embedding_dim = onnxruntime::narrow<uint32_t>(cos_cache->Shape()[1]);
  const auto num_heads = hidden_size / head_size;

  // Rotary embeddings are calculated in a pair-wise fashion. Use the shape
  // [batch, sequence, heads, half_rotary_dim_complement] to unfold the global index in shader.
  const TensorShape global_shape({static_cast<int64_t>(batch_size),
                                  static_cast<int64_t>(sequence_length),
                                  static_cast<int64_t>(num_heads),
                                  static_cast<int64_t>(head_size - half_rotary_embedding_dim)});
  const auto rank = global_shape.NumDimensions();
  std::vector<uint32_t> global_dims(rank);
  std::vector<uint32_t> global_strides(rank);
  for (size_t j = 0; j < rank; ++j) {
    global_dims[j] = onnxruntime::narrow<uint32_t>(global_shape[j]);
    global_strides[j] = onnxruntime::narrow<uint32_t>(global_shape.SizeFromDimension(j + 1));
  }
  const auto output_size = onnxruntime::narrow<uint32_t>(global_shape.Size());

  RotaryEmbeddingProgram program(rotary_interleaved, use_seqlens_for_position);
  program
      .CacheHint(rotary_interleaved, use_seqlens_for_position)
      .AddInputs({{input, ProgramTensorMetadataDependency::TypeAndRank},
                  {position_ids_or_seqlens, ProgramTensorMetadataDependency::TypeAndRank},
                  {cos_cache, ProgramTensorMetadataDependency::Rank},
                  {sin_cache, ProgramTensorMetadataDependency::Rank}})
      .AddOutput({output, ProgramTensorMetadataDependency::None})
      .SetDispatchGroupSize((output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({{scale},
                            {gsl::make_span(global_dims)},
                            {gsl::make_span(global_strides)},
                            {gsl::make_span(input_output_strides)}});
  if (!use_seqlens_for_position) {
    program.AddIndices(TensorShape{1, 1});
  }
  return context.RunProgram(program);
}

Status RotaryEmbedding::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {
  const auto* input = context.Input<Tensor>(0);
  const auto input_shape = input->Shape();
  const auto* position_ids = context.Input<Tensor>(1);
  const auto* cos_cache = context.Input<Tensor>(2);
  const auto* sin_cache = context.Input<Tensor>(3);
  auto* output = context.Output(0, input_shape);

  const auto batch_size = onnxruntime::narrow<uint32_t>(input->Shape()[0]);
  const auto batch_stride = onnxruntime::narrow<uint32_t>(input_shape.SizeFromDimension(1));
  const auto sequence_length = onnxruntime::narrow<uint32_t>(input_shape[input_shape.NumDimensions() - 2]);
  const auto hidden_size = batch_stride / sequence_length;
  const auto half_rotary_embedding_dim = onnxruntime::narrow<uint32_t>(cos_cache->Shape()[1]);
  const auto head_size = rotary_embedding_dim_ == 0 ? half_rotary_embedding_dim * 2 : hidden_size / num_heads_;

  // position_ids bounds validation is handled by shader-side defense-in-depth checks
  // (OOB position_ids → pass-through input unchanged). Host-side value scanning is not possible
  // because WebGPU program inputs must be GPU buffers (InputMemoryType(OrtMemTypeCPUInput) is
  // incompatible with AddInputs).

  const auto input_output_strides =
      input_shape.NumDimensions() == 3
          ? std::vector<uint32_t>({batch_stride, hidden_size, head_size, 1})
          : (input_shape.NumDimensions() == 4
                 ? std::vector<uint32_t>({batch_stride, head_size, sequence_length * head_size, 1})
                 : std::vector<uint32_t>({}));

  return RunRotaryEmbedding(context, input, position_ids, cos_cache, sin_cache, output,
                            static_cast<int>(batch_size), static_cast<int>(sequence_length),
                            static_cast<int>(hidden_size), static_cast<int>(head_size),
                            scale_, interleaved_, /*use_seqlens_for_position=*/false, input_output_strides);
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
