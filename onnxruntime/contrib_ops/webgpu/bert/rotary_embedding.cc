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
  const auto& position_ids = shader.AddInput("position_ids", ShaderUsage::UseUniform);
  const auto& cos_cache = shader.AddInput("cos_cache", ShaderUsage::UseUniform);
  const auto& sin_cache = shader.AddInput("sin_cache", ShaderUsage::UseUniform);
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform);
  // TODO: remove output_indices.
  const auto& output_indices = shader.AddIndices("output_indices", ShaderUsage::None);
  const auto interleaved_str = interleaved_ ? "true" : "false";
  shader.MainFunctionBody() << "  let half_rotary_emb_dim = uniforms.cos_cache_shape[1];\n"
                               "  let bsnh = global_idx / uniforms.global_stride % uniforms.global_shape;\n"
                               "  let size = uniforms.global_shape[0] * uniforms.global_stride[0];\n"
                               "  if (global_idx >= size) { return; }\n"
                               "  if (bsnh[3] < half_rotary_emb_dim) {\n"
                            << "    let position_ids_idx = " << position_ids.BroadcastedIndicesToOffset("bsnh.xy", output_indices) << ";\n"
                            << "    let position_id = u32(" << position_ids.GetByOffset("position_ids_idx") << ") + select(0, bsnh[1], position_ids_idx == 0);\n"
                            << "    let i = dot(bsnh, uniforms.input_output_stride) + select(0, bsnh[3], " << interleaved_str << ");\n"
                            << "    let j = i + select(half_rotary_emb_dim, 1, " << interleaved_str << ");\n"
                            << "    let re = " << input.GetByOffset("i") << " * " << cos_cache.GetByIndices("vec2<u32>(position_id, bsnh[3])") << " - " << input.GetByOffset("j") << " * " << sin_cache.GetByIndices("vec2<u32>(position_id, bsnh[3])") << ";\n"
                            << "    " << output.SetByOffset("i", "re") << "\n"
                            << "    let im = " << input.GetByOffset("i") << " * " << sin_cache.GetByIndices("vec2<u32>(position_id, bsnh[3])") << " + " << input.GetByOffset("j") + " * " << cos_cache.GetByIndices("vec2<u32>(position_id, bsnh[3])") << ";\n"
                            << "    " << output.SetByOffset("j", "im") << "\n"
                            << "  } else { \n"
                               "    let k = dot(bsnh, uniforms.input_output_stride) + half_rotary_emb_dim;\n"
                            << "    " << output.SetByOffset("k", input.GetByOffset("k")) << "\n"
                            << "  }";

  return Status::OK();
}

Status FusedQKRotaryEmbeddingProgram::GenerateShaderCode(ShaderHelper& shader) const {
  // Inputs
  const auto& q_input = shader.AddInput("q_input", ShaderUsage::UseUniform);
  const auto& k_input = shader.AddInput("k_input", ShaderUsage::UseUniform);
  const auto& seqlens = shader.AddInput("seqlens", ShaderUsage::UseUniform);
  const auto& cos_cache = shader.AddInput("cos_cache", ShaderUsage::UseUniform);
  const auto& sin_cache = shader.AddInput("sin_cache", ShaderUsage::UseUniform);
  // Outputs
  const auto& q_output = shader.AddOutput("q_output", ShaderUsage::UseUniform);
  const auto& k_output = shader.AddOutput("k_output", ShaderUsage::UseUniform);

  const auto interleaved_str = interleaved_ ? "true" : "false";

  shader.MainFunctionBody()
      << "  if (global_idx >= uniforms.q_domain_size) { return; }\n"
      << "  let half_rotary_dim = uniforms.cos_cache_shape[1];\n"
      << "  let bsnh = global_idx / uniforms.q_global_stride % uniforms.q_global_shape;\n"
      << "  if (bsnh[3] < half_rotary_dim) {\n"
      << "    let batch_idx = bsnh[0];\n"
      << "    let sequence_idx = bsnh[1];\n"
      << "    let seqlen_i = " << seqlens.GetByOffset("batch_idx") << ";\n"
      << "    let seqlen = u32(seqlen_i);\n"
      << "    let total_seqlen = seqlen + 1u;\n"
      << "    let past_seqlen = total_seqlen - uniforms.q_global_shape[1];\n"
      << "    let position_id = past_seqlen + sequence_idx;\n"
      << "    let cos_v = " << cos_cache.GetByIndices("vec2<u32>(position_id, bsnh[3])") << ";\n"
      << "    let sin_v = " << sin_cache.GetByIndices("vec2<u32>(position_id, bsnh[3])") << ";\n"
      << "    let qi = dot(bsnh, uniforms.q_input_output_stride) + select(0u, bsnh[3], " << interleaved_str << ");\n"
      << "    let qj = qi + select(half_rotary_dim, 1u, " << interleaved_str << ");\n"
      << "    let q_re = " << q_input.GetByOffset("qi") << " * cos_v - " << q_input.GetByOffset("qj") << " * sin_v;\n"
      << "    " << q_output.SetByOffset("qi", "q_re") << "\n"
      << "    let q_im = " << q_input.GetByOffset("qi") << " * sin_v + " << q_input.GetByOffset("qj") << " * cos_v;\n"
      << "    " << q_output.SetByOffset("qj", "q_im") << "\n"
      // Conditionally process Key (only for heads that exist in K domain)
      << "    if (bsnh[2] < uniforms.k_global_shape[2]) {\n"
      << "      let ki = dot(bsnh, uniforms.k_input_output_stride) + select(0u, bsnh[3], " << interleaved_str << ");\n"
      << "      let kj = ki + select(half_rotary_dim, 1u, " << interleaved_str << ");\n"
      << "      let k_re = " << k_input.GetByOffset("ki") << " * cos_v - " << k_input.GetByOffset("kj") << " * sin_v;\n"
      << "      " << k_output.SetByOffset("ki", "k_re") << "\n"
      << "      let k_im = " << k_input.GetByOffset("ki") << " * sin_v + " << k_input.GetByOffset("kj") << " * cos_v;\n"
      << "      " << k_output.SetByOffset("kj", "k_im") << "\n"
      << "    }\n"
      << "  } else {\n"
      << "    let qk = dot(bsnh, uniforms.q_input_output_stride) + half_rotary_dim;\n"
      << "    " << q_output.SetByOffset("qk", q_input.GetByOffset("qk")) << "\n"
      // Conditionally process Key (only for heads that exist in K domain)
      << "    if (bsnh[2] < uniforms.k_global_shape[2]) {\n"
      << "      let kk = dot(bsnh, uniforms.k_input_output_stride) + half_rotary_dim;\n"
      << "      " << k_output.SetByOffset("kk", k_input.GetByOffset("kk")) << "\n"
      << "    }\n"
      << "  }\n";
  return Status::OK();
}

Status SplitPackedQKVWithRotaryEmbeddingProgram::GenerateShaderCode(ShaderHelper& sh) const {
  // Inputs
  const auto& packed_qkv = sh.AddInput("packed_qkv", ShaderUsage::UseUniform);
  const auto& seqlens = sh.AddInput("seqlens", ShaderUsage::UseUniform);
  const auto& cos_cache = sh.AddInput("cos_cache", ShaderUsage::UseUniform);
  const auto& sin_cache = sh.AddInput("sin_cache", ShaderUsage::UseUniform);

  // Outputs
  const auto& query = sh.AddOutput("query", ShaderUsage::UseUniform);
  const auto& key = sh.AddOutput("key", ShaderUsage::UseUniform);
  const auto& value = sh.AddOutput("val", ShaderUsage::UseUniform);

  const auto interleaved_str = interleaved_ ? "true" : "false";

  sh.MainFunctionBody()
      << sh.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.dispatch_size")
      << "  // Dispatch: batch * seq * num_heads * (half_rotary_dim + need_copy_dim)\n"
      << "  // work_per_head = half_rotary_dim + (head_size - 2 * half_rotary_dim)\n"
      << "  let work_per_head = uniforms.head_size - uniforms.half_rotary_dim;\n"
      << "  let total_work = uniforms.num_heads * work_per_head;\n"
      << "  \n"
      << "  let batch_idx = global_idx / (uniforms.sequence_length * total_work);\n"
      << "  let remainder1 = global_idx % (uniforms.sequence_length * total_work);\n"
      << "  let seq_idx = remainder1 / total_work;\n"
      << "  let remainder2 = remainder1 % total_work;\n"
      << "  let head_idx = remainder2 / work_per_head;\n"
      << "  let in_head_idx = remainder2 % work_per_head;\n"
      << "\n"
      << "  // Calculate base offset in packed_qkv for this token\n"
      << "  // Layout per token: [Q(hidden_size), K(kv_hidden_size), V(kv_hidden_size)]\n"
      << "  let token_size = uniforms.hidden_size + 2u * uniforms.kv_hidden_size;\n"
      << "  let base_offset = batch_idx * uniforms.sequence_length * token_size + seq_idx * token_size;\n"
      << "\n"
      << "  // Calculate position_id (needed for rotary embedding)\n"
      << "  let seqlen_i = " << seqlens.GetByOffset("batch_idx") << ";\n"
      << "  let seqlen = u32(seqlen_i);\n"
      << "  var position_id: u32 = 0u;\n"
      << "  if (uniforms.first_prompt_flag == 1u) {\n"
      << "    let total_seqlen = seqlen + 1u;\n"
      << "    position_id = select(1u, seq_idx, seq_idx < total_seqlen);\n"
      << "  } else if (uniforms.subsequent_prompt_flag == 1u) {\n"
      << "    let total_seqlen = seqlen + 1u;\n"
      << "    let past_seqlen = total_seqlen - uniforms.sequence_length;\n"
      << "    let cand = past_seqlen + seq_idx;\n"
      << "    position_id = select(1u, cand, cand < total_seqlen);\n"
      << "  } else {\n"
      << "    position_id = seqlen;\n"
      << "  }\n"
      << "\n"
      << "  if (in_head_idx < uniforms.half_rotary_dim) {\n"
      << "    // Process a rotary pair (i, j)\n"
      << "    let cos_v = " << cos_cache.GetByIndices("vec2<u32>(position_id, in_head_idx)") << ";\n"
      << "    let sin_v = " << sin_cache.GetByIndices("vec2<u32>(position_id, in_head_idx)") << ";\n"
      << "\n"
      << "    // Calculate actual indices in the head for i and j\n"
      << "    let idx_i = select(in_head_idx, in_head_idx, " << interleaved_str << ");\n"
      << "    let idx_j = select(in_head_idx + uniforms.half_rotary_dim, in_head_idx + 1u, " << interleaved_str << ");\n"
      << "\n"
      << "    // Process Q pair\n"
      << "    let q_base = base_offset + head_idx * uniforms.head_size;\n"
      << "    let q_i_offset = q_base + idx_i;\n"
      << "    let q_j_offset = q_base + idx_j;\n"
      << "    let q_i = " << packed_qkv.GetByOffset("q_i_offset") << ";\n"
      << "    let q_j = " << packed_qkv.GetByOffset("q_j_offset") << ";\n"
      << "    let q_re = q_i * cos_v - q_j * sin_v;\n"
      << "    let q_im = q_i * sin_v + q_j * cos_v;\n"
      << "    " << query.SetByIndices("vec3<u32>(batch_idx, seq_idx, head_idx * uniforms.head_size + idx_i)", "q_re") << ";\n"
      << "    " << query.SetByIndices("vec3<u32>(batch_idx, seq_idx, head_idx * uniforms.head_size + idx_j)", "q_im") << ";\n"
      << "\n"
      << "    // Process K and V pairs if within kv_num_heads\n"
      << "    if (head_idx < uniforms.kv_num_heads) {\n"
      << "      let k_base = base_offset + uniforms.hidden_size + head_idx * uniforms.head_size;\n"
      << "      let k_i_offset = k_base + idx_i;\n"
      << "      let k_j_offset = k_base + idx_j;\n"
      << "      let k_i = " << packed_qkv.GetByOffset("k_i_offset") << ";\n"
      << "      let k_j = " << packed_qkv.GetByOffset("k_j_offset") << ";\n"
      << "      let k_re = k_i * cos_v - k_j * sin_v;\n"
      << "      let k_im = k_i * sin_v + k_j * cos_v;\n"
      << "      " << key.SetByIndices("vec3<u32>(batch_idx, seq_idx, head_idx * uniforms.head_size + idx_i)", "k_re") << ";\n"
      << "      " << key.SetByIndices("vec3<u32>(batch_idx, seq_idx, head_idx * uniforms.head_size + idx_j)", "k_im") << ";\n"
      << "\n"
      << "      // V doesn't need rotary, just copy the pair\n"
      << "      let v_base = base_offset + uniforms.hidden_size + uniforms.kv_hidden_size + head_idx * uniforms.head_size;\n"
      << "      let v_i = " << packed_qkv.GetByOffset("v_base + idx_i") << ";\n"
      << "      let v_j = " << packed_qkv.GetByOffset("v_base + idx_j") << ";\n"
      << "      " << value.SetByIndices("vec3<u32>(batch_idx, seq_idx, head_idx * uniforms.head_size + idx_i)", "v_i") << ";\n"
      << "      " << value.SetByIndices("vec3<u32>(batch_idx, seq_idx, head_idx * uniforms.head_size + idx_j)", "v_j") << ";\n"
      << "    }\n"
      << "  } else {\n"
      << "    // Process non-rotary elements (direct copy)\n"
      << "    // actual_idx = 2 * half_rotary_dim + (in_head_idx - half_rotary_dim)\n"
      << "    let actual_idx = uniforms.half_rotary_dim + in_head_idx;\n"
      << "\n"
      << "    // Copy Q\n"
      << "    let q_offset = base_offset + head_idx * uniforms.head_size + actual_idx;\n"
      << "    let q_data = " << packed_qkv.GetByOffset("q_offset") << ";\n"
      << "    " << query.SetByIndices("vec3<u32>(batch_idx, seq_idx, head_idx * uniforms.head_size + actual_idx)", "q_data") << ";\n"
      << "\n"
      << "    // Copy K and V if within kv_num_heads\n"
      << "    if (head_idx < uniforms.kv_num_heads) {\n"
      << "      let k_offset = base_offset + uniforms.hidden_size + head_idx * uniforms.head_size + actual_idx;\n"
      << "      let k_data = " << packed_qkv.GetByOffset("k_offset") << ";\n"
      << "      " << key.SetByIndices("vec3<u32>(batch_idx, seq_idx, head_idx * uniforms.head_size + actual_idx)", "k_data") << ";\n"
      << "\n"
      << "      let v_offset = base_offset + uniforms.hidden_size + uniforms.kv_hidden_size + head_idx * uniforms.head_size + actual_idx;\n"
      << "      let v_data = " << packed_qkv.GetByOffset("v_offset") << ";\n"
      << "      " << value.SetByIndices("vec3<u32>(batch_idx, seq_idx, head_idx * uniforms.head_size + actual_idx)", "v_data") << ";\n"
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

  // Rotary embeddings will be calculated in a pair-wise fashion. In accordance, use the shape
  // [batch size, sequence length, num of heads, num of pairs to rotate + num of dims to copy]
  // to unfold the global index in shader.
  const TensorShape global_shape({batch_size,
                                  sequence_length,
                                  hidden_size / head_size,
                                  head_size - half_rotary_embedding_dim});

  const auto rank = global_shape.NumDimensions();
  std::vector<uint32_t> global_dims(rank);
  std::vector<uint32_t> global_strides(rank);
  for (size_t j = 0; j < rank; ++j) {
    global_dims[j] = onnxruntime::narrow<uint32_t>(global_shape[j]);
    global_strides[j] = onnxruntime::narrow<uint32_t>(global_shape.SizeFromDimension(j + 1));
  }

  const auto output_size = onnxruntime::narrow<const uint32_t>(global_shape.Size());
  RotaryEmbeddingProgram program{interleaved_};
  const auto input_output_strides =
      input_shape.NumDimensions() == 3
          ? std::vector<uint32_t>({batch_stride, hidden_size, head_size, 1})
          : (input_shape.NumDimensions() == 4
                 ? std::vector<uint32_t>({batch_stride, head_size, sequence_length * head_size, 1})
                 : std::vector<uint32_t>({}));

  program
      .CacheHint(interleaved_)
      .AddInputs({{input, ProgramTensorMetadataDependency::TypeAndRank},
                  {position_ids, ProgramTensorMetadataDependency::Rank},
                  {cos_cache, ProgramTensorMetadataDependency::Rank},
                  {sin_cache, ProgramTensorMetadataDependency::Rank}})
      .AddOutput({output, ProgramTensorMetadataDependency::None})
      .SetDispatchGroupSize((output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({{scale_},
                            {gsl::make_span(global_dims)},
                            {gsl::make_span(global_strides)},
                            {gsl::make_span(input_output_strides)}})
      .AddIndices(TensorShape{1, 1});
  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
