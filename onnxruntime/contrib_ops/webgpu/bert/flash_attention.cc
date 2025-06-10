// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/bert/multihead_attention_helper.h"
#include "contrib_ops/webgpu/bert/flash_attention.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"

#include "core/providers/webgpu/webgpu_supported_types.h"

using namespace onnxruntime::webgpu;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;
using namespace onnxruntime::contrib::multihead_attention_helper;

namespace onnxruntime {
namespace contrib {
namespace webgpu {

Status CopyKVCacheProgram::GenerateShaderCode(ShaderHelper& shader) const {
  // Expectations are
  //    qkv have same number of heads and hidden dimension (head size).
  //    qkv are in BSNH format.
  //            B - batch size but shader only supports batch_size 1.
  //            S - current sequence length but shader supports only S = 1.
  //            N - number of heads.
  //            H - head size or hidden dimension for each qkv head.
  //  KV cache is stored as BN(total_sequence_length)H
  //  Attention bias is in BN(total_sequence_length)
  const auto& key = shader.AddInput("key", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias | ShaderUsage::UseIndicesTypeAlias);
  shader.AddInput("value", ShaderUsage::UseUniform);
  const auto& present_key = shader.AddOutput("present_key", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);
  const auto& present_value = shader.AddOutput("present_value", ShaderUsage::UseUniform);
  const auto& copy_kv_shape = shader.AddIndices("copy_kv_shape");

  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.copy_size")
                            << "  let output_indices = " << copy_kv_shape.OffsetToIndices("global_idx") << ";\n"
                            << "  let head_size_id = output_indices[3];\n"
                               "  let sequence_id = output_indices[2];\n"
                               "  let num_head_id = output_indices[1];\n"
                               "  let batch = output_indices[0];\n";
  if (has_past_) {
    shader.MainFunctionBody() << "let past_sequence_length = uniforms.past_sequence_length;\n";
    if (past_present_share_buffer_) {
      shader.MainFunctionBody() << "  let present_offset = " << present_key.IndicesToOffset("present_key_indices_t(batch, num_head_id, past_sequence_length + sequence_id, head_size_id)") << ";\n"
                                << "  let offset = " << key.IndicesToOffset(kv_BNSH_ ? "key_indices_t(batch, num_head_id, sequence_id, head_size_id)" : "key_indices_t(batch, sequence_id, num_head_id, head_size_id)") << ";\n"
                                << "  " << present_key.SetByOffset("present_offset", "key[offset]") << ";\n"
                                << "  " << present_value.SetByOffset("present_offset", "value[offset]") << ";\n";
    } else {
      const auto& past_key = shader.AddInput("past_key", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias | ShaderUsage::UseIndicesTypeAlias);
      shader.AddInput("past_value", ShaderUsage::UseUniform);
      shader.MainFunctionBody() << "let present_offset = global_idx;"
                                << "if (sequence_id < past_sequence_length) {\n"
                                << "  let pastOffset = " << past_key.IndicesToOffset("past_key_indices_t(batch, num_head_id, sequence_id, head_size_id)") << ";\n"
                                << "  " << present_key.SetByOffset("present_offset", "past_key[pastOffset]") << ";\n"
                                << "  " << present_value.SetByOffset("present_offset", "past_value[pastOffset]") << ";\n"
                                << "} else {\n"
                                << "  let offset = " << key.IndicesToOffset(kv_BNSH_ ? "key_indices_t(batch, num_head_id, sequence_id - past_sequence_length, head_size_id)" : "key_indices_t(batch, sequence_id - past_sequence_length, num_head_id, head_size_id)") << ";\n"
                                << "  " << present_key.SetByOffset("present_offset", "key[offset]") << ";\n"
                                << "  " << present_value.SetByOffset("present_offset", "value[offset]") << ";\n"
                                << "}";
    }
  } else {
    shader.MainFunctionBody() << "  let present_offset = " << (past_present_share_buffer_ ? present_key.IndicesToOffset("output_indices") : "global_idx") << ";\n"
                              << "let offset = " << key.IndicesToOffset(kv_BNSH_ ? "key_indices_t(batch, num_head_id, sequence_id, head_size_id)" : "key_indices_t(batch, sequence_id, num_head_id, head_size_id)") << ";\n"
                              << present_key.SetByOffset("present_offset", "key[offset]") << ";\n"
                              << present_value.SetByOffset("present_offset", "value[offset]") << ";\n";
  }
  return Status::OK();
}

Status CopyKVCache(onnxruntime::webgpu::ComputeContext& context, const WebgpuAttentionParameters& parameters,
                   const Tensor* K, const Tensor* past_key, Tensor* present_key,
                   const Tensor* V, const Tensor* past_value, Tensor* present_value) {
  // CopyKVCache takes past key/value and current key/value and copies them to present key and value.
  // This makes it so that FlashAttention only needs to look at present key and value, and saves
  // number of input buffers in the shader, which we run out of (<=8) without this optimization.
  const int components = parameters.head_size_ % 4 == 0 ? 4 : (parameters.head_size_ % 2 == 0 ? 2 : 1);
  bool has_past = (parameters.total_sequence_length_ - parameters.kv_sequence_length_) > 0;
  // parameters.total_sequence_length_ is past_sequence_length + kv_sequence_length.
  // parameters.kv_num_heads_ may be smaller than parameters.num_heads_ when parameters.is_gqa_ is true.
  int num_heads = parameters.is_gqa_ ? parameters.kv_num_heads_ : parameters.num_heads_;
  // Only copy the new kv data for static kv cache
  int copy_sequence_length = has_past && parameters.past_present_share_buffer_ ? parameters.kv_sequence_length_ : parameters.total_sequence_length_;
  TensorShape copy_kv_shape{parameters.batch_size_, num_heads, copy_sequence_length, parameters.head_size_ / components};
  int64_t copy_size = copy_kv_shape.Size();
  CopyKVCacheProgram program{"CopyKVCache", has_past, parameters.qkv_format_ == Q_K_V_BSNH_BNSH_BNSH, parameters.past_present_share_buffer_};
  if (parameters.qkv_format_ == Q_K_V_BSNH_BNSH_BNSH) {
    program.AddInputs({{K, ProgramTensorMetadataDependency::TypeAndRank, components},
                       {V, ProgramTensorMetadataDependency::TypeAndRank, components}});
  } else {
    ORT_ENFORCE(parameters.qkv_format_ == Q_K_V_BSNH, "qkv format ", parameters.qkv_format_, " is not supported yet in CopyKVCache.");
    // Reshape (batch_size, kv_sequence_length, kv_hidden_size) to (batch_size, kv_sequence_length, num_head, head_size)
    TensorShape reshaped_KV_shape{parameters.batch_size_, parameters.kv_sequence_length_, num_heads, parameters.head_size_ / components};
    program.AddInputs({{K, ProgramTensorMetadataDependency::TypeAndRank, reshaped_KV_shape, components},
                       {V, ProgramTensorMetadataDependency::TypeAndRank, reshaped_KV_shape, components}});
  }
  if (has_past && !parameters.past_present_share_buffer_) {
    program.AddInputs({{past_key, ProgramTensorMetadataDependency::TypeAndRank, components},
                       {past_value, ProgramTensorMetadataDependency::TypeAndRank, components}});
  }
  program.AddOutputs({{present_key, ProgramTensorMetadataDependency::Rank, components},
                      {present_value, ProgramTensorMetadataDependency::Rank, components}})
      .AddIndices(std::move(copy_kv_shape));
  program.SetDispatchGroupSize(static_cast<uint32_t>((copy_size + 63) / 64))
      .SetWorkgroupSize(64)
      .CacheHint(has_past, parameters.qkv_format_, parameters.past_present_share_buffer_)
      .AddUniformVariables({{static_cast<uint32_t>(copy_size)},
                            // Note that when parameters.past_present_share_buffer_ is true, parameters.past_sequence_length_ will become to
                            // max_sequence_length. To get a valid past_sequence_length, we use total_sequence_length - kv_sequence_length.
                            {static_cast<uint32_t>(parameters.total_sequence_length_ - parameters.kv_sequence_length_)}});

  return context.RunProgram(program);
}

Status FlashAttentionProgram::GenerateShaderCode(ShaderHelper& shader) const {
  // Expectations are
  //    qkv have same number of heads and hidden dimension (head size).
  //    qkv are in BSNH format.
  //            B - batch size but shader only supports batch_size 1.
  //            S - current sequence length but shader supports only S = 1.
  //            N - number of heads.
  //            H - head size or hidden dimension for each qkv head.
  //  KV cache is stored as BN(total_sequence_length)H
  //  Attention bias is in BN(new_sequence_length)(total_sequence_length)
  //
  //  Expectation is that present_key, and present_value contain past key and values since
  //  we are out of storage buffers a shader can have and both past/present cant be passed.
  // The hidden size of each q head should be a multiple of 4 because shader uses vectorized loads.
  shader.AddInput("q", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  shader.AddInput("present_key", ShaderUsage::UseUniform);
  shader.AddInput("present_value", ShaderUsage::UseUniform);
  if (has_attention_bias_) {
    shader.AddInput("attention_bias", ShaderUsage::UseUniform);
  }
  shader.AddOutput("output", ShaderUsage::UseUniform);

  shader.AdditionalImplementation() << "const qkv_head_size: u32 = " << qkv_head_size_ << ";\n"
                                    << "const num_heads: u32 =" << qkv_num_heads_ << ";\n";

  shader.AdditionalImplementation() << R"HELPER_FN(
  // For max performance max_k_step should be the same as sg_size, however we might run out of registers
  // for qk_1, qk_2 .. qk_(sg_size). So we cap it at max_k_step (16).
  const max_k_step: u32 = 16u;
  const vec_factor: u32 = 4u;
  const qkv_head_size_vec: u32 = qkv_head_size / vec_factor;
  const min_value = f32(-3.402823e+38f);;

  // Default SHM usage limit is 16KB in Dawn.
  // vec4<f16> * qkv_head_size_vec * max_k_step = 8 * (128/4) * 16 = 4KB. 128 is head_size for phi4.
  var<workgroup> k_tile : array<array<q_value_t, qkv_head_size_vec>, max_k_step>;
  var<workgroup> v_tile : array<array<q_value_t, qkv_head_size_vec>, max_k_step>;

  // Private memory per lane.
  var<private> q_tile : array<q_value_t, qkv_head_size_vec>;
  fn loadq(q_idx_global : u32, head_idx: u32)
  {
      // Stored as float16[batch_size,sequence_length,3072] the inputs as per onnx MHA
      // This is the layout if TransferBSDToBNSH has not been run.
      let offset = q_idx_global * (qkv_head_size_vec) * num_heads + qkv_head_size_vec * head_idx;
      // Stored as BNSH - which is what webgpu uses after TransferBSDToBNSH has been run.
      //let offset = head_idx * uniforms.new_sequence_length * qkv_head_size_vec + q_idx_global * qkv_head_size_vec;
      for (var idx:u32 = 0; idx < qkv_head_size_vec; idx++)
      {
          q_tile[idx] = q[idx+offset];
      }
  }
  fn loadk(k_start : u32, head_idx: u32, local_idx: u32, k_step: u32)
  {
      // Stored as float16[batch_size,num_heads,present_sequence_length,96]
      let offset = head_idx * uniforms.present_sequence_length * qkv_head_size_vec + k_start * qkv_head_size_vec;
      for (var idx:u32 = local_idx; idx < qkv_head_size_vec*k_step; idx+=workgroup_size_x)
      {
          let slot = u32(idx/qkv_head_size_vec);
          let val = select(q_value_t(0), present_key[offset+idx], k_start + slot < uniforms.total_sequence_length);
          k_tile[slot][idx%qkv_head_size_vec] = val;
      }
  }
  fn loadv(v_start : u32, head_idx: u32, local_idx: u32, k_step: u32)
  {
      // Stored as float16[batch_size,num_heads,present_sequence_length,96]
      let offset = head_idx * uniforms.present_sequence_length * qkv_head_size_vec + v_start * qkv_head_size_vec;
      for (var idx:u32 = local_idx; idx < qkv_head_size_vec*k_step; idx+=workgroup_size_x)
      {
          let slot = u32(idx/qkv_head_size_vec);
          let val  = select(q_value_t(0), present_value[offset+idx], v_start + slot < uniforms.total_sequence_length);
          v_tile[slot][idx%qkv_head_size_vec] = val;
      }
  }
)HELPER_FN";

  if (is_qualcomm_) {
    shader.AdditionalImplementation() << R"HELPER_FN(
  const half_qkv_head_size_vec = qkv_head_size_vec / 2u;

  // Move half of o_tile from private memory into workgroup memory to reduce register pressure.
  // Note that register spill was observed on Qualcomm if whole o_tile is on private memory.
  // vec4<f16> * half_qkv_head_size_vec * workgroup_size_x = 8 * (128/4/2) * 64 = 8KB.
  var<workgroup> o_tile_r : array<array<vec4<f32>, half_qkv_head_size_vec>, workgroup_size_x>;

  // Private memory per lane.
  var<private> o_tile : array<vec4<f32>, half_qkv_head_size_vec>;
  fn writeo(o_idx_global: u32, head_idx: u32, local_idx: u32)
  {
      // Stored as float16[batch_size,sequence_length,3072]
      let offset = o_idx_global * num_heads * qkv_head_size_vec + head_idx * qkv_head_size_vec;
      for (var idx:u32 = 0; idx < half_qkv_head_size_vec; idx ++)
      {
          output[offset+idx] = q_value_t(o_tile[idx]);
          output[offset+idx+half_qkv_head_size_vec] = q_value_t(o_tile_r[local_idx][idx]);
      }
  }
    )HELPER_FN";
  } else {
    shader.AdditionalImplementation() << R"HELPER_FN(
  // Private memory per lane.
  var<private> o_tile : array<vec4<f32>, qkv_head_size_vec>;
  fn writeo(o_idx_global: u32, head_idx: u32)
  {
      // Stored as float16[batch_size,sequence_length,3072]
      let offset = o_idx_global * num_heads * qkv_head_size_vec + head_idx * qkv_head_size_vec;
      for (var idx:u32 = 0; idx < qkv_head_size_vec; idx ++)
      {
          output[offset+idx] = q_value_t(o_tile[idx]);
      }
  }
    )HELPER_FN";
  }

  if (has_attention_bias_) {
    shader.AdditionalImplementation() << R"HELPER_FN(
      fn loadAttentionBias(q_idx_global : u32, k_idx_global : u32, head_idx: u32) -> vec4<q_element_t>
      {
          // Stored as float16[batch_size,num_heads,new_seq_length,total_sequence_length]
          if (q_idx_global >= uniforms.new_sequence_length  || k_idx_global >= uniforms.total_sequence_length) {
              return vec4<q_element_t>(0);
          }
          let offset_base = head_idx * uniforms.new_sequence_length * uniforms.total_sequence_length + q_idx_global * uniforms.total_sequence_length;
          let offset = offset_base + k_idx_global;
          let offset_max = offset_base + uniforms.total_sequence_length;
          let c1 = q_element_t(attention_bias[min(offset, offset_max)]);
          let c2 = q_element_t(attention_bias[min(offset+1, offset_max)]);
          let c3 = q_element_t(attention_bias[min(offset+2, offset_max)]);
          let c4 = q_element_t(attention_bias[min(offset+3, offset_max)]);
          return vec4<q_element_t>(c1,c2,c3,c4);
      }
    )HELPER_FN";
  } else {
    shader.AdditionalImplementation() << R"HELPER_FN(
      fn loadAttentionBias(q_idx_global : u32, k_idx_global : u32, head_idx: u32) -> vec4<q_element_t>
      {
        return vec4<q_element_t>(0);
      }
    )HELPER_FN";
  }

  // Shader is designed to be dispatched as Dispatch(num_heads, new_sequence_length / workgroup_size_x, 1)
  // Each lane/thread is responsible for a single q.
  shader.MainFunctionBody() << R"MAIN_FN(
  let head_idx = u32(workgroup_idx / uniforms.num_seq_tile);
  let capped_sg_id = min(sg_id, max_k_step - 1u);
  let capped_sg_size = min(sg_size, max_k_step);

  // Load Q
  let q_idx_global = (workgroup_idx % uniforms.num_seq_tile) * workgroup_size_x + local_idx;
  let valid_q = q_idx_global < uniforms.new_sequence_length;
  if (valid_q)
  {
    loadq(q_idx_global, head_idx);
  }

  var previous_max : f32 = min_value;
  var previous_denom : f32 = 0;

  for(var k_start = 0u; k_start < uniforms.total_sequence_length; k_start+=capped_sg_size)
  {
    workgroupBarrier();
    loadk(k_start, head_idx / uniforms.n_reps, local_idx, capped_sg_size);
    loadv(k_start, head_idx / uniforms.n_reps, local_idx, capped_sg_size);
    workgroupBarrier();

    // Compute QKt
    var qk_1:vec4<f32>;
    var qk_2:vec4<f32>;
    var qk_3:vec4<f32>;
    var qk_4:vec4<f32>;
    if (sg_size > 8)
    {
      for (var i:u32 = 0u; i < qkv_head_size_vec; i++)
      {
        var k_local = vec4<f32>(k_tile[capped_sg_id][i]);
        var q_own = vec4<f32>(q_tile[i]);
        qk_1[0] += dot(q_own, subgroupShuffle(k_local, 0));
        qk_1[1] += dot(q_own, subgroupShuffle(k_local, 1));
        qk_1[2] += dot(q_own, subgroupShuffle(k_local, 2));
        qk_1[3] += dot(q_own, subgroupShuffle(k_local, 3));
        qk_2[0] += dot(q_own, subgroupShuffle(k_local, 4));
        qk_2[1] += dot(q_own, subgroupShuffle(k_local, 5));
        qk_2[2] += dot(q_own, subgroupShuffle(k_local, 6));
        qk_2[3] += dot(q_own, subgroupShuffle(k_local, 7));
        qk_3[0] += dot(q_own, subgroupShuffle(k_local, 8));
        qk_3[1] += dot(q_own, subgroupShuffle(k_local, 9));
        qk_3[2] += dot(q_own, subgroupShuffle(k_local, 10));
        qk_3[3] += dot(q_own, subgroupShuffle(k_local, 11));
        qk_4[0] += dot(q_own, subgroupShuffle(k_local, 12));
        qk_4[1] += dot(q_own, subgroupShuffle(k_local, 13));
        qk_4[2] += dot(q_own, subgroupShuffle(k_local, 14));
        qk_4[3] += dot(q_own, subgroupShuffle(k_local, 15));
      }
    }
    else
    {
      for (var i:u32 = 0u; i < qkv_head_size_vec; i++)
      {
        var k_local = vec4<f32>(k_tile[capped_sg_id][i]);
        var q_own = vec4<f32>(q_tile[i]);
        qk_1[0] += dot(q_own, subgroupShuffle(k_local, 0));
        qk_1[1] += dot(q_own, subgroupShuffle(k_local, 1));
        qk_1[2] += dot(q_own, subgroupShuffle(k_local, 2));
        qk_1[3] += dot(q_own, subgroupShuffle(k_local, 3));
        qk_2[0] += dot(q_own, subgroupShuffle(k_local, 4));
        qk_2[1] += dot(q_own, subgroupShuffle(k_local, 5));
        qk_2[2] += dot(q_own, subgroupShuffle(k_local, 6));
        qk_2[3] += dot(q_own, subgroupShuffle(k_local, 7));
      }
    }

    qk_1 = qk_1 * uniforms.alpha + vec4<f32>(loadAttentionBias(q_idx_global, k_start, head_idx));
    qk_2 = qk_2 * uniforms.alpha + vec4<f32>(loadAttentionBias(q_idx_global, k_start+4, head_idx));
    if (sg_size > 8)
    {
      qk_3 = qk_3 * uniforms.alpha + vec4<f32>(loadAttentionBias(q_idx_global, k_start+8, head_idx));
      qk_4 = qk_4 * uniforms.alpha + vec4<f32>(loadAttentionBias(q_idx_global, k_start+12, head_idx));
    }

    let seq_causal_length = select(uniforms.total_sequence_length, uniforms.past_sequence_length + q_idx_global + 1, uniforms.is_gqa > 0);
    // Neuter qk values where K is out of bounds.
    qk_1[0] = select(min_value, qk_1[0], k_start+0 < seq_causal_length);
    qk_1[1] = select(min_value, qk_1[1], k_start+1 < seq_causal_length);
    qk_1[2] = select(min_value, qk_1[2], k_start+2 < seq_causal_length);
    qk_1[3] = select(min_value, qk_1[3], k_start+3 < seq_causal_length);
    qk_2[0] = select(min_value, qk_2[0], k_start+4 < seq_causal_length);
    qk_2[1] = select(min_value, qk_2[1], k_start+5 < seq_causal_length);
    qk_2[2] = select(min_value, qk_2[2], k_start+6 < seq_causal_length);
    qk_2[3] = select(min_value, qk_2[3], k_start+7 < seq_causal_length);
    if (sg_size > 8)
    {
      qk_3[0] = select(min_value, qk_3[0], k_start+8 < seq_causal_length);
      qk_3[1] = select(min_value, qk_3[1], k_start+9 < seq_causal_length);
      qk_3[2] = select(min_value, qk_3[2], k_start+10 < seq_causal_length);
      qk_3[3] = select(min_value, qk_3[3], k_start+11 < seq_causal_length);
      qk_4[0] = select(min_value, qk_4[0], k_start+12 < seq_causal_length);
      qk_4[1] = select(min_value, qk_4[1], k_start+13 < seq_causal_length);
      qk_4[2] = select(min_value, qk_4[2], k_start+14 < seq_causal_length);
      qk_4[3] = select(min_value, qk_4[3], k_start+15 < seq_causal_length);
    }
)MAIN_FN";
  //
  // Compute SoftMax as per Flash Attention technique.
  //
  // Crux of Flash Attention is here, that allows for partial softmax computation,
  // direct update of output and merging with previous results.
  // https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf
  // Where b is the block size of the tile. Xi is storing QKtranspose for the ith tile.
  // mi_local is the max of Xi. Note: _ in this notation means what follows is a
  // subscript. max_j=1:b (Xi[j]) is the max of Xi[j] for j=1 to b.
  //
  // for i = 1, #tiles do
  //  Xi = Q[k,:] Kt[:, (i-1) b : i b]
  //  mi_local= max_j=1:b (Xi[j])
  //  Mi = max(M_(i-1), mi_local)
  //  d'_i = d'_(i-1) * e^(M_(i-1)-M_i) + Σ_j=1:b e^(Xi[j]-Mi)
  //  o'_i = o'_(i-1) * d'_(i-1) * e^(M_(i-1)-M_i) / d'_i + Σ_j=1:b (e^(Xi[j]-Mi) / d'_i) V[j + (i - 1)b,:]
  // end
  //
  // In the code below:
  // dleft is the first term of d'_i expression above : d'_(i-1) * e^(M_(i-1)-M_i).
  // sum is the second term of the same expression    : Σ_j=1:b e^(Xi[j]-Mi)
  // o_ratio is the part of the first term of o'_i expression above : d'_(i-1) * e^(M_(i-1)-M_i) / d'_i
  //
  shader.MainFunctionBody() << R"MAIN_FN(
    var local_max_temp = max(qk_1, qk_2);
    if (sg_size > 8)
    {
      local_max_temp = max(local_max_temp, qk_3);
      local_max_temp = max(local_max_temp, qk_4);
    }
    let local_max = max(max(local_max_temp.x, local_max_temp.y),max(local_max_temp.z, local_max_temp.w));
    let new_max = max(previous_max, local_max);
    qk_1 = exp(qk_1 - new_max);
    qk_2 = exp(qk_2 - new_max);
    if (sg_size > 8) {
      qk_3 = exp(qk_3 - new_max);
      qk_4 = exp(qk_4 - new_max);
    }
    let sum_vec = qk_1 + qk_2 + qk_3 + qk_4;
    let sum = sum_vec.x + sum_vec.y + sum_vec.z + sum_vec.w;

    // Compute lhs term of update di prime and the compute di prime.
    let dleft = previous_denom * exp(previous_max-new_max);
    var d = dleft + sum;
    d = select(d,f32(0.0000001),d==0);
    qk_1 = qk_1 / d;
    qk_2 = qk_2 / d;
    if (sg_size > 8) {
      qk_3 = qk_3 / d;
      qk_4 = qk_4 / d;
    }
    previous_max = new_max;
    previous_denom = d;
    let o_ratio = dleft / d;

)MAIN_FN";

  if (is_qualcomm_) {
    shader.MainFunctionBody() << R"MAIN_FN(
    if (sg_size > 8) {
      for (var i:u32 = 0; i < half_qkv_head_size_vec; i++)
      {
          var val = vec4<f32>(v_tile[capped_sg_id][i]);
          var sum = subgroupShuffle(val, 0) * qk_1[0];
          sum += subgroupShuffle(val, 1) * qk_1[1];
          sum += subgroupShuffle(val, 2) * qk_1[2];
          sum += subgroupShuffle(val, 3) * qk_1[3];
          sum += subgroupShuffle(val, 4) * qk_2[0];
          sum += subgroupShuffle(val, 5) * qk_2[1];
          sum += subgroupShuffle(val, 6) * qk_2[2];
          sum += subgroupShuffle(val, 7) * qk_2[3];
          sum += subgroupShuffle(val, 8) * qk_3[0];
          sum += subgroupShuffle(val, 9) * qk_3[1];
          sum += subgroupShuffle(val, 10) * qk_3[2];
          sum += subgroupShuffle(val, 11) * qk_3[3];
          sum += subgroupShuffle(val, 12) * qk_4[0];
          sum += subgroupShuffle(val, 13) * qk_4[1];
          sum += subgroupShuffle(val, 14) * qk_4[2];
          sum += subgroupShuffle(val, 15) * qk_4[3];
          o_tile[i] = o_tile[i] * o_ratio + sum;

          val = vec4<f32>(v_tile[capped_sg_id][half_qkv_head_size_vec + i]);
          sum = subgroupShuffle(val, 0) * qk_1[0];
          sum += subgroupShuffle(val, 1) * qk_1[1];
          sum += subgroupShuffle(val, 2) * qk_1[2];
          sum += subgroupShuffle(val, 3) * qk_1[3];
          sum += subgroupShuffle(val, 4) * qk_2[0];
          sum += subgroupShuffle(val, 5) * qk_2[1];
          sum += subgroupShuffle(val, 6) * qk_2[2];
          sum += subgroupShuffle(val, 7) * qk_2[3];
          sum += subgroupShuffle(val, 8) * qk_3[0];
          sum += subgroupShuffle(val, 9) * qk_3[1];
          sum += subgroupShuffle(val, 10) * qk_3[2];
          sum += subgroupShuffle(val, 11) * qk_3[3];
          sum += subgroupShuffle(val, 12) * qk_4[0];
          sum += subgroupShuffle(val, 13) * qk_4[1];
          sum += subgroupShuffle(val, 14) * qk_4[2];
          sum += subgroupShuffle(val, 15) * qk_4[3];
          o_tile_r[local_idx][i] = o_tile_r[local_idx][i] * o_ratio + sum;
      }
    }
    else
    {
      for (var i:u32 = 0; i < half_qkv_head_size_vec; i++)
      {
          var val = vec4<f32>(v_tile[capped_sg_id][i]);
          var sum = subgroupShuffle(val, 0) * qk_1[0];
          sum += subgroupShuffle(val, 1) * qk_1[1];
          sum += subgroupShuffle(val, 2) * qk_1[2];
          sum += subgroupShuffle(val, 3) * qk_1[3];
          sum += subgroupShuffle(val, 4) * qk_2[0];
          sum += subgroupShuffle(val, 5) * qk_2[1];
          sum += subgroupShuffle(val, 6) * qk_2[2];
          sum += subgroupShuffle(val, 7) * qk_2[3];
          o_tile[i] = o_tile[i] * o_ratio + sum;

          val = vec4<f32>(v_tile[capped_sg_id][half_qkv_head_size_vec + i]);
          sum = subgroupShuffle(val, 0) * qk_1[0];
          sum += subgroupShuffle(val, 1) * qk_1[1];
          sum += subgroupShuffle(val, 2) * qk_1[2];
          sum += subgroupShuffle(val, 3) * qk_1[3];
          sum += subgroupShuffle(val, 4) * qk_2[0];
          sum += subgroupShuffle(val, 5) * qk_2[1];
          sum += subgroupShuffle(val, 6) * qk_2[2];
          sum += subgroupShuffle(val, 7) * qk_2[3];
          o_tile_r[local_idx][i] = o_tile_r[local_idx][i] * o_ratio + sum;
      }
    }
  }

  if (valid_q) {
    writeo(q_idx_global, head_idx, local_idx);
  }
)MAIN_FN";
  } else {
    shader.MainFunctionBody() << R"MAIN_FN(
    if (sg_size > 8) {
      for (var i:u32 = 0; i < qkv_head_size_vec; i++)
      {
          var val = vec4<f32>(v_tile[capped_sg_id][i]);
          var sum = subgroupShuffle(val, 0) * qk_1[0];
          sum += subgroupShuffle(val, 1) * qk_1[1];
          sum += subgroupShuffle(val, 2) * qk_1[2];
          sum += subgroupShuffle(val, 3) * qk_1[3];
          sum += subgroupShuffle(val, 4) * qk_2[0];
          sum += subgroupShuffle(val, 5) * qk_2[1];
          sum += subgroupShuffle(val, 6) * qk_2[2];
          sum += subgroupShuffle(val, 7) * qk_2[3];
          sum += subgroupShuffle(val, 8) * qk_3[0];
          sum += subgroupShuffle(val, 9) * qk_3[1];
          sum += subgroupShuffle(val, 10) * qk_3[2];
          sum += subgroupShuffle(val, 11) * qk_3[3];
          sum += subgroupShuffle(val, 12) * qk_4[0];
          sum += subgroupShuffle(val, 13) * qk_4[1];
          sum += subgroupShuffle(val, 14) * qk_4[2];
          sum += subgroupShuffle(val, 15) * qk_4[3];
          o_tile[i] = o_tile[i] * o_ratio + sum;
      }
    }
    else
    {
      for (var i:u32 = 0; i < qkv_head_size_vec; i++)
      {
          var val = vec4<f32>(v_tile[capped_sg_id][i]);
          var sum = subgroupShuffle(val, 0) * qk_1[0];
          sum += subgroupShuffle(val, 1) * qk_1[1];
          sum += subgroupShuffle(val, 2) * qk_1[2];
          sum += subgroupShuffle(val, 3) * qk_1[3];
          sum += subgroupShuffle(val, 4) * qk_2[0];
          sum += subgroupShuffle(val, 5) * qk_2[1];
          sum += subgroupShuffle(val, 6) * qk_2[2];
          sum += subgroupShuffle(val, 7) * qk_2[3];
          o_tile[i] = o_tile[i] * o_ratio + sum;
      }
    }
  }

  if (valid_q) {
    writeo(q_idx_global, head_idx);
  }
)MAIN_FN";
  }

  return Status::OK();
}

Status FlashAttentionDecodeQKTProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("q", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  shader.AddInput("present_key", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);
  if (has_attention_bias_) {
    shader.AddInput("attention_bias", ShaderUsage::UseUniform);
  }
  shader.AddOutput("output", ShaderUsage::UseUniform);
  shader.AddOutput("metadata", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);
  // Note that this shader adopts similar algorithm with dp4a generation shader.
  //
  // This algorithm works to compute dot product of keys with queries parallelly, by processing on the k (head_size) dimension at each step amongst tile_size_k_vec threads,
  // and utilizing the remaining threads in the workgroup to process additional rows of |present_key| in parallel (such that the values in shared memory (tile_q) for |q| can be reused).
  // For each load of q, the tile_size_k_vec threads also reload |present_key| tile_size/sub_tile_count times to compute partial dot products of other |present_key| rows
  // in order to complete all tile_size |present_key| rows in this workgroup and also reusing the loaded in register values of |q|.
  constexpr int tile_size_k_vec = 8;

  // 1. Each workgroup processes one row of |q| and tile_size rows of |present_key|
  //
  // 2. Computation Process:
  //    - Reads [tile_size][tile_size_k_vec] block of |present_key| data at a time
  //    - Each thread within workgroup computes dot products of 4 A*B elements since each k represents 4 elements of |present_key|
  //    - Stores intermediate results in shared memory (inner_qk_values)
  //    - Iterates through columns (head_size_vec) accumulating results in inner_qk_values
  //    - Performs final reduction sum in inner_qk_values for output
  shader.AdditionalImplementation() << "const tile_size = " << tile_size_ << "u;\n"
                                    << "const tile_size_k_vec = " << tile_size_k_vec << "u;\n"
                                    << "const sub_tile_count = " << WorkgroupSizeX() / tile_size_k_vec << "u;\n";
  shader.AdditionalImplementation() << R"ADDNL_FN(
var<workgroup> tile_q: array<q_value_t, tile_size_k_vec>;
var<workgroup> inner_qk_values: array<array<f32, tile_size_k_vec>, tile_size>;
var<workgroup> tile_qk: array<f32, tile_size>;
)ADDNL_FN";

  if (has_attention_bias_) {
    shader.AdditionalImplementation() << R"HELPER_FN(
      fn loadAttentionBias(idx: u32) -> q_element_t
      {
        return attention_bias[idx];
      }
    )HELPER_FN";
  } else {
    shader.AdditionalImplementation() << R"HELPER_FN(
      fn loadAttentionBias(idx: u32) -> q_element_t
      {
        return q_element_t(0);
      }
    )HELPER_FN";
  }

  shader.MainFunctionBody() << R"MAIN_FN(
    let local_row = u32(local_idx / tile_size_k_vec);
    let local_col = local_idx % tile_size_k_vec;
    let total_seq_offset = (workgroup_idx % uniforms.num_total_seq_length_tile) * tile_size;
    let head_idx = u32(workgroup_idx / uniforms.num_total_seq_length_tile);
    let q_offset = head_idx * uniforms.head_size_vec;
    var total_sequence_length = uniforms.total_sequence_length;
    let present_offset = u32(head_idx / uniforms.n_reps) * uniforms.present_sequence_length * uniforms.head_size_vec;
    for (var k: u32 = 0u; k < uniforms.head_size_vec; k += tile_size_k_vec) {
      if (local_idx < tile_size_k_vec && k + local_idx < uniforms.head_size_vec) {
        tile_q[local_idx] = q[q_offset + k + local_idx];
      }
      workgroupBarrier();
      let q_data = vec4<f32>(tile_q[local_col]);
      if (k + local_col < uniforms.head_size_vec) {
        for (var row_offset = 0u; row_offset < tile_size; row_offset += sub_tile_count) {
          if (total_seq_offset + row_offset + local_row < total_sequence_length) {
            inner_qk_values[row_offset + local_row][local_col] += dot(vec4<f32>(present_key[present_offset + (total_seq_offset + row_offset + local_row) * uniforms.head_size_vec + k + local_col]), q_data);
          }
        }
      }
      workgroupBarrier();
    }

    if (local_idx < tile_size && total_seq_offset + local_idx < total_sequence_length && head_idx < uniforms.num_heads) {
      var sum = f32(0);
      for (var i = 0u; i < tile_size_k_vec; i++) {
        sum += inner_qk_values[local_idx][i];
      }

      let output_idx = head_idx * total_sequence_length + total_seq_offset + local_idx;
      sum = sum * uniforms.alpha + f32(loadAttentionBias(output_idx));
      tile_qk[local_idx] = sum;
      output[output_idx] = sum;
    }
    workgroupBarrier();

    if (head_idx >= uniforms.num_heads) {
      return;
    }

    if (local_idx == 0u) {
      // Calculate the max and sum in current split.
      var l_max = f32(-3.402823e+38f);
      var l_sum = f32(0);
      for (var i = 0u; i < tile_size && (total_seq_offset + i) < total_sequence_length; i++) {
        l_max = max(l_max, f32(tile_qk[i]));
      }
      for (var i = 0u; i < tile_size && (total_seq_offset + i) < total_sequence_length; i++) {
        l_sum += exp(f32(tile_qk[i]) - l_max);
      }
      let meta_offset = head_idx * uniforms.num_total_seq_length_tile + workgroup_idx % uniforms.num_total_seq_length_tile;
      metadata[meta_offset] = metadata_value_t(l_max, l_sum);
    }
)MAIN_FN";

  return Status::OK();
}

Status ComputeFlashAttentionDecodeQKT(onnxruntime::webgpu::ComputeContext& context, const Tensor* Q,
                                      const Tensor* attention_bias, Tensor* output, Tensor* present_key, Tensor* metadata,
                                      const WebgpuAttentionParameters& parameters, uint32_t num_total_seq_length_tile, uint32_t tile_size) {
  const float alpha = parameters.scale_ == 0.0f ? 1.f / sqrt(static_cast<float>(parameters.head_size_))
                                                : parameters.scale_;

  const bool has_attention_bias = attention_bias != nullptr;
  const int components = 4;

  FlashAttentionDecodeQKTProgram program{"FlashAttentionDecodeQKT", has_attention_bias, tile_size};
  program.AddInputs({{Q, ProgramTensorMetadataDependency::TypeAndRank, components},
                     {present_key, ProgramTensorMetadataDependency::TypeAndRank, components}});
  if (has_attention_bias) {
    program.AddInput({attention_bias, ProgramTensorMetadataDependency::TypeAndRank});
  }
  program.AddOutputs({{output, ProgramTensorMetadataDependency::Rank},
                      {metadata, ProgramTensorMetadataDependency::Rank, 2}});

  const uint32_t vectorized_head_size = parameters.head_size_ / components;
  program.SetDispatchGroupSize(parameters.num_heads_ * num_total_seq_length_tile)
      .SetWorkgroupSize(64)
      .CacheHint(tile_size, has_attention_bias)
      .AddUniformVariables({{static_cast<uint32_t>(vectorized_head_size)},
                            {static_cast<uint32_t>(parameters.total_sequence_length_)},
                            {static_cast<float>(alpha)},
                            // present_sequence_length is used to index into the KV cache, for static kv cache it is the max sequence length.
                            {static_cast<uint32_t>(parameters.is_gqa_ ? parameters.seqlen_present_kv_cache_ : parameters.total_sequence_length_)},
                            {static_cast<uint32_t>(parameters.n_reps)},
                            {num_total_seq_length_tile},
                            {static_cast<uint32_t>(parameters.num_heads_)}});

  return context.RunProgram(program);
}

Status FlashAttentionDecodeSplitVxProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("metadata", ShaderUsage::UseUniform);
  shader.AddInput("qk", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);
  shader.AddInput("present_value", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  shader.AddOutput("out_split_vx", ShaderUsage::UseUniform);

  // Note that this shader adopts similar algorithm with dp4a generation shader.
  //
  // This algorithm works to compute dot product of v with qk parallelly, by processing on the head_size dimension at each step amongst tile_size_k_vec threads,
  // and utilizing the remaining threads in the workgroup to process additional rows of |present_value| in parallel (such that the values in shared memory (tile_qk) for |qk| can be reused).
  // The tile_size_k_vec threads also reload |present_value| tile_size/sub_tile_count times to compute partial dot products of other |present_value| rows
  // in order to complete all tile_size |present_value| rows in this workgroup and also reusing the values in tile_qk.
  //
  // The difference with FlashAttentionDecodeQKTProgram is that the dot products go through the rows (total_sequence_length) of |present_value| instead of columns (head_size_vec).
  // And each workgroup only calculate current tile_size's dot products instead of iterating the whole row |total_sequence_length|.
  // That's why this shader is a split shader. The final reduce will be done in FlashAttentionDecodeReduceProgram.
  constexpr int tile_size_k_vec = 8;

  shader.AdditionalImplementation() << "const head_size_vec = " << head_size_vec_ << "u;\n"
                                    << "const tile_size = " << tile_size_ << "u;\n"
                                    << "const tile_size_k_vec = " << tile_size_k_vec << "u;\n"
                                    << "const sub_tile_count = " << WorkgroupSizeX() / tile_size_k_vec << "u;\n";
  shader.AdditionalImplementation() << R"HELPER_FN(
var<workgroup> tile_qk: array<present_value_element_t, tile_size>;
var<workgroup> tile_output: array<present_value_value_t, head_size_vec>;
var<workgroup> qkv_values: array<array<present_value_value_t, tile_size_k_vec>, sub_tile_count>;

  )HELPER_FN";

  // TODO: Ideally, there should only be two shaders FlashAttentionDecodeSplitVx and FlashAttentionDecodeVxReduce, which can also reduce the intermediate memory.
  // The FlashAttentionDecodeQKT can be merged into split shader and do the final softmax adjustment in the reduce shader. However, some issues are met that when
  // the total sequence length exceeds some value, the result will become garbage. Since it can't be resolved in a short time, leave it as TODO to fix it in future.
  shader.MainFunctionBody() << R"MAIN_FN(
    let local_row = u32(local_idx / tile_size_k_vec);
    let local_col = local_idx % tile_size_k_vec;
    let total_seq_offset = (workgroup_idx % uniforms.num_total_seq_length_tile) * tile_size;
    let head_idx = u32(workgroup_idx / uniforms.num_total_seq_length_tile);
    var total_sequence_length = uniforms.total_sequence_length;
    let present_offset = u32(head_idx / uniforms.n_reps) * uniforms.head_size_vec * uniforms.present_sequence_length;

    // Calculate the global max and sum in qk.
    if (head_idx < uniforms.num_heads)
    {
      var g_max = f32(-3.402823e+38f);
      var g_sum = f32(0);
      for (var i = 0u; i < uniforms.num_total_seq_length_tile; i++)
      {
        let meta_offset = head_idx * uniforms.num_total_seq_length_tile + i;
        g_max = max(g_max, metadata[meta_offset].x);
      }
      for (var i = 0u; i < uniforms.num_total_seq_length_tile; i++)
      {
        let meta_offset = head_idx * uniforms.num_total_seq_length_tile + i;
        let m_value = metadata[meta_offset];
        g_sum += exp(m_value.x - g_max) * m_value.y;
      }

      if (total_seq_offset + local_idx < total_sequence_length) {
        tile_qk[local_idx] = present_value_element_t(exp(qk[head_idx * total_sequence_length + total_seq_offset + local_idx] - g_max) / g_sum);
      }
    }
    for (var k: u32 = 0u; k < uniforms.head_size_vec; k += tile_size_k_vec) {
      var value = present_value_value_t(0);
      qkv_values[local_row][local_col] = present_value_value_t(0);
      workgroupBarrier();

      if (k + local_col < uniforms.head_size_vec) {
        for (var row_offset = 0u; row_offset < tile_size; row_offset += sub_tile_count) {
          if (total_seq_offset + row_offset + local_row < total_sequence_length) {
            value += present_value[present_offset + (total_seq_offset + row_offset + local_row) * uniforms.head_size_vec + k + local_col] * tile_qk[row_offset + local_row];
          }
        }
      }

      qkv_values[local_row][local_col] = value;
      workgroupBarrier();

      if (local_idx < tile_size_k_vec) {
        for (var i = 0u; i < sub_tile_count; i++) {
          tile_output[k + local_idx] += qkv_values[i][local_idx];
        }
      }
      workgroupBarrier();
    }

    if (head_idx >= uniforms.num_heads) {
      return;
    }

    for (var i = local_idx; i < uniforms.head_size_vec; i += workgroup_size_x) {
      let out_offset = head_idx * uniforms.num_total_seq_length_tile * uniforms.head_size_vec + (workgroup_idx % uniforms.num_total_seq_length_tile) * uniforms.head_size_vec + i;
      out_split_vx[out_offset] = tile_output[i];
    }
)MAIN_FN";

  return Status::OK();
}

Status ComputeFlashAttentionDecodeSplitVxScore(onnxruntime::webgpu::ComputeContext& context,
                                               const Tensor* metadata,
                                               const Tensor* qk,
                                               Tensor* out_split_vx,
                                               Tensor* present_value,
                                               const WebgpuAttentionParameters& parameters,
                                               uint32_t num_total_seq_length_tile,
                                               uint32_t tile_size) {
  const int components = 4;
  int head_size_vec = parameters.v_head_size_ / components;
  FlashAttentionDecodeSplitVxProgram program{"FlashAttentionDecodeSplitVx", tile_size, head_size_vec};
  program.AddInputs({{metadata, ProgramTensorMetadataDependency::TypeAndRank, 2},
                     {qk, ProgramTensorMetadataDependency::TypeAndRank},
                     {present_value, ProgramTensorMetadataDependency::TypeAndRank, components}});
  program.AddOutputs({{out_split_vx, ProgramTensorMetadataDependency::TypeAndRank, components}});  // [B, N, split_k, head_size]
  program.SetDispatchGroupSize(parameters.num_heads_ * num_total_seq_length_tile)
      .CacheHint(tile_size, head_size_vec)
      .SetWorkgroupSize(64)
      .AddUniformVariables({{static_cast<uint32_t>(parameters.total_sequence_length_)},
                            {static_cast<uint32_t>(head_size_vec)},
                            {static_cast<uint32_t>(parameters.is_gqa_ ? parameters.seqlen_present_kv_cache_ : parameters.total_sequence_length_)},
                            {static_cast<uint32_t>(parameters.n_reps)},
                            num_total_seq_length_tile,
                            {static_cast<uint32_t>(parameters.num_heads_)}});

  return context.RunProgram(program);
}

Status FlashAttentionDecodeVxReduceProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("input", ShaderUsage::UseUniform);
  shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);

  // Inputs are splits of the GQA output, split into num_total_seq_length_tiles rows.
  // This shader needs to add these splits across the row dimension to arrive at the final result. The column is head size wide.
  // The reduction achieves maximum parallelization by splitting this task first into tile_size columns that each workgroup is responsible for.
  // Then within each workgroup the task of summation over the num_total_seq_length_tile for the tile_size columns is further split in two ways.
  // First across the row dimension to have WORKGROUP_SIZE/TILE_SIZE parallel computations of summation of TILE_SIZE rows.
  // Then across the column dimension where each thread is responsible for 1 column of the TILE_SIZE columns the workgroup is resposible for.
  shader.AdditionalImplementation() << "const TILE_SIZE = " << tile_size_ << ";\n";
  shader.AdditionalImplementation() << R"HELPER_FN(
var<workgroup> tile_input: array<array<output_value_t, TILE_SIZE>, TILE_SIZE>;
  )HELPER_FN";

  shader.MainFunctionBody() << R"MAIN_FN(
    let head_size_offset = (workgroup_idx % uniforms.num_head_size_tile) * TILE_SIZE;
    let head_idx = u32(workgroup_idx / uniforms.num_head_size_tile);
    let in_offset = head_idx * uniforms.num_total_seq_length_tile * uniforms.head_size_vec;
    var value = output_value_t(0);
    let local_row = u32(local_idx / TILE_SIZE);
    let local_col = local_idx % TILE_SIZE;

    if (head_size_offset + local_col < uniforms.head_size_vec) {
      for (var r = 0u; r < uniforms.num_total_seq_length_tile; r += TILE_SIZE) {
        if (r + local_row < uniforms.num_total_seq_length_tile) {
          value += input[in_offset + (r + local_row) * uniforms.head_size_vec + head_size_offset + local_col];
        }
      }
    }

    tile_input[local_row][local_col] = value;
    workgroupBarrier();

    if (head_idx >= uniforms.num_heads) {
      return;
    }

    if (local_idx < TILE_SIZE && head_size_offset + local_idx < uniforms.head_size_vec) {
      value = output_value_t(0);
      for (var i = 0u; i < TILE_SIZE; i++) {
        value += tile_input[i][local_idx];
      }
      let output_id = head_idx * uniforms.head_size_vec + head_size_offset + local_idx;
      output[output_id] = value;
    }
)MAIN_FN";

  return Status::OK();
}

Status ComputeFlashAttentionDecodeVxReduce(onnxruntime::webgpu::ComputeContext& context,
                                           const Tensor* out_split_vx,
                                           Tensor* output,
                                           const WebgpuAttentionParameters& parameters,
                                           uint32_t num_total_seq_length_tile) {
  const int components = 4;
  constexpr int tile_size = 8;
  int tile_head_size = tile_size * components;
  FlashAttentionDecodeVxReduceProgram program{"FlashAttentionDecodeVxReduce", tile_size};
  program.AddInputs({{out_split_vx, ProgramTensorMetadataDependency::TypeAndRank, components}});
  program.AddOutputs({{output, ProgramTensorMetadataDependency::TypeAndRank, components}});
  const uint32_t num_head_size_tile = static_cast<uint32_t>((parameters.v_head_size_ + tile_head_size - 1) / tile_head_size);
  program.SetDispatchGroupSize(parameters.num_heads_ * num_head_size_tile)
      .CacheHint(tile_size)
      .SetWorkgroupSize(tile_size * tile_size)
      .AddUniformVariables({{static_cast<uint32_t>(parameters.v_head_size_ / components)},
                            num_total_seq_length_tile,
                            {num_head_size_tile},
                            {static_cast<uint32_t>(parameters.num_heads_)}});

  return context.RunProgram(program);
}

Status ApplyFlashAttention(const Tensor* Q, const Tensor* K, const Tensor* V, const Tensor* attention_bias,
                           Tensor* output, const Tensor* past_key, Tensor* present_key, const Tensor* past_value, Tensor* present_value,
                           const WebgpuAttentionParameters& parameters, onnxruntime::webgpu::ComputeContext& context) {
  ORT_RETURN_IF_ERROR(CopyKVCache(context, parameters, K, past_key, present_key, V, past_value, present_value));

  if (parameters.sequence_length_ > 1) {
    const uint32_t tile_size = 64;
    bool has_attention_bias = attention_bias != nullptr;
    bool is_qualcomm = context.AdapterInfo().vendor == std::string_view{"qualcomm"};
    FlashAttentionProgram program{"FlashAttention", has_attention_bias, is_qualcomm, parameters.head_size_, parameters.num_heads_};
    program.AddInputs({{Q, ProgramTensorMetadataDependency::TypeAndRank, 4},
                       {present_key, ProgramTensorMetadataDependency::TypeAndRank, 4},
                       {present_value, ProgramTensorMetadataDependency::TypeAndRank, 4}});
    if (has_attention_bias) {
      program.AddInputs({{attention_bias, ProgramTensorMetadataDependency::TypeAndRank}});
    }
    program.AddOutputs({{output, ProgramTensorMetadataDependency::TypeAndRank, 4}});
    const float alpha = parameters.scale_ == 0.0f ? 1.f / sqrt(static_cast<float>(parameters.head_size_))
                                                  : parameters.scale_;
    const uint32_t num_seq_tile = (parameters.sequence_length_ + tile_size - 1) / tile_size;
    program.SetDispatchGroupSize(parameters.num_heads_ * num_seq_tile)
        .SetWorkgroupSize(tile_size)
        .CacheHint(has_attention_bias, parameters.head_size_, parameters.num_heads_, is_qualcomm)
        .AddUniformVariables({{static_cast<uint32_t>(parameters.sequence_length_)},
                              {static_cast<uint32_t>(parameters.total_sequence_length_)},
                              {static_cast<uint32_t>(parameters.past_present_share_buffer_ ? parameters.past_sequence_length_ : parameters.total_sequence_length_)},
                              {static_cast<uint32_t>(parameters.total_sequence_length_ - parameters.kv_sequence_length_)},
                              {static_cast<uint32_t>(parameters.is_gqa_ ? 1 : 0)},
                              {static_cast<uint32_t>(parameters.n_reps)},
                              {alpha},
                              {num_seq_tile}});

    return context.RunProgram(program);
  }

  const TensorShapeVector qk_dims({parameters.batch_size_, parameters.num_heads_,
                                   parameters.sequence_length_, parameters.total_sequence_length_});
  const TensorShape qk_shape(qk_dims);
  Tensor qk = context.CreateGPUTensor(DataTypeImpl::GetType<float>(), qk_shape);
  constexpr uint32_t tile_size = 64;
  const uint32_t num_total_seq_length_tile = (parameters.total_sequence_length_ + tile_size - 1) / tile_size;
  // The metadata is used to store the max and sum of each tile.
  const TensorShapeVector metadata_dims({parameters.batch_size_, parameters.num_heads_,
                                         num_total_seq_length_tile, 2});
  const TensorShape metadata_shape(metadata_dims);
  Tensor metadata = context.CreateGPUTensor(DataTypeImpl::GetType<float>(), metadata_shape);
  ORT_RETURN_IF_ERROR(ComputeFlashAttentionDecodeQKT(context, Q, attention_bias, &qk, present_key, &metadata,
                                                     parameters, num_total_seq_length_tile, tile_size));

  const TensorShapeVector out_split_vx_dims({parameters.batch_size_, parameters.num_heads_, num_total_seq_length_tile, parameters.head_size_});
  const TensorShape out_split_vx_shape(out_split_vx_dims);
  Tensor out_split_vx = context.CreateGPUTensor(Q->DataType(), out_split_vx_shape);
  ORT_RETURN_IF_ERROR(ComputeFlashAttentionDecodeSplitVxScore(context, &metadata, &qk, &out_split_vx, present_value, parameters, num_total_seq_length_tile, tile_size));
  ORT_RETURN_IF_ERROR(ComputeFlashAttentionDecodeVxReduce(context, &out_split_vx, output, parameters, num_total_seq_length_tile));

  return Status::OK();
}

bool CanApplyFlashAttention(const Tensor* bias, const Tensor* present_key, const Tensor* present_value,
                            const WebgpuAttentionParameters& parameters, onnxruntime::webgpu::ComputeContext& context) {
  return parameters.batch_size_ == 1 &&
         !parameters.is_packed_qkv_ &&
         parameters.head_size_ == parameters.v_head_size_ &&
         bias == nullptr &&
         context.HasFeature(wgpu::FeatureName::Subgroups) &&
         present_key != nullptr && present_value != nullptr && present_key->SizeInBytes() > 0 &&
         present_value->SizeInBytes() > 0 &&
         ((context.AdapterInfo().vendor == std::string_view{"qualcomm"} && parameters.head_size_ % 8 == 0) || parameters.head_size_ % 4 == 0);
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
