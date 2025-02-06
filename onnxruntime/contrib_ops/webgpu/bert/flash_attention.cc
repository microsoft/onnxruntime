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
  shader.AddInput("key", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  shader.AddInput("value", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  if (has_past_) {
    shader.AddInput("past_key", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
    shader.AddInput("past_value", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  }
  shader.AddOutput("present_key", ShaderUsage::UseUniform);
  shader.AddOutput("present_value", ShaderUsage::UseUniform);

  shader.MainFunctionBody() << "let headIdx = workgroup_id.z;\n"
                            << "let kIdx = workgroup_id.x;\n"
                            << "let presentKeyOffset = headIdx * num_workgroups.x * uniforms.vectorized_head_size + (kIdx)*uniforms.vectorized_head_size;\n";
  if (has_past_) {
    shader.MainFunctionBody() << "if (kIdx < uniforms.past_sequence_length) {\n"
                              << "  let pastKeyOffset = headIdx * uniforms.past_sequence_length * uniforms.vectorized_head_size + (kIdx)*uniforms.vectorized_head_size;\n"
                              << "  for (var w: u32 = 0u; w < uniforms.vectorized_head_size; w ++) {\n"
                              << "    present_key[presentKeyOffset+w] = past_key[pastKeyOffset+w];\n"
                              << "    present_value[presentKeyOffset+w] = past_value[pastKeyOffset+w];\n"
                              << "  }\n"
                              << "}\n"
                              << "else if (kIdx >= uniforms.past_sequence_length) {\n";
  } else {
    shader.MainFunctionBody() << "if (kIdx >= uniforms.past_sequence_length) {\n";
  }
  shader.MainFunctionBody() << "  let nkIdx = kIdx - uniforms.past_sequence_length;\n"
                            << "  // Assumes kv have BSNH layout. num_workgroups.z is the num_head as per the dispatch requirement.\n"
                            << "  let nOffset = nkIdx * uniforms.vectorized_head_size * num_workgroups.z + headIdx*uniforms.vectorized_head_size;\n"
                            << "  // Assumes kv have BNSH layout.\n"
                            << "  // let nOffset = headIdx * uniforms.kv_sequence_length * uniforms.vectorized_head_size + nkIdx * uniforms.vectorized_head_size;\n"
                            << "  for (var w: u32 = 0u; w < uniforms.vectorized_head_size; w ++) {\n"
                            << "    present_key[presentKeyOffset+w] = key[nOffset+w];\n"
                            << "    present_value[presentKeyOffset+w] = value[nOffset+w];\n"
                            << "  }\n"
                            << "}\n";

  return Status::OK();
}

Status CopyKVCache(onnxruntime::webgpu::ComputeContext& context, const WebgpuAttentionParameters& parameters,
                   const Tensor* K, const Tensor* past_key, Tensor* present_key,
                   const Tensor* V, const Tensor* past_value, Tensor* present_value,
                   int past_sequence_length, int total_sequence_length) {
  // CopyKVCache takes past key/value and current key/value and copies them to present key and value.
  // This makes it so that FlashAttention only needs to look at present key and value, and saves
  // number of input buffers in the shader, which we run out of (<=8) without this optimization.
  const int components = parameters.head_size_ % 4 == 0 ? 4 : (parameters.head_size_ % 2 == 0 ? 2 : 1);
  bool has_past = (past_sequence_length != 0);
  CopyKVCacheProgram program{"CopyKVCache", has_past};
  if (has_past) {
    program.AddInputs({{K, ProgramTensorMetadataDependency::TypeAndRank, components},
                       {V, ProgramTensorMetadataDependency::TypeAndRank, components},
                       {past_key, ProgramTensorMetadataDependency::TypeAndRank, components},
                       {past_value, ProgramTensorMetadataDependency::TypeAndRank, components}});
  } else {
    program.AddInputs({{K, ProgramTensorMetadataDependency::TypeAndRank, components},
                       {V, ProgramTensorMetadataDependency::TypeAndRank, components}});
  }

  program.AddOutputs({{present_key, ProgramTensorMetadataDependency::Rank, components},
                      {present_value, ProgramTensorMetadataDependency::Rank, components}});

  program.SetDispatchGroupSize(total_sequence_length, 1, parameters.num_heads_)
      .SetWorkgroupSize(1)
      .CacheHint(std::to_string(components) + std::to_string(has_past))
      .AddUniformVariables({{static_cast<uint32_t>(past_sequence_length)},
                            {static_cast<uint32_t>(parameters.kv_sequence_length_)},
                            {static_cast<uint32_t>(parameters.head_size_ / components)}});

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
  const min_value : q_element_t = q_element_t(-65504.0);

  // Default SHM usage limit is 16KB in Dawn.
  var<workgroup> k_tile : array<array<q_value_t, qkv_head_size_vec>, max_k_step>; // 96 * 2 * 16 = 3KB.
  var<workgroup> v_tile : array<array<q_value_t, qkv_head_size_vec>, max_k_step>; // 96 * 2 * 16 = 3KB.

  // Private memory per lane.
  var<private> q_tile : array<q_value_t, qkv_head_size_vec>;
  var<private> o_tile : array<q_value_t, qkv_head_size_vec>;
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
          let val = select(q_value_t(0), present_key[offset+idx], k_start + slot < uniforms.present_sequence_length);
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
          let val  = select(q_value_t(0), present_value[offset+idx], v_start + slot < uniforms.present_sequence_length);
          v_tile[slot][idx%qkv_head_size_vec] = val;
      }
  }
  fn writeo(o_idx_global: u32, head_idx: u32)
  {
      // Stored as float16[batch_size,sequence_length,3072]
      let offset = o_idx_global * num_heads * qkv_head_size_vec + head_idx * qkv_head_size_vec;
      for (var idx:u32 = 0; idx < qkv_head_size_vec; idx ++)
      {
          output[offset+idx] = o_tile[idx];
      }
  }
)HELPER_FN";

  if (has_attention_bias_) {
    shader.AdditionalImplementation() << R"HELPER_FN(
      fn loadAttentionBias(q_idx_global : u32, k_idx_global : u32, head_idx: u32) -> vec4<q_element_t>
      {
          // Stored as float16[batch_size,num_heads,new_seq_length,total_sequence_length]
          if (q_idx_global >= uniforms.new_sequence_length  || k_idx_global >= uniforms.present_sequence_length) {
              return vec4<q_element_t>(0);
          }
          let offset_base = head_idx * uniforms.new_sequence_length * uniforms.present_sequence_length + q_idx_global * uniforms.present_sequence_length;
          let offset = offset_base + k_idx_global;
          let offset_max = offset_base + uniforms.present_sequence_length;
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
  let head_idx = workgroup_id.x;
  let capped_sg_id = min(sg_id, max_k_step);
  let capped_sg_size = min(sg_size, max_k_step);

  // Load Q
  let q_idx_global = workgroup_id.y * workgroup_size_x + local_idx;
  let valid_q = q_idx_global < uniforms.new_sequence_length;
  if (valid_q)
  {
    loadq(q_idx_global, head_idx);
  }

  var previous_max : q_element_t = min_value;
  var previous_denom : q_element_t = 0;

  for(var k_start = 0u; k_start < uniforms.present_sequence_length; k_start+=capped_sg_size)
  {
    workgroupBarrier();
    loadk(k_start, head_idx, local_idx, capped_sg_size);
    loadv(k_start, head_idx, local_idx, capped_sg_size);
    workgroupBarrier();

    // Compute QKt
    var qk_1:vec4<q_element_t>;
    var qk_2:vec4<q_element_t>;
    var qk_3:vec4<q_element_t>;
    var qk_4:vec4<q_element_t>;
    if (sg_size > 8)
    {
      for (var i:u32 = 0u; i < qkv_head_size_vec; i++)
      {
        var k_local = k_tile[capped_sg_id][i];
        var q_own = q_tile[i];
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
        var k_local = k_tile[capped_sg_id][i];
        var q_own = q_tile[i];
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

    qk_1 = qk_1 * q_element_t(uniforms.alpha) + loadAttentionBias(q_idx_global, k_start, head_idx);
    qk_2 = qk_2 * q_element_t(uniforms.alpha) + loadAttentionBias(q_idx_global, k_start+4, head_idx);
    if (sg_size > 8)
    {
      qk_3 = qk_3 * q_element_t(uniforms.alpha) + loadAttentionBias(q_idx_global, k_start+8, head_idx);
      qk_4 = qk_4 * q_element_t(uniforms.alpha) + loadAttentionBias(q_idx_global, k_start+12, head_idx);
    }

    // Neuter qk values where K is out of bounds.
    qk_1[1] = select(min_value, qk_1[1], k_start+1 < uniforms.present_sequence_length);
    qk_1[2] = select(min_value, qk_1[2], k_start+2 < uniforms.present_sequence_length);
    qk_1[3] = select(min_value, qk_1[3], k_start+3 < uniforms.present_sequence_length);
    qk_2[0] = select(min_value, qk_2[0], k_start+4 < uniforms.present_sequence_length);
    qk_2[1] = select(min_value, qk_2[1], k_start+5 < uniforms.present_sequence_length);
    qk_2[2] = select(min_value, qk_2[2], k_start+6 < uniforms.present_sequence_length);
    qk_2[3] = select(min_value, qk_2[3], k_start+7 < uniforms.present_sequence_length);
    if (sg_size > 8)
    {
      qk_3[0] = select(min_value, qk_3[0], k_start+8 < uniforms.present_sequence_length);
      qk_3[1] = select(min_value, qk_3[1], k_start+9 < uniforms.present_sequence_length);
      qk_3[2] = select(min_value, qk_3[2], k_start+10 < uniforms.present_sequence_length);
      qk_3[3] = select(min_value, qk_3[3], k_start+11 < uniforms.present_sequence_length);
      qk_4[0] = select(min_value, qk_4[0], k_start+12 < uniforms.present_sequence_length);
      qk_4[1] = select(min_value, qk_4[1], k_start+13 < uniforms.present_sequence_length);
      qk_4[2] = select(min_value, qk_4[2], k_start+14 < uniforms.present_sequence_length);
      qk_4[3] = select(min_value, qk_4[3], k_start+15 < uniforms.present_sequence_length);
    }

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
    var local_max_temp = max(qk_1, qk_2);
    if (sg_size > 8)
    {
      local_max_temp = max(local_max_temp, qk_3);
      local_max_temp = max(local_max_temp, qk_4);
    }
    let local_max = max(max(local_max_temp.x, local_max_temp.y),max(local_max_temp.z, local_max_temp.w));
    let new_max = max(previous_max, local_max);
    qk_1 = q_value_t(exp(vec4<f32>(qk_1) - f32(new_max)));
    qk_2 = q_value_t(exp(vec4<f32>(qk_2) - f32(new_max)));
    if (sg_size > 8) {
      qk_3 = q_value_t(exp(vec4<f32>(qk_3) - f32(new_max)));
      qk_4 = q_value_t(exp(vec4<f32>(qk_4) - f32(new_max)));
    }
    let sum_vec = qk_1 + qk_2 + qk_3 + qk_4;
    let sum = sum_vec.x + sum_vec.y + sum_vec.z + sum_vec.w;
    // Compute lhs term of update di prime and the compute di prime.
    let dleft = previous_denom * exp(previous_max-new_max);
    var d = dleft + sum;
    d = select(d,q_element_t(0.0000001),d==0);
    qk_1 = qk_1 / d;
    qk_2 = qk_2 / d;
    if (sg_size > 8) {
      qk_3 = qk_3 / d;
      qk_4 = qk_4 / d;
    }
    previous_max = new_max;
    previous_denom = d;
    let o_ratio = dleft / d;

    if (sg_size > 8) {
      for (var i:u32 = 0; i < qkv_head_size_vec; i++)
      {
          var val = select(vec4<q_element_t>(0), v_tile[capped_sg_id][i], k_start + capped_sg_id < uniforms.present_sequence_length);
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
          var val = select(vec4<q_element_t>(0), v_tile[capped_sg_id][i], k_start + capped_sg_id < uniforms.present_sequence_length);
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

  return Status::OK();
}

Status ApplyFlashAttention(const Tensor* Q, const Tensor* K, const Tensor* V, const Tensor* attention_bias,
                           Tensor* output, const Tensor* past_key, Tensor* present_key, const Tensor* past_value, Tensor* present_value,
                           const WebgpuAttentionParameters& parameters, onnxruntime::webgpu::ComputeContext& context) {
  ORT_RETURN_IF_ERROR(CopyKVCache(context, parameters, K, past_key, present_key, V, past_value, present_value, parameters.past_sequence_length_, parameters.total_sequence_length_));

  const uint32_t tile_size = 64;
  bool has_attention_bias = attention_bias != nullptr;
  FlashAttentionProgram program{"FlashAttention", has_attention_bias, parameters.head_size_, parameters.num_heads_};
  program.AddInputs({{Q, ProgramTensorMetadataDependency::TypeAndRank, 4},
                     {present_key, ProgramTensorMetadataDependency::TypeAndRank, 4},
                     {present_value, ProgramTensorMetadataDependency::TypeAndRank, 4},
                     {attention_bias, ProgramTensorMetadataDependency::TypeAndRank}});
  program.AddOutputs({{output, ProgramTensorMetadataDependency::TypeAndRank, 4}});
  const float alpha = parameters.scale_ == 0.0f ? 1.f / sqrt(static_cast<float>(parameters.head_size_))
                                                : parameters.scale_;
  std::string cache_hint = std::to_string(has_attention_bias) +
                           std::to_string(parameters.head_size_) +
                           std::to_string(parameters.num_heads_);
  program.SetDispatchGroupSize(parameters.num_heads_, (parameters.sequence_length_ + tile_size - 1) / tile_size, 1)
      .SetWorkgroupSize(tile_size)
      .CacheHint(cache_hint)
      .AddUniformVariables({{static_cast<uint32_t>(parameters.sequence_length_)},
                            {static_cast<uint32_t>(parameters.total_sequence_length_)},
                            {alpha}});

  return context.RunProgram(program);
}

bool CanApplyFlashAttention(const Tensor* bias, const Tensor* present_key, const Tensor* present_value,
                            const WebgpuAttentionParameters& parameters, onnxruntime::webgpu::ComputeContext& context) {
  return parameters.batch_size_ == 1 &&
         bias == nullptr &&
         parameters.sequence_length_ > 1 &&
         context.Device().HasFeature(wgpu::FeatureName::Subgroups) &&
         present_key != nullptr && present_value != nullptr && present_key->SizeInBytes() > 0 &&
         present_value->SizeInBytes() > 0 && parameters.head_size_ % 4 == 0;
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
