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
  CopyKVCacheProgram program{"CopyKVCache", components, has_past};
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
  constexpr int vectorization_size = 4;
  shader.AddInput("q", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  shader.AddInput("present_key", ShaderUsage::UseUniform);
  shader.AddInput("present_value", ShaderUsage::UseUniform);
  if (has_attention_bias_) {
    shader.AddInput("attention_bias", ShaderUsage::UseUniform);
  }
  shader.AddOutput("output", ShaderUsage::UseUniform);

  // SUBGROUP_SIZE has to be the same as sg_size. For intel this will be 16.
  // TILE_SIZE is the number of groups sharing the k_tile.
  // TILE_SIZE has to be <= SUBGROUP_SIZE. Ideal perf of computeSoftMax is when
  // TILE_SIZE == SUBGROUP_SIZE. This is a sperate constant from SUBGROUP_SIZE
  // because SUBGROUP_SIZE * TILE_SIZE has to be <= 256 as per webgpu
  // gpu limits. For Intel this TILE_SIZE will be 16.
  // Change precision_t to be f32 below to run dotproduct/ softmax in fp32 precision.
  shader.AdditionalImplementation() << "const SUBGROUP_SIZE: u32 = " << subgroup_size_ << ";\n"
                                    << "const TILE_SIZE: u32 = " << tile_size_ << ";\n"
                                    << "const VECTOR_SIZE: u32 = " << vectorization_size << ";\n"
                                    << "const QKV_HEAD_SIZE: u32 = " << qkv_head_size_ << ";\n"
                                    << "const QKV_HEAD_VECTORIZED_SIZE: u32 = QKV_HEAD_SIZE / VECTOR_SIZE;\n"
                                    << "const NUM_HEADS: u32 = " << qkv_num_heads_ << ";\n"
                                    << "alias precision_t = q_element_t;\n"
                                    << "const MIN_VALUE : precision_t = precision_t(-65504.0h);\n";

  // Best to keep SHM usage per workgroup < 128KB, from intel docs for Intel Iris Xe GPU.
  // "The SLM is a 128KB High Bandwidth Memory (HBM) accessible from the EUs in the subslice"
  // GPU afterwhich workgroups will be unscheduled to make space for memory.
  shader.AdditionalImplementation() << ""
                                    << "var<workgroup> q_tile : array<array<q_value_t, QKV_HEAD_VECTORIZED_SIZE>, TILE_SIZE>; // 96 * 2 * 16 = 3KB.\n"
                                    << "var<workgroup> k_tile : array<array<q_value_t, QKV_HEAD_VECTORIZED_SIZE>, TILE_SIZE>; // 96 * 2 * 16 = 3KB.\n"
                                    << "var<workgroup> v_tile : array<array<q_value_t, QKV_HEAD_VECTORIZED_SIZE>, TILE_SIZE>; // 96 * 2 * 16 = 3KB.\n"
                                    << "var<workgroup> o_tile : array<array<q_value_t, QKV_HEAD_VECTORIZED_SIZE>, TILE_SIZE>; // 96 * 2 * 16 = 3KB.\n"
                                    << "var<workgroup> qk_tile : array<array<precision_t, TILE_SIZE>, TILE_SIZE>; // 16 * 2 * 16 = 512\n"
                                    << "var<workgroup> max_tile : array<precision_t, TILE_SIZE>; // 2 * 16 = 32\n"
                                    << "var<workgroup> denom_tile : array<precision_t, TILE_SIZE>; // 2 * 16 = 32\n"
                                    << "var<workgroup> o_ratio : array<precision_t, TILE_SIZE>; // 2 * 16 = 32\n";

  shader.AdditionalImplementation() << R"HELPER_FN(
fn loadq(slot: u32, q_idx_global : u32, head_idx: u32, sg_id : u32, sg_size : u32)
{
    // Stored as float16[batch_size,sequence_length,3072] the inputs as per onnx MHA
    // This is the layout if TransferBSDToBNSH has not been run.
    let offset = q_idx_global * (QKV_HEAD_VECTORIZED_SIZE) * NUM_HEADS + QKV_HEAD_VECTORIZED_SIZE * head_idx;
    // Stored as BNSH - which is what webgpu uses after TransferBSDToBNSH has been run.
    // let offset = head_idx * uniforms.new_sequence_length * QKV_HEAD_VECTORIZED_SIZE + q_idx_global * QKV_HEAD_VECTORIZED_SIZE;
    for (var idx:u32 = sg_id; idx < QKV_HEAD_VECTORIZED_SIZE; idx+= sg_size)
    {
        var value = q[idx+offset];
        q_tile[slot][idx] = value;
    }
}
fn loadk(slot: u32, k_idx_global : u32, head_idx: u32, sg_id: u32, sg_size: u32)
{
    // Stored as float16[batch_size,num_heads,present_sequence_length,96]
    let offset = head_idx * uniforms.present_sequence_length * QKV_HEAD_VECTORIZED_SIZE + k_idx_global * QKV_HEAD_VECTORIZED_SIZE;
    for (var idx:u32 = sg_id; idx < QKV_HEAD_VECTORIZED_SIZE; idx+=sg_size)
    {
        var value = present_key[idx+offset];
        k_tile[slot][idx] = value;
    }
}
fn loadv(slot: u32, v_idx_global : u32, head_idx: u32, sg_id: u32, sg_size: u32)
{
    // Stored as float16[batch_size,num_heads,present_sequence_length,96]
    let offset = head_idx * uniforms.present_sequence_length * QKV_HEAD_VECTORIZED_SIZE + v_idx_global * QKV_HEAD_VECTORIZED_SIZE;
    for (var idx:u32 = sg_id; idx < QKV_HEAD_VECTORIZED_SIZE; idx+=sg_size)
    {
        v_tile[slot][idx] = present_value[idx+offset];
    }
}
fn loadAttentionBias(q_row: u32, q_idx_global : u32, k_col: u32, k_idx_global : u32, head_idx: u32)
{
    // Stored as float16[batch_size,num_heads,new_seq_length,total_sequence_length]
    if (q_idx_global >= uniforms.new_sequence_length  || k_idx_global >= uniforms.present_sequence_length || k_col >= TILE_SIZE) {
        qk_tile[q_row][k_col] = 0.0;
        return;
    }
    let offset = head_idx * uniforms.new_sequence_length * uniforms.present_sequence_length + q_idx_global * uniforms.present_sequence_length + k_idx_global;
    qk_tile[q_row][k_col] = precision_t(attention_bias[offset]);
}
fn writeo(slot: u32, o_idx_global : u32, head_idx: u32, sg_id : u32, sg_size : u32)
{
    // Stored as float16[batch_size,sequence_length,3072]
    let offset = o_idx_global * NUM_HEADS * QKV_HEAD_VECTORIZED_SIZE + head_idx * QKV_HEAD_VECTORIZED_SIZE;
    for (var idx:u32 = sg_id; idx < QKV_HEAD_VECTORIZED_SIZE; idx += sg_size)
    {
        let value = o_tile[slot][idx];
        output[offset+idx] = value;
    }
}
fn computeDotProduct(q_idx: u32, k_idx: u32, sg_id: u32, sg_size : u32)
{
    var sum:vec4<precision_t> = vec4<precision_t>(0, 0, 0, 0);
    // idx is not initialized to sg_id to ensure uniformity because the loop uses
    // subgroupAdd and unused lanes need to be initialized with 0 for correctness.
    for (var idx:u32 = 0; idx < QKV_HEAD_VECTORIZED_SIZE; idx+= sg_size)
    {
        var result = vec4<precision_t>(0);
        let sg_idx = idx+sg_id;
        if (sg_idx < QKV_HEAD_VECTORIZED_SIZE)
        {
            result = vec4<precision_t>(q_tile[q_idx][sg_idx])*vec4<precision_t>(k_tile[k_idx][sg_idx]);
        }
        sum += subgroupAdd(result);
    }
    if (sg_id == 0)
    {
        let single_sum : precision_t = sum.x + sum.y + sum.z + sum.w;
        let sqrt_dk = precision_t(uniforms.alpha);
        let value = single_sum * sqrt_dk;
        qk_tile[q_idx][k_idx] += value;
    }
}
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
fn computeSoftMax(q_idx: u32, sg_id:u32, enabled:bool)
{
    var x : precision_t = MIN_VALUE;
    if (enabled){
        x = qk_tile[q_idx][sg_id];
    }
    var max_value = subgroupMax(x);
    max_value = max(max_tile[q_idx], max_value);
    let sub = x - max_value;
    var value:precision_t = 0;
    if (enabled) {
        value = exp(sub);
    }
    let sum = subgroupAdd(value);
    // Compute lhs term of update di prime and the compute di prime.
    let dleft = denom_tile[q_idx] * exp(max_tile[q_idx]-max_value);
    var d = dleft + sum;
    if (d == 0)
    {
        // Avoid division by zero by setting d to a really small value.
        // Note: Removing this protection has had no negative effect on any
        // of the prompts tried so far. This is a safety net.
        d = precision_t(0.0000001h);
    }
    qk_tile[q_idx][sg_id] = value / d;
    if (sg_id == 0)
    {
        max_tile[q_idx] = max_value;
        denom_tile[q_idx] = d;
        o_ratio[q_idx] = dleft / d;
    }
}
fn computeO(q_idx: u32, sg_id:u32, enabled:bool)
{
    var attn = precision_t(0);
    if (enabled)
    {
      attn = qk_tile[q_idx][sg_id];
    }
    for (var i:u32 = 0; i < QKV_HEAD_VECTORIZED_SIZE; i++)
    {
        let val = vec4<precision_t>(v_tile[sg_id][i]);
        var intermediate = attn * val;
        let sum = subgroupAdd(intermediate);
        if (sg_id == 0)
        {
            let o_ratio = o_ratio[q_idx];
            let old_o = vec4<precision_t>(o_tile[q_idx][i]);
            let new_o = ( o_ratio * old_o) +  sum;
            o_tile[q_idx][i] = q_value_t(new_o);
        }
    }
}
)HELPER_FN";

  // Shader is designed to be dispatched as Dispatch(num_heads, new_sequence_length / TILE_SIZE, 1)
  // Each workgroup is responsible for a range of q values (TILE_SIZE) and visits all Ks for those q's.
  // Each workgroup has TILE_SIZE waves, with each wave having subgroup size number of lanes (threads).
  // Synchronization between lanes in a wave is free, with various subgroup* functions, and this shader
  // uses that. Synchronization between waves requires calling workgroupBarrier.
  shader.MainFunctionBody() << R"MAIN_FN(
let head_idx = workgroup_id.x;
// It is always the case that 0 <= wave_id < TILE_SIZE
// Each wave has sg_size lanes (subgroup threads).
let wave_id = u32(local_idx / sg_size);

let q_idx_start = workgroup_id.y * TILE_SIZE;
let q_idx_global = q_idx_start + wave_id;
let q_idx_global_using_wave_valid = q_idx_global < uniforms.new_sequence_length;
if (q_idx_global_using_wave_valid)
{
  // Each invocation (wave_id) gets lane threads (subgroup threads) and is responsible for 1 query.
  loadq(wave_id, q_idx_global, head_idx, sg_id, sg_size);
}
if (sg_id == 0)
{
  max_tile[wave_id] = MIN_VALUE;
}
for(var k_start = 0u; k_start < uniforms.present_sequence_length; k_start+=TILE_SIZE)
{
    // Insert barrier before updating shared memory the workgroup shares.
    workgroupBarrier();
    let k_idx_global = k_start+wave_id;
    let k_idx_global_using_wave_valid = k_idx_global < uniforms.present_sequence_length;
    if (k_idx_global_using_wave_valid) {
        // Leveraging the subgroup lanes for parallelism, load into slot wave_id
        // K/V values from k_start+wave_id.
        loadk(wave_id, k_idx_global, head_idx, sg_id, sg_size);
        loadv(wave_id, k_idx_global, head_idx, sg_id, sg_size);
    }
    // Next, we want for every q row (wave_id) to populate bias for new sequence length
    // (k_start+sg_id). loadAttentionBias handles range checking q_idx_global,
    // and sg_id, (k_start+sg_id).
    loadAttentionBias(wave_id, q_idx_global, sg_id, k_start+sg_id, head_idx);
    // Insert barrier before workgroup starts reading the shared memory.
    workgroupBarrier();

    //if (k_idx_global_using_wave_valid)
    {
      // Iterate over Q rather than K because for the case of new_seq 1, there is a single query
      // and context length of K by iterating over Q using the waves for K, this step can use all
      // the waves in the workgroup, instead of leaving them idle.
      for (var q_idx = 0u; q_idx < TILE_SIZE && q_idx_start + q_idx < uniforms.new_sequence_length; q_idx++)
      {
          // Leveraging the subgroups for parallelism, compute dot product of QK.
          // We validate q_idx,wave_id to be less than TILE_SIZE, computeDotProduct only needs to
          // validate sg_id as being less than QKV_HEAD_VECTORIZED_SIZE.
          computeDotProduct(q_idx, wave_id, sg_id, sg_size);
      }
    }
    // Insert barrier before SoftMax reads the dot product values across K.
    workgroupBarrier();

    let wave_lane_valid:bool = q_idx_global_using_wave_valid && sg_id < TILE_SIZE && sg_id + k_start < uniforms.present_sequence_length;
    computeSoftMax(wave_id, sg_id, wave_lane_valid);
    computeO(wave_id, sg_id, wave_lane_valid);
}
workgroupBarrier();
if (q_idx_global_using_wave_valid)
{
  writeo(wave_id, q_idx_global, head_idx, sg_id, sg_size);
}
)MAIN_FN";

  return Status::OK();
}

Status ApplyFlashAttention(const Tensor* Q, const Tensor* K, const Tensor* V, const Tensor* attention_bias,
                           Tensor* output, const Tensor* past_key, Tensor* present_key, const Tensor* past_value, Tensor* present_value,
                           const WebgpuAttentionParameters& parameters, onnxruntime::webgpu::ComputeContext& context) {
  ORT_RETURN_IF_ERROR(CopyKVCache(context, parameters, K, past_key, present_key, V, past_value, present_value, parameters.past_sequence_length_, parameters.total_sequence_length_));

  const uint32_t subgroup_size = 16;
  const uint32_t tile_size = subgroup_size;
  bool has_attention_bias = attention_bias != nullptr;
  FlashAttentionProgram program{"FlashAttention", has_attention_bias, subgroup_size, tile_size, parameters.head_size_, parameters.num_heads_};
  program.AddInputs({{Q, ProgramTensorMetadataDependency::TypeAndRank, 4},
                     {present_key, ProgramTensorMetadataDependency::TypeAndRank, 4},
                     {present_value, ProgramTensorMetadataDependency::TypeAndRank, 4},
                     {attention_bias, ProgramTensorMetadataDependency::TypeAndRank}});
  program.AddOutputs({{output, ProgramTensorMetadataDependency::TypeAndRank, 4}});
  const float alpha = parameters.scale_ == 0.0f ? 1.f / sqrt(static_cast<float>(parameters.head_size_))
                                                : parameters.scale_;
  std::string cache_hint = std::to_string(has_attention_bias) +
                           std::to_string(subgroup_size) +
                           std::to_string(tile_size) +
                           std::to_string(parameters.head_size_) +
                           std::to_string(parameters.num_heads_);
  program.SetDispatchGroupSize(parameters.num_heads_, (parameters.sequence_length_ + tile_size - 1) / tile_size, 1)
      .SetWorkgroupSize(subgroup_size * subgroup_size)
      .CacheHint(cache_hint)
      .AddUniformVariables({{static_cast<uint32_t>(parameters.sequence_length_)},
                            {static_cast<uint32_t>(parameters.total_sequence_length_)},
                            {alpha}});

  return context.RunProgram(program);
}

bool CanApplyFlashAttention(const Tensor* bias, const Tensor* present_key, const Tensor* present_value,
                            const WebgpuAttentionParameters& parameters, onnxruntime::webgpu::ComputeContext& context) {
  // The min subgroup size affects the block size while going through the sequence length.
  // 16 is the smallest size tested, smaller sized would impact performance.
  // Checking for this also ensures that we dont run flash attention where subgroup is not supported.
  constexpr int kMinSupportedSubgroupSize = 16;
  // Workgroup size is set to be (subgroup_size * subgroup_size), check that it is allowed.
  // Flash attention is written only to support batch_size of 1, algorithm can be extended to support
  // batch_size > 1. What bias is used for is not clear, so it is not implemented in the shader.
  // The Flash attention implementation is vectorized, to keep things simple, only vec4 is implemented -
  // this implies that head_size has to be a multiple of 4.
  return context.DeviceLimits().maxComputeWorkgroupSizeX >= (kMinSupportedSubgroupSize * kMinSupportedSubgroupSize) &&
         parameters.batch_size_ == 1 &&
         bias == nullptr &&
         present_key != nullptr && present_value != nullptr && present_key->SizeInBytes() > 0 &&
         present_value->SizeInBytes() > 0 && parameters.head_size_ % 4 == 0;
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
