// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/bert/group_query_attention_helper.h"
#include "contrib_ops/webgpu/bert/attention_common.h"
#include "contrib_ops/webgpu/bert/group_query_attention.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "contrib_ops/webgpu/bert/rotary_embedding.h"
#include "contrib_ops/webgpu/bert/flash_attention.h"

#include "core/common/narrow.h"
#include "core/providers/webgpu/nn/layer_norm.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/shader_helper.h"

using namespace onnxruntime::webgpu;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;
using namespace onnxruntime::contrib::group_query_attention_helper;

namespace onnxruntime {
namespace contrib {
namespace webgpu {

Status SplitPackedQKVWithRotaryEmbeddingProgram::GenerateShaderCode(ShaderHelper& sh) const {
  const auto& packed_qkv = sh.AddInput("packed_qkv", ShaderUsage::UseUniform);
  const auto& seqlens = sh.AddInput("seqlens", ShaderUsage::UseUniform);
  const auto& cos_cache = sh.AddInput("cos_cache", ShaderUsage::UseUniform);
  const auto& sin_cache = sh.AddInput("sin_cache", ShaderUsage::UseUniform);

  const auto& query = sh.AddOutput("query", ShaderUsage::UseUniform);
  const auto& key = sh.AddOutput("key", ShaderUsage::UseUniform);
  const auto& val = sh.AddOutput("val", ShaderUsage::UseUniform);

  return WGSL_TEMPLATE_APPLY(sh, "bert/split_packed_qkv_with_rotary_embedding.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(interleaved, interleaved_),
                             WGSL_TEMPLATE_PARAMETER(multi_rotary_cache_concat_offset, multi_rotary_cache_concat_offset_),
                             WGSL_TEMPLATE_PARAMETER(use_multi_rotary_cache_concat, multi_rotary_cache_concat_offset_ > 0),
                             WGSL_TEMPLATE_VARIABLE(cos_cache, cos_cache),
                             WGSL_TEMPLATE_VARIABLE(key, key),
                             WGSL_TEMPLATE_VARIABLE(packed_qkv, packed_qkv),
                             WGSL_TEMPLATE_VARIABLE(query, query),
                             WGSL_TEMPLATE_VARIABLE(seqlens, seqlens),
                             WGSL_TEMPLATE_VARIABLE(sin_cache, sin_cache),
                             WGSL_TEMPLATE_VARIABLE(val, val));
}

// Split packed QKV with Q/K rotary embedding fusion
Status RunSplitPackedQKVWithRotaryEmbedding(onnxruntime::webgpu::ComputeContext& context,
                                            const WebgpuAttentionParameters& params,
                                            const Tensor* packedQKV,
                                            const Tensor* seqlen_k,
                                            const Tensor* cos_cache,
                                            const Tensor* sin_cache,
                                            Tensor* query,
                                            Tensor* key,
                                            Tensor* val) {
  const auto half_rotary_embedding_dim = gsl::narrow_cast<uint32_t>(cos_cache->Shape()[1]);
  const auto head_size = params.head_size_;

  int components = 1;
  // Currently we only support vectorization when RoPE is not interleaved
  if (!params.rotary_interleaved_) {
    if ((params.head_size_ % 4 == 0) && (half_rotary_embedding_dim % 4 == 0)) {
      components = 4;
    } else if ((params.head_size_ % 2 == 0) && (half_rotary_embedding_dim % 2 == 0)) {
      components = 2;
    }
  }

  // Adjust dimensions for vectorization
  const auto half_rotary_embedding_dim_vec = half_rotary_embedding_dim / components;
  const auto head_size_vec = head_size / components;

  // Dispatch: batch_size * sequence_length * num_heads * (half_rotary_dim + need_copy_dim)
  // work_per_head = half_rotary_dim + (head_size - 2 * half_rotary_dim)
  //               = head_size - half_rotary_dim
  const auto work_per_head_vec = head_size_vec - half_rotary_embedding_dim_vec;
  auto dispatch_size = static_cast<uint32_t>(params.batch_size_ * params.sequence_length_ * params.num_heads_ * work_per_head_vec);

  const uint32_t multi_rotary_cache_concat_offset = context.MultiRotaryCacheConcatOffset();
  SplitPackedQKVWithRotaryEmbeddingProgram program(params.rotary_interleaved_, multi_rotary_cache_concat_offset);
  program
      .CacheHint(params.rotary_interleaved_, multi_rotary_cache_concat_offset)
      .AddInput({packedQKV, ProgramTensorMetadataDependency::TypeAndRank, components})
      .AddInputs({
          {seqlen_k, ProgramTensorMetadataDependency::TypeAndRank},
          {cos_cache, ProgramTensorMetadataDependency::Rank, components},
          {sin_cache, ProgramTensorMetadataDependency::Rank, components},
      })
      .AddOutputs({{query, ProgramTensorMetadataDependency::None, components},
                   {key, ProgramTensorMetadataDependency::None, components},
                   {val, ProgramTensorMetadataDependency::None, components}})
      .AddUniformVariables({
          {static_cast<uint32_t>(params.sequence_length_)},
          {static_cast<uint32_t>(params.hidden_size_ / components)},
          {static_cast<uint32_t>(params.kv_hidden_size_ / components)},
          {static_cast<uint32_t>(params.num_heads_)},
          {static_cast<uint32_t>(params.kv_num_heads_)},
          {static_cast<uint32_t>(head_size_vec)},
          {static_cast<uint32_t>(half_rotary_embedding_dim_vec)},
          {static_cast<uint32_t>(dispatch_size)},
      })
      .SetDispatchGroupSize((dispatch_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE);
  return context.RunProgram(program);
}

// Fused Q/K rotary embedding. When q_norm_weight and k_norm_weight are non-null, a per-head
// RMS normalization (Q[c] *= inverseSqrt(mean(Q[..]^2)+eps) * q_norm_weight[c]; same for K)
// is fused into the rotary kernel ahead of the rotation. This decode-only fast path replaces
// the standalone SimplifiedLayerNormalization dispatches that GroupQueryAttentionPreNormFusion
// folds away.
Status RunFusedQKRotaryEmbedding(onnxruntime::webgpu::ComputeContext& context,
                                 const WebgpuAttentionParameters& params,
                                 const Tensor* query_in,
                                 const Tensor* key_in,
                                 const Tensor* seqlen_k,
                                 const Tensor* cos_cache,
                                 const Tensor* sin_cache,
                                 Tensor* query_out,
                                 Tensor* key_out,
                                 const Tensor* q_norm_weight = nullptr,
                                 const Tensor* k_norm_weight = nullptr,
                                 float qk_norm_epsilon = 0.0f) {
  const auto half_rotary_embedding_dim = gsl::narrow_cast<uint32_t>(cos_cache->Shape()[1]);
  const auto head_size = params.head_size_;

  // Build Q domain
  const auto hidden_size_q = params.hidden_size_;
  const TensorShape q_global_shape({params.batch_size_, params.sequence_length_,
                                    hidden_size_q / head_size,
                                    static_cast<int64_t>(head_size - half_rotary_embedding_dim)});
  const auto rank = q_global_shape.NumDimensions();
  std::vector<uint32_t> q_global_dims(rank);
  std::vector<uint32_t> q_global_strides(rank);
  for (size_t j = 0; j < rank; ++j) {
    q_global_dims[j] = gsl::narrow_cast<uint32_t>(q_global_shape[j]);
    q_global_strides[j] = gsl::narrow_cast<uint32_t>(q_global_shape.SizeFromDimension(j + 1));
  }

  // Build K domain
  const auto hidden_size_k = params.kv_hidden_size_;
  const TensorShape k_global_shape({params.batch_size_, params.sequence_length_,
                                    hidden_size_k / head_size,
                                    static_cast<int64_t>(head_size - half_rotary_embedding_dim)});
  std::vector<uint32_t> k_global_dims(rank);
  for (size_t j = 0; j < rank; ++j) {
    k_global_dims[j] = gsl::narrow_cast<uint32_t>(k_global_shape[j]);
  }

  const auto q_domain_size = gsl::narrow_cast<uint32_t>(q_global_shape.Size());

  const auto q_input_output_strides = std::vector<uint32_t>(
      {gsl::narrow_cast<uint32_t>(query_in->Shape().SizeFromDimension(1)),
       gsl::narrow_cast<uint32_t>(hidden_size_q),
       gsl::narrow_cast<uint32_t>(head_size),
       1u});

  const auto k_input_output_strides = std::vector<uint32_t>(
      {gsl::narrow_cast<uint32_t>(key_in->Shape().SizeFromDimension(1)),
       gsl::narrow_cast<uint32_t>(hidden_size_k),
       gsl::narrow_cast<uint32_t>(head_size),
       1u});

  // Dispatch computations only over the Q domain, and fuse K write operations using a head-index-based condition.
  const bool has_qk_norm = (q_norm_weight != nullptr) && (k_norm_weight != nullptr);
  FusedQKRotaryEmbeddingProgram program(params.rotary_interleaved_, has_qk_norm);
  // When has_qk_norm is true the shader binds q_input/k_input with UseElementTypeAlias, so
  // the per-input cache dependency must include Type for both. Without TypeAndRank on
  // key_in the shader-validation in Debug builds fails with "Input dependency is not set
  // for Type, but type alias for element type or value type is used."
  const auto k_input_dep = has_qk_norm
                               ? ProgramTensorMetadataDependency::TypeAndRank
                               : ProgramTensorMetadataDependency::Rank;
  program
      .CacheHint(params.rotary_interleaved_, has_qk_norm)
      .AddInputs({
          {query_in, ProgramTensorMetadataDependency::TypeAndRank},
          {key_in, k_input_dep},
          {seqlen_k, ProgramTensorMetadataDependency::TypeAndRank},
          {cos_cache, ProgramTensorMetadataDependency::Rank},
          {sin_cache, ProgramTensorMetadataDependency::Rank},
      })
      .AddOutputs({
          {query_out, ProgramTensorMetadataDependency::None},
          {key_out, ProgramTensorMetadataDependency::None},
      })
      .SetDispatchGroupSize((q_domain_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({
          {params.scale_},
          {gsl::make_span(q_global_dims)},
          {gsl::make_span(q_global_strides)},
          {gsl::make_span(q_input_output_strides)},
          {gsl::make_span(k_global_dims)},
          {gsl::make_span(k_input_output_strides)},
          {q_domain_size},
          {static_cast<uint32_t>(head_size)},
          {qk_norm_epsilon},
      });

  if (has_qk_norm) {
    program.AddInputs({
        {q_norm_weight, ProgramTensorMetadataDependency::Type},
        {k_norm_weight, ProgramTensorMetadataDependency::Type},
    });
  }

  return context.RunProgram(program);
}

Status GroupQueryAttention::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {
  const Tensor* query = context.Input<Tensor>(0);
  const Tensor* key = context.Input<Tensor>(1);
  const Tensor* value = context.Input<Tensor>(2);
  const Tensor* past_key = context.Input<Tensor>(3);
  const Tensor* past_value = context.Input<Tensor>(4);
  const Tensor* seqlen_k = context.Input<Tensor>(5);
  const Tensor* total_seqlen_tensor = context.Input<Tensor>(6);
  const Tensor* cos_cache = context.Input<Tensor>(7);
  const Tensor* sin_cache = context.Input<Tensor>(8);
  const Tensor* position_ids = context.Input<Tensor>(9);  // TODO: support sliding window
  const Tensor* attention_bias = context.Input<Tensor>(10);
  const Tensor* head_sink = context.Input<Tensor>(11);
  // Inputs 12 and 13 are k_scale / v_scale (KV-cache quant). Not consumed by WebGPU yet.
  // Inputs 14 and 15 are q_norm_weight / k_norm_weight, populated by
  // GroupQueryAttentionPreNormFusion. WebGPU supports these inputs for the configurations
  // validated below (do_rotary, non-packed Q/K/V).
  const Tensor* q_norm_weight = context.InputCount() > 14 ? context.Input<Tensor>(14) : nullptr;
  const Tensor* k_norm_weight = context.InputCount() > 15 ? context.Input<Tensor>(15) : nullptr;
  const bool has_qk_norm = (q_norm_weight != nullptr) && (k_norm_weight != nullptr);
  // The current fused prologue only supports the Qwen3-style configuration that
  // GroupQueryAttentionPreNormFusion targets: do_rotary, non-packed Q/K/V. Reject any
  // other configuration so downstream rewrites cannot land silently.
  if ((q_norm_weight != nullptr) ^ (k_norm_weight != nullptr)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "GroupQueryAttention: q_norm_weight and k_norm_weight must be provided together.");
  }
  if (has_qk_norm && !do_rotary_) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "GroupQueryAttention: q_norm_weight / k_norm_weight require do_rotary=1.");
  }

  GroupQueryAttentionParameters params = {};
  ORT_RETURN_IF_ERROR(group_query_attention_helper::CheckInputs(query,
                                                                key,
                                                                value,
                                                                past_key,
                                                                past_value,
                                                                cos_cache,
                                                                sin_cache,
                                                                &params,
                                                                num_heads_,
                                                                kv_num_heads_,
                                                                seqlen_k,
                                                                total_seqlen_tensor,
                                                                scale_,
                                                                softcap_,
                                                                0,
                                                                onnxruntime::narrow<int>(context.DeviceLimits().maxComputeInvocationsPerWorkgroup)));
  params.use_smooth_softmax = use_smooth_softmax_;
  params.rotary_interleaved = rotary_interleaved_;

  ORT_RETURN_IF_ERROR(group_query_attention_helper::CheckCustomAttentionInputs(position_ids,
                                                                               attention_bias,
                                                                               head_sink,
                                                                               params));

  ORT_RETURN_IF_ERROR(group_query_attention_helper::CheckNoQKOutput(
      context.OutputCount(),
      static_cast<int>(Info().GetAttrOrDefault<int64_t>("qk_output", static_cast<int64_t>(QKOutputType::NO_OUTPUT)))));

  WebgpuAttentionParameters parameters(params);
  if (has_qk_norm && parameters.is_packed_qkv_) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "GroupQueryAttention: q_norm_weight / k_norm_weight are not supported when QKV is packed.");
  }
  if (has_qk_norm) {
    // The fused rotary shader multiplies q/k elements by q/k_norm_weight values without
    // inserting casts between storage element types. Enforce dtype parity so hand-authored
    // models fail with a clear INVALID_ARGUMENT instead of a shader compile error.
    if (q_norm_weight->DataType() != query->DataType()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "GroupQueryAttention: q_norm_weight element type must match query element type.");
    }
    if (k_norm_weight->DataType() != key->DataType()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "GroupQueryAttention: k_norm_weight element type must match key element type.");
    }

    // The fused prologue indexes q/k_norm_weight as a 1-D tensor of length head_size. Validate
    // shape here so a hand-authored model with a wrong shape fails with INVALID_ARGUMENT instead
    // of silently reading the wrong offsets (or out of bounds).
    const auto& q_norm_shape = q_norm_weight->Shape();
    if (!(q_norm_shape.NumDimensions() == 1 &&
          q_norm_shape[0] == static_cast<int64_t>(parameters.head_size_))) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "GroupQueryAttention: q_norm_weight must be a 1-D tensor of shape [head_size=",
                             parameters.head_size_, "], got ", q_norm_shape.ToString(), ".");
    }
    const auto& k_norm_shape = k_norm_weight->Shape();
    if (!(k_norm_shape.NumDimensions() == 1 &&
          k_norm_shape[0] == static_cast<int64_t>(parameters.head_size_))) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "GroupQueryAttention: k_norm_weight must be a 1-D tensor of shape [head_size=",
                             parameters.head_size_, "], got ", k_norm_shape.ToString(), ".");
    }
  }
  TensorShapeVector output_shape(3);
  output_shape[0] = static_cast<int64_t>(parameters.batch_size_);
  output_shape[1] = static_cast<int64_t>(parameters.sequence_length_);
  output_shape[2] = static_cast<int64_t>(parameters.hidden_size_);
  Tensor* output = context.Output(0, output_shape);
  std::vector<int64_t> present_dims{
      parameters.batch_size_,
      kv_num_heads_,
      parameters.seqlen_present_kv_cache_,
      parameters.head_size_};
  std::vector<int64_t> present_kv_shape(present_dims);
  Tensor* present_key = context.Output(1, present_kv_shape);
  Tensor* present_value = context.Output(2, present_kv_shape);

  // When present_key/present_value outputs are not requested (nullptr), this is a
  // KV-shared layer. Flash attention will create internal GPU buffers as needed.
  parameters.past_present_share_buffer_ = present_key != nullptr && present_value != nullptr &&
                                          past_key != nullptr && past_value != nullptr &&
                                          past_key->DataRaw() == present_key->DataRaw() &&
                                          past_value->DataRaw() == present_value->DataRaw();

  ORT_ENFORCE(parameters.total_sequence_length_ <= parameters.seqlen_present_kv_cache_, "Total sequence length cannot be greater than the existing KV cache length.");

  Tensor qSplit;
  Tensor kSplit;
  Tensor vSplit;

  Tensor qRotary;
  Tensor kRotary;

  // kv_sequence_length==0 fast path: K/V inputs are empty (shared KV layer).
  // Skip all K/V processing; only apply RoPE to Q if needed.
  // Use past_key/past_value directly as the KV context.
  const bool kv_empty = (parameters.kv_sequence_length_ == 0);

  // Use a sliding window if the total sequence exceeds the window's length.
  bool use_sliding_window = (local_window_size_ != -1 && local_window_size_ < parameters.total_sequence_length_);
  bool will_use_flash_attention = false;
  // For kv_empty layers (shared KV), sliding window is irrelevant — there's no new KV to window
  // over, the layer reuses another layer's already-computed KV cache. Flash attention is required
  // for these layers, so we bypass the sliding window check to allow it.
  if (!use_smooth_softmax_ && (!use_sliding_window || kv_empty)) {
    // Create a temporary parameters copy with is_packed_qkv_ set to false to check if flash attention can be applied after unpacking
    WebgpuAttentionParameters temp_params = parameters;
    temp_params.is_packed_qkv_ = false;
    will_use_flash_attention = CanApplyFlashAttention(temp_params, context, seqlen_k);
  }

  if (kv_empty) {
    // KV inputs are empty - shared KV layer. Only need to optionally apply RoPE to Q.
    ORT_ENFORCE(!parameters.is_packed_qkv_, "Packed QKV is not supported with kv_sequence_length==0 (shared KV layers).");
    if (do_rotary_) {
      // Apply RoPE to Q only — K doesn't need rotation since we reuse another layer's already-rotated KV cache.
      qRotary = context.CreateGPUTensor(query->DataType(), query->Shape());
      // Query is BSD (3 dims): [batch, sequence, hidden]. Strides for bsnh layout:
      // {batch_stride, hidden_size, head_size, 1}.
      const auto batch_stride = static_cast<uint32_t>(parameters.sequence_length_ * parameters.hidden_size_);
      const std::vector<uint32_t> q_input_output_strides{
          batch_stride,
          static_cast<uint32_t>(parameters.hidden_size_),
          static_cast<uint32_t>(parameters.head_size_),
          1u};
      ORT_RETURN_IF_ERROR(RunRotaryEmbedding(context,
                                             query, seqlen_k, cos_cache, sin_cache, &qRotary,
                                             parameters.batch_size_, parameters.sequence_length_,
                                             parameters.hidden_size_, parameters.head_size_,
                                             parameters.scale_, parameters.rotary_interleaved_,
                                             /*use_seqlens_for_position=*/true, q_input_output_strides));
      query = &qRotary;
    }
  } else if (parameters.is_packed_qkv_ && do_rotary_) {
    // Use the ultimate fused operation when FlashAttention and static KV cache is enabled.
    if (will_use_flash_attention && parameters.past_present_share_buffer_) {
      // Directly call ApplyFlashAttention with fused split/rotary/copyKV enabled
      // query points to packed QKV, K and V are nullptr since they're not needed
      return ApplyFlashAttention(query, nullptr, nullptr, attention_bias, output, past_key, present_key, past_value,
                                 present_value, parameters, context, seqlen_k, cos_cache, sin_cache, head_sink);
    }
    // Fused: splitQKV + rotary QK
    qSplit = context.CreateGPUTensor(query->DataType(), TensorShape({parameters.batch_size_, parameters.sequence_length_, parameters.hidden_size_}));
    kSplit = context.CreateGPUTensor(query->DataType(), TensorShape({parameters.batch_size_, parameters.sequence_length_, parameters.kv_hidden_size_}));
    vSplit = context.CreateGPUTensor(query->DataType(), TensorShape({parameters.batch_size_, parameters.sequence_length_, parameters.kv_hidden_size_}));
    ORT_RETURN_IF_ERROR(RunSplitPackedQKVWithRotaryEmbedding(context, parameters,
                                                             query, seqlen_k,
                                                             cos_cache, sin_cache,
                                                             &qSplit, &kSplit, &vSplit));
    parameters.is_packed_qkv_ = false;
    parameters.qkv_format_ = Q_K_V_BSNH;
    query = &qSplit;
    key = &kSplit;
    value = &vSplit;
  } else {
    if (parameters.is_packed_qkv_) {
      // splitQKV
      qSplit = context.CreateGPUTensor(query->DataType(), TensorShape({parameters.batch_size_, parameters.sequence_length_, parameters.hidden_size_}));
      kSplit = context.CreateGPUTensor(query->DataType(), TensorShape({parameters.batch_size_, parameters.sequence_length_, parameters.kv_hidden_size_}));
      vSplit = context.CreateGPUTensor(query->DataType(), TensorShape({parameters.batch_size_, parameters.sequence_length_, parameters.kv_hidden_size_}));
      ORT_RETURN_IF_ERROR(SplitPackedQKV(context, parameters, query, &qSplit, &kSplit, &vSplit, parameters.kv_hidden_size_));
      parameters.is_packed_qkv_ = false;
      parameters.qkv_format_ = Q_K_V_BSNH;
      query = &qSplit;
      key = &kSplit;
      value = &vSplit;
    }
    if (do_rotary_) {
      // Per-head RMS normalization handling for Qwen3-style models (GQA inputs 14/15).
      //   - Decode (sequence_length == 1): fold the norm into the FusedQKRotaryEmbedding
      //     kernel. Each thread re-reads its head's head_size channels (Approach A); no
      //     reductions, no shared memory. Sub-microsecond overhead vs ~60us/layer SLN savings.
      //   - Prefill (sequence_length > 1): fall back to two standalone SimplifiedLayerNorm
      //     dispatches into scratch tensors, then run the unfused FusedQKRotaryEmbedding.
      //     Matches the pre-fusion graph timing exactly so prefill cannot regress.
      Tensor qNorm;
      Tensor kNorm;
      const Tensor* q_for_rotary = query;
      const Tensor* k_for_rotary = key;
      const Tensor* q_norm_for_fused = nullptr;
      const Tensor* k_norm_for_fused = nullptr;
      const bool decode_norm_fast_path = has_qk_norm && parameters.sequence_length_ == 1;
      if (has_qk_norm && !decode_norm_fast_path) {
        qNorm = context.CreateGPUTensor(query->DataType(), query->Shape());
        kNorm = context.CreateGPUTensor(key->DataType(), key->Shape());
        const uint32_t q_norm_count =
            static_cast<uint32_t>(parameters.batch_size_) *
            static_cast<uint32_t>(parameters.sequence_length_) *
            static_cast<uint32_t>(parameters.num_heads_);
        const uint32_t k_norm_count =
            static_cast<uint32_t>(parameters.batch_size_) *
            static_cast<uint32_t>(parameters.sequence_length_) *
            static_cast<uint32_t>(parameters.kv_num_heads_);
        ORT_RETURN_IF_ERROR(onnxruntime::webgpu::RunLayerNormProgram(
            context, query, q_norm_weight, /*bias=*/nullptr, qk_norm_epsilon_,
            q_norm_count, static_cast<int64_t>(parameters.head_size_),
            /*simplified=*/true, &qNorm, /*mean=*/nullptr, /*inv_std_dev=*/nullptr));
        ORT_RETURN_IF_ERROR(onnxruntime::webgpu::RunLayerNormProgram(
            context, key, k_norm_weight, /*bias=*/nullptr, qk_norm_epsilon_,
            k_norm_count, static_cast<int64_t>(parameters.head_size_),
            /*simplified=*/true, &kNorm, /*mean=*/nullptr, /*inv_std_dev=*/nullptr));
        q_for_rotary = &qNorm;
        k_for_rotary = &kNorm;
      } else if (decode_norm_fast_path) {
        q_norm_for_fused = q_norm_weight;
        k_norm_for_fused = k_norm_weight;
      }
      // rotary QK
      qRotary = context.CreateGPUTensor(q_for_rotary->DataType(), q_for_rotary->Shape());
      kRotary = context.CreateGPUTensor(k_for_rotary->DataType(), k_for_rotary->Shape());
      ORT_RETURN_IF_ERROR(RunFusedQKRotaryEmbedding(context, parameters,
                                                    q_for_rotary, k_for_rotary,
                                                    seqlen_k,
                                                    cos_cache, sin_cache,
                                                    &qRotary, &kRotary,
                                                    q_norm_for_fused, k_norm_for_fused,
                                                    qk_norm_epsilon_));
      query = &qRotary;
      key = &kRotary;
    } else if (has_qk_norm) {
      // Defensive: do_rotary_ guard above should make this unreachable, but keep it
      // explicit so a future schema/config drift surfaces as a clear error.
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, NOT_IMPLEMENTED,
          "GroupQueryAttention: q/k norm weights require do_rotary=1 (no rotary, no norm path).");
    }
  }

  if (will_use_flash_attention) {
    return ApplyFlashAttention(query, key, value, attention_bias, output, past_key, present_key, past_value,
                               present_value, parameters, context, seqlen_k, nullptr, nullptr, head_sink);
  }

  // Non-flash attention path does not support kv_sequence_length==0 (shared KV layers).
  if (kv_empty) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "WebGPU non-flash attention path does not support kv_sequence_length==0 (shared KV layers). "
                           "Flash attention is required for KV-shared decoder layers.");
  }

  TensorShapeVector q_new_dims({parameters.batch_size_, parameters.num_heads_,
                                parameters.sequence_length_, parameters.head_size_});
  TensorShape q_new_shape(q_new_dims);
  Tensor Q = context.CreateGPUTensor(query->DataType(), q_new_shape);
  ORT_RETURN_IF_ERROR(TransferBSDToBNSH(
      context, parameters.num_heads_, parameters.sequence_length_, parameters.head_size_, query, nullptr, 0, &Q));
  if (parameters.qkv_format_ == Q_K_V_BSNH_BNSH_BNSH) {  // key and value in BNSH format
    return ApplyAttention(&Q, key, value, attention_bias, past_key, past_value, output, present_key,
                          present_value, nullptr, parameters, context, head_sink, seqlen_k, local_window_size_);
  }

  TensorShapeVector k_new_dims({parameters.batch_size_, parameters.kv_num_heads_,
                                parameters.kv_sequence_length_, parameters.head_size_});
  TensorShape k_new_shape(k_new_dims);
  Tensor K = context.CreateGPUTensor(key->DataType(), k_new_shape);
  ORT_RETURN_IF_ERROR(TransferBSDToBNSH(context, parameters.kv_num_heads_, parameters.kv_sequence_length_,
                                        parameters.head_size_, key, nullptr, 0, &K));

  TensorShapeVector v_new_dims({parameters.batch_size_, parameters.kv_num_heads_,
                                parameters.kv_sequence_length_, parameters.v_head_size_});
  TensorShape v_new_shape(v_new_dims);
  Tensor V = context.CreateGPUTensor(value->DataType(), v_new_shape);
  ORT_RETURN_IF_ERROR(TransferBSDToBNSH(context, parameters.kv_num_heads_, parameters.kv_sequence_length_,
                                        parameters.v_head_size_, value, nullptr, 0, &V));
  return ApplyAttention(&Q, &K, &V, attention_bias, past_key, past_value, output, present_key,
                        present_value, nullptr, parameters, context, head_sink, seqlen_k, local_window_size_);
}

KernelCreateInfo CreateGroupQueryAttentionKernelInfo(bool enable_graph_capture) {
  KernelDefBuilder builder;
  builder.SetName("GroupQueryAttention")
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Provider(kWebGpuExecutionProvider)
      .TypeConstraint("T", WebGpuSupportedFloatTypes())
      .MayInplace(3, 1)
      .MayInplace(4, 2);

  // Only set InputMemoryType to CPU when graph capture is disabled
  if (!enable_graph_capture) {
    builder.InputMemoryType(OrtMemTypeCPUInput, 6);
  }

  return KernelCreateInfo(
      builder.Build(),
      [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
        out = std::make_unique<GroupQueryAttention>(info);
        return Status::OK();
      });
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
