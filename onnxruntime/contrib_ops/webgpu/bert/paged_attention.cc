// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <limits>

#include "contrib_ops/webgpu/bert/paged_attention.h"

#include "contrib_ops/cpu/bert/attention_parameters.h"
#include "contrib_ops/cpu/bert/paged_attention_helper.h"
#include "contrib_ops/webgpu/bert/attention_common.h"
#include "contrib_ops/webgpu/bert/flash_attention.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "core/common/logging/logging.h"
#include "core/framework/tensorprotoutils.h"
#include "core/providers/webgpu/webgpu_utils.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

// v1 registers only against T = float16, S = int32. See
// docs/design/webgpu_paged_attention.md §3 for the schema surface and
// §5 for the phased delivery plan.
//
// MayInplace hints the ORT framework that inputs 3/4 (key_cache/value_cache)
// may share buffers with outputs 1/2 (key_cache_out/value_cache_out). The
// framework honours this for OpTester and (in practice) for GenAI IO-binding,
// which is how the KV-cache aliasing contract is realised. Same pattern as
// GroupQueryAttention on both CUDA and WebGPU.
ONNX_OPERATOR_KERNEL_EX(
    PagedAttention,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>())
        .TypeConstraint("T_CACHE", DataTypeImpl::GetTensorType<MLFloat16>())
        .TypeConstraint("T_KV_SCALE", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("S", DataTypeImpl::GetTensorType<int32_t>())
        .MayInplace(3, 1)
        .MayInplace(4, 2),
    PagedAttention);

Status ScatterKVToPagedCacheProgram::GenerateShaderCode(ShaderHelper& sh) const {
  const auto& key = sh.AddInput("key", ShaderUsage::UseUniform);
  const auto& value = sh.AddInput("value", ShaderUsage::UseUniform);
  const auto& cumulative_sequence_length = sh.AddInput("cumulative_sequence_length", ShaderUsage::UseUniform);
  const auto& past_seqlens = sh.AddInput("past_seqlens", ShaderUsage::UseUniform);
  const auto& block_table = sh.AddInput("block_table", ShaderUsage::UseUniform);
  const auto& key_cache = sh.AddOutput("key_cache", ShaderUsage::UseUniform);
  const auto& value_cache = sh.AddOutput("value_cache", ShaderUsage::UseUniform);
  return WGSL_TEMPLATE_APPLY(sh, "bert/paged_attention_scatter_kv.wgsl.template",
                             WGSL_TEMPLATE_VARIABLE(block_table, block_table),
                             WGSL_TEMPLATE_VARIABLE(cumulative_sequence_length, cumulative_sequence_length),
                             WGSL_TEMPLATE_VARIABLE(key, key),
                             WGSL_TEMPLATE_VARIABLE(key_cache, key_cache),
                             WGSL_TEMPLATE_VARIABLE(past_seqlens, past_seqlens),
                             WGSL_TEMPLATE_VARIABLE(value, value),
                             WGSL_TEMPLATE_VARIABLE(value_cache, value_cache));
}

// Dispatch the scatter program.
//
// `key_cache_out` and `value_cache_out` are the ORT tensors written to by the
// program — because PagedAttention aliases input caches to output caches
// (validated at Compute time), these are actually the same GPU buffers as the
// cache inputs. Writes only touch the slots computed from block_table, so
// other entries in the cache remain intact.
static Status RunScatterKVToPagedCache(onnxruntime::webgpu::ComputeContext& context,
                                       const PagedAttentionParameters& parameters,
                                       const Tensor* key,
                                       const Tensor* value,
                                       const Tensor* cumulative_seqlens_q,
                                       const Tensor* past_seqlens,
                                       const Tensor* block_table,
                                       Tensor* key_cache_out,
                                       Tensor* value_cache_out) {
  const uint32_t token_count = static_cast<uint32_t>(parameters.token_count);
  const uint32_t batch_size = static_cast<uint32_t>(parameters.batch_size);
  const uint32_t kv_num_heads = static_cast<uint32_t>(parameters.kv_num_heads);
  const uint32_t head_size = static_cast<uint32_t>(parameters.head_size);
  const uint32_t block_size = static_cast<uint32_t>(parameters.block_size);
  const uint32_t dispatch_size = token_count * kv_num_heads * head_size;

  ScatterKVToPagedCacheProgram program{};
  program
      .AddInputs({
          {key, ProgramTensorMetadataDependency::TypeAndRank},
          {value, ProgramTensorMetadataDependency::TypeAndRank},
          {cumulative_seqlens_q, ProgramTensorMetadataDependency::TypeAndRank},
          {past_seqlens, ProgramTensorMetadataDependency::TypeAndRank},
          {block_table, ProgramTensorMetadataDependency::TypeAndRank},
      })
      .AddOutputs({
          {key_cache_out, ProgramTensorMetadataDependency::TypeAndRank},
          {value_cache_out, ProgramTensorMetadataDependency::TypeAndRank},
      })
      .AddUniformVariables({
          {token_count},
          {batch_size},
          {kv_num_heads},
          {head_size},
          {block_size},
          {dispatch_size},
      })
      .SetDispatchGroupSize((dispatch_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE);
  return context.RunProgram(program);
}

Status PagedAttentionRotaryProgram::GenerateShaderCode(ShaderHelper& sh) const {
  const auto& input = sh.AddInput("input", ShaderUsage::UseUniform);
  const auto& cos_cache = sh.AddInput("cos_cache", ShaderUsage::UseUniform);
  const auto& sin_cache = sh.AddInput("sin_cache", ShaderUsage::UseUniform);
  const auto& cumulative_sequence_length = sh.AddInput("cumulative_sequence_length", ShaderUsage::UseUniform);
  const auto& past_seqlens = sh.AddInput("past_seqlens", ShaderUsage::UseUniform);
  const auto& output = sh.AddOutput("output", ShaderUsage::UseUniform);
  return WGSL_TEMPLATE_APPLY(sh, "bert/paged_attention_rotary.wgsl.template",
                             WGSL_TEMPLATE_VARIABLE(cos_cache, cos_cache),
                             WGSL_TEMPLATE_VARIABLE(cumulative_sequence_length, cumulative_sequence_length),
                             WGSL_TEMPLATE_VARIABLE(input, input),
                             WGSL_TEMPLATE_VARIABLE(output, output),
                             WGSL_TEMPLATE_VARIABLE(past_seqlens, past_seqlens),
                             WGSL_TEMPLATE_VARIABLE(sin_cache, sin_cache));
}

// Dispatch the rotary embedding program for one 2-D packed token tensor
// (input : (token_count, n_heads * head_size)). Rotated Q and K each call
// this once with the appropriate `n_heads` (num_heads for Q, kv_num_heads
// for K). V is not rotated.
static Status RunRotaryEmbedding(onnxruntime::webgpu::ComputeContext& context,
                                 const PagedAttentionParameters& parameters,
                                 uint32_t n_heads,
                                 bool interleaved,
                                 const Tensor* input,
                                 const Tensor* cos_cache,
                                 const Tensor* sin_cache,
                                 const Tensor* cumulative_seqlens_q,
                                 const Tensor* past_seqlens,
                                 Tensor* output) {
  const uint32_t token_count = static_cast<uint32_t>(parameters.token_count);
  const uint32_t batch_size = static_cast<uint32_t>(parameters.batch_size);
  const uint32_t head_size = static_cast<uint32_t>(parameters.head_size);
  const uint32_t rotary_dim = static_cast<uint32_t>(parameters.rotary_dim);
  const uint32_t interleaved_u = interleaved ? 1u : 0u;
  const uint32_t dispatch_size = token_count * n_heads * head_size;

  PagedAttentionRotaryProgram program{};
  program
      .AddInputs({
          {input, ProgramTensorMetadataDependency::TypeAndRank},
          {cos_cache, ProgramTensorMetadataDependency::TypeAndRank},
          {sin_cache, ProgramTensorMetadataDependency::TypeAndRank},
          {cumulative_seqlens_q, ProgramTensorMetadataDependency::TypeAndRank},
          {past_seqlens, ProgramTensorMetadataDependency::TypeAndRank},
      })
      .AddOutputs({
          {output, ProgramTensorMetadataDependency::TypeAndRank},
      })
      .AddUniformVariables({
          {batch_size},
          {n_heads},
          {head_size},
          {rotary_dim},
          {interleaved_u},
          {dispatch_size},
      })
      .SetDispatchGroupSize((dispatch_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE);
  return context.RunProgram(program);
}

Status PagedAttentionSplitPackedQKVProgram::GenerateShaderCode(ShaderHelper& sh) const {
  const auto& input = sh.AddInput("input", ShaderUsage::UseUniform);
  const auto& q_out = sh.AddOutput("q_out", ShaderUsage::UseUniform);
  const auto& k_out = sh.AddOutput("k_out", ShaderUsage::UseUniform);
  const auto& v_out = sh.AddOutput("v_out", ShaderUsage::UseUniform);
  return WGSL_TEMPLATE_APPLY(sh, "bert/paged_attention_split_packed_qkv.wgsl.template",
                             WGSL_TEMPLATE_VARIABLE(input, input),
                             WGSL_TEMPLATE_VARIABLE(k_out, k_out),
                             WGSL_TEMPLATE_VARIABLE(q_out, q_out),
                             WGSL_TEMPLATE_VARIABLE(v_out, v_out));
}

// Dispatch the split-packed-QKV program: slice the packed query into three
// standalone Q, K, V tensors (each with the non-packed hidden layout) so the
// rest of the pipeline can consume them unchanged.
static Status RunSplitPackedQKV(onnxruntime::webgpu::ComputeContext& context,
                                const PagedAttentionParameters& parameters,
                                const Tensor* packed_qkv,
                                Tensor* q_out,
                                Tensor* k_out,
                                Tensor* v_out) {
  const uint32_t token_count = static_cast<uint32_t>(parameters.token_count);
  const uint32_t q_hidden_size = static_cast<uint32_t>(parameters.hidden_size);
  const uint32_t kv_hidden_size = static_cast<uint32_t>(parameters.kv_hidden_size);
  const uint32_t packed_hidden_size = q_hidden_size + 2u * kv_hidden_size;
  const uint32_t dispatch_size = token_count * packed_hidden_size;

  PagedAttentionSplitPackedQKVProgram program{};
  program
      .AddInputs({
          {packed_qkv, ProgramTensorMetadataDependency::TypeAndRank},
      })
      .AddOutputs({
          {q_out, ProgramTensorMetadataDependency::TypeAndRank},
          {k_out, ProgramTensorMetadataDependency::TypeAndRank},
          {v_out, ProgramTensorMetadataDependency::TypeAndRank},
      })
      .AddUniformVariables({
          {token_count},
          {q_hidden_size},
          {kv_hidden_size},
          {packed_hidden_size},
          {dispatch_size},
      })
      .SetDispatchGroupSize((dispatch_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE);
  return context.RunProgram(program);
}

Status PagedAttentionGatherKVProgram::GenerateShaderCode(ShaderHelper& sh) const {
  const auto& key_cache = sh.AddInput("key_cache", ShaderUsage::UseUniform);
  const auto& value_cache = sh.AddInput("value_cache", ShaderUsage::UseUniform);
  const auto& cumulative_sequence_length =
      sh.AddInput("cumulative_sequence_length", ShaderUsage::UseUniform);
  const auto& past_seqlens = sh.AddInput("past_seqlens", ShaderUsage::UseUniform);
  const auto& block_table = sh.AddInput("block_table", ShaderUsage::UseUniform);
  const auto& k_padded = sh.AddOutput("k_padded", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);
  const auto& v_padded = sh.AddOutput("v_padded", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);
  return WGSL_TEMPLATE_APPLY(sh, "bert/paged_attention_gather_kv.wgsl.template",
                             WGSL_TEMPLATE_VARIABLE(block_table, block_table),
                             WGSL_TEMPLATE_VARIABLE(cumulative_sequence_length, cumulative_sequence_length),
                             WGSL_TEMPLATE_VARIABLE(k_padded, k_padded),
                             WGSL_TEMPLATE_VARIABLE(key_cache, key_cache),
                             WGSL_TEMPLATE_VARIABLE(past_seqlens, past_seqlens),
                             WGSL_TEMPLATE_VARIABLE(v_padded, v_padded),
                             WGSL_TEMPLATE_VARIABLE(value_cache, value_cache));
}

// Materialize a padded BNSH (batch, kv_num_heads, max_kv_len, head_size)
// view of the paged K and V caches so ApplyFlashAttention can consume them
// as its present_key / present_value tensors. Slots beyond each batch's
// effective total_kv_len are zero-filled.
static Status RunGatherKV(onnxruntime::webgpu::ComputeContext& context,
                          const PagedAttentionParameters& parameters,
                          uint32_t max_kv_len,
                          const Tensor* key_cache,
                          const Tensor* value_cache,
                          const Tensor* cumulative_seqlens_q,
                          const Tensor* past_seqlens,
                          const Tensor* block_table,
                          Tensor* k_padded,
                          Tensor* v_padded) {
  const uint32_t batch_size = static_cast<uint32_t>(parameters.batch_size);
  const uint32_t kv_num_heads = static_cast<uint32_t>(parameters.kv_num_heads);
  const uint32_t head_size = static_cast<uint32_t>(parameters.head_size);
  const uint32_t block_size = static_cast<uint32_t>(parameters.block_size);
  const uint32_t dispatch_size = batch_size * kv_num_heads * max_kv_len * head_size;

  PagedAttentionGatherKVProgram program{};
  program
      .AddInputs({
          {key_cache, ProgramTensorMetadataDependency::TypeAndRank},
          {value_cache, ProgramTensorMetadataDependency::TypeAndRank},
          {cumulative_seqlens_q, ProgramTensorMetadataDependency::TypeAndRank},
          {past_seqlens, ProgramTensorMetadataDependency::TypeAndRank},
          {block_table, ProgramTensorMetadataDependency::TypeAndRank},
      })
      .AddOutputs({
          {k_padded, ProgramTensorMetadataDependency::TypeAndRank},
          {v_padded, ProgramTensorMetadataDependency::TypeAndRank},
      })
      .AddUniformVariables({
          {batch_size},
          {kv_num_heads},
          {head_size},
          {block_size},
          {max_kv_len},
          {dispatch_size},
      })
      .SetDispatchGroupSize((dispatch_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE);
  return context.RunProgram(program);
}

Status PagedAttentionUnpackQueryProgram::GenerateShaderCode(ShaderHelper& sh) const {
  const auto& input = sh.AddInput("input", ShaderUsage::UseUniform);
  const auto& cumulative_sequence_length =
      sh.AddInput("cumulative_sequence_length", ShaderUsage::UseUniform);
  const auto& output = sh.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);
  return WGSL_TEMPLATE_APPLY(sh, "bert/paged_attention_unpack_query.wgsl.template",
                             WGSL_TEMPLATE_VARIABLE(cumulative_sequence_length, cumulative_sequence_length),
                             WGSL_TEMPLATE_VARIABLE(input, input),
                             WGSL_TEMPLATE_VARIABLE(output, output));
}

// Unpack the packed varlen query tensor into a padded BSNH
// (batch, max_seqlen_q, num_heads, head_size) tensor for FlashAttention.
// Pad rows (s >= seq_len_b) are zero-filled; their FA outputs are discarded
// by the repack kernel.
static Status RunUnpackQuery(onnxruntime::webgpu::ComputeContext& context,
                             const PagedAttentionParameters& parameters,
                             uint32_t max_seqlen_q,
                             const Tensor* query,
                             const Tensor* cumulative_seqlens_q,
                             Tensor* q_padded) {
  const uint32_t num_heads = static_cast<uint32_t>(parameters.num_heads);
  const uint32_t head_size = static_cast<uint32_t>(parameters.head_size);
  const uint32_t batch_size = static_cast<uint32_t>(parameters.batch_size);
  const uint32_t dispatch_size = batch_size * max_seqlen_q * num_heads * head_size;

  PagedAttentionUnpackQueryProgram program{};
  program
      .AddInputs({
          {query, ProgramTensorMetadataDependency::TypeAndRank},
          {cumulative_seqlens_q, ProgramTensorMetadataDependency::TypeAndRank},
      })
      .AddOutputs({
          {q_padded, ProgramTensorMetadataDependency::TypeAndRank},
      })
      .AddUniformVariables({
          {num_heads},
          {head_size},
          {max_seqlen_q},
          {dispatch_size},
      })
      .SetDispatchGroupSize((dispatch_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE);
  return context.RunProgram(program);
}

Status PagedAttentionRepackOutputProgram::GenerateShaderCode(ShaderHelper& sh) const {
  const auto& input = sh.AddInput("input", ShaderUsage::UseUniform);
  const auto& cumulative_sequence_length =
      sh.AddInput("cumulative_sequence_length", ShaderUsage::UseUniform);
  const auto& output = sh.AddOutput("output", ShaderUsage::UseUniform);
  return WGSL_TEMPLATE_APPLY(sh, "bert/paged_attention_repack_output.wgsl.template",
                             WGSL_TEMPLATE_VARIABLE(cumulative_sequence_length, cumulative_sequence_length),
                             WGSL_TEMPLATE_VARIABLE(input, input),
                             WGSL_TEMPLATE_VARIABLE(output, output));
}

Status PagedAttentionPackMetadataProgram::GenerateShaderCode(ShaderHelper& sh) const {
  const auto& cumulative_sequence_length =
      sh.AddInput("cumulative_sequence_length", ShaderUsage::UseUniform);
  const auto& past_seqlens = sh.AddInput("past_seqlens", ShaderUsage::UseUniform);
  const auto& output = sh.AddOutput("output", ShaderUsage::UseUniform);
  return WGSL_TEMPLATE_APPLY(sh, "bert/paged_attention_pack_metadata.wgsl.template",
                             WGSL_TEMPLATE_VARIABLE(cumulative_sequence_length, cumulative_sequence_length),
                             WGSL_TEMPLATE_VARIABLE(output, output),
                             WGSL_TEMPLATE_VARIABLE(past_seqlens, past_seqlens));
}

static Status RunPackMetadata(onnxruntime::webgpu::ComputeContext& context,
                              uint32_t batch_size,
                              const Tensor* cumulative_sequence_length,
                              const Tensor* past_seqlens,
                              Tensor* output) {
  const uint32_t dispatch_size = batch_size + 1u;
  PagedAttentionPackMetadataProgram program{};
  program
      .AddInputs({
          {cumulative_sequence_length, ProgramTensorMetadataDependency::TypeAndRank},
          {past_seqlens, ProgramTensorMetadataDependency::TypeAndRank},
      })
      .AddOutputs({
          {output, ProgramTensorMetadataDependency::TypeAndRank},
      })
      .AddUniformVariables({
          {batch_size},
          {dispatch_size},
      })
      .SetDispatchGroupSize((dispatch_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE);
  return context.RunProgram(program);
}

// Inverse of RunUnpackQuery: pull the valid (s < seq_len_b) slots out of the
// padded BSNH attention output and write them into the packed varlen
// (token_count, hidden_size) layout PagedAttention's caller expects.
static Status RunRepackOutput(onnxruntime::webgpu::ComputeContext& context,
                              const PagedAttentionParameters& parameters,
                              const Tensor* padded_output,
                              const Tensor* cumulative_seqlens_q,
                              Tensor* output) {
  const uint32_t batch_size = static_cast<uint32_t>(parameters.batch_size);
  const uint32_t num_heads = static_cast<uint32_t>(parameters.num_heads);
  const uint32_t head_size = static_cast<uint32_t>(parameters.head_size);
  const uint32_t hidden_size = static_cast<uint32_t>(parameters.hidden_size);
  const uint32_t token_count = static_cast<uint32_t>(parameters.token_count);
  const uint32_t dispatch_size = token_count * hidden_size;

  PagedAttentionRepackOutputProgram program{};
  program
      .AddInputs({
          {padded_output, ProgramTensorMetadataDependency::TypeAndRank},
          {cumulative_seqlens_q, ProgramTensorMetadataDependency::TypeAndRank},
      })
      .AddOutputs({
          {output, ProgramTensorMetadataDependency::TypeAndRank},
      })
      .AddUniformVariables({
          {batch_size},
          {num_heads},
          {head_size},
          {hidden_size},
          {dispatch_size},
      })
      .SetDispatchGroupSize((dispatch_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE);
  return context.RunProgram(program);
}

PagedAttention::PagedAttention(const OpKernelInfo& info) : WebGpuKernel(info) {
  int64_t num_heads = 0;
  int64_t kv_num_heads = 0;
  ORT_ENFORCE(info.GetAttr("num_heads", &num_heads).IsOK() && num_heads > 0,
              "num_heads must be provided and > 0.");
  ORT_ENFORCE(info.GetAttr("kv_num_heads", &kv_num_heads).IsOK() &&
                  kv_num_heads > 0 && num_heads % kv_num_heads == 0,
              "kv_num_heads must be provided, > 0, and evenly divide num_heads.");
  num_heads_ = static_cast<int>(num_heads);
  kv_num_heads_ = static_cast<int>(kv_num_heads);
  local_window_size_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("local_window_size", -1));
  do_rotary_ = info.GetAttrOrDefault<int64_t>("do_rotary", 0) == 1;
  rotary_interleaved_ = info.GetAttrOrDefault<int64_t>("rotary_interleaved", 0) == 1;
  has_explicit_scale_ = info.GetAttr<float>("scale", &scale_).IsOK();
  if (!has_explicit_scale_) {
    scale_ = info.GetAttrOrDefault<float>("scale", 0.0f);
  }
  softcap_ = info.GetAttrOrDefault<float>("softcap", 0.0f);
  qk_norm_epsilon_ = info.GetAttrOrDefault<float>("qk_norm_epsilon", 1e-6f);
  k_quant_type_ = info.GetAttrOrDefault<std::string>("k_quant_type", "NONE");
  v_quant_type_ = info.GetAttrOrDefault<std::string>("v_quant_type", "NONE");
  k_cache_dtype_ = info.GetAttrOrDefault<std::string>("k_cache_dtype", "");
  v_cache_dtype_ = info.GetAttrOrDefault<std::string>("v_cache_dtype", "");
  kv_cache_layout_ = info.GetAttrOrDefault<std::string>("kv_cache_layout", "SEPARATE");
  v_head_size_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("v_head_size", 0));
  rotary_offset_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("rotary_offset", 0));
  use_smooth_softmax_ = info.GetAttrOrDefault<int64_t>("use_smooth_softmax", 0) == 1;
}

Status PagedAttention::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {
  // See docs/design/webgpu_paged_attention.md for the phased plan and reuse strategy.
  const Tensor* query = context.Input<Tensor>(0);
  const Tensor* key = context.Input<Tensor>(1);
  const Tensor* value = context.Input<Tensor>(2);
  const Tensor* key_cache = context.Input<Tensor>(3);
  const Tensor* value_cache = context.Input<Tensor>(4);
  const Tensor* cumulative_seqlens_q = context.Input<Tensor>(5);
  const Tensor* past_seqlens = context.Input<Tensor>(6);
  const Tensor* block_table = context.Input<Tensor>(7);
  const Tensor* cos_cache = context.InputCount() > 8 ? context.Input<Tensor>(8) : nullptr;
  const Tensor* sin_cache = context.InputCount() > 9 ? context.Input<Tensor>(9) : nullptr;
  const Tensor* slot_mapping = context.InputCount() > 10 ? context.Input<Tensor>(10) : nullptr;
  const Tensor* head_sink = context.InputCount() > 11 ? context.Input<Tensor>(11) : nullptr;
  const Tensor* q_norm_weight = context.InputCount() > 12 ? context.Input<Tensor>(12) : nullptr;
  const Tensor* k_norm_weight = context.InputCount() > 13 ? context.Input<Tensor>(13) : nullptr;
  const Tensor* k_scale = context.InputCount() > 14 ? context.Input<Tensor>(14) : nullptr;
  const Tensor* v_scale = context.InputCount() > 15 ? context.Input<Tensor>(15) : nullptr;
  const Tensor* attention_metadata = context.InputCount() > 16 ? context.Input<Tensor>(16) : nullptr;

  PagedAttentionParameters parameters{};
  const KVQuantizationType k_quant_type = StringToKVQuantizationType(k_quant_type_);
  const KVQuantizationType v_quant_type = StringToKVQuantizationType(v_quant_type_);
  const KVCacheDataType k_cache_dtype = StringToKVCacheDataType(k_cache_dtype_);
  const KVCacheDataType v_cache_dtype = StringToKVCacheDataType(v_cache_dtype_);
  const bool is_latent_kv = (kv_cache_layout_ == "LATENT");

  // 0 for max_threads_per_block disables the num_heads >= max_threads_per_block guard,
  // which is CUDA-block-size specific and not meaningful for WebGPU dispatch.
  ORT_RETURN_IF_ERROR(paged_attention_helper::CheckInputs(query,
                                                          key,
                                                          value,
                                                          key_cache,
                                                          value_cache,
                                                          cumulative_seqlens_q,
                                                          past_seqlens,
                                                          block_table,
                                                          cos_cache,
                                                          sin_cache,
                                                          slot_mapping,
                                                          head_sink,
                                                          q_norm_weight,
                                                          k_norm_weight,
                                                          k_scale,
                                                          v_scale,
                                                          attention_metadata,
                                                          &parameters,
                                                          num_heads_,
                                                          kv_num_heads_,
                                                          scale_,
                                                          softcap_,
                                                          qk_norm_epsilon_,
                                                          k_quant_type,
                                                          v_quant_type,
                                                          k_cache_dtype,
                                                          v_cache_dtype,
                                                          KVCacheDataType::FLOAT16,
                                                          is_latent_kv,
                                                          v_head_size_,
                                                          rotary_offset_,
                                                          has_explicit_scale_,
                                                          /*max_threads_per_block*/ 0));
  parameters.local_window_size = local_window_size_;
  parameters.do_rotary = do_rotary_;
  parameters.rotary_interleaved = rotary_interleaved_;

  // Feature guards. softcap and local_window_size are rejected until FA gains
  // the corresponding shader-side support (tracked in the design doc).
  if (softcap_ != 0.0f) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "PagedAttention (WebGPU): non-zero softcap is not supported yet.");
  }
  if (local_window_size_ != -1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "PagedAttention (WebGPU): local_window_size != -1 is not supported yet.");
  }
  if (kv_cache_layout_ != "SEPARATE") {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "PagedAttention (WebGPU): kv_cache_layout='", kv_cache_layout_,
                           "' is not supported yet.");
  }
  if (v_head_size_ != 0 && v_head_size_ != parameters.head_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "PagedAttention (WebGPU): v_head_size != head_size is not supported yet.");
  }
  if (rotary_offset_ != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "PagedAttention (WebGPU): rotary_offset != 0 is not supported yet.");
  }
  if (use_smooth_softmax_) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "PagedAttention (WebGPU): use_smooth_softmax is not supported yet.");
  }
  if (k_quant_type_ != "NONE" || v_quant_type_ != "NONE") {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "PagedAttention (WebGPU): quantized KV cache is not supported yet.");
  }
  if ((!k_cache_dtype_.empty() && k_cache_dtype_ != "float16") ||
      (!v_cache_dtype_.empty() && v_cache_dtype_ != "float16")) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "PagedAttention (WebGPU): only float16 KV cache dtype is supported.");
  }
  if (context.KvCacheQuantizationEnabled()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "PagedAttention (WebGPU): KV cache quantization is not supported yet.");
  }
  if (slot_mapping != nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "PagedAttention (WebGPU): slot_mapping input is not supported yet.");
  }
  if (head_sink != nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "PagedAttention (WebGPU): head_sink input is not supported yet.");
  }
  if (q_norm_weight != nullptr || k_norm_weight != nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "PagedAttention (WebGPU): q_norm_weight/k_norm_weight inputs are not supported yet.");
  }
  if (k_scale != nullptr || v_scale != nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "PagedAttention (WebGPU): k_scale/v_scale inputs are not supported yet.");
  }
  if (attention_metadata != nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "PagedAttention (WebGPU): attention_metadata input is not supported yet.");
  }

  if (do_rotary_ && (cos_cache == nullptr || sin_cache == nullptr)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "cos_cache and sin_cache must be passed to PagedAttention when do_rotary = 1");
  }

  // Output 0: packed attention output, shape [token_count, q_hidden_size].
  TensorShapeVector output_shape{static_cast<int64_t>(parameters.token_count),
                                 static_cast<int64_t>(parameters.hidden_size)};
  Tensor* output = context.Output(0, output_shape);

  // Outputs 1 and 2 (key_cache_out, value_cache_out) are optional in the
  // schema. If both are omitted, update the input caches in place, matching
  // CUDA. Otherwise both must be supplied.
  //
  // MayInplace(3, 1) and MayInplace(4, 2) on the kernel def let the ORT
  // allocation planner reuse the input buffers as output buffers when it can.
  // The planner requires UseCount(input) == 1 (see allocation_planner.cc), so
  // this fires for typical graph-internal edges but NOT when key_cache /
  // value_cache come in as graph inputs (as in OpTester-based unit tests).
  //
  // For the non-aliased case we fall back to copying the input caches into
  // freshly allocated output cache buffers before the scatter runs, so the
  // untouched slots in the output correctly reflect the past cache contents.
  // This mirrors the past-vs-present handling in GroupQueryAttention.
  TensorShapeVector cache_shape{static_cast<int64_t>(parameters.num_blocks),
                                static_cast<int64_t>(parameters.block_size),
                                static_cast<int64_t>(parameters.kv_num_heads),
                                static_cast<int64_t>(parameters.head_size)};
  Tensor* key_cache_out = context.OutputCount() > 1 ? context.Output(1, cache_shape) : nullptr;
  Tensor* value_cache_out = context.OutputCount() > 2 ? context.Output(2, cache_shape) : nullptr;
  ORT_RETURN_IF((key_cache_out == nullptr) != (value_cache_out == nullptr),
                "PagedAttention (WebGPU): key_cache_out and value_cache_out must be both present or both absent.");
  if (key_cache_out == nullptr) {
    key_cache_out = const_cast<Tensor*>(key_cache);
    value_cache_out = const_cast<Tensor*>(value_cache);
  }

  const bool key_cache_aliased = (key_cache->DataRaw() == key_cache_out->MutableDataRaw());
  const bool value_cache_aliased = (value_cache->DataRaw() == value_cache_out->MutableDataRaw());

  // Non-aliased path (OpTester and any future non-IO-bound consumer): materialize
  // the input caches into the freshly allocated output tensors so untouched slots
  // are preserved through the op. Done before the empty-query fast path so the
  // outputs are correctly initialized even when there is no scatter work to do.
  if (!key_cache_aliased || !value_cache_aliased) {
    LOGS_DEFAULT(WARNING) << "PagedAttention (WebGPU): cache outputs are not aliased with cache inputs; "
                             "falling back to a GPU cache copy. Configure IO-binding to alias the cache "
                             "buffers in production to avoid this per-run copy.";
  }
  if (!key_cache_aliased) {
    ORT_RETURN_IF_ERROR(context.CopyTensor(*key_cache, *key_cache_out));
  }
  if (!value_cache_aliased) {
    ORT_RETURN_IF_ERROR(context.CopyTensor(*value_cache, *value_cache_out));
  }

  // Empty-query fast path: output is [0, hidden_size] and the cache outputs
  // already reflect the input caches (either via aliasing or via the copies
  // above), so there is no kernel work to do.
  if (parameters.token_count == 0) {
    return Status::OK();
  }

  // Packed-QKV: materialize three standalone Q/K/V tensors from the packed
  // input so the rest of the routine can use the non-packed path unchanged.
  // The CUDA kernel handles packed-QKV as an addressing mode inside its
  // rotary/scatter kernels; we trade some bandwidth for kernel reuse.
  Tensor packed_q_tensor;
  Tensor packed_k_tensor;
  Tensor packed_v_tensor;
  if (parameters.is_packed_qkv) {
    const auto* dtype = query->DataType();
    packed_q_tensor = context.CreateGPUTensor(
        dtype, TensorShape({parameters.token_count, parameters.hidden_size}));
    packed_k_tensor = context.CreateGPUTensor(
        dtype, TensorShape({parameters.token_count, parameters.kv_hidden_size}));
    packed_v_tensor = context.CreateGPUTensor(
        dtype, TensorShape({parameters.token_count, parameters.kv_hidden_size}));
    ORT_RETURN_IF_ERROR(RunSplitPackedQKV(context, parameters, query,
                                          &packed_q_tensor, &packed_k_tensor,
                                          &packed_v_tensor));
    // Re-point the local Q/K/V so the rest of the routine sees the
    // non-packed layout and needs no further branching.
    query = &packed_q_tensor;
    key = &packed_k_tensor;
    value = &packed_v_tensor;
  }

  // Fallback attention: gather paged K/V into padded BNSH, unpack varlen Q
  // into LEFT-aligned padded BSNH, dispatch ApplyFlashAttention, then repack.
  // See docs/design/webgpu_paged_attention.md §4.
  // Pack the two int32 metadata tensors, then perform one D→H sync to derive
  // max_seqlen_q, max_kv_len, and the per-batch seqlen_k / seqlens_q values.
  const auto* int32_type = DataTypeImpl::GetType<int32_t>();
  const int64_t batch_size_i64 = static_cast<int64_t>(parameters.batch_size);
  const int64_t packed_metadata_size = 2 * batch_size_i64 + 1;

  Tensor packed_metadata_gpu = context.CreateGPUTensor(
      int32_type, TensorShape({packed_metadata_size}));
  ORT_RETURN_IF_ERROR(RunPackMetadata(context, static_cast<uint32_t>(parameters.batch_size),
                                      cumulative_seqlens_q, past_seqlens,
                                      &packed_metadata_gpu));

  Tensor packed_metadata_cpu = context.CreateCPUTensor(
      int32_type, TensorShape({packed_metadata_size}));
  ORT_RETURN_IF_ERROR(context.CopyTensor(packed_metadata_gpu, packed_metadata_cpu));
  const int32_t* cum_ptr = packed_metadata_cpu.Data<int32_t>();
  const int32_t* past_ptr = cum_ptr + batch_size_i64 + 1;

  // Compute per-batch effective lengths and the tightest max_seqlen_q /
  // max_kv_len bounds. FA's seqlens_k convention is the LAST VALID KV INDEX
  // (0-based), so entry b is (past + q_len - 1); the shader reads it back as
  // u32(seqlens_k[b]) + 1u. seqlens_q is the raw per-batch new-Q length.
  Tensor seqlen_k_cpu = context.CreateCPUTensor(int32_type, TensorShape({batch_size_i64}));
  int32_t* seqlen_k_ptr = seqlen_k_cpu.MutableData<int32_t>();
  Tensor seqlens_q_cpu = context.CreateCPUTensor(int32_type, TensorShape({batch_size_i64}));
  int32_t* seqlens_q_ptr = seqlens_q_cpu.MutableData<int32_t>();
  int32_t max_seqlen_q_i = 0;
  int32_t max_kv_len_i = 0;
  const int64_t cache_capacity = static_cast<int64_t>(parameters.block_size) *
                                 static_cast<int64_t>(parameters.max_num_blocks_per_seq);
  if (cum_ptr[0] != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "PagedAttention (WebGPU): cumulative_sequence_length must start at 0.");
  }
  for (int b = 0; b < parameters.batch_size; ++b) {
    const int64_t cum_lo = static_cast<int64_t>(cum_ptr[b]);
    const int64_t cum_hi = static_cast<int64_t>(cum_ptr[b + 1]);
    if (cum_hi < cum_lo) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "PagedAttention (WebGPU): cumulative_sequence_length must be non-decreasing.");
    }
    const int64_t q_len = cum_hi - cum_lo;
    const int64_t past_len = static_cast<int64_t>(past_ptr[b]);
    if (past_len < 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "PagedAttention (WebGPU): past_seqlens must be non-negative.");
    }
    const int64_t total_kv_len = past_len + q_len;
    if (total_kv_len > cache_capacity) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "PagedAttention (WebGPU): past_seqlens + query length exceeds the KV cache capacity.");
    }
    if (total_kv_len > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "PagedAttention (WebGPU): total KV sequence length exceeds int32 range.");
    }
    // Keep -1 when total_kv_len is zero: the shader adds 1 after converting
    // this last-valid-index sentinel to u32, intentionally producing zero.
    seqlen_k_ptr[b] = static_cast<int32_t>(total_kv_len - 1);
    seqlens_q_ptr[b] = static_cast<int32_t>(q_len);  // Raw per-batch new-Q length.
    if (q_len > max_seqlen_q_i) {
      max_seqlen_q_i = static_cast<int32_t>(q_len);
    }
    if (total_kv_len > max_kv_len_i) {
      max_kv_len_i = static_cast<int32_t>(total_kv_len);
    }
  }
  if (cum_ptr[parameters.batch_size] != parameters.token_count) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "PagedAttention (WebGPU): cumulative_sequence_length must end at token_count.");
  }
  const uint32_t max_seqlen_q = static_cast<uint32_t>(max_seqlen_q_i);
  const uint32_t max_kv_len = static_cast<uint32_t>(max_kv_len_i);

  if (do_rotary_) {
    const int64_t required_cache_length = static_cast<int64_t>(max_kv_len);
    if (cos_cache->Shape()[0] < required_cache_length) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "PagedAttention (WebGPU): cos_cache dimension 0 must be at least max KV length ",
                             required_cache_length, ".");
    }
    if (sin_cache->Shape()[0] < required_cache_length) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "PagedAttention (WebGPU): sin_cache dimension 0 must be at least max KV length ",
                             required_cache_length, ".");
    }
  }

  // Empty-attention fast path.
  if (max_seqlen_q == 0 || max_kv_len == 0) {
    context.FillZero(*output);
    return Status::OK();
  }

  const uint64_t max_storage_buffer_binding_size = context.DeviceLimits().maxStorageBufferBindingSize;
  const uint64_t kv_padded_bytes = static_cast<uint64_t>(parameters.batch_size) *
                                   static_cast<uint64_t>(parameters.kv_num_heads) *
                                   static_cast<uint64_t>(max_kv_len) *
                                   static_cast<uint64_t>(parameters.head_size) * sizeof(MLFloat16);
  const uint64_t q_padded_bytes = static_cast<uint64_t>(parameters.batch_size) *
                                  static_cast<uint64_t>(max_seqlen_q) *
                                  static_cast<uint64_t>(parameters.hidden_size) * sizeof(MLFloat16);
  if (kv_padded_bytes > max_storage_buffer_binding_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "PagedAttention (WebGPU): k_padded/v_padded scratch requires ",
                           kv_padded_bytes, " bytes, exceeding maxStorageBufferBindingSize of ",
                           max_storage_buffer_binding_size, ".");
  }
  if (q_padded_bytes > max_storage_buffer_binding_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "PagedAttention (WebGPU): q_padded/output_padded scratch requires ",
                           q_padded_bytes, " bytes, exceeding maxStorageBufferBindingSize of ",
                           max_storage_buffer_binding_size, ".");
  }

  // Fused rotary path. Rotate Q and K into scratch tensors, then scatter the
  // rotated K + untouched V into the paged cache. Metadata validation above
  // must complete before either shader can use device-derived cache positions.
  const Tensor* query_for_fa = query;
  const Tensor* key_for_scatter = key;
  Tensor rotated_query_tensor;
  Tensor rotated_key_tensor;
  if (do_rotary_) {
    rotated_query_tensor = context.CreateGPUTensor(query->DataType(), query->Shape());
    ORT_RETURN_IF_ERROR(RunRotaryEmbedding(context, parameters,
                                           static_cast<uint32_t>(parameters.num_heads),
                                           rotary_interleaved_,
                                           query, cos_cache, sin_cache,
                                           cumulative_seqlens_q, past_seqlens,
                                           &rotated_query_tensor));
    query_for_fa = &rotated_query_tensor;

    rotated_key_tensor = context.CreateGPUTensor(key->DataType(), key->Shape());
    ORT_RETURN_IF_ERROR(RunRotaryEmbedding(context, parameters,
                                           static_cast<uint32_t>(parameters.kv_num_heads),
                                           rotary_interleaved_,
                                           key, cos_cache, sin_cache,
                                           cumulative_seqlens_q, past_seqlens,
                                           &rotated_key_tensor));
    key_for_scatter = &rotated_key_tensor;
  }

  ORT_RETURN_IF_ERROR(RunScatterKVToPagedCache(context, parameters, key_for_scatter, value,
                                               cumulative_seqlens_q, past_seqlens,
                                               block_table, key_cache_out, value_cache_out));

  const auto* dtype = query->DataType();

  Tensor seqlen_k_gpu = context.CreateGPUTensor(int32_type, TensorShape({batch_size_i64}));
  ORT_RETURN_IF_ERROR(context.CopyTensor(seqlen_k_cpu, seqlen_k_gpu));

  Tensor seqlens_q_gpu = context.CreateGPUTensor(int32_type, TensorShape({batch_size_i64}));
  ORT_RETURN_IF_ERROR(context.CopyTensor(seqlens_q_cpu, seqlens_q_gpu));

  Tensor k_padded = context.CreateGPUTensor(
      dtype, TensorShape({batch_size_i64,
                          static_cast<int64_t>(parameters.kv_num_heads),
                          static_cast<int64_t>(max_kv_len),
                          static_cast<int64_t>(parameters.head_size)}));
  Tensor v_padded = context.CreateGPUTensor(
      dtype, TensorShape({batch_size_i64,
                          static_cast<int64_t>(parameters.kv_num_heads),
                          static_cast<int64_t>(max_kv_len),
                          static_cast<int64_t>(parameters.head_size)}));
  Tensor q_padded = context.CreateGPUTensor(
      dtype, TensorShape({batch_size_i64,
                          static_cast<int64_t>(max_seqlen_q),
                          static_cast<int64_t>(parameters.num_heads),
                          static_cast<int64_t>(parameters.head_size)}));
  Tensor output_padded = context.CreateGPUTensor(
      dtype, TensorShape({batch_size_i64,
                          static_cast<int64_t>(max_seqlen_q),
                          static_cast<int64_t>(parameters.num_heads),
                          static_cast<int64_t>(parameters.head_size)}));

  // Gather paged K/V into padded BNSH. The gather reads from the just-scattered
  // key_cache_out / value_cache_out so it sees new tokens + past cache.
  ORT_RETURN_IF_ERROR(RunGatherKV(context, parameters, max_kv_len,
                                  key_cache_out, value_cache_out,
                                  cumulative_seqlens_q, past_seqlens,
                                  block_table, &k_padded, &v_padded));

  ORT_RETURN_IF_ERROR(RunUnpackQuery(context, parameters, max_seqlen_q,
                                     query_for_fa, cumulative_seqlens_q, &q_padded));

  // WebgpuAttentionParameters via the GQA constructor so is_gqa_ is set.
  // kv_sequence_length = 0 triggers FA's kv_empty aliasing path (K=V=nullptr;
  // FA aliases present_key/value to past_key/value).
  GroupQueryAttentionParameters gqa_params{};
  gqa_params.batch_size = parameters.batch_size;
  gqa_params.sequence_length = static_cast<int>(max_seqlen_q);
  gqa_params.kv_sequence_length = 0;
  gqa_params.total_sequence_length = static_cast<int>(max_kv_len);
  gqa_params.hidden_size = parameters.hidden_size;
  gqa_params.head_size = parameters.head_size;
  gqa_params.v_hidden_size = parameters.kv_hidden_size;
  gqa_params.v_head_size = parameters.head_size;
  gqa_params.num_heads = parameters.num_heads;
  gqa_params.is_unidirectional = true;
  gqa_params.past_present_share_buffer = false;
  gqa_params.do_rotary = false;  // Q/K already rotated above.
  gqa_params.scale = parameters.scale;
  gqa_params.kv_num_heads = parameters.kv_num_heads;
  gqa_params.kv_hidden_size = parameters.kv_hidden_size;
  gqa_params.seqlen_past_kv_cache = 0;
  gqa_params.seqlen_present_kv_cache = static_cast<int>(max_kv_len);
  gqa_params.qkv_format = Q_K_V_BSNH;
  gqa_params.mask_type = MASK_NONE;
  WebgpuAttentionParameters fa_params(gqa_params);
  ORT_RETURN_IF_NOT(CanApplyFlashAttention(fa_params, context),
                    "PagedAttention (WebGPU): input configuration is not supported by FlashAttention.");

  ORT_RETURN_IF_ERROR(ApplyFlashAttention(
      &q_padded,
      /*K=*/nullptr, /*V=*/nullptr, /*attention_bias=*/nullptr,
      &output_padded,
      /*past_key=*/&k_padded, /*present_key=*/nullptr,
      /*past_value=*/&v_padded, /*present_value=*/nullptr,
      fa_params, context, &seqlen_k_gpu,
      /*cos_cache=*/nullptr, /*sin_cache=*/nullptr, /*head_sink=*/nullptr,
      /*total_seqlen=*/nullptr, /*seqlens_q=*/&seqlens_q_gpu));

  ORT_RETURN_IF_ERROR(RunRepackOutput(context, parameters, &output_padded,
                                      cumulative_seqlens_q, output));

  return Status::OK();
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
