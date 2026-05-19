// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/bert/turboquant_attention.h"
#include "contrib_ops/webgpu/bert/flash_attention.h"
#include "contrib_ops/webgpu/bert/attention_common.h"
#include "core/platform/env_var.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

namespace {

// Slot byte sizes for the packed cache layout — must match Option A's
// `slot_bytes` calculation in turboquant_kv_fusion.cc.
//   K slot: ceil(D * key_bits / 8) bytes + 2 bytes (vec_norm fp16)
//   V slot: ceil(D * value_bits / 8) bytes + 4 bytes (v_scale fp16 + v_zero fp16)
inline uint32_t KeySlotBytes(uint32_t head_dim, uint32_t key_bits) {
  return ((head_dim * key_bits) + 7) / 8 + 2;
}
inline uint32_t ValueSlotBytes(uint32_t head_dim, uint32_t value_bits) {
  return ((head_dim * value_bits) + 7) / 8 + 4;
}
inline uint32_t SlotBytes(uint32_t head_dim, uint32_t key_bits, uint32_t value_bits) {
  uint32_t k = KeySlotBytes(head_dim, key_bits);
  uint32_t v = ValueSlotBytes(head_dim, value_bits);
  return k > v ? k : v;
}

}  // namespace

Status TurboQuantEncodeProgram::GenerateShaderCode(ShaderHelper& sh) const {
  // Bindings — the AddInput/AddOutput calls register the storage buffers
  // with the shader; the WGSL template references them by name.  Order must
  // match what the template expects (the WGSL emitter binds in declaration
  // order).
  sh.AddInput("K_in", ShaderUsage::UseUniform);
  sh.AddInput("V_in", ShaderUsage::UseUniform);
  sh.AddInput("k_codebook", ShaderUsage::UseUniform);
  // Past cache (uint32 view of past_key / past_value).  The shader's
  // copy-past branch reads from these; for prompt step (past_seq=0) the
  // shader never indexes them but the bindings must still exist.
  sh.AddInput("k_cache_past", ShaderUsage::UseUniform);
  sh.AddInput("v_cache_past", ShaderUsage::UseUniform);
  sh.AddOutput("k_cache", ShaderUsage::UseUniform);
  sh.AddOutput("v_cache", ShaderUsage::UseUniform);

  uint32_t slot_bytes = SlotBytes(head_dim_, key_bits_, value_bits_);
  // u32 view of the cache; slot must be 4-byte aligned for clean indexing.
  uint32_t slot_u32 = (slot_bytes + 3) / 4;
  uint32_t key_packed_u32 = ((head_dim_ * key_bits_) / 8 + 3) / 4;
  uint32_t value_packed_u32 = ((head_dim_ * value_bits_) / 8 + 3) / 4;

  // Parameters must be in ALPHABETICAL order (the wgsl-gen tool emits the
  // struct fields alphabetically).  norm_correction is not used in the
  // encode template, so it's not part of the param struct.
  return WGSL_TEMPLATE_APPLY(
      sh, "bert/turboquant_encode.wgsl.template",
      WGSL_TEMPLATE_PARAMETER(head_dim, head_dim_),
      WGSL_TEMPLATE_PARAMETER(key_bits, key_bits_),
      WGSL_TEMPLATE_PARAMETER(key_packed_u32, key_packed_u32),
      WGSL_TEMPLATE_PARAMETER(slot_u32, slot_u32),
      WGSL_TEMPLATE_PARAMETER(value_bits, value_bits_),
      WGSL_TEMPLATE_PARAMETER(value_packed_u32, value_packed_u32));
}

Status TurboQuantDecodeProgram::GenerateShaderCode(ShaderHelper& sh) const {
  sh.AddInput("k_cache", ShaderUsage::UseUniform);
  sh.AddInput("v_cache", ShaderUsage::UseUniform);
  sh.AddInput("k_codebook", ShaderUsage::UseUniform);
  sh.AddOutput("K_out", ShaderUsage::UseUniform);
  sh.AddOutput("V_out", ShaderUsage::UseUniform);

  uint32_t slot_bytes = SlotBytes(head_dim_, key_bits_, value_bits_);
  uint32_t slot_u32 = (slot_bytes + 3) / 4;
  uint32_t key_packed_u32 = ((head_dim_ * key_bits_) / 8 + 3) / 4;
  uint32_t value_packed_u32 = ((head_dim_ * value_bits_) / 8 + 3) / 4;

  return WGSL_TEMPLATE_APPLY(
      sh, "bert/turboquant_decode.wgsl.template",
      WGSL_TEMPLATE_PARAMETER(head_dim, head_dim_),
      WGSL_TEMPLATE_PARAMETER(key_bits, key_bits_),
      WGSL_TEMPLATE_PARAMETER(key_packed_u32, key_packed_u32),
      WGSL_TEMPLATE_PARAMETER(norm_correction, norm_correction_ ? 1u : 0u),
      WGSL_TEMPLATE_PARAMETER(slot_u32, slot_u32),
      WGSL_TEMPLATE_PARAMETER(value_bits, value_bits_),
      WGSL_TEMPLATE_PARAMETER(value_packed_u32, value_packed_u32));
}

Status RunTurboQuantAttention(onnxruntime::webgpu::ComputeContext& context,
                              const WebgpuAttentionParameters& params,
                              const Tensor* query,
                              const Tensor* key,
                              const Tensor* value,
                              const Tensor* past_key,
                              const Tensor* past_value,
                              const Tensor* k_codebook,
                              const Tensor* hadamard,
                              const Tensor* attention_bias,
                              const Tensor* head_sink,
                              const Tensor* seqlen_k,
                              const Tensor* cos_cache,
                              const Tensor* sin_cache,
                              Tensor* present_key,
                              Tensor* present_value,
                              Tensor* output,
                              uint32_t key_bits,
                              uint32_t value_bits,
                              bool norm_correction,
                              int local_window_size) {
  ORT_UNUSED_PARAMETER(hadamard);
  ORT_UNUSED_PARAMETER(seqlen_k);  // we use params.seqlen_present_kv_cache_
  ORT_UNUSED_PARAMETER(local_window_size);

  const uint32_t head_dim = static_cast<uint32_t>(params.head_size_);
  const uint32_t batch_size = static_cast<uint32_t>(params.batch_size_);
  const uint32_t kv_num_heads = static_cast<uint32_t>(params.kv_num_heads_);
  const uint32_t new_seq_len = static_cast<uint32_t>(params.sequence_length_);
  const uint32_t past_seq_len = static_cast<uint32_t>(params.seqlen_past_kv_cache_);
  const uint32_t total_seq_len = static_cast<uint32_t>(params.total_sequence_length_);
  const uint32_t max_seq_len = static_cast<uint32_t>(params.seqlen_present_kv_cache_);

  // ORT WebGPU's ShaderVariableHelper rejects uint8 storage bindings (no WGSL
  // type maps to a single u8).  Alias the packed uint8 cache as uint32 view
  // tensors with shape (..., slot_u32) so bindings map to `array<u32>` — which
  // is exactly how the WGSL templates read them (`k_cache[i]` returns u32).
  const uint32_t k_slot_bytes = (head_dim * key_bits + 7) / 8 + 2;
  const uint32_t v_slot_bytes = (head_dim * value_bits + 7) / 8 + 4;
  const uint32_t slot_bytes = k_slot_bytes > v_slot_bytes ? k_slot_bytes : v_slot_bytes;
  const int64_t slot_u32 = static_cast<int64_t>((slot_bytes + 3) / 4);
  const TensorShape present_u32_shape{
      static_cast<int64_t>(batch_size),
      static_cast<int64_t>(kv_num_heads),
      static_cast<int64_t>(max_seq_len),
      slot_u32,
  };
  Tensor k_cache_present_u32(DataTypeImpl::GetType<uint32_t>(), present_u32_shape,
                             present_key->MutableDataRaw(), present_key->Location());
  Tensor v_cache_present_u32(DataTypeImpl::GetType<uint32_t>(), present_u32_shape,
                             present_value->MutableDataRaw(), present_value->Location());

  // Past cache uint32 view.  WGPU rejects (a) zero-sized storage bindings and
  // (b) the same buffer being bound as both readonly-input and writable-output
  // in one pass.  So for prompt step (past_seq_len = 0) we allocate a tiny
  // dummy uint32 buffer that the shader's copy branch never indexes.
  const int64_t past_dim2 = static_cast<int64_t>(past_seq_len > 0 ? past_seq_len : 1);
  const TensorShape past_u32_shape{
      static_cast<int64_t>(batch_size),
      static_cast<int64_t>(kv_num_heads),
      past_dim2,
      slot_u32,
  };
  const bool have_past = past_seq_len > 0 && past_key != nullptr && past_value != nullptr;
  Tensor dummy_past_k, dummy_past_v;
  if (!have_past) {
    dummy_past_k = context.CreateGPUTensor(DataTypeImpl::GetType<uint32_t>(), past_u32_shape);
    dummy_past_v = context.CreateGPUTensor(DataTypeImpl::GetType<uint32_t>(), past_u32_shape);
  }
  Tensor k_cache_past_u32(DataTypeImpl::GetType<uint32_t>(), past_u32_shape,
                          have_past ? const_cast<void*>(past_key->DataRaw())
                                    : dummy_past_k.MutableDataRaw(),
                          have_past ? past_key->Location() : dummy_past_k.Location());
  Tensor v_cache_past_u32(DataTypeImpl::GetType<uint32_t>(), past_u32_shape,
                          have_past ? const_cast<void*>(past_value->DataRaw())
                                    : dummy_past_v.MutableDataRaw(),
                          have_past ? past_value->Location() : dummy_past_v.Location());

  // Step 1: encode (with copy-past + encode-fresh fused).  After this runs,
  // present_key / present_value contain the full packed cache for this step.
  TurboQuantEncodeProgram encode{head_dim, key_bits, value_bits, norm_correction};
  encode.AddInputs({
      {key, ProgramTensorMetadataDependency::TypeAndRank},
      {value, ProgramTensorMetadataDependency::TypeAndRank},
      {k_codebook, ProgramTensorMetadataDependency::TypeAndRank},
      {&k_cache_past_u32, ProgramTensorMetadataDependency::TypeAndRank},
      {&v_cache_past_u32, ProgramTensorMetadataDependency::TypeAndRank},
  });
  encode.AddOutputs({
      {&k_cache_present_u32, ProgramTensorMetadataDependency::TypeAndRank},
      {&v_cache_present_u32, ProgramTensorMetadataDependency::TypeAndRank},
  });
  encode.SetWorkgroupSize(head_dim);
  encode.SetDispatchGroupSize(batch_size * kv_num_heads * total_seq_len);
  encode.AddUniformVariables({
      {batch_size},
      {new_seq_len},
      {kv_num_heads},
      {max_seq_len},
      {past_seq_len},
  });
  ORT_RETURN_IF_ERROR(context.RunProgram(encode));

  // ApplyFlashAttention requires WebGPU subgroups (Chrome 136+).  When that
  // feature is missing — Safari WebKit, Firefox today, older Chrome — we
  // route through the non-FA `ApplyAttention` path instead.  The fallback
  // costs us the fused-attention speedup but keeps TurboQuant correctness
  // intact, and it's the only browser path that exists for those engines.
  const bool has_subgroups = context.HasFeature(wgpu::FeatureName::Subgroups);
  // Escape hatch: set ORT_TQ_DISABLE_FA=1 to force the ApplyAttention
  // fallback even on subgroups-capable adapters.  Useful for diagnosing
  // the Safari / Firefox code path on dev machines that have Chrome's
  // subgroups feature.  Does not change KV-cache layout or quality —
  // only routes scoring through the non-flash kernels.
  // Use ORT's portable env reader instead of std::getenv — MSVC treats the
  // latter as deprecated under /sdl, which fails the build with -Werror.
  const bool force_fallback = !onnxruntime::detail::GetEnvironmentVar("ORT_TQ_DISABLE_FA").empty();
  const bool use_fa = has_subgroups && !force_fallback;

  // Step 2: prompt step uses Option ε — run standard FA on fresh fp16 K/V
  // (when subgroups are available).  The cache is now correctly populated
  // for the next step regardless of which attention kernel we pick.
  if (past_seq_len == 0) {
    if (use_fa) {
      return ApplyFlashAttention(query, key, value, attention_bias, output,
                                 nullptr, nullptr, nullptr, nullptr,
                                 params, context, /*seqlens_k=*/nullptr,
                                 cos_cache, sin_cache, head_sink);
    }
    // No-subgroups fallback: transfer Q/K/V to BNSH and call ApplyAttention.
    const TensorShape q_bnsh{
        static_cast<int64_t>(batch_size),
        static_cast<int64_t>(params.num_heads_),
        static_cast<int64_t>(new_seq_len),
        static_cast<int64_t>(head_dim),
    };
    const TensorShape kv_bnsh{
        static_cast<int64_t>(batch_size),
        static_cast<int64_t>(kv_num_heads),
        static_cast<int64_t>(new_seq_len),
        static_cast<int64_t>(head_dim),
    };
    Tensor q_bnsh_t = context.CreateGPUTensor(query->DataType(), q_bnsh);
    Tensor k_bnsh_t = context.CreateGPUTensor(key->DataType(), kv_bnsh);
    Tensor v_bnsh_t = context.CreateGPUTensor(value->DataType(), kv_bnsh);
    ORT_RETURN_IF_ERROR(TransferBSDToBNSH(
        context, params.num_heads_, new_seq_len, head_dim, query, nullptr, 0, &q_bnsh_t));
    ORT_RETURN_IF_ERROR(TransferBSDToBNSH(
        context, kv_num_heads, new_seq_len, head_dim, key, nullptr, 0, &k_bnsh_t));
    ORT_RETURN_IF_ERROR(TransferBSDToBNSH(
        context, kv_num_heads, new_seq_len, head_dim, value, nullptr, 0, &v_bnsh_t));
    WebgpuAttentionParameters aa_params = params;
    aa_params.qkv_format_ = Q_K_V_BNSH;
    aa_params.past_sequence_length_ = 0;
    aa_params.seqlen_past_kv_cache_ = 0;
    aa_params.total_sequence_length_ = static_cast<int>(new_seq_len);
    return ApplyAttention(&q_bnsh_t, &k_bnsh_t, &v_bnsh_t, attention_bias,
                          /*past_key=*/nullptr, /*past_value=*/nullptr,
                          output, /*present_key=*/nullptr, /*present_value=*/nullptr,
                          /*output_qk=*/nullptr, aa_params, context,
                          head_sink, /*seqlen_k=*/nullptr, local_window_size);
  }

  // Step 3: decode all present slots → fp16 K/V scratch in BNSH layout.
  const TensorShape kv_scratch_shape{
      static_cast<int64_t>(batch_size),
      static_cast<int64_t>(kv_num_heads),
      static_cast<int64_t>(total_seq_len),
      static_cast<int64_t>(head_dim),
  };
  Tensor k_scratch = context.CreateGPUTensor(query->DataType(), kv_scratch_shape);
  Tensor v_scratch = context.CreateGPUTensor(query->DataType(), kv_scratch_shape);

  TurboQuantDecodeProgram decode{head_dim, key_bits, value_bits, norm_correction};
  decode.AddInputs({
      {&k_cache_present_u32, ProgramTensorMetadataDependency::TypeAndRank},
      {&v_cache_present_u32, ProgramTensorMetadataDependency::TypeAndRank},
      {k_codebook, ProgramTensorMetadataDependency::TypeAndRank},
  });
  decode.AddOutputs({
      {&k_scratch, ProgramTensorMetadataDependency::TypeAndRank},
      {&v_scratch, ProgramTensorMetadataDependency::TypeAndRank},
  });
  decode.SetWorkgroupSize(head_dim);
  decode.SetDispatchGroupSize(batch_size * kv_num_heads * total_seq_len);
  decode.AddUniformVariables({
      {batch_size},
      {total_seq_len},
      {kv_num_heads},
      {max_seq_len},
  });
  ORT_RETURN_IF_ERROR(context.RunProgram(decode));

  // Step 4: attention on the dequantised K/V scratch.
  //
  // k_scratch / v_scratch are BNSH-laid-out (this is the natural cache layout
  // and matches what FA's `CopyKVCache` writes to its internal present_key).
  if (use_fa) {
    // FA path: tell it to read K/V as BNSH via Q_K_V_BSNH_BNSH_BNSH (Q stays
    // BSNH).  Override kv_sequence_length to the FULL present length and
    // clear past_seq so CopyKVCache treats k_scratch as the entire present
    // cache.
    WebgpuAttentionParameters fa_params = params;
    fa_params.qkv_format_ = Q_K_V_BSNH_BNSH_BNSH;
    fa_params.kv_sequence_length_ = static_cast<int>(total_seq_len);
    fa_params.past_sequence_length_ = 0;
    fa_params.seqlen_past_kv_cache_ = 0;
    return ApplyFlashAttention(query, &k_scratch, &v_scratch, attention_bias, output,
                               nullptr, nullptr, nullptr, nullptr,
                               fa_params, context, /*seqlens_k=*/nullptr,
                               cos_cache, sin_cache, head_sink);
  }

  // No-subgroups fallback: transfer Q (BSNH) to BNSH and call ApplyAttention.
  // K/V scratch are already BNSH from the decode shader, so they pass through
  // as-is.  We feed the entire present cache as K/V and set past_seq=0 so
  // ApplyAttention doesn't try to do its own past-merging.
  const TensorShape q_bnsh{
      static_cast<int64_t>(batch_size),
      static_cast<int64_t>(params.num_heads_),
      static_cast<int64_t>(new_seq_len),
      static_cast<int64_t>(head_dim),
  };
  Tensor q_bnsh_t = context.CreateGPUTensor(query->DataType(), q_bnsh);
  ORT_RETURN_IF_ERROR(TransferBSDToBNSH(
      context, params.num_heads_, new_seq_len, head_dim, query, nullptr, 0, &q_bnsh_t));
  WebgpuAttentionParameters aa_params = params;
  aa_params.qkv_format_ = Q_K_V_BNSH;
  aa_params.kv_sequence_length_ = static_cast<int>(total_seq_len);
  aa_params.past_sequence_length_ = 0;
  aa_params.seqlen_past_kv_cache_ = 0;
  return ApplyAttention(&q_bnsh_t, &k_scratch, &v_scratch, attention_bias,
                        /*past_key=*/nullptr, /*past_value=*/nullptr,
                        output, /*present_key=*/nullptr, /*present_value=*/nullptr,
                        /*output_qk=*/nullptr, aa_params, context,
                        head_sink, /*seqlen_k=*/nullptr, local_window_size);
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
