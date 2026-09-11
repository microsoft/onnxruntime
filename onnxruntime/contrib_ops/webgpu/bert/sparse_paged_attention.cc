// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/bert/sparse_paged_attention.h"

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <string>

#include "contrib_ops/cpu/bert/attention_parameters.h"
#include "contrib_ops/cpu/bert/paged_attention_helper.h"
#include "contrib_ops/webgpu/bert/paged_attention.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "core/common/logging/logging.h"
#include "core/providers/webgpu/webgpu_utils.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

namespace {

// One workgroup per (token, head). 64 lanes matches the WebGPU default and the
// reduction below assumes a power of two.
constexpr uint32_t kAttentionWorkgroupSize = 64;

// The staged shaders keep two head_size-sized FP32 arrays plus one
// workgroup-size-sized reduction array in workgroup memory.
constexpr int kMaxSupportedHeadSize = 512;

// Upper bound for every sanitized position the shaders derive from
// caller-owned i32 buffers (past_seqlens, cumulative_sequence_length). Staying
// below 2^30 lets the shaders add two sanitized values without overflowing i32.
constexpr uint64_t kMaxSanitizedPosition = 0x3FFFFFFF;

// Number of logical positions the block table can address. Anything at or
// beyond this has no physical block, so clamping to it is lossless.
uint32_t MainCachePositionBound(const PagedAttentionParameters& parameters) {
  const uint64_t positions = static_cast<uint64_t>(std::max(parameters.max_num_blocks_per_seq, 0)) *
                             static_cast<uint64_t>(std::max(parameters.block_size, 0));
  return static_cast<uint32_t>(std::min(positions, kMaxSanitizedPosition));
}

// A buffer larger than maxStorageBufferBindingSize is bound as several
// consecutive segments (see ProgramManager::CalculateSegmentsForInputsAndOutputs),
// so it consumes more than one storage binding in the bind group layout.
uint32_t StorageBindingSegments(const Tensor* tensor, uint64_t max_binding_size) {
  if (tensor == nullptr) {
    return 0;
  }
  const uint64_t bytes = tensor->SizeInBytes();
  if (max_binding_size == 0 || bytes <= max_binding_size) {
    return 1;
  }
  return static_cast<uint32_t>((bytes + max_binding_size - 1) / max_binding_size);
}

struct StageBinding {
  const Tensor* tensor;
  const char* name;
  // Optional bindings are read with raw indexing in the templates, because the
  // WGSL template system cannot take a conditionally added variable. Raw
  // indexing only ever reaches the first segment of a segmented buffer, so such
  // a binding must stay within one segment.
  bool raw_indexed = false;
};

// Validate a stage's bind group against the device *before* the shader is
// generated, so the caller gets an actionable NOT_IMPLEMENTED instead of a
// ShaderHelper/Dawn validation failure deep inside program creation. Counting
// tensors is not sufficient: a buffer beyond maxStorageBufferBindingSize takes
// several bindings, so a large KV cache or selection buffer can exhaust
// maxStorageBuffersPerShaderStage even when the tensor count looks fine.
Status CheckStageStorageBindings(const onnxruntime::webgpu::ComputeContext& context,
                                 const char* stage_name,
                                 std::initializer_list<StageBinding> bindings) {
  const uint64_t max_binding_size = context.DeviceLimits().maxStorageBufferBindingSize;
  const uint32_t max_bindings = context.DeviceLimits().maxStorageBuffersPerShaderStage;
  uint32_t binding_count = 0;
  for (const StageBinding& binding : bindings) {
    const uint32_t segments = StorageBindingSegments(binding.tensor, max_binding_size);
    if (segments > 1 && binding.raw_indexed) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                             "SparsePagedAttention (WebGPU): the ", stage_name, " stage binds '",
                             binding.name, "' as ", segments,
                             " buffer segments because it is larger than the ", max_binding_size,
                             " byte maxStorageBufferBindingSize; segmented bindings are not "
                             "supported for this input.");
    }
    binding_count += segments;
  }
  if (binding_count > max_bindings) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "SparsePagedAttention (WebGPU): the ", stage_name, " stage needs ",
                           binding_count, " storage buffer bindings (buffers larger than the ",
                           max_binding_size,
                           " byte maxStorageBufferBindingSize are bound as multiple segments), "
                           "but the device supports only ",
                           max_bindings, " per shader stage.");
  }
  return Status::OK();
}

}  // namespace

// v1 registers only the type combination the WebGPU shaders actually implement:
// float16 activations and a float16 KV cache. The schema also permits bfloat16
// and an int8 cache; those are rejected at kernel-lookup time rather than
// silently falling back to a different numeric behavior.
//
// MayInplace hints the ORT framework that inputs 3/4 (key_cache/value_cache)
// may share buffers with outputs 1/2 (key_cache_out/value_cache_out), which is
// how the KV-cache aliasing contract is realized in production IO-binding.
ONNX_OPERATOR_KERNEL_EX(
    SparsePagedAttention,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>())
        .TypeConstraint("T_CACHE", DataTypeImpl::GetTensorType<MLFloat16>())
        .TypeConstraint("T_AUX", DataTypeImpl::GetTensorType<MLFloat16>())
        .TypeConstraint("T_KV_SCALE", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("S", DataTypeImpl::GetTensorType<int32_t>())
        .InputMemoryType(OrtMemTypeCPUInput, 21)
        .MayInplace(3, 1)
        .MayInplace(4, 2),
    SparsePagedAttention);

Status SparsePagedAttentionScatterKVProgram::GenerateShaderCode(ShaderHelper& sh) const {
  const auto& key = sh.AddInput("key", ShaderUsage::UseUniform);
  const auto& value = sh.AddInput("value", ShaderUsage::UseUniform);
  const auto& slot_mapping = sh.AddInput("slot_mapping", ShaderUsage::UseUniform);
  const auto& key_cache = sh.AddOutput("key_cache", ShaderUsage::UseUniform);
  const auto& value_cache = sh.AddOutput("value_cache", ShaderUsage::UseUniform);
  return WGSL_TEMPLATE_APPLY(sh, "bert/sparse_paged_attention_scatter_kv.wgsl.template",
                             WGSL_TEMPLATE_VARIABLE(key, key),
                             WGSL_TEMPLATE_VARIABLE(key_cache, key_cache),
                             WGSL_TEMPLATE_VARIABLE(slot_mapping, slot_mapping),
                             WGSL_TEMPLATE_VARIABLE(value, value),
                             WGSL_TEMPLATE_VARIABLE(value_cache, value_cache));
}

Status SparsePagedAttentionTokenMetaProgram::GenerateShaderCode(ShaderHelper& sh) const {
  const auto& cumulative_sequence_length =
      sh.AddInput("cumulative_sequence_length", ShaderUsage::UseUniform);
  const auto& past_seqlens = sh.AddInput("past_seqlens", ShaderUsage::UseUniform);
  // Optional bindings are read with raw indexing in the template, so they are
  // added to the shader but not passed as template variables.
  if (has_auxiliary_lengths_) {
    sh.AddInput("auxiliary_lengths", ShaderUsage::UseUniform);
  }
  const auto& token_meta = sh.AddOutput("token_meta", ShaderUsage::UseUniform);
  return WGSL_TEMPLATE_APPLY(sh, "bert/sparse_paged_attention_token_meta.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(has_auxiliary_lengths, has_auxiliary_lengths_),
                             WGSL_TEMPLATE_VARIABLE(cumulative_sequence_length, cumulative_sequence_length),
                             WGSL_TEMPLATE_VARIABLE(past_seqlens, past_seqlens),
                             WGSL_TEMPLATE_VARIABLE(token_meta, token_meta));
}

Status SparsePagedAttentionMainProgram::GenerateShaderCode(ShaderHelper& sh) const {
  const auto& query = sh.AddInput("query", ShaderUsage::UseUniform);
  const auto& key_cache = sh.AddInput("key_cache", ShaderUsage::UseUniform);
  const auto& value_cache = sh.AddInput("value_cache", ShaderUsage::UseUniform);
  const auto& token_meta = sh.AddInput("token_meta", ShaderUsage::UseUniform);
  const auto& block_table = sh.AddInput("block_table", ShaderUsage::UseUniform);
  if (use_selected_) {
    sh.AddInput("selected_indices", ShaderUsage::UseUniform);
    sh.AddInput("selected_counts", ShaderUsage::UseUniform);
  }
  const auto& partial = sh.AddOutput("partial", ShaderUsage::UseUniform);
  return WGSL_TEMPLATE_APPLY(sh, "bert/sparse_paged_attention_main.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(dedup_selected, dedup_selected_),
                             WGSL_TEMPLATE_PARAMETER(is_causal, is_causal_),
                             WGSL_TEMPLATE_PARAMETER(qkv_head_size, head_size_),
                             WGSL_TEMPLATE_PARAMETER(use_local_window, use_local_window_),
                             WGSL_TEMPLATE_PARAMETER(use_selected, use_selected_),
                             WGSL_TEMPLATE_VARIABLE(block_table, block_table),
                             WGSL_TEMPLATE_VARIABLE(key_cache, key_cache),
                             WGSL_TEMPLATE_VARIABLE(partial, partial),
                             WGSL_TEMPLATE_VARIABLE(query, query),
                             WGSL_TEMPLATE_VARIABLE(token_meta, token_meta),
                             WGSL_TEMPLATE_VARIABLE(value_cache, value_cache));
}

Status SparsePagedAttentionAuxiliaryProgram::GenerateShaderCode(ShaderHelper& sh) const {
  const auto& query = sh.AddInput("query", ShaderUsage::UseUniform);
  const auto& auxiliary_key = sh.AddInput("auxiliary_key", ShaderUsage::UseUniform);
  if (!auxiliary_kv_shared_) {
    sh.AddInput("auxiliary_value", ShaderUsage::UseUniform);
  }
  const auto& token_meta = sh.AddInput("token_meta", ShaderUsage::UseUniform);
  const auto& selected_indices = sh.AddInput("selected_indices", ShaderUsage::UseUniform);
  const auto& selected_counts = sh.AddInput("selected_counts", ShaderUsage::UseUniform);
  const auto& partial = sh.AddOutput("partial", ShaderUsage::UseUniform);
  return WGSL_TEMPLATE_APPLY(sh, "bert/sparse_paged_attention_auxiliary.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(auxiliary_kv_shared, auxiliary_kv_shared_),
                             WGSL_TEMPLATE_PARAMETER(qkv_head_size, head_size_),
                             WGSL_TEMPLATE_VARIABLE(auxiliary_key, auxiliary_key),
                             WGSL_TEMPLATE_VARIABLE(partial, partial),
                             WGSL_TEMPLATE_VARIABLE(query, query),
                             WGSL_TEMPLATE_VARIABLE(selected_counts, selected_counts),
                             WGSL_TEMPLATE_VARIABLE(selected_indices, selected_indices),
                             WGSL_TEMPLATE_VARIABLE(token_meta, token_meta));
}

Status SparsePagedAttentionFinalizeProgram::GenerateShaderCode(ShaderHelper& sh) const {
  if (has_main_) {
    sh.AddInput("partial_main", ShaderUsage::UseUniform);
  }
  if (has_auxiliary_) {
    sh.AddInput("partial_auxiliary", ShaderUsage::UseUniform);
  }
  if (has_head_sink_) {
    sh.AddInput("head_sink", ShaderUsage::UseUniform);
  }
  const auto& output = sh.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);
  return WGSL_TEMPLATE_APPLY(sh, "bert/sparse_paged_attention_finalize.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(has_auxiliary, has_auxiliary_),
                             WGSL_TEMPLATE_PARAMETER(has_head_sink, has_head_sink_),
                             WGSL_TEMPLATE_PARAMETER(has_main, has_main_),
                             WGSL_TEMPLATE_VARIABLE(output, output));
}

namespace {

Status RunSparseScatterKVWithSlotMapping(onnxruntime::webgpu::ComputeContext& context,
                                         const PagedAttentionParameters& parameters,
                                         const Tensor* key,
                                         const Tensor* value,
                                         const Tensor* slot_mapping,
                                         Tensor* key_cache_out,
                                         Tensor* value_cache_out) {
  const uint32_t kv_num_heads = static_cast<uint32_t>(parameters.kv_num_heads);
  const uint32_t head_size = static_cast<uint32_t>(parameters.head_size);
  const uint32_t cache_slot_count =
      static_cast<uint32_t>(parameters.num_blocks) * static_cast<uint32_t>(parameters.block_size);
  const uint32_t dispatch_size =
      static_cast<uint32_t>(parameters.token_count) * kv_num_heads * head_size;

  SparsePagedAttentionScatterKVProgram program{};
  program
      .AddInputs({
          {key, ProgramTensorMetadataDependency::TypeAndRank},
          {value, ProgramTensorMetadataDependency::TypeAndRank},
          {slot_mapping, ProgramTensorMetadataDependency::TypeAndRank},
      })
      .AddOutputs({
          {key_cache_out, ProgramTensorMetadataDependency::TypeAndRank},
          {value_cache_out, ProgramTensorMetadataDependency::TypeAndRank},
      })
      .AddUniformVariables({
          {kv_num_heads},
          {head_size},
          {cache_slot_count},
          {dispatch_size},
      })
      .SetDispatchGroupSize((dispatch_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE);
  return context.RunProgram(program);
}

Status RunSparseTokenMeta(onnxruntime::webgpu::ComputeContext& context,
                          const PagedAttentionParameters& parameters,
                          uint32_t auxiliary_capacity,
                          const Tensor* cumulative_seqlens_q,
                          const Tensor* past_seqlens,
                          const Tensor* auxiliary_lengths,
                          Tensor* token_meta) {
  const uint32_t batch_size = static_cast<uint32_t>(parameters.batch_size);
  const uint32_t dispatch_size = static_cast<uint32_t>(parameters.token_count);

  SparsePagedAttentionTokenMetaProgram program{auxiliary_lengths != nullptr};
  program.AddInputs({
      {cumulative_seqlens_q, ProgramTensorMetadataDependency::TypeAndRank},
      {past_seqlens, ProgramTensorMetadataDependency::TypeAndRank},
  });
  if (auxiliary_lengths != nullptr) {
    program.AddInputs({{auxiliary_lengths, ProgramTensorMetadataDependency::TypeAndRank}});
  }
  program
      .AddOutputs({
          {token_meta, ProgramTensorMetadataDependency::TypeAndRank},
      })
      .CacheHint(auxiliary_lengths != nullptr)
      .AddUniformVariables({
          {batch_size},
          {auxiliary_capacity},
          {MainCachePositionBound(parameters)},
          {dispatch_size},
      })
      .SetDispatchGroupSize((dispatch_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE);
  return context.RunProgram(program);
}

Status RunSparseMainPartial(onnxruntime::webgpu::ComputeContext& context,
                            const PagedAttentionParameters& parameters,
                            bool is_causal,
                            bool use_local_window,
                            bool use_selected,
                            bool dedup_selected,
                            uint32_t max_selected_entries,
                            float scale,
                            const Tensor* query,
                            const Tensor* key_cache,
                            const Tensor* value_cache,
                            const Tensor* token_meta,
                            const Tensor* block_table,
                            const Tensor* selected_indices,
                            const Tensor* selected_counts,
                            Tensor* partial) {
  const uint32_t num_heads = static_cast<uint32_t>(parameters.num_heads);
  const uint32_t workgroup_count = static_cast<uint32_t>(parameters.token_count) * num_heads;
  // local_window_size <= 0 means "no window bound"; the shader spells that 0.
  const uint32_t local_window_size =
      parameters.local_window_size > 0 ? static_cast<uint32_t>(parameters.local_window_size) : 0u;

  SparsePagedAttentionMainProgram program{parameters.head_size, is_causal, use_local_window,
                                          use_selected, dedup_selected};
  program.AddInputs({
      {query, ProgramTensorMetadataDependency::TypeAndRank},
      {key_cache, ProgramTensorMetadataDependency::TypeAndRank},
      {value_cache, ProgramTensorMetadataDependency::TypeAndRank},
      {token_meta, ProgramTensorMetadataDependency::TypeAndRank},
      {block_table, ProgramTensorMetadataDependency::TypeAndRank},
  });
  if (use_selected) {
    program.AddInputs({
        {selected_indices, ProgramTensorMetadataDependency::TypeAndRank},
        {selected_counts, ProgramTensorMetadataDependency::TypeAndRank},
    });
  }
  program
      .AddOutputs({
          {partial, ProgramTensorMetadataDependency::TypeAndRank},
      })
      // Every value baked into the generated WGSL must appear here: the four
      // #params, the compile-time head size, and the workgroup size.
      .CacheHint(parameters.head_size, is_causal, use_local_window, use_selected, dedup_selected,
                 kAttentionWorkgroupSize)
      .AddUniformVariables({
          {num_heads},
          {static_cast<uint32_t>(parameters.kv_num_heads)},
          {static_cast<uint32_t>(parameters.block_size)},
          {static_cast<uint32_t>(parameters.num_blocks)},
          {static_cast<uint32_t>(parameters.max_num_blocks_per_seq)},
          {MainCachePositionBound(parameters)},
          {max_selected_entries},
          {local_window_size},
          {workgroup_count},
          {scale},
          {parameters.softcap},
      })
      .SetDispatchGroupSize(workgroup_count)
      .SetWorkgroupSize(kAttentionWorkgroupSize);
  return context.RunProgram(program);
}

Status RunSparseAuxiliaryPartial(onnxruntime::webgpu::ComputeContext& context,
                                 const PagedAttentionParameters& parameters,
                                 bool auxiliary_kv_shared,
                                 uint32_t auxiliary_capacity,
                                 uint32_t auxiliary_num_heads,
                                 uint32_t max_selected_entries,
                                 float scale,
                                 const Tensor* query,
                                 const Tensor* auxiliary_key,
                                 const Tensor* auxiliary_value,
                                 const Tensor* token_meta,
                                 const Tensor* selected_indices,
                                 const Tensor* selected_counts,
                                 Tensor* partial) {
  const uint32_t num_heads = static_cast<uint32_t>(parameters.num_heads);
  const uint32_t workgroup_count = static_cast<uint32_t>(parameters.token_count) * num_heads;

  SparsePagedAttentionAuxiliaryProgram program{parameters.head_size, auxiliary_kv_shared};
  program.AddInputs({
      {query, ProgramTensorMetadataDependency::TypeAndRank},
      {auxiliary_key, ProgramTensorMetadataDependency::TypeAndRank},
  });
  if (!auxiliary_kv_shared) {
    program.AddInputs({{auxiliary_value, ProgramTensorMetadataDependency::TypeAndRank}});
  }
  program
      .AddInputs({
          {token_meta, ProgramTensorMetadataDependency::TypeAndRank},
          {selected_indices, ProgramTensorMetadataDependency::TypeAndRank},
          {selected_counts, ProgramTensorMetadataDependency::TypeAndRank},
      })
      .AddOutputs({
          {partial, ProgramTensorMetadataDependency::TypeAndRank},
      })
      .CacheHint(parameters.head_size, auxiliary_kv_shared, kAttentionWorkgroupSize)
      .AddUniformVariables({
          {num_heads},
          {static_cast<uint32_t>(parameters.kv_num_heads)},
          {auxiliary_capacity},
          {auxiliary_num_heads},
          {max_selected_entries},
          {workgroup_count},
          {scale},
          {parameters.softcap},
      })
      .SetDispatchGroupSize(workgroup_count)
      .SetWorkgroupSize(kAttentionWorkgroupSize);
  return context.RunProgram(program);
}

Status RunSparseFinalize(onnxruntime::webgpu::ComputeContext& context,
                         const PagedAttentionParameters& parameters,
                         const Tensor* partial_main,
                         const Tensor* partial_auxiliary,
                         const Tensor* head_sink,
                         Tensor* output) {
  const uint32_t num_heads = static_cast<uint32_t>(parameters.num_heads);
  const uint32_t head_size = static_cast<uint32_t>(parameters.head_size);
  const uint32_t dispatch_size =
      static_cast<uint32_t>(parameters.token_count) * num_heads * head_size;

  SparsePagedAttentionFinalizeProgram program{partial_main != nullptr, partial_auxiliary != nullptr,
                                              head_sink != nullptr};
  if (partial_main != nullptr) {
    program.AddInputs({{partial_main, ProgramTensorMetadataDependency::TypeAndRank}});
  }
  if (partial_auxiliary != nullptr) {
    program.AddInputs({{partial_auxiliary, ProgramTensorMetadataDependency::TypeAndRank}});
  }
  if (head_sink != nullptr) {
    program.AddInputs({{head_sink, ProgramTensorMetadataDependency::TypeAndRank}});
  }
  program
      .AddOutputs({
          {output, ProgramTensorMetadataDependency::TypeAndRank},
      })
      .CacheHint(partial_main != nullptr, partial_auxiliary != nullptr, head_sink != nullptr)
      .AddUniformVariables({
          {num_heads},
          {head_size},
          {dispatch_size},
      })
      .SetDispatchGroupSize((dispatch_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE);
  return context.RunProgram(program);
}

}  // namespace

SparsePagedAttention::SparsePagedAttention(const OpKernelInfo& info) : WebGpuKernel(info) {
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
  is_causal_ = info.GetAttrOrDefault<int64_t>("is_causal", 1) == 1;
  do_rotary_ = info.GetAttrOrDefault<int64_t>("do_rotary", 0) == 1;
  rotary_interleaved_ = info.GetAttrOrDefault<int64_t>("rotary_interleaved", 0) == 1;
  rotary_offset_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("rotary_offset", 0));
  has_explicit_scale_ = info.GetAttr<float>("scale", &scale_).IsOK();
  if (!has_explicit_scale_) {
    scale_ = info.GetAttrOrDefault<float>("scale", 0.0f);
  }
  softcap_ = info.GetAttrOrDefault<float>("softcap", 0.0f);
  qk_norm_epsilon_ = info.GetAttrOrDefault<float>("qk_norm_epsilon", 1e-6f);
  k_quant_type_ = info.GetAttrOrDefault<std::string>("k_quant_type", "NONE");
  v_quant_type_ = info.GetAttrOrDefault<std::string>("v_quant_type", "NONE");

  const std::string attention_mode =
      info.GetAttrOrDefault<std::string>("attention_mode", "selected_only");
  ORT_ENFORCE(attention_mode == "selected_only" || attention_mode == "local_plus_selected",
              "'attention_mode' must be 'selected_only' or 'local_plus_selected'.");
  attention_mode_ = attention_mode == "selected_only" ? SparseAttentionMode::kSelectedOnly
                                                      : SparseAttentionMode::kLocalPlusSelected;

  const std::string selected_kv_source =
      info.GetAttrOrDefault<std::string>("selected_kv_source", "main");
  ORT_ENFORCE(selected_kv_source == "main" || selected_kv_source == "auxiliary",
              "'selected_kv_source' must be 'main' or 'auxiliary'.");
  selected_kv_source_ = selected_kv_source == "main" ? SparseSelectedKvSource::kMain
                                                     : SparseSelectedKvSource::kAuxiliary;
  ORT_ENFORCE(info.GetAttrOrDefault<std::string>("auxiliary_cache_layout", "contiguous") ==
                  "contiguous",
              "Only auxiliary_cache_layout='contiguous' is supported.");
  auxiliary_kv_shared_ = info.GetAttrOrDefault<int64_t>("auxiliary_kv_shared", 0) == 1;
}

Status SparsePagedAttention::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {
  const Tensor* query = context.Input<Tensor>(0);
  const Tensor* key = context.Input<Tensor>(1);
  const Tensor* value = context.Input<Tensor>(2);
  const Tensor* key_cache = context.Input<Tensor>(3);
  const Tensor* value_cache = context.Input<Tensor>(4);
  const Tensor* cumulative_seqlens_q = context.Input<Tensor>(5);
  const Tensor* past_seqlens = context.Input<Tensor>(6);
  const Tensor* block_table = context.Input<Tensor>(7);
  const Tensor* slot_mapping = context.InputCount() > 8 ? context.Input<Tensor>(8) : nullptr;
  const Tensor* selected_indices = context.InputCount() > 9 ? context.Input<Tensor>(9) : nullptr;
  const Tensor* selected_counts = context.InputCount() > 10 ? context.Input<Tensor>(10) : nullptr;
  const Tensor* auxiliary_key = context.InputCount() > 11 ? context.Input<Tensor>(11) : nullptr;
  const Tensor* auxiliary_value = context.InputCount() > 12 ? context.Input<Tensor>(12) : nullptr;
  const Tensor* auxiliary_lengths = context.InputCount() > 13 ? context.Input<Tensor>(13) : nullptr;
  const Tensor* cos_cache = context.InputCount() > 14 ? context.Input<Tensor>(14) : nullptr;
  const Tensor* sin_cache = context.InputCount() > 15 ? context.Input<Tensor>(15) : nullptr;
  const Tensor* head_sink = context.InputCount() > 16 ? context.Input<Tensor>(16) : nullptr;
  const Tensor* q_norm_weight = context.InputCount() > 17 ? context.Input<Tensor>(17) : nullptr;
  const Tensor* k_norm_weight = context.InputCount() > 18 ? context.Input<Tensor>(18) : nullptr;
  const Tensor* k_scale = context.InputCount() > 19 ? context.Input<Tensor>(19) : nullptr;
  const Tensor* v_scale = context.InputCount() > 20 ? context.Input<Tensor>(20) : nullptr;
  const Tensor* attention_metadata = context.InputCount() > 21 ? context.Input<Tensor>(21) : nullptr;

  PagedAttentionParameters parameters{};
  const KVQuantizationType k_quant_type = StringToKVQuantizationType(k_quant_type_);
  const KVQuantizationType v_quant_type = StringToKVQuantizationType(v_quant_type_);

  // attention_metadata is intentionally not forwarded: the shared helper only
  // understands the PagedAttention 3-entry layout, while SparsePagedAttention
  // defines a 5-entry layout that is validated below. 0 for max_threads_per_block
  // disables the CUDA-block-size specific guard.
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
                                                          /*attention_metadata*/ static_cast<const Tensor*>(nullptr),
                                                          &parameters,
                                                          num_heads_,
                                                          kv_num_heads_,
                                                          scale_,
                                                          softcap_,
                                                          qk_norm_epsilon_,
                                                          k_quant_type,
                                                          v_quant_type,
                                                          KVCacheDataType::DEFAULT,
                                                          KVCacheDataType::DEFAULT,
                                                          KVCacheDataType::FLOAT16,
                                                          /*is_latent_kv*/ false,
                                                          /*v_head_size*/ 0,
                                                          rotary_offset_,
                                                          has_explicit_scale_,
                                                          /*max_threads_per_block*/ 0));
  parameters.local_window_size = local_window_size_;
  parameters.is_causal = is_causal_;
  parameters.do_rotary = do_rotary_;
  parameters.rotary_interleaved = rotary_interleaved_;

  // Feature guards. Every rejected configuration produces an explicit error
  // rather than silently computing something else.
  if (k_quant_type_ != "NONE" || v_quant_type_ != "NONE") {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "SparsePagedAttention (WebGPU): quantized KV cache is not supported yet.");
  }
  if (k_scale != nullptr || v_scale != nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "SparsePagedAttention (WebGPU): k_scale/v_scale inputs are not supported yet.");
  }
  if (context.KvCacheQuantizationEnabled()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "SparsePagedAttention (WebGPU): KV cache quantization is not supported yet.");
  }
  if (q_norm_weight != nullptr || k_norm_weight != nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "SparsePagedAttention (WebGPU): q_norm_weight/k_norm_weight inputs are not supported yet.");
  }
  if (rotary_offset_ != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "SparsePagedAttention (WebGPU): rotary_offset != 0 is not supported yet.");
  }
  if (do_rotary_ && (cos_cache == nullptr || sin_cache == nullptr)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "cos_cache and sin_cache are required when do_rotary=1.");
  }
  if (parameters.head_size > kMaxSupportedHeadSize) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "SparsePagedAttention (WebGPU): head_size ", parameters.head_size,
                           " exceeds the supported maximum of ", kMaxSupportedHeadSize, ".");
  }

  // Selection inputs are mandatory for this op in every mode.
  if (selected_indices == nullptr || selected_counts == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "selected_indices and selected_counts are required inputs.");
  }
  const auto& selected_dims = selected_indices->Shape().GetDims();
  if (selected_dims.size() != 2 || selected_dims[0] != parameters.token_count) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "selected_indices must have shape (token_count, max_selected_entries).");
  }
  const int64_t max_selected_entries = selected_dims[1];
  const auto& count_dims = selected_counts->Shape().GetDims();
  if (count_dims.size() != 1 || count_dims[0] != parameters.token_count) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "selected_counts must have shape (token_count).");
  }

  int64_t auxiliary_capacity = 0;
  int64_t auxiliary_num_heads = 0;
  const bool selected_from_auxiliary = selected_kv_source_ == SparseSelectedKvSource::kAuxiliary;
  if (selected_from_auxiliary) {
    if (auxiliary_key == nullptr || auxiliary_lengths == nullptr) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "auxiliary_key and auxiliary_lengths are required when "
                             "selected_kv_source='auxiliary'.");
    }
    if (!auxiliary_kv_shared_ && auxiliary_value == nullptr) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "auxiliary_value is required when auxiliary_kv_shared=0.");
    }
    if (auxiliary_kv_shared_ && auxiliary_value != nullptr) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "auxiliary_value must be absent when auxiliary_kv_shared=1.");
    }
    const auto& aux_dims = auxiliary_key->Shape().GetDims();
    if (aux_dims.size() != 4 || aux_dims[0] != parameters.batch_size ||
        (aux_dims[2] != 1 && aux_dims[2] != kv_num_heads_) ||
        aux_dims[3] != parameters.head_size) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "auxiliary_key must have shape (batch_size, capacity, 1 or "
                             "kv_num_heads, head_size).");
    }
    auxiliary_capacity = aux_dims[1];
    auxiliary_num_heads = aux_dims[2];
    if (auxiliary_value != nullptr && auxiliary_value->Shape() != auxiliary_key->Shape()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "auxiliary_value must have the same shape as auxiliary_key.");
    }
    const auto& aux_length_dims = auxiliary_lengths->Shape().GetDims();
    if (aux_length_dims.size() != 1 || aux_length_dims[0] != parameters.batch_size) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "auxiliary_lengths must have shape (batch_size).");
    }
  } else if (auxiliary_key != nullptr || auxiliary_value != nullptr ||
             auxiliary_lengths != nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Auxiliary inputs must be absent when selected_kv_source='main'.");
  }

  // attention_metadata lives on CPU (see InputMemoryType above), so inspecting
  // it costs nothing. Selection and block tables are never read on the host.
  if (attention_metadata != nullptr) {
    const auto& dims = attention_metadata->Shape().GetDims();
    if (dims.size() != 1 || dims[0] != 5) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "attention_metadata must have shape (5): [max_query_len_bound, "
                             "max_local_main_len_bound, max_selected_entries_bound, "
                             "max_auxiliary_len_bound, max_combined_attention_len_bound].");
    }
    const int32_t* metadata = attention_metadata->Data<int32_t>();
    for (int i = 0; i < 5; ++i) {
      if (metadata[i] < 0) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "attention_metadata entries must be non-negative.");
      }
    }
    if (metadata[2] > 0 && metadata[2] < max_selected_entries) {
      // The bound is advisory: the shader still honours the device-resident
      // selected_counts. Reject only a self-inconsistent contract.
      LOGS_DEFAULT(VERBOSE) << "SparsePagedAttention (WebGPU): attention_metadata "
                               "max_selected_entries_bound is smaller than the selected_indices "
                               "width; the device-resident counts remain authoritative.";
    }
  }

  TensorShapeVector output_shape{static_cast<int64_t>(parameters.token_count),
                                 static_cast<int64_t>(parameters.hidden_size)};
  Tensor* output = context.Output(0, output_shape);

  // Outputs 1/2 are optional. When they are present but the allocation planner
  // could not alias them onto the cache inputs (OpTester and any non-IO-bound
  // consumer), materialize the input caches into them first so untouched slots
  // survive the op.
  Tensor* key_cache_out = context.OutputCount() > 1 ? context.Output(1, key_cache->Shape()) : nullptr;
  Tensor* value_cache_out = context.OutputCount() > 2 ? context.Output(2, value_cache->Shape()) : nullptr;
  ORT_RETURN_IF((key_cache_out == nullptr) != (value_cache_out == nullptr),
                "SparsePagedAttention (WebGPU): key_cache_out and value_cache_out must be both "
                "present or both absent.");
  if (key_cache_out == nullptr) {
    key_cache_out = const_cast<Tensor*>(key_cache);
    value_cache_out = const_cast<Tensor*>(value_cache);
  }

  const bool key_cache_aliased = (key_cache->DataRaw() == key_cache_out->MutableDataRaw());
  const bool value_cache_aliased = (value_cache->DataRaw() == value_cache_out->MutableDataRaw());
  if (!key_cache_aliased || !value_cache_aliased) {
    LOGS_DEFAULT(WARNING) << "SparsePagedAttention (WebGPU): cache outputs are not aliased with "
                             "cache inputs; falling back to a GPU cache copy. Configure IO-binding "
                             "to alias the cache buffers in production to avoid this per-run copy.";
  }
  if (!key_cache_aliased) {
    ORT_RETURN_IF_ERROR(context.CopyTensor(*key_cache, *key_cache_out));
  }
  if (!value_cache_aliased) {
    ORT_RETURN_IF_ERROR(context.CopyTensor(*value_cache, *value_cache_out));
  }

  if (parameters.token_count == 0) {
    return Status::OK();
  }

  const uint64_t max_storage_buffer_binding_size = context.DeviceLimits().maxStorageBufferBindingSize;
  const uint64_t partial_bytes = static_cast<uint64_t>(parameters.token_count) *
                                 static_cast<uint64_t>(parameters.num_heads) *
                                 (static_cast<uint64_t>(parameters.head_size) + 2u) * sizeof(float);
  // The finalize stage reads the partial scratch with raw indexing (an optional
  // binding cannot be a template variable), which only reaches the first
  // segment, so the scratch must fit a single binding.
  if (partial_bytes > max_storage_buffer_binding_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "SparsePagedAttention (WebGPU): the partial softmax scratch requires ",
                           partial_bytes, " bytes, exceeding maxStorageBufferBindingSize of ",
                           max_storage_buffer_binding_size, ".");
  }
  const uint64_t workgroup_storage_bytes =
      (2ull * static_cast<uint64_t>(parameters.head_size) + kAttentionWorkgroupSize) * sizeof(float);
  if (workgroup_storage_bytes > context.DeviceLimits().maxComputeWorkgroupStorageSize) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "SparsePagedAttention (WebGPU): head_size ", parameters.head_size,
                           " requires ", workgroup_storage_bytes,
                           " bytes of workgroup storage, exceeding the device limit of ",
                           context.DeviceLimits().maxComputeWorkgroupStorageSize, ".");
  }
  // Storage-binding budget. Counting tensors is not enough: the program manager
  // splits any buffer larger than maxStorageBufferBindingSize into several
  // consecutive bindings, so a KV cache or a selection buffer past that limit
  // silently inflates the bind group. Every stage below is therefore validated
  // against its actual segment count, immediately before it is dispatched.

  // Packed-QKV: materialize standalone Q/K/V so the staged pipeline below sees
  // a single layout. Reuses the PagedAttention split program.
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
    ORT_RETURN_IF_ERROR(CheckStageStorageBindings(
        context, "packed-QKV split",
        {{query, "query"},
         {&packed_q_tensor, "query"},
         {&packed_k_tensor, "key"},
         {&packed_v_tensor, "value"}}));
    ORT_RETURN_IF_ERROR(RunPagedAttentionSplitPackedQKV(context, parameters, query,
                                                        &packed_q_tensor, &packed_k_tensor,
                                                        &packed_v_tensor));
    query = &packed_q_tensor;
    key = &packed_k_tensor;
    value = &packed_v_tensor;
  }

  const Tensor* query_for_attention = query;
  const Tensor* key_for_scatter = key;
  Tensor rotated_query_tensor;
  Tensor rotated_key_tensor;
  if (do_rotary_) {
    rotated_query_tensor = context.CreateGPUTensor(query->DataType(), query->Shape());
    rotated_key_tensor = context.CreateGPUTensor(key->DataType(), key->Shape());
    ORT_RETURN_IF_ERROR(CheckStageStorageBindings(
        context, "rotary embedding",
        {{query, "query"},
         {cos_cache, "cos_cache"},
         {sin_cache, "sin_cache"},
         {cumulative_seqlens_q, "cumulative_sequence_length"},
         {past_seqlens, "past_seqlens"},
         {&rotated_query_tensor, "output"}}));
    ORT_RETURN_IF_ERROR(CheckStageStorageBindings(
        context, "rotary embedding",
        {{key, "key"},
         {cos_cache, "cos_cache"},
         {sin_cache, "sin_cache"},
         {cumulative_seqlens_q, "cumulative_sequence_length"},
         {past_seqlens, "past_seqlens"},
         {&rotated_key_tensor, "output"}}));
    ORT_RETURN_IF_ERROR(RunPagedAttentionRotaryEmbedding(
        context, parameters, static_cast<uint32_t>(parameters.num_heads), rotary_interleaved_,
        query, cos_cache, sin_cache, cumulative_seqlens_q, past_seqlens, &rotated_query_tensor));
    query_for_attention = &rotated_query_tensor;

    ORT_RETURN_IF_ERROR(RunPagedAttentionRotaryEmbedding(
        context, parameters, static_cast<uint32_t>(parameters.kv_num_heads), rotary_interleaved_,
        key, cos_cache, sin_cache, cumulative_seqlens_q, past_seqlens, &rotated_key_tensor));
    key_for_scatter = &rotated_key_tensor;
  }

  // Write the new K/V into the main cache. slot_mapping, when supplied, owns
  // the placement (including -1 = "do not store this token"); otherwise the
  // block_table-derived placement of PagedAttention applies.
  if (slot_mapping != nullptr) {
    ORT_RETURN_IF_ERROR(CheckStageStorageBindings(
        context, "KV scatter",
        {{key_for_scatter, "key"},
         {value, "value"},
         {slot_mapping, "slot_mapping"},
         {key_cache_out, "key_cache"},
         {value_cache_out, "value_cache"}}));
    ORT_RETURN_IF_ERROR(RunSparseScatterKVWithSlotMapping(context, parameters, key_for_scatter,
                                                          value, slot_mapping, key_cache_out,
                                                          value_cache_out));
  } else {
    ORT_RETURN_IF_ERROR(CheckStageStorageBindings(
        context, "KV scatter",
        {{key_for_scatter, "key"},
         {value, "value"},
         {cumulative_seqlens_q, "cumulative_sequence_length"},
         {past_seqlens, "past_seqlens"},
         {block_table, "block_table"},
         {key_cache_out, "key_cache"},
         {value_cache_out, "value_cache"}}));
    ORT_RETURN_IF_ERROR(RunPagedAttentionScatterKVToPagedCache(
        context, parameters, key_for_scatter, value, cumulative_seqlens_q, past_seqlens,
        block_table, key_cache_out, value_cache_out));
  }

  const auto* int32_type = DataTypeImpl::GetType<int32_t>();
  Tensor token_meta = context.CreateGPUTensor(
      int32_type, TensorShape({static_cast<int64_t>(parameters.token_count), 4}));
  ORT_RETURN_IF_ERROR(CheckStageStorageBindings(
      context, "token metadata",
      {{cumulative_seqlens_q, "cumulative_sequence_length"},
       {past_seqlens, "past_seqlens"},
       {selected_from_auxiliary ? auxiliary_lengths : nullptr, "auxiliary_lengths",
        /*raw_indexed*/ true},
       {&token_meta, "token_meta"}}));
  ORT_RETURN_IF_ERROR(RunSparseTokenMeta(context, parameters,
                                         static_cast<uint32_t>(auxiliary_capacity),
                                         cumulative_seqlens_q, past_seqlens,
                                         selected_from_auxiliary ? auxiliary_lengths : nullptr,
                                         &token_meta));

  const bool local_plus_selected = attention_mode_ == SparseAttentionMode::kLocalPlusSelected;
  // Which stages run:
  //   selected_only      + main      -> main partial over the selected entries
  //   selected_only      + auxiliary -> auxiliary partial only
  //   local_plus_selected+ main      -> main partial over local window + selected (de-duplicated)
  //   local_plus_selected+ auxiliary -> main partial over the local window, plus auxiliary partial
  const bool run_main = local_plus_selected || !selected_from_auxiliary;
  const bool run_auxiliary = selected_from_auxiliary;
  const bool main_uses_selected = !selected_from_auxiliary;
  const float scale = parameters.scale == 0.0f
                          ? 1.0f / std::sqrt(static_cast<float>(parameters.head_size))
                          : parameters.scale;

  const auto* float_type = DataTypeImpl::GetType<float>();
  const TensorShape partial_shape({static_cast<int64_t>(parameters.token_count),
                                   static_cast<int64_t>(parameters.num_heads),
                                   static_cast<int64_t>(parameters.head_size) + 2});
  Tensor partial_main;
  Tensor partial_auxiliary;

  if (run_main) {
    partial_main = context.CreateGPUTensor(float_type, partial_shape);
    ORT_RETURN_IF_ERROR(CheckStageStorageBindings(
        context, "main partial attention",
        {{query_for_attention, "query"},
         {key_cache_out, "key_cache"},
         {value_cache_out, "value_cache"},
         {&token_meta, "token_meta"},
         {block_table, "block_table"},
         {main_uses_selected ? selected_indices : nullptr, "selected_indices",
          /*raw_indexed*/ true},
         {main_uses_selected ? selected_counts : nullptr, "selected_counts",
          /*raw_indexed*/ true},
         {&partial_main, "partial"}}));
    ORT_RETURN_IF_ERROR(RunSparseMainPartial(
        context, parameters, is_causal_, local_plus_selected, main_uses_selected,
        local_plus_selected && main_uses_selected,
        static_cast<uint32_t>(max_selected_entries), scale, query_for_attention, key_cache_out,
        value_cache_out, &token_meta, block_table,
        main_uses_selected ? selected_indices : nullptr,
        main_uses_selected ? selected_counts : nullptr, &partial_main));
  }
  if (run_auxiliary) {
    partial_auxiliary = context.CreateGPUTensor(float_type, partial_shape);
    ORT_RETURN_IF_ERROR(CheckStageStorageBindings(
        context, "auxiliary partial attention",
        {{query_for_attention, "query"},
         {auxiliary_key, "auxiliary_key"},
         {auxiliary_kv_shared_ ? nullptr : auxiliary_value, "auxiliary_value",
          /*raw_indexed*/ true},
         {&token_meta, "token_meta"},
         {selected_indices, "selected_indices"},
         {selected_counts, "selected_counts"},
         {&partial_auxiliary, "partial"}}));
    ORT_RETURN_IF_ERROR(RunSparseAuxiliaryPartial(
        context, parameters, auxiliary_kv_shared_, static_cast<uint32_t>(auxiliary_capacity),
        static_cast<uint32_t>(auxiliary_num_heads), static_cast<uint32_t>(max_selected_entries),
        scale, query_for_attention, auxiliary_key, auxiliary_value, &token_meta, selected_indices,
        selected_counts, &partial_auxiliary));
  }

  ORT_RETURN_IF_ERROR(CheckStageStorageBindings(
      context, "softmax finalize",
      {{run_main ? &partial_main : nullptr, "partial_main", /*raw_indexed*/ true},
       {run_auxiliary ? &partial_auxiliary : nullptr, "partial_auxiliary", /*raw_indexed*/ true},
       {head_sink, "head_sink", /*raw_indexed*/ true},
       {output, "output"}}));
  return RunSparseFinalize(context, parameters, run_main ? &partial_main : nullptr,
                           run_auxiliary ? &partial_auxiliary : nullptr, head_sink, output);
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
