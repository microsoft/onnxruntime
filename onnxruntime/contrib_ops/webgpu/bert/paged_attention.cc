// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/bert/paged_attention.h"

#include "contrib_ops/cpu/bert/attention_parameters.h"
#include "contrib_ops/cpu/bert/paged_attention_helper.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
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
  const uint32_t kv_hidden_size = static_cast<uint32_t>(parameters.kv_hidden_size);
  const uint32_t block_size = static_cast<uint32_t>(parameters.block_size);
  const uint32_t max_num_blocks_per_seq = static_cast<uint32_t>(parameters.max_num_blocks_per_seq);
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
          {kv_hidden_size},
          {block_size},
          {max_num_blocks_per_seq},
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
          {token_count},
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
  scale_ = info.GetAttrOrDefault<float>("scale", 0.0f);
  softcap_ = info.GetAttrOrDefault<float>("softcap", 0.0f);
}

Status PagedAttention::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {
  // See docs/design/webgpu_paged_attention.md §5 for the phased plan.
  // Phase 1a wires up input parsing, shape validation, output allocation,
  // and the aliased-vs-non-aliased cache branching. Actual kernel dispatch
  // (decode / prefill) lands in Phase 1b.3 / 1b.4.
  const Tensor* query = context.Input<Tensor>(0);
  const Tensor* key = context.Input<Tensor>(1);
  const Tensor* value = context.Input<Tensor>(2);
  const Tensor* key_cache = context.Input<Tensor>(3);
  const Tensor* value_cache = context.Input<Tensor>(4);
  const Tensor* cumulative_seqlens_q = context.Input<Tensor>(5);
  const Tensor* past_seqlens = context.Input<Tensor>(6);
  const Tensor* block_table = context.Input<Tensor>(7);
  const Tensor* cos_cache = context.Input<Tensor>(8);
  const Tensor* sin_cache = context.Input<Tensor>(9);

  PagedAttentionParameters parameters{};
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
                                                          &parameters,
                                                          num_heads_,
                                                          kv_num_heads_,
                                                          scale_,
                                                          softcap_,
                                                          /*max_threads_per_block*/ 0));
  parameters.local_window_size = local_window_size_;
  parameters.do_rotary = do_rotary_;
  parameters.rotary_interleaved = rotary_interleaved_;

  if (do_rotary_ && (cos_cache == nullptr || sin_cache == nullptr)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "cos_cache and sin_cache must be passed to PagedAttention when do_rotary = 1");
  }

  // Output 0: packed attention output, shape [token_count, q_hidden_size].
  TensorShapeVector output_shape{static_cast<int64_t>(parameters.token_count),
                                 static_cast<int64_t>(parameters.hidden_size)};
  Tensor* output = context.Output(0, output_shape);

  // Outputs 1 and 2 (key_cache_out, value_cache_out) are optional and per the
  // schema alias inputs 3 and 4 in the fast path. In production GenAI wires
  // this via IO-binding — the same OrtValue is bound to input 3 and output 1
  // (and 4/2) so key_cache->DataRaw() == key_cache_out->MutableDataRaw() and
  // the scatter writes land directly in the caller's cache buffers.
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
  Tensor* key_cache_out = context.Output(1, cache_shape);
  Tensor* value_cache_out = context.Output(2, cache_shape);
  ORT_ENFORCE(key_cache_out != nullptr && value_cache_out != nullptr,
              "PagedAttention (WebGPU): key_cache_out and value_cache_out outputs "
              "are required (both must be present or both absent per the schema).");

  const bool key_cache_aliased = (key_cache->DataRaw() == key_cache_out->MutableDataRaw());
  const bool value_cache_aliased = (value_cache->DataRaw() == value_cache_out->MutableDataRaw());

  // Empty-query fast path: output is [0, hidden_size] and caches are pre-aliased,
  // so there is no kernel work to do.
  if (parameters.token_count == 0) {
    return Status::OK();
  }

  // Packed-QKV support (Phase 1b.2b): the CUDA kernel treats packed-QKV as a
  // special addressing mode inside the rotary + scatter kernels. On WebGPU we
  // instead materialize three standalone Q, K, V tensors from the packed input
  // and then fall through to the same non-packed rotary + scatter pipeline
  // from Phase 1b.2. That trades some memory bandwidth for a clean reuse of
  // the existing kernels; perf-focused fusion can revisit this in Phase 1c.
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

  // Scatter new K,V tokens into the paged cache. In the aliased fast path this
  // writes directly into the caller's cache buffers. In the non-aliased path
  // (OpTester and any future non-IO-bound consumer) we first materialize the
  // input caches into the freshly allocated output tensors so untouched slots
  // are preserved through the op.
  if (!key_cache_aliased) {
    ORT_RETURN_IF_ERROR(context.CopyTensor(*key_cache, *key_cache_out));
  }
  if (!value_cache_aliased) {
    ORT_RETURN_IF_ERROR(context.CopyTensor(*value_cache, *value_cache_out));
  }

  // Phase 1b.2: fused rotary path. Mirrors paged_attention_impl.cu:
  //   1) rotate query into a workspace (in production this feeds the
  //      attention kernel; in Phase 1b.2 there is no attention yet, so we
  //      route rotated Q into `output` — the same buffer 1b.3 will overwrite
  //      with attention results — so tests can validate Q rotation directly
  //      instead of leaving it unexercised until 1b.3);
  //   2) rotate key into a temp workspace tensor;
  //   3) scatter the rotated key alongside the untouched value into the
  //      paged cache using the same ScatterKVToPagedCacheProgram from 1b.1.
  //
  // When do_rotary_ == false, key is scattered directly (Phase 1b.1 flow)
  // and `output` is filled with zeros as a well-defined placeholder for the
  // attention result until 1b.3 lands.
  const Tensor* key_for_scatter = key;
  Tensor rotated_key_tensor;
  if (do_rotary_) {
    ORT_RETURN_IF_ERROR(RunRotaryEmbedding(context, parameters,
                                           static_cast<uint32_t>(parameters.num_heads),
                                           rotary_interleaved_,
                                           query, cos_cache, sin_cache,
                                           cumulative_seqlens_q, past_seqlens,
                                           output));

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

  // TODO(Phase 1b.3 / 1b.4): dispatch the attention kernel over the newly
  // updated paged cache. Until that lands, `output` holds either a
  // well-defined zero (do_rotary_ == false) or the rotated query
  // (do_rotary_ == true) so tests can validate the pre-attention pipeline.
  if (!do_rotary_) {
    context.FillZero(*output);
  }
  return Status::OK();
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
