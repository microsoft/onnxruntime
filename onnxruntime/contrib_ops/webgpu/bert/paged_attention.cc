// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/bert/paged_attention.h"

#include "contrib_ops/cpu/bert/attention_parameters.h"
#include "contrib_ops/cpu/bert/paged_attention_helper.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "core/framework/tensorprotoutils.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

// v1 registers only against T = float16, S = int32. See
// docs/design/webgpu_paged_attention.md §3 for the schema surface and
// §5 for the phased delivery plan.
ONNX_OPERATOR_KERNEL_EX(
    PagedAttention,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>())
        .TypeConstraint("S", DataTypeImpl::GetTensorType<int32_t>()),
    PagedAttention);

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
  // and cache-alias enforcement. Actual kernel dispatch (decode / prefill)
  // lands in Phase 1b.
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
  // schema always alias inputs 3 and 4. GenAI wires this via IO-binding: the
  // same OrtValue is bound to input 3 and output 1 (and 4/2). We do not
  // declare Alias() on the KernelDef — matching the CUDA implementation — and
  // instead enforce pointer equality at Compute time. If the caller has not
  // set up the aliasing, this hard-errors with a specific message rather than
  // silently writing to the wrong buffer.
  TensorShapeVector cache_shape{static_cast<int64_t>(parameters.num_blocks),
                                static_cast<int64_t>(parameters.block_size),
                                static_cast<int64_t>(parameters.kv_num_heads),
                                static_cast<int64_t>(parameters.head_size)};
  Tensor* key_cache_out = context.Output(1, cache_shape);
  Tensor* value_cache_out = context.Output(2, cache_shape);

  if (key_cache_out != nullptr &&
      key_cache->DataRaw() != key_cache_out->MutableDataRaw()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "key_cache and key_cache_out must be the same buffer "
                           "(IO-bind the same OrtValue to input 3 and output 1).");
  }
  if (value_cache_out != nullptr &&
      value_cache->DataRaw() != value_cache_out->MutableDataRaw()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "value_cache and value_cache_out must be the same buffer "
                           "(IO-bind the same OrtValue to input 4 and output 2).");
  }

  // Empty-query fast path: output is [0, hidden_size] and caches are pre-aliased,
  // so there is no kernel work to do.
  if (parameters.token_count == 0) {
    return Status::OK();
  }

  // Silence unused-variable warnings until Phase 1b consumes these.
  (void)output;
  (void)block_table;
  (void)cumulative_seqlens_q;
  (void)past_seqlens;

  // Decode-vs-prefill routing heuristic: token_count == batch_size means each
  // sequence contributed exactly one query token (pure decode / continuous
  // batching). Any other token_count means at least one sequence is in prefill
  // or a mixed decode+prefill batch — both take the gather-then-flash path in
  // Phase 1 (see design doc §5, §6).
  const bool is_pure_decode = parameters.token_count == parameters.batch_size;
  if (is_pure_decode) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "PagedAttention (WebGPU): decode kernel dispatch is not "
                           "yet implemented. See docs/design/webgpu_paged_attention.md §5 "
                           "(Phase 1b).");
  }
  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                         "PagedAttention (WebGPU): prefill (gather-then-flash) kernel "
                         "dispatch is not yet implemented. See docs/design/webgpu_paged_attention.md §5 "
                         "(Phase 1b).");
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
