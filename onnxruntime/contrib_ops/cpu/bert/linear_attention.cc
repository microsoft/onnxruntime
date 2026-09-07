// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/bert/linear_attention.h"
#include "contrib_ops/cpu/bert/linear_attention_helper.h"

#include "core/framework/allocator.h"
#include "core/framework/tensorprotoutils.h"
#include "core/common/safeint.h"
#include "core/mlas/inc/mlas.h"
#include "core/platform/threadpool.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include <cstring>

using onnxruntime::concurrency::ThreadPool;

namespace onnxruntime {
namespace contrib {

// These ops are internal-only, so register outside of onnx
// Note: Only float is registered for CPU. The op schema allows float16/bfloat16
// for CUDA compatibility, but the CPU kernel computes in float32 internally.
// MLFloat16 CPU support would require input/output conversion buffers
// (MlasConvertHalfToFloatBuffer / MlasConvertFloatToHalfBuffer).
//
// The recurrence itself lives in MLAS (MlasLinearAttention, see
// core/mlas/lib/linear_attention.cpp), which owns the threading, the state
// layout and the ISA dispatch. This kernel is plumbing: it validates inputs,
// derives the head geometry, allocates the output/state/scratch buffers, and
// hands MLAS a descriptor.
#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      LinearAttention,                                            \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCpuExecutionProvider,                                      \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      LinearAttention<T>);

REGISTER_KERNEL_TYPED(float)

template <typename T>
LinearAttention<T>::LinearAttention(const OpKernelInfo& info) : OpKernel(info) {
  int64_t q_num_heads = 0;
  ORT_ENFORCE(info.GetAttr("q_num_heads", &q_num_heads).IsOK() &&
                  q_num_heads > 0 &&
                  q_num_heads <= std::numeric_limits<int>::max(),
              "q_num_heads must be an integer in [1, INT_MAX]");
  q_num_heads_ = static_cast<int>(q_num_heads);

  int64_t kv_num_heads = 0;
  ORT_ENFORCE(info.GetAttr("kv_num_heads", &kv_num_heads).IsOK() &&
                  kv_num_heads > 0 &&
                  kv_num_heads <= std::numeric_limits<int>::max(),
              "kv_num_heads must be an integer in [1, INT_MAX]");
  kv_num_heads_ = static_cast<int>(kv_num_heads);

  update_rule_ = info.GetAttrOrDefault<std::string>("update_rule", "gated_delta");
  ORT_ENFORCE(update_rule_ == "linear" || update_rule_ == "gated" ||
                  update_rule_ == "delta" || update_rule_ == "gated_delta",
              "update_rule must be one of: linear, gated, delta, gated_delta");

  if (update_rule_ == "linear") {
    rule_ = MlasLinearAttentionRuleLinear;
  } else if (update_rule_ == "gated") {
    rule_ = MlasLinearAttentionRuleGated;
  } else if (update_rule_ == "delta") {
    rule_ = MlasLinearAttentionRuleDelta;
  } else {
    rule_ = MlasLinearAttentionRuleGatedDelta;
  }

  scale_ = info.GetAttrOrDefault<float>("scale", 0.0f);

  int64_t chunk_size = info.GetAttrOrDefault<int64_t>("chunk_size", 64);
  // chunk_size_ reserved for future chunk-parallel prefill algorithm; not yet used.
  chunk_size_ = static_cast<int>(chunk_size);

  ORT_ENFORCE(info.GetAttrOrDefault<int64_t>("state_window", 0) == 0,
              "CPU LinearAttention does not support state_window > 0 (CUDA EP only)");
  state_window_ = 0;
}

template <typename T>
Status LinearAttention<T>::Compute(OpKernelContext* context) const {
  // ==== Input Retrieval ====
  const Tensor* query_tensor = context->Input<Tensor>(0);
  const Tensor* key_tensor = context->Input<Tensor>(1);         // optional
  const Tensor* value_tensor = context->Input<Tensor>(2);       // optional
  const Tensor* past_state_tensor = context->Input<Tensor>(3);  // optional
  const Tensor* decay_tensor = context->Input<Tensor>(4);       // optional
  const Tensor* beta_tensor = context->Input<Tensor>(5);        // optional

  ORT_RETURN_IF_NOT(query_tensor != nullptr, "query input is required");

  const auto& query_shape = query_tensor->Shape();
  ORT_RETURN_IF_NOT(query_shape.NumDimensions() == 3,
                    "query must be 3D [B, T, H*D], got ", query_shape.NumDimensions(), "D");

  const int64_t batch_size = query_shape[0];
  const int64_t seq_len = query_shape[1];
  const int64_t query_hidden = query_shape[2];

  // ==== Determine d_k and d_v ====
  ORT_RETURN_IF_NOT(key_tensor != nullptr && value_tensor != nullptr,
                    "key and value inputs are required");

  int64_t d_k, d_v;
  int n_k_heads;
  const float* q_data;
  const float* k_data;
  const float* v_data;

  {
    const auto& key_shape = key_tensor->Shape();
    const auto& value_shape = value_tensor->Shape();
    ORT_RETURN_IF_NOT(key_shape.NumDimensions() == 3 && value_shape.NumDimensions() == 3,
                      "key and value must be 3D");
    ORT_RETURN_IF_NOT(key_shape[0] == batch_size && value_shape[0] == batch_size,
                      "batch size mismatch");
    ORT_RETURN_IF_NOT(key_shape[1] == seq_len && value_shape[1] == seq_len,
                      "sequence length mismatch");

    d_k = query_hidden / q_num_heads_;
    ORT_RETURN_IF_NOT(query_hidden == q_num_heads_ * d_k,
                      "query hidden size must be divisible by q_num_heads");
    ORT_RETURN_IF_NOT(key_shape[2] % d_k == 0,
                      "key hidden size must be divisible by d_k");
    n_k_heads = static_cast<int>(key_shape[2] / d_k);
    d_v = value_shape[2] / kv_num_heads_;
    ORT_RETURN_IF_NOT(value_shape[2] == kv_num_heads_ * d_v,
                      "value hidden size must be divisible by kv_num_heads");

    q_data = query_tensor->Data<float>();
    k_data = key_tensor->Data<float>();
    v_data = value_tensor->Data<float>();
  }

  // ==== Determine scale ====
  float s = scale_;
  if (s == 0.0f) {
    s = 1.0f / std::sqrt(static_cast<float>(d_k));
  }

  // ==== Validate optional inputs based on update_rule ====
  bool needs_decay = (rule_ == MlasLinearAttentionRuleGated ||
                      rule_ == MlasLinearAttentionRuleGatedDelta);
  bool needs_beta = (rule_ == MlasLinearAttentionRuleDelta ||
                     rule_ == MlasLinearAttentionRuleGatedDelta);

  ORT_RETURN_IF_NOT(!needs_decay || decay_tensor != nullptr,
                    "decay input is required for update_rule=", update_rule_);
  ORT_RETURN_IF_NOT(!needs_beta || beta_tensor != nullptr,
                    "beta input is required for update_rule=", update_rule_);

  const float* decay_data = decay_tensor ? decay_tensor->Data<float>() : nullptr;
  const float* beta_data = beta_tensor ? beta_tensor->Data<float>() : nullptr;

  MLAS_LINEAR_ATTENTION_DECAY_LAYOUT decay_layout = MlasLinearAttentionDecayNone;
  if (decay_tensor != nullptr) {
    const auto& decay_shape = decay_tensor->Shape();
    ORT_RETURN_IF_NOT(decay_shape.NumDimensions() == 3,
                      "decay must be rank 3 (B, T, ...), got rank ", decay_shape.NumDimensions());
    ORT_RETURN_IF_NOT(decay_shape[0] == batch_size && decay_shape[1] == seq_len,
                      "decay dims 0/1 must match (B=", batch_size, ", T=", seq_len,
                      "), got (", decay_shape[0], ", ", decay_shape[1], ")");
    int64_t decay_last = decay_shape[2];
    if (decay_last == kv_num_heads_ * d_k) {
      decay_layout = MlasLinearAttentionDecayPerKeyDim;
    } else {
      ORT_RETURN_IF_NOT(decay_last == kv_num_heads_,
                        "decay last dim must be H_kv or H_kv*d_k");
      decay_layout = MlasLinearAttentionDecayPerHead;
    }
  }

  MLAS_LINEAR_ATTENTION_BETA_LAYOUT beta_layout = MlasLinearAttentionBetaNone;
  if (beta_tensor != nullptr) {
    const auto& beta_shape = beta_tensor->Shape();
    ORT_RETURN_IF_NOT(beta_shape.NumDimensions() == 3,
                      "beta must be rank 3 (B, T, ...), got rank ", beta_shape.NumDimensions());
    ORT_RETURN_IF_NOT(beta_shape[0] == batch_size && beta_shape[1] == seq_len,
                      "beta dims 0/1 must match (B=", batch_size, ", T=", seq_len,
                      "), got (", beta_shape[0], ", ", beta_shape[1], ")");
    int64_t beta_last = beta_shape[2];
    if (beta_last == kv_num_heads_) {
      beta_layout = MlasLinearAttentionBetaPerHead;
    } else {
      ORT_RETURN_IF_NOT(beta_last == 1, "beta last dim must be H_kv or 1");
      beta_layout = MlasLinearAttentionBetaShared;
    }
  }

  // MLAS only consumes decay/beta that the rule actually uses; a tensor
  // supplied for a rule that ignores it stays ignored.
  if (!needs_decay) {
    decay_data = nullptr;
    decay_layout = MlasLinearAttentionDecayNone;
  }
  if (!needs_beta) {
    beta_data = nullptr;
    beta_layout = MlasLinearAttentionBetaNone;
  }

  // ==== Initialize state: write directly into output present_state ====
  // state_window_ is always 0 on CPU, so this is the legacy (B, H_kv, d_k, d_v)
  // shape. MLAS updates this buffer in place and never initializes it, so the
  // seeding below is mandatory.
  TensorShape state_shape;
  ORT_RETURN_IF_ERROR(linear_attention_helper::CheckInputs(
      state_window_, static_cast<int>(batch_size), kv_num_heads_,
      static_cast<int>(d_k), static_cast<int>(d_v), past_state_tensor, state_shape));
  Tensor* present_state_tensor = context->Output(1, state_shape);
  float* state_data = present_state_tensor->MutableData<float>();
  int64_t state_per_head = d_k * d_v;
  int64_t total_state = batch_size * kv_num_heads_ * state_per_head;

  if (past_state_tensor != nullptr) {
    const float* ps_data = past_state_tensor->Data<float>();
    if (ps_data != state_data) {
      std::memcpy(state_data, ps_data, static_cast<size_t>(total_state) * sizeof(float));
    }
  } else {
    std::memset(state_data, 0, static_cast<size_t>(total_state) * sizeof(float));
  }

  // ==== Allocate output ====
  // Output hidden dim: max(q_num_heads, kv_num_heads) * d_v
  // Standard GQA: q_num_heads * d_v; Inverse GQA: kv_num_heads * d_v
  int64_t output_hidden = static_cast<int64_t>(
      MlasLinearAttentionOutputHiddenSize(q_num_heads_, kv_num_heads_, static_cast<int>(d_v)));
  TensorShape output_shape({batch_size, seq_len, output_hidden});
  Tensor* output_tensor = context->Output(0, output_shape);
  float* output_data = output_tensor->MutableData<float>();

  // ==== Validate the GQA head mapping ====
  // Standard GQA: q_num_heads >= kv_num_heads, multiple Q heads per KV group.
  // Inverse GQA: q_num_heads < kv_num_heads (e.g., Qwen3.5 9B: n_k=16, n_kv=32).
  // Also n_k_heads may differ from both (K has its own head count).
  // MLAS derives the actual mappings; the checks here exist to report a Status.
  if (q_num_heads_ >= kv_num_heads_) {
    ORT_RETURN_IF_NOT(q_num_heads_ % kv_num_heads_ == 0,
                      "q_num_heads must be divisible by kv_num_heads");
  } else {
    ORT_RETURN_IF_NOT(kv_num_heads_ % q_num_heads_ == 0,
                      "kv_num_heads must be divisible by q_num_heads (inverse GQA)");
  }

  // K-to-KV head mapping: when n_k < kv_num_heads, multiple KV heads share one K head
  ORT_RETURN_IF_NOT(kv_num_heads_ % n_k_heads == 0,
                    "kv_num_heads must be divisible by n_k_heads");

  // ==== Hand off to MLAS ====
  // Work is partitioned over (batch, kv_head) pairs: each pair owns a disjoint
  // state matrix, and the sequential token dependency lives entirely within a
  // single pair.
  auto* tp = context->GetOperatorThreadPool();

  MlasLinearAttentionArgs args;
  args.batch_size = static_cast<int>(batch_size);
  args.sequence_length = static_cast<int>(seq_len);
  args.q_num_heads = q_num_heads_;
  args.kv_num_heads = kv_num_heads_;
  args.k_num_heads = n_k_heads;
  args.k_head_size = static_cast<int>(d_k);
  args.v_head_size = static_cast<int>(d_v);
  args.rule = rule_;
  args.decay_layout = decay_layout;
  args.beta_layout = beta_layout;
  args.scale = s;

  const int thread_count = std::max(1, ThreadPool::DegreeOfParallelism(tp));
  args.thread_count = thread_count;
  args.buffer_size_per_thread =
      MlasLinearAttentionBufferSizePerThread(args.k_head_size, args.v_head_size);

  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));
  const size_t buffer_bytes = SafeInt<size_t>(args.buffer_size_per_thread) * thread_count;
  IAllocatorUniquePtr<void> buffer = IAllocator::MakeUniquePtr<void>(allocator, buffer_bytes);
  args.buffer = reinterpret_cast<float*>(buffer.get());

  args.query = q_data;
  args.key = k_data;
  args.value = v_data;
  args.decay = decay_data;
  args.beta = beta_data;
  args.state = state_data;
  args.output = output_data;

  MlasLinearAttention(&args, tp);

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
