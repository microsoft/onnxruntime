// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/varlen_linear_attention.h"
#include "contrib_ops/cpu/bert/linear_attention_helper.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_type_conversion.h"

#include <limits>

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;  // CudaKernel, Stream, GetDeviceProp, ToCudaType

#define REGISTER_KERNEL_TYPED(T)                                        \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                        \
      VarlenLinearAttention,                                            \
      kMSDomain,                                                        \
      1,                                                                \
      T,                                                                \
      kCudaExecutionProvider,                                           \
      (*KernelDefBuilder::Create())                                     \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())        \
          .TypeConstraint("S", DataTypeImpl::GetTensorType<T>())        \
          .TypeConstraint("M", DataTypeImpl::GetTensorType<int32_t>()), \
      VarlenLinearAttention<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

template <typename T>
VarlenLinearAttention<T>::VarlenLinearAttention(const OpKernelInfo& info) : CudaKernel(info) {
  int64_t q_num_heads = 0;
  ORT_ENFORCE(info.GetAttr("q_num_heads", &q_num_heads).IsOK() &&
                  q_num_heads > 0 && q_num_heads <= std::numeric_limits<int>::max(),
              "q_num_heads must be in [1, INT_MAX]");
  q_num_heads_ = static_cast<int>(q_num_heads);

  int64_t kv_num_heads = 0;
  ORT_ENFORCE(info.GetAttr("kv_num_heads", &kv_num_heads).IsOK() &&
                  kv_num_heads > 0 && kv_num_heads <= std::numeric_limits<int>::max(),
              "kv_num_heads must be in [1, INT_MAX]");
  kv_num_heads_ = static_cast<int>(kv_num_heads);

  update_rule_ = info.GetAttrOrDefault<std::string>("update_rule", "gated_delta");
  ORT_ENFORCE(update_rule_ == "linear" || update_rule_ == "gated" ||
                  update_rule_ == "delta" || update_rule_ == "gated_delta",
              "update_rule must be one of: linear, gated, delta, gated_delta");
  scale_ = info.GetAttrOrDefault<float>("scale", 0.0f);

  int64_t chunk_size = info.GetAttrOrDefault<int64_t>("chunk_size", 64);
  // chunk_size_ reserved for a future chunk-parallel prefill algorithm; not yet used.
  chunk_size_ = static_cast<int>(chunk_size);

  ORT_THROW_IF_ERROR(linear_attention_helper::ParseStateWindow(info, state_window_));
}

template <typename T>
Status VarlenLinearAttention<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* query_tensor = context->Input<Tensor>(0);
  const Tensor* key_tensor = context->Input<Tensor>(1);
  const Tensor* value_tensor = context->Input<Tensor>(2);
  const Tensor* cu_seqlens_tensor = context->Input<Tensor>(3);
  const Tensor* past_state_tensor = context->Input<Tensor>(4);  // optional
  const Tensor* decay_tensor = context->Input<Tensor>(5);       // optional
  const Tensor* beta_tensor = context->Input<Tensor>(6);        // optional

  ORT_RETURN_IF_NOT(query_tensor != nullptr && key_tensor != nullptr && value_tensor != nullptr,
                    "query, key and value inputs are required");
  ORT_RETURN_IF_NOT(cu_seqlens_tensor != nullptr, "cumulative_sequence_length input is required");

  const auto& query_shape = query_tensor->Shape();
  ORT_RETURN_IF_NOT(query_shape.NumDimensions() == 2,
                    "query must be rank 2 (total_tokens, q_num_heads * d_k), got rank ",
                    query_shape.NumDimensions());

  const int64_t total_tokens_64 = query_shape[0];
  const int64_t query_hidden_64 = query_shape[1];
  ORT_RETURN_IF_NOT(total_tokens_64 <= std::numeric_limits<int>::max() &&
                        query_hidden_64 <= std::numeric_limits<int>::max(),
                    "query dimensions are too large for the CUDA kernel");

  const auto& cu_seqlens_shape = cu_seqlens_tensor->Shape();
  ORT_RETURN_IF_NOT(cu_seqlens_shape.NumDimensions() == 1,
                    "cumulative_sequence_length must be rank 1 (batch_size + 1), got rank ",
                    cu_seqlens_shape.NumDimensions());
  ORT_RETURN_IF_NOT(cu_seqlens_shape[0] >= 2,
                    "cumulative_sequence_length must have at least 2 elements (batch_size >= 1), got ",
                    cu_seqlens_shape[0]);
  // batch_size = cu_seqlens.Shape()[0] - 1. Never derived from the packed token dimension: that
  // is the whole point of a ragged batch, where total_tokens has no fixed relationship to
  // batch_size beyond total_tokens >= batch_size.
  const int64_t batch_size_64 = cu_seqlens_shape[0] - 1;
  ORT_RETURN_IF_NOT(batch_size_64 <= std::numeric_limits<int>::max(),
                    "batch size is too large for the CUDA kernel");
  ORT_RETURN_IF_NOT(total_tokens_64 >= batch_size_64,
                    "total_tokens must be at least batch_size because every sequence must contain a token");
  const int batch_size = static_cast<int>(batch_size_64);

  const auto& key_shape = key_tensor->Shape();
  const auto& value_shape = value_tensor->Shape();
  ORT_RETURN_IF_NOT(key_shape.NumDimensions() == 2 && value_shape.NumDimensions() == 2,
                    "key and value must be rank 2 (total_tokens, hidden)");
  ORT_RETURN_IF_NOT(key_shape[1] <= std::numeric_limits<int>::max() &&
                        value_shape[1] <= std::numeric_limits<int>::max(),
                    "key and value dimensions are too large for the CUDA kernel");
  ORT_RETURN_IF_NOT(key_shape[0] == total_tokens_64 && value_shape[0] == total_tokens_64,
                    "key and value token dimensions must match query");
  ORT_RETURN_IF_NOT(query_hidden_64 > 0 && query_hidden_64 % q_num_heads_ == 0,
                    "query last dim (", query_hidden_64, ") must be positive and divisible by q_num_heads (",
                    q_num_heads_, ")");
  ORT_RETURN_IF_NOT(value_shape[1] > 0 && value_shape[1] % kv_num_heads_ == 0,
                    "value last dim (", value_shape[1], ") must be positive and divisible by kv_num_heads (",
                    kv_num_heads_, ")");
  const int d_k = static_cast<int>(query_hidden_64 / q_num_heads_);
  int d_v = static_cast<int>(value_shape[1]) / kv_num_heads_;
  ORT_RETURN_IF_NOT(key_shape[1] > 0 && key_shape[1] % d_k == 0,
                    "key last dim (", key_shape[1], ") must be divisible by d_k (", d_k, ")");
  int n_k_heads = static_cast<int>(key_shape[1]) / d_k;

  // GQA head mapping validations
  if (q_num_heads_ >= kv_num_heads_) {
    ORT_ENFORCE(q_num_heads_ % kv_num_heads_ == 0,
                "q_num_heads must be divisible by kv_num_heads");
  } else {
    ORT_ENFORCE(kv_num_heads_ % q_num_heads_ == 0,
                "kv_num_heads must be divisible by q_num_heads (inverse GQA)");
  }
  ORT_ENFORCE(kv_num_heads_ % n_k_heads == 0,
              "kv_num_heads must be divisible by n_k_heads");

  float s = scale_;
  if (s == 0.0f) {
    s = 1.0f / std::sqrt(static_cast<float>(d_k));
  }

  bool needs_decay = (update_rule_ == "gated" || update_rule_ == "gated_delta");
  bool needs_beta = (update_rule_ == "delta" || update_rule_ == "gated_delta");
  bool needs_retrieval = (update_rule_ == "delta" || update_rule_ == "gated_delta");

  ORT_ENFORCE(!needs_decay || decay_tensor != nullptr,
              "decay input is required for update_rule=", update_rule_);
  ORT_ENFORCE(!needs_beta || beta_tensor != nullptr,
              "beta input is required for update_rule=", update_rule_);

  bool decay_per_key_dim = false;
  if (decay_tensor != nullptr) {
    const auto& decay_shape = decay_tensor->Shape();
    ORT_RETURN_IF_NOT(decay_shape.NumDimensions() == 2,
                      "decay must be rank 2 (total_tokens, ...), got rank ", decay_shape.NumDimensions());
    ORT_RETURN_IF_NOT(decay_shape[0] == total_tokens_64,
                      "decay token dimension must match query");
    const int64_t decay_last = decay_shape[1];
    if (decay_last == static_cast<int64_t>(kv_num_heads_) * d_k) {
      decay_per_key_dim = true;
    } else {
      ORT_RETURN_IF_NOT(decay_last == kv_num_heads_,
                        "decay last dim must be H_kv or H_kv*d_k");
    }
  }

  bool beta_per_head = false;
  if (beta_tensor != nullptr) {
    const auto& beta_shape = beta_tensor->Shape();
    ORT_RETURN_IF_NOT(beta_shape.NumDimensions() == 2,
                      "beta must be rank 2 (total_tokens, ...), got rank ", beta_shape.NumDimensions());
    ORT_RETURN_IF_NOT(beta_shape[0] == total_tokens_64,
                      "beta token dimension must match query");
    const int64_t beta_last = beta_shape[1];
    if (beta_last == kv_num_heads_) {
      beta_per_head = true;
    } else {
      ORT_RETURN_IF_NOT(beta_last == 1, "beta last dim must be H_kv or 1");
    }
  }

  // Allocate outputs
  const int64_t output_hidden_64 = static_cast<int64_t>(std::max(q_num_heads_, kv_num_heads_)) * d_v;
  ORT_RETURN_IF_NOT(output_hidden_64 <= std::numeric_limits<int>::max(),
                    "output hidden dimension is too large for the CUDA kernel");
  TensorShape output_shape({total_tokens_64, output_hidden_64});
  Tensor* output_tensor = context->Output(0, output_shape);

  // past_state / present_state are [batch_size, H_kv, d_k, d_v], or [W, batch_size, H_kv, d_k,
  // d_v] when state_window_ = W > 0. batch_size here is the number of packed sequences (from
  // cu_seqlens), not total_tokens.
  const int state_slots = state_window_ > 0 ? state_window_ : 1;
  TensorShape state_shape;
  ORT_RETURN_IF_ERROR(linear_attention_helper::CheckInputs(
      state_window_, batch_size, kv_num_heads_, d_k, d_v, past_state_tensor, state_shape));
  Tensor* present_state_tensor = context->Output(1, state_shape);

  T* present_state_data = present_state_tensor->MutableData<T>();
  const T* initial_state_data = present_state_data;

  // Every sequence's own length decides which window slots it writes (right-aligned: slot
  // t + W - L), so which slots stay untouched varies per sequence -- unlike the dense op, no
  // single contiguous prefix is safe to skip. Zero the whole fresh present_state buffer
  // unconditionally so every slot this call does not write is well-defined, whether or not
  // past_state was provided.
  CUDA_RETURN_IF_ERROR(cudaMemsetAsync(
      present_state_data, 0, present_state_tensor->SizeInBytes(), Stream(context)));

  if (past_state_tensor != nullptr) {
    initial_state_data = past_state_tensor->Data<T>();
  }

  // total_tokens == batch_size is a host-visible shape fact under the trusted offsets contract:
  // every sequence contributes at least one token, so this can only hold when every sequence
  // contributes exactly one. It selects the dedicated fast path without reading cu_seqlens.
  const bool all_ones = (total_tokens_64 == batch_size_64);

  typedef typename OrtToCudaType<T>::type CudaT;

  return LaunchVarlenLinearAttentionKernel<CudaT>(
      Stream(context),
      reinterpret_cast<const CudaT*>(query_tensor->Data<T>()),
      reinterpret_cast<const CudaT*>(key_tensor->Data<T>()),
      reinterpret_cast<const CudaT*>(value_tensor->Data<T>()),
      decay_tensor ? reinterpret_cast<const CudaT*>(decay_tensor->Data<T>()) : nullptr,
      beta_tensor ? reinterpret_cast<const CudaT*>(beta_tensor->Data<T>()) : nullptr,
      reinterpret_cast<CudaT*>(output_tensor->MutableData<T>()),
      reinterpret_cast<const CudaT*>(initial_state_data),
      reinterpret_cast<CudaT*>(present_state_data),
      cu_seqlens_tensor->Data<int32_t>(),
      batch_size,
      all_ones,
      q_num_heads_,
      kv_num_heads_,
      n_k_heads,
      d_k,
      d_v,
      s,
      needs_decay,
      decay_per_key_dim,
      needs_beta,
      beta_per_head,
      needs_retrieval,
      GetDeviceProp().multiProcessorCount,
      GetDeviceProp().maxThreadsPerBlock,
      GetDeviceProp().sharedMemPerBlockOptin,
      state_slots);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
