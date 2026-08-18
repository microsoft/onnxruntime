// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/linear_attention.h"
#include "contrib_ops/cuda/bert/linear_attention_impl.h"
#include "contrib_ops/cpu/bert/linear_attention_helper.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_type_conversion.h"
#include "core/platform/env_var_utils.h"

#include <limits>

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;  // CudaKernel, Stream, GetDeviceProp, ToCudaType

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      LinearAttention,                                            \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      LinearAttention<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

template <typename T>
LinearAttention<T>::LinearAttention(const OpKernelInfo& info) : CudaKernel(info) {
  int64_t q_num_heads = 0;
  ORT_ENFORCE(info.GetAttr("q_num_heads", &q_num_heads).IsOK() && q_num_heads > 0);
  q_num_heads_ = static_cast<int>(q_num_heads);

  int64_t kv_num_heads = 0;
  ORT_ENFORCE(info.GetAttr("kv_num_heads", &kv_num_heads).IsOK() && kv_num_heads > 0);
  kv_num_heads_ = static_cast<int>(kv_num_heads);

  update_rule_ = info.GetAttrOrDefault<std::string>("update_rule", "gated_delta");
  ORT_ENFORCE(update_rule_ == "linear" || update_rule_ == "gated" ||
                  update_rule_ == "delta" || update_rule_ == "gated_delta",
              "update_rule must be one of: linear, gated, delta, gated_delta");
  scale_ = info.GetAttrOrDefault<float>("scale", 0.0f);

  int64_t chunk_size = info.GetAttrOrDefault<int64_t>("chunk_size", 64);
  // chunk_size_ reserved for future chunk-parallel prefill algorithm; not yet used.
  chunk_size_ = static_cast<int>(chunk_size);

  // Only the trailing states are ever consumed (speculative-decoding rollback), while one state
  // per token costs d_k*d_v per token per layer -- ~88 GB for a 2.8k prefill on a 30-layer model.
  // A window caps both the allocation and the write traffic; 0 keeps the plain 4D single state.
  ORT_THROW_IF_ERROR(linear_attention_helper::ParseStateWindow(info, state_window_));

  decode_seq_threshold_ =
      ParseEnvironmentVariableWithDefault<int>("ORT_LINEAR_ATTENTION_COL_SEQ_THRESHOLD", 16);
  row_split_ = ParseEnvironmentVariableWithDefault<int>("ORT_LINEAR_ATTENTION_ROW_SPLIT", 8);
}

template <typename T>
Status LinearAttention<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* query_tensor = context->Input<Tensor>(0);
  const Tensor* key_tensor = context->Input<Tensor>(1);         // optional
  const Tensor* value_tensor = context->Input<Tensor>(2);       // optional
  const Tensor* past_state_tensor = context->Input<Tensor>(3);  // optional
  const Tensor* decay_tensor = context->Input<Tensor>(4);       // optional
  const Tensor* beta_tensor = context->Input<Tensor>(5);        // optional

  ORT_RETURN_IF_NOT(query_tensor != nullptr, "query input is required");

  const auto& query_shape = query_tensor->Shape();
  ORT_RETURN_IF_NOT(query_shape.NumDimensions() == 3, "query must be 3D");

  const int64_t batch_size_64 = query_shape[0];
  const int64_t seq_len_64 = query_shape[1];
  const int64_t query_hidden_64 = query_shape[2];
  ORT_RETURN_IF_NOT(batch_size_64 <= std::numeric_limits<int>::max() &&
                        seq_len_64 <= std::numeric_limits<int>::max() &&
                        query_hidden_64 <= std::numeric_limits<int>::max(),
                    "query dimensions are too large for the CUDA kernel");
  const int batch_size = static_cast<int>(batch_size_64);
  const int seq_len = static_cast<int>(seq_len_64);

  ORT_RETURN_IF_NOT(seq_len > 0, "sequence length must be positive, got ", seq_len);

  ORT_RETURN_IF_NOT(key_tensor != nullptr && value_tensor != nullptr, "key and value inputs are required");

  const auto& key_shape = key_tensor->Shape();
  const auto& value_shape = value_tensor->Shape();

  ORT_RETURN_IF_NOT(key_shape.NumDimensions() == 3 && value_shape.NumDimensions() == 3,
                    "key and value must be 3D");
  ORT_RETURN_IF_NOT(key_shape[2] <= std::numeric_limits<int>::max() &&
                        value_shape[2] <= std::numeric_limits<int>::max(),
                    "key and value dimensions are too large for the CUDA kernel");
  ORT_RETURN_IF_NOT(key_shape[0] == query_shape[0] && value_shape[0] == query_shape[0],
                    "key and value batch dimensions must match query");
  ORT_RETURN_IF_NOT(key_shape[1] == query_shape[1] && value_shape[1] == query_shape[1],
                    "key and value sequence dimensions must match query");
  ORT_RETURN_IF_NOT(query_hidden_64 > 0 && query_hidden_64 % q_num_heads_ == 0,
                    "query last dim (", query_hidden_64, ") must be positive and divisible by q_num_heads (",
                    q_num_heads_, ")");
  ORT_RETURN_IF_NOT(value_shape[2] > 0 && value_shape[2] % kv_num_heads_ == 0,
                    "value last dim (", value_shape[2], ") must be positive and divisible by kv_num_heads (",
                    kv_num_heads_, ")");
  const int d_k = static_cast<int>(query_hidden_64 / q_num_heads_);
  int d_v = static_cast<int>(value_shape[2]) / kv_num_heads_;
  ORT_RETURN_IF_NOT(key_shape[2] > 0 && key_shape[2] % d_k == 0,
                    "key last dim (", key_shape[2], ") must be divisible by d_k (", d_k, ")");
  int n_k_heads = static_cast<int>(key_shape[2]) / d_k;

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
    ORT_RETURN_IF_NOT(decay_shape.NumDimensions() == 3,
                      "decay must be rank 3 (B, T, ...), got rank ", decay_shape.NumDimensions());
    ORT_RETURN_IF_NOT(decay_shape[0] == batch_size && decay_shape[1] == seq_len,
                      "decay batch/sequence dimensions must match query");
    const int64_t decay_last = decay_shape[2];
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
    ORT_RETURN_IF_NOT(beta_shape.NumDimensions() == 3,
                      "beta must be rank 3 (B, T, ...), got rank ", beta_shape.NumDimensions());
    ORT_RETURN_IF_NOT(beta_shape[0] == batch_size && beta_shape[1] == seq_len,
                      "beta batch/sequence dimensions must match query");
    const int64_t beta_last = beta_shape[2];
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
  int output_hidden = static_cast<int>(output_hidden_64);
  TensorShape output_shape({batch_size, seq_len, output_hidden});
  Tensor* output_tensor = context->Output(0, output_shape);

  // past_state / present_state are [B, H_kv, d_k, d_v], or [W, B, H_kv, d_k, d_v] when
  // state_window_ = W > 0. Right-aligned: token t lands in slot t + W - seq_len, so slot W-1
  // always holds the state after the last token (and is the slot past_state is read from) and,
  // when W < seq_len, only the last W states are written.
  const int state_slots = state_window_ > 0 ? state_window_ : 1;
  TensorShape state_shape;
  ORT_RETURN_IF_ERROR(linear_attention_helper::CheckInputs(
      state_window_, batch_size, kv_num_heads_, d_k, d_v, past_state_tensor, state_shape));
  Tensor* present_state_tensor = context->Output(1, state_shape);

  T* present_state_data = present_state_tensor->MutableData<T>();
  const T* initial_state_data = present_state_data;

  // If past_state is nullptr, zero-init the buffer used as the initial state. Only slot W-1 is
  // actually read by the kernel, but zeroing the whole thing also defines the slots below
  // W - seq_len, which the kernel deliberately leaves alone when the window is wider than the
  // sequence (that is what bounds the per-step work).
  if (past_state_tensor == nullptr) {
    CUDA_RETURN_IF_ERROR(cudaMemsetAsync(
        present_state_data, 0,
        present_state_tensor->SizeInBytes(),
        Stream(context)));
  } else {
    initial_state_data = past_state_tensor->Data<T>();

    // present_state is a freshly allocated output that never aliases past_state, so the slots the
    // kernel skips would otherwise be uninitialized device memory. Zero them instead.
    if (state_window_ > 0 && state_slots > seq_len) {
      const size_t leading_bytes = static_cast<size_t>(state_slots - seq_len) *
                                   (present_state_tensor->SizeInBytes() / static_cast<size_t>(state_slots));
      if (leading_bytes > 0) {
        CUDA_RETURN_IF_ERROR(cudaMemsetAsync(present_state_data, 0, leading_bytes, Stream(context)));
      }
    }
  }

  typedef typename OrtToCudaType<T>::type CudaT;

  return LaunchLinearAttentionKernel<CudaT>(
      Stream(context),
      reinterpret_cast<const CudaT*>(query_tensor->Data<T>()),
      reinterpret_cast<const CudaT*>(key_tensor->Data<T>()),
      reinterpret_cast<const CudaT*>(value_tensor->Data<T>()),
      decay_tensor ? reinterpret_cast<const CudaT*>(decay_tensor->Data<T>()) : nullptr,
      beta_tensor ? reinterpret_cast<const CudaT*>(beta_tensor->Data<T>()) : nullptr,
      reinterpret_cast<CudaT*>(output_tensor->MutableData<T>()),
      reinterpret_cast<const CudaT*>(initial_state_data),
      reinterpret_cast<CudaT*>(present_state_data),
      batch_size,
      seq_len,
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
      decode_seq_threshold_,
      row_split_,
      GetDeviceProp().multiProcessorCount,
      GetDeviceProp().maxThreadsPerBlock,
      GetDeviceProp().sharedMemPerBlockOptin,
      state_slots);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
