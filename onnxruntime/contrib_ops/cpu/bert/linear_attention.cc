// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/bert/linear_attention.h"

#include "core/framework/tensorprotoutils.h"
#include "core/common/safeint.h"
#include "core/platform/threadpool.h"

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace onnxruntime {
namespace contrib {

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

#undef REGISTER_KERNEL_TYPED

namespace {
LinearAttentionUpdateRule ParseUpdateRule(const std::string& rule) {
  if (rule == "linear") {
    return LinearAttentionUpdateRule::kLinear;
  } else if (rule == "gated") {
    return LinearAttentionUpdateRule::kGated;
  } else if (rule == "delta") {
    return LinearAttentionUpdateRule::kDelta;
  } else if (rule == "gated_delta") {
    return LinearAttentionUpdateRule::kGatedDelta;
  }
  ORT_THROW("Unknown update_rule: ", rule, ". Must be one of: linear, gated, delta, gated_delta");
}
}  // namespace

template <typename T>
LinearAttention<T>::LinearAttention(const OpKernelInfo& info) : OpKernel(info) {
  std::string update_rule_str = info.GetAttrOrDefault<std::string>("update_rule", "gated_delta");
  update_rule_ = ParseUpdateRule(update_rule_str);

  scale_ = info.GetAttrOrDefault<float>("scale", 0.0f);
}

template <typename T>
Status LinearAttention<T>::Compute(OpKernelContext* context) const {
  const Tensor* query = context->Input<Tensor>(0);
  const Tensor* key = context->Input<Tensor>(1);
  const Tensor* value = context->Input<Tensor>(2);
  const Tensor* past_state = context->Input<Tensor>(3);  // optional
  const Tensor* decay = context->Input<Tensor>(4);        // optional
  const Tensor* beta = context->Input<Tensor>(5);         // optional

  // Validate inputs
  ORT_RETURN_IF_NOT(query != nullptr, "query input is required");
  ORT_RETURN_IF_NOT(key != nullptr, "key input is required");
  ORT_RETURN_IF_NOT(value != nullptr, "value input is required");

  const auto& q_shape = query->Shape();
  const auto& k_shape = key->Shape();
  const auto& v_shape = value->Shape();

  ORT_RETURN_IF_NOT(q_shape.NumDimensions() == 4, "query must be 4D: (B, H, T, d_k)");
  ORT_RETURN_IF_NOT(k_shape.NumDimensions() == 4, "key must be 4D: (B, H, T, d_k)");
  ORT_RETURN_IF_NOT(v_shape.NumDimensions() == 4, "value must be 4D: (B, H, T, d_v)");

  const int64_t batch_size = q_shape[0];
  const int64_t num_heads = q_shape[1];
  const int64_t seq_len = q_shape[2];
  const int64_t key_dim = q_shape[3];
  const int64_t value_dim = v_shape[3];

  ORT_RETURN_IF_NOT(k_shape[0] == batch_size && k_shape[1] == num_heads &&
                        k_shape[2] == seq_len && k_shape[3] == key_dim,
                    "key shape must match query in (B, H, T, d_k)");
  ORT_RETURN_IF_NOT(v_shape[0] == batch_size && v_shape[1] == num_heads && v_shape[2] == seq_len,
                    "value shape must match query in (B, H, T)");

  if (past_state != nullptr) {
    const auto& ps_shape = past_state->Shape();
    ORT_RETURN_IF_NOT(ps_shape.NumDimensions() == 4, "past_state must be 4D: (B, H, d_k, d_v)");
    ORT_RETURN_IF_NOT(ps_shape[0] == batch_size && ps_shape[1] == num_heads &&
                          ps_shape[2] == key_dim && ps_shape[3] == value_dim,
                      "past_state shape must be (B, H, d_k, d_v)");
  }

  // Validate decay is present for gated modes
  if (update_rule_ == LinearAttentionUpdateRule::kGated ||
      update_rule_ == LinearAttentionUpdateRule::kGatedDelta) {
    ORT_RETURN_IF_NOT(decay != nullptr, "decay input is required for gated and gated_delta update rules");
    const auto& decay_shape = decay->Shape();
    ORT_RETURN_IF_NOT(decay_shape.NumDimensions() == 4, "decay must be 4D");
    ORT_RETURN_IF_NOT(decay_shape[0] == batch_size && decay_shape[1] == num_heads &&
                          decay_shape[2] == seq_len,
                      "decay shape must match (B, H, T, ...)");
    // decay's last dim can be 1 (per-head scalar) or key_dim (per-key-dimension)
    ORT_RETURN_IF_NOT(decay_shape[3] == 1 || decay_shape[3] == key_dim,
                      "decay last dimension must be 1 or key_dim");
  }

  // Validate beta is present for delta modes
  if (update_rule_ == LinearAttentionUpdateRule::kDelta ||
      update_rule_ == LinearAttentionUpdateRule::kGatedDelta) {
    ORT_RETURN_IF_NOT(beta != nullptr, "beta input is required for delta and gated_delta update rules");
    const auto& beta_shape = beta->Shape();
    ORT_RETURN_IF_NOT(beta_shape.NumDimensions() == 4, "beta must be 4D");
    ORT_RETURN_IF_NOT(beta_shape[0] == batch_size && beta_shape[1] == num_heads &&
                          beta_shape[2] == seq_len && beta_shape[3] == 1,
                      "beta shape must be (B, H, T, 1)");
  }

  // Determine scale
  float scale = scale_;
  if (scale == 0.0f) {
    scale = 1.0f / std::sqrt(static_cast<float>(key_dim));
  }

  // Allocate outputs
  TensorShape output_shape({batch_size, num_heads, seq_len, value_dim});
  Tensor* output = context->Output(0, output_shape);

  TensorShape state_shape({batch_size, num_heads, key_dim, value_dim});
  Tensor* present_state = context->Output(1, state_shape);

  const T* q_data = query->Data<T>();
  const T* k_data = key->Data<T>();
  const T* v_data = value->Data<T>();
  const T* decay_data = (decay != nullptr) ? decay->Data<T>() : nullptr;
  const T* beta_data = (beta != nullptr) ? beta->Data<T>() : nullptr;

  T* output_data = output->MutableData<T>();
  T* state_data = present_state->MutableData<T>();

  // Initialize state from past_state or zero
  if (past_state != nullptr) {
    std::memcpy(state_data, past_state->Data<T>(),
                static_cast<size_t>(batch_size * num_heads * key_dim * value_dim) * sizeof(T));
  } else {
    std::memset(state_data, 0,
                static_cast<size_t>(batch_size * num_heads * key_dim * value_dim) * sizeof(T));
  }

  const bool is_gated = (update_rule_ == LinearAttentionUpdateRule::kGated ||
                         update_rule_ == LinearAttentionUpdateRule::kGatedDelta);
  const bool is_delta = (update_rule_ == LinearAttentionUpdateRule::kDelta ||
                         update_rule_ == LinearAttentionUpdateRule::kGatedDelta);

  const int64_t decay_key_stride = (decay != nullptr) ? decay->Shape()[3] : 0;

  // Process each batch and head
  for (int64_t b = 0; b < batch_size; ++b) {
    for (int64_t h = 0; h < num_heads; ++h) {
      // State pointer: state[b, h, :, :] = (key_dim x value_dim)
      T* S = state_data + (b * num_heads + h) * key_dim * value_dim;

      for (int64_t t = 0; t < seq_len; ++t) {
        // q[b, h, t, :] = (key_dim,)
        const T* q_t = q_data + ((b * num_heads + h) * seq_len + t) * key_dim;
        // k[b, h, t, :] = (key_dim,)
        const T* k_t = k_data + ((b * num_heads + h) * seq_len + t) * key_dim;
        // v[b, h, t, :] = (value_dim,)
        const T* v_t = v_data + ((b * num_heads + h) * seq_len + t) * value_dim;
        // output[b, h, t, :] = (value_dim,)
        T* o_t = output_data + ((b * num_heads + h) * seq_len + t) * value_dim;

        // Compute per-key-dim decay factors: g[k] = exp(decay[b, h, t, k])
        // decay_data may be broadcastable (last dim 1 or key_dim)
        std::vector<T> g(static_cast<size_t>(key_dim), static_cast<T>(1));
        if (is_gated && decay_data != nullptr) {
          const T* decay_t = decay_data + ((b * num_heads + h) * seq_len + t) * decay_key_stride;
          for (int64_t dk = 0; dk < key_dim; ++dk) {
            int64_t decay_idx = (decay_key_stride == 1) ? 0 : dk;
            g[static_cast<size_t>(dk)] = static_cast<T>(std::exp(static_cast<float>(decay_t[decay_idx])));
          }
        }

        // Step 1: Apply decay to state: S = diag(g) * S
        if (is_gated) {
          for (int64_t dk = 0; dk < key_dim; ++dk) {
            for (int64_t dv = 0; dv < value_dim; ++dv) {
              S[dk * value_dim + dv] *= g[static_cast<size_t>(dk)];
            }
          }
        }

        if (is_delta) {
          // Step 2 (Delta modes): Compute retrieved = S^T k
          // retrieved[dv] = sum_k S[k, dv] * k_t[k]
          std::vector<T> retrieved(static_cast<size_t>(value_dim), static_cast<T>(0));
          for (int64_t dk = 0; dk < key_dim; ++dk) {
            for (int64_t dv = 0; dv < value_dim; ++dv) {
              retrieved[static_cast<size_t>(dv)] += S[dk * value_dim + dv] * k_t[dk];
            }
          }

          // Step 3: Compute delta = beta * (v - retrieved)
          const T* beta_t = beta_data + ((b * num_heads + h) * seq_len + t);
          T beta_val = *beta_t;

          // Step 4: Update state: S += k outer delta = k outer (beta * (v - retrieved))
          for (int64_t dk = 0; dk < key_dim; ++dk) {
            for (int64_t dv = 0; dv < value_dim; ++dv) {
              T delta = beta_val * (v_t[dv] - retrieved[static_cast<size_t>(dv)]);
              S[dk * value_dim + dv] += k_t[dk] * delta;
            }
          }
        } else {
          // Non-delta modes: S += k outer v
          for (int64_t dk = 0; dk < key_dim; ++dk) {
            for (int64_t dv = 0; dv < value_dim; ++dv) {
              S[dk * value_dim + dv] += k_t[dk] * v_t[dv];
            }
          }
        }

        // Step 5: Compute output: o = scale * q^T S
        // o[dv] = scale * sum_k q_t[k] * S[k, dv]
        for (int64_t dv = 0; dv < value_dim; ++dv) {
          T sum = static_cast<T>(0);
          for (int64_t dk = 0; dk < key_dim; ++dk) {
            sum += q_t[dk] * S[dk * value_dim + dv];
          }
          o_t[dv] = static_cast<T>(static_cast<float>(sum) * scale);
        }
      }
    }
  }

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
