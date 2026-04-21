// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/bert/linear_attention.h"

#include "core/framework/tensorprotoutils.h"
#include "core/common/safeint.h"
#include "core/mlas/inc/mlas.h"
#include "core/platform/threadpool.h"

#include <cmath>
#include <vector>

using onnxruntime::concurrency::ThreadPool;

namespace onnxruntime {
namespace contrib {

// These ops are internal-only, so register outside of onnx
// Note: Only float is registered for CPU. The op schema allows float16/bfloat16
// for CUDA compatibility, but the CPU kernel computes in float32 internally.
// MLFloat16 CPU support would require input/output conversion buffers
// (MlasConvertHalfToFloatBuffer / MlasConvertFloatToHalfBuffer).
//
// MLAS usage: MlasGemm is used for retrieval (S^T @ k), state update (k ⊗ delta),
// and query readout (S^T @ q) when d_k * d_v >= 4096. Smaller dimensions use
// scalar loops to avoid MLAS overhead.
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
  ORT_ENFORCE(info.GetAttr("q_num_heads", &q_num_heads).IsOK() && q_num_heads > 0,
              "q_num_heads must be a positive integer");
  q_num_heads_ = static_cast<int>(q_num_heads);

  int64_t kv_num_heads = 0;
  ORT_ENFORCE(info.GetAttr("kv_num_heads", &kv_num_heads).IsOK() && kv_num_heads > 0,
              "kv_num_heads must be a positive integer");
  kv_num_heads_ = static_cast<int>(kv_num_heads);

  update_rule_ = info.GetAttrOrDefault<std::string>("update_rule", "gated_delta");
  ORT_ENFORCE(update_rule_ == "linear" || update_rule_ == "gated" ||
                  update_rule_ == "delta" || update_rule_ == "gated_delta",
              "update_rule must be one of: linear, gated, delta, gated_delta");

  scale_ = info.GetAttrOrDefault<float>("scale", 0.0f);

  int64_t chunk_size = info.GetAttrOrDefault<int64_t>("chunk_size", 64);
  // chunk_size_ reserved for future chunk-parallel prefill algorithm; not yet used.
  chunk_size_ = static_cast<int>(chunk_size);
}

namespace {

// Process a single (batch, kv_head) pair across all timesteps.
// This is the hot inner loop — called once per (b, h_kv) combination,
// potentially from different threads.
void ProcessHead(
    float* S,                 // State matrix [d_k, d_v], in-place updated
    const float* q_data,      // Packed Q: (B, T, H_q*d_k)
    const float* k_data,      // Packed K: (B, T, n_k*d_k)
    const float* v_data,      // Packed V: (B, T, H_kv*d_v)
    const float* decay_data,  // Decay gates (may be nullptr)
    const float* beta_data,   // Beta rates (may be nullptr)
    float* output_data,       // Output: (B, T, H_q*d_v)
    float* retrieved_buf,     // Pre-allocated scratch buffer [d_v]
    int64_t batch_idx,
    int h_kv,
    int h_k,  // Key head index (may differ from h_kv when n_k != kv_num_heads)
    int64_t seq_len,
    int64_t d_k,
    int64_t d_v,
    int q_num_heads,
    int kv_num_heads,
    int n_k_heads,
    int heads_per_group,
    int64_t output_hidden,
    float scale,
    bool needs_decay,
    bool decay_per_key_dim,
    bool needs_beta,
    bool beta_per_head,
    bool needs_retrieval) {
  const size_t dk = static_cast<size_t>(d_k);
  const size_t dv = static_cast<size_t>(d_v);
  const bool use_mlas = (d_k * d_v >= 4096);

  for (int64_t t = 0; t < seq_len; ++t) {
    // Pointers into packed 3D tensors at position [batch_idx, t, head*d]
    const float* kt = k_data + (batch_idx * seq_len + t) * (n_k_heads * d_k) + h_k * d_k;
    const float* vt = v_data + (batch_idx * seq_len + t) * (kv_num_heads * d_v) + h_kv * d_v;

    // ---- Step 1: Apply decay S *= exp(g_t) ----
    if (needs_decay) {
      if (decay_per_key_dim) {
        const float* gt = decay_data + (batch_idx * seq_len + t) * (kv_num_heads * d_k) + h_kv * d_k;
        for (int64_t i = 0; i < d_k; ++i) {
          float exp_g = std::exp(gt[i]);
          // Scale row i of S by exp_g
          for (int64_t j = 0; j < d_v; ++j) {
            S[i * d_v + j] *= exp_g;
          }
        }
      } else {
        const float* gt = decay_data + (batch_idx * seq_len + t) * kv_num_heads + h_kv;
        float exp_g = std::exp(gt[0]);
        for (int64_t i = 0; i < d_k * d_v; ++i) {
          S[i] *= exp_g;
        }
      }
    }

    // ---- Step 2: Retrieval = S^T @ k_t ----
    if (needs_retrieval) {
      if (use_mlas) {
        MlasGemm(
            CblasNoTrans,
            CblasNoTrans,
            1,
            dv,
            dk,
            1.0f,
            kt,
            dk,
            S,
            dv,
            0.0f,
            retrieved_buf,
            dv,
            nullptr,
            nullptr);
      } else {
        for (int64_t j = 0; j < d_v; ++j) {
          float acc = 0.0f;
          for (int64_t i = 0; i < d_k; ++i) {
            acc += S[i * d_v + j] * kt[i];
          }
          retrieved_buf[static_cast<size_t>(j)] = acc;
        }
      }
    }

    // ---- Step 3: State update ----
    if (needs_beta) {
      float bt;
      if (beta_per_head) {
        bt = beta_data[(batch_idx * seq_len + t) * kv_num_heads + h_kv];
      } else {
        bt = beta_data[(batch_idx * seq_len + t) * 1];
      }
      // Compute delta = beta * (v_t - retrieved) in-place into retrieved_buf
      for (size_t j = 0; j < dv; ++j) {
        retrieved_buf[j] = bt * (vt[j] - retrieved_buf[j]);
      }
      // S += k_t outer delta
      if (use_mlas) {
        MlasGemm(
            CblasNoTrans,
            CblasNoTrans,
            dk,
            dv,
            1,
            1.0f,
            kt,
            1,
            retrieved_buf,
            dv,
            1.0f,
            S,
            dv,
            nullptr,
            nullptr);
      } else {
        for (int64_t i = 0; i < d_k; ++i) {
          float* s_row = S + i * d_v;
          const float ki = kt[i];
          for (int64_t j = 0; j < d_v; ++j) {
            s_row[j] += ki * retrieved_buf[static_cast<size_t>(j)];
          }
        }
      }
    } else {
      // linear/gated: S += k_t outer v_t
      if (use_mlas) {
        MlasGemm(
            CblasNoTrans,
            CblasNoTrans,
            dk,
            dv,
            1,
            1.0f,
            kt,
            1,
            vt,
            dv,
            1.0f,
            S,
            dv,
            nullptr,
            nullptr);
      } else {
        for (int64_t i = 0; i < d_k; ++i) {
          float* s_row = S + i * d_v;
          const float ki = kt[i];
          for (int64_t j = 0; j < d_v; ++j) {
            s_row[j] += ki * vt[j];
          }
        }
      }
    }

    // ---- Step 4: Query readout for each q head in this kv group ----
    // o_t = scale * q_t^T @ S -> [1, d_v]
    //
    // Standard GQA (heads_per_group > 0): multiple Q heads share this KV state.
    // Inverse GQA (heads_per_group == 0): multiple KV states share one Q head.
    if (heads_per_group > 0) {
      for (int g = 0; g < heads_per_group; ++g) {
        int h_q = h_kv * heads_per_group + g;
        const float* qt = q_data + (batch_idx * seq_len + t) * (q_num_heads * d_k) + h_q * d_k;
        float* ot = output_data + (batch_idx * seq_len + t) * output_hidden + h_q * d_v;

        if (use_mlas) {
          // Use alpha=1.0 to hit the MLAS M=1 gemv fast path, then scale output.
          MlasGemm(
              CblasNoTrans,
              CblasNoTrans,
              1,
              dv,
              dk,
              1.0f,
              qt,
              dk,
              S,
              dv,
              0.0f,
              ot,
              dv,
              nullptr,
              nullptr);
          if (scale != 1.0f) {
            for (size_t j = 0; j < dv; ++j) {
              ot[j] *= scale;
            }
          }
        } else {
          for (int64_t j = 0; j < d_v; ++j) {
            float acc = 0.0f;
            for (int64_t i = 0; i < d_k; ++i) {
              acc += qt[i] * S[i * d_v + j];
            }
            ot[j] = scale * acc;
          }
        }
      }
    } else {
      // Inverse GQA: this KV head's Q is determined by h_kv * q_num / kv_num
      int h_q = h_kv * q_num_heads / kv_num_heads;
      const float* qt = q_data + (batch_idx * seq_len + t) * (q_num_heads * d_k) + h_q * d_k;
      float* ot = output_data + (batch_idx * seq_len + t) * output_hidden + h_kv * d_v;

      if (use_mlas) {
        // Use alpha=1.0 to hit the MLAS M=1 gemv fast path, then scale output.
        MlasGemm(
            CblasNoTrans,
            CblasNoTrans,
            1,
            dv,
            dk,
            1.0f,
            qt,
            dk,
            S,
            dv,
            0.0f,
            ot,
            dv,
            nullptr,
            nullptr);
        if (scale != 1.0f) {
          for (size_t j = 0; j < dv; ++j) {
            ot[j] *= scale;
          }
        }
      } else {
        for (int64_t j = 0; j < d_v; ++j) {
          float acc = 0.0f;
          for (int64_t i = 0; i < d_k; ++i) {
            acc += qt[i] * S[i * d_v + j];
          }
          ot[j] = scale * acc;
        }
      }
    }
  }
}

}  // anonymous namespace

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
  bool needs_decay = (update_rule_ == "gated" || update_rule_ == "gated_delta");
  bool needs_beta = (update_rule_ == "delta" || update_rule_ == "gated_delta");
  bool needs_retrieval = (update_rule_ == "delta" || update_rule_ == "gated_delta");

  ORT_RETURN_IF_NOT(!needs_decay || decay_tensor != nullptr,
                    "decay input is required for update_rule=", update_rule_);
  ORT_RETURN_IF_NOT(!needs_beta || beta_tensor != nullptr,
                    "beta input is required for update_rule=", update_rule_);

  const float* decay_data = decay_tensor ? decay_tensor->Data<float>() : nullptr;
  const float* beta_data = beta_tensor ? beta_tensor->Data<float>() : nullptr;

  bool decay_per_key_dim = false;
  if (decay_tensor != nullptr) {
    const auto& decay_shape = decay_tensor->Shape();
    ORT_RETURN_IF_NOT(decay_shape.NumDimensions() == 3,
                      "decay must be rank 3 (B, T, ...), got rank ", decay_shape.NumDimensions());
    ORT_RETURN_IF_NOT(decay_shape[0] == batch_size && decay_shape[1] == seq_len,
                      "decay dims 0/1 must match (B=", batch_size, ", T=", seq_len,
                      "), got (", decay_shape[0], ", ", decay_shape[1], ")");
    int64_t decay_last = decay_shape[2];
    if (decay_last == kv_num_heads_ * d_k) {
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
                      "beta dims 0/1 must match (B=", batch_size, ", T=", seq_len,
                      "), got (", beta_shape[0], ", ", beta_shape[1], ")");
    int64_t beta_last = beta_shape[2];
    if (beta_last == kv_num_heads_) {
      beta_per_head = true;
    } else {
      ORT_RETURN_IF_NOT(beta_last == 1, "beta last dim must be H_kv or 1");
    }
  }

  // ==== Initialize state: write directly into output present_state ====
  // present_state: (B, H_kv, d_k, d_v)
  TensorShape state_shape({batch_size, static_cast<int64_t>(kv_num_heads_), d_k, d_v});
  Tensor* present_state_tensor = context->Output(1, state_shape);
  float* state_data = present_state_tensor->MutableData<float>();
  int64_t state_per_head = d_k * d_v;
  int64_t total_state = batch_size * kv_num_heads_ * state_per_head;

  if (past_state_tensor != nullptr) {
    const auto& ps_shape = past_state_tensor->Shape();
    ORT_RETURN_IF_NOT(ps_shape.NumDimensions() == 4 &&
                          ps_shape[0] == batch_size &&
                          ps_shape[1] == kv_num_heads_ &&
                          ps_shape[2] == d_k &&
                          ps_shape[3] == d_v,
                      "past_state must be (B, H_kv, d_k, d_v)");
    const float* ps_data = past_state_tensor->Data<float>();
    std::memcpy(state_data, ps_data, static_cast<size_t>(total_state) * sizeof(float));
  } else {
    std::memset(state_data, 0, static_cast<size_t>(total_state) * sizeof(float));
  }

  // ==== Allocate output ====
  // Output hidden dim: max(q_num_heads, kv_num_heads) * d_v
  // Standard GQA: q_num_heads * d_v; Inverse GQA: kv_num_heads * d_v
  int64_t output_hidden = std::max(q_num_heads_, kv_num_heads_) * d_v;
  TensorShape output_shape({batch_size, seq_len, output_hidden});
  Tensor* output_tensor = context->Output(0, output_shape);
  float* output_data = output_tensor->MutableData<float>();

  // ==== GQA head mapping ====
  // Standard GQA: q_num_heads >= kv_num_heads, multiple Q heads per KV group.
  // Inverse GQA: q_num_heads < kv_num_heads (e.g., Qwen3.5 9B: n_k=16, n_kv=32).
  // Also n_k_heads may differ from both (K has its own head count).
  int heads_per_group;  // Q heads per KV group (0 if inverse GQA)
  if (q_num_heads_ >= kv_num_heads_) {
    ORT_RETURN_IF_NOT(q_num_heads_ % kv_num_heads_ == 0,
                      "q_num_heads must be divisible by kv_num_heads");
    heads_per_group = q_num_heads_ / kv_num_heads_;
  } else {
    ORT_RETURN_IF_NOT(kv_num_heads_ % q_num_heads_ == 0,
                      "kv_num_heads must be divisible by q_num_heads (inverse GQA)");
    heads_per_group = 0;  // signals inverse GQA to ProcessHead
  }

  // K-to-KV head mapping: when n_k < kv_num_heads, multiple KV heads share one K head
  ORT_RETURN_IF_NOT(kv_num_heads_ % n_k_heads == 0,
                    "kv_num_heads must be divisible by n_k_heads");
  int kv_per_k_head = kv_num_heads_ / n_k_heads;

  // ==== Thread-parallel over (batch, kv_head) pairs ====
  // Each (b, h_kv) pair is fully independent — the state matrix for each
  // head is disjoint, and the sequential token dependency is within a
  // single head only. This gives us batch_size * kv_num_heads parallel tasks.
  int64_t total_tasks = batch_size * kv_num_heads_;

  // Cost estimate: per task processes seq_len tokens, each doing ~3*d_k*d_v FLOPs
  double cost_per_task = static_cast<double>(seq_len) * static_cast<double>(d_k * d_v) * 3.0;

  auto* tp = context->GetOperatorThreadPool();

  ThreadPool::TryParallelFor(
      tp,
      static_cast<std::ptrdiff_t>(total_tasks),
      cost_per_task,
      [&](std::ptrdiff_t first, std::ptrdiff_t last) {
        // Pre-allocate scratch buffer per thread-batch to avoid malloc in hot loop
        std::vector<float> retrieved_buf(static_cast<size_t>(d_v));

        for (std::ptrdiff_t task = first; task < last; ++task) {
          int64_t b = task / kv_num_heads_;
          int h_kv = static_cast<int>(task % kv_num_heads_);
          int h_k = h_kv / kv_per_k_head;  // map KV head to K head

          float* S = state_data + (b * kv_num_heads_ + h_kv) * state_per_head;

          ProcessHead(
              S, q_data, k_data, v_data, decay_data, beta_data, output_data,
              retrieved_buf.data(),
              b, h_kv, h_k, seq_len, d_k, d_v,
              q_num_heads_, kv_num_heads_, n_k_heads, heads_per_group, output_hidden,
              s, needs_decay, decay_per_key_dim, needs_beta, beta_per_head,
              needs_retrieval);
        }
      });

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
