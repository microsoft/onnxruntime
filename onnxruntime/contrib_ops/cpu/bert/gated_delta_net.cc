// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/bert/gated_delta_net.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "core/common/safeint.h"
#include "core/framework/tensor.h"
#include "core/platform/threadpool.h"

namespace onnxruntime {
namespace contrib {

namespace {

inline float SoftplusFloat(float x) {
  return x > 20.0f ? x : std::log1pf(std::exp(x));
}

inline float SigmoidFloat(float x) {
  if (x >= 0.0f) {
    return 1.0f / (1.0f + std::exp(-x));
  }
  const float e = std::exp(x);
  return e / (1.0f + e);
}

inline float LoadGateElement(const Tensor* gate, int64_t index) {
  if (gate->IsDataType<float>()) {
    return gate->Data<float>()[index];
  }
  if (gate->IsDataType<MLFloat16>()) {
    return static_cast<float>(gate->Data<MLFloat16>()[index]);
  }
  return static_cast<float>(gate->Data<BFloat16>()[index]);
}

inline void ResolveSequenceRange(const int32_t* cu_seqlens, int b, int64_t total_tokens,
                                 int64_t uniform_len, int64_t* start, int64_t* len) {
  if (cu_seqlens == nullptr) {
    *start = static_cast<int64_t>(b) * uniform_len;
    *len = uniform_len;
    return;
  }
  int64_t s = static_cast<int64_t>(cu_seqlens[b]);
  int64_t e = static_cast<int64_t>(cu_seqlens[b + 1]);
  s = std::clamp<int64_t>(s, 0, total_tokens);
  e = std::clamp<int64_t>(e, 0, total_tokens);
  *start = s;
  *len = e > s ? (e - s) : 0;
}

}  // namespace

#define REGISTER_KERNEL_TYPED(T)                                           \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                           \
      GatedDeltaNet,                                                       \
      kMSDomain,                                                           \
      1,                                                                   \
      T,                                                                   \
      kCpuExecutionProvider,                                               \
      (*KernelDefBuilder::Create())                                        \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())           \
          .TypeConstraint("TS", DataTypeImpl::GetTensorType<float>())      \
          .TypeConstraint("TG", {DataTypeImpl::GetTensorType<float>(),     \
                                 DataTypeImpl::GetTensorType<MLFloat16>(), \
                                 DataTypeImpl::GetTensorType<BFloat16>()}) \
          .TypeConstraint("TI", DataTypeImpl::GetTensorType<int32_t>())    \
          .MayInplace(6, 1),                                               \
      GatedDeltaNet<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

#undef REGISTER_KERNEL_TYPED

template <typename T>
GatedDeltaNet<T>::GatedDeltaNet(const OpKernelInfo& info) : OpKernel(info) {
  const std::string rule = info.GetAttrOrDefault<std::string>("update_rule", "gated_delta");
  if (rule == "linear") {
    update_rule_ = UpdateRule::kLinear;
  } else if (rule == "gated") {
    update_rule_ = UpdateRule::kGated;
  } else if (rule == "delta") {
    update_rule_ = UpdateRule::kDelta;
  } else if (rule == "gated_delta") {
    update_rule_ = UpdateRule::kGatedDelta;
  } else {
    ORT_THROW("update_rule must be one of: linear, gated, delta, gated_delta");
  }

  const std::string gate = info.GetAttrOrDefault<std::string>("gate_activation", "none");
  if (gate == "none") {
    gate_activation_ = GateActivation::kNone;
  } else if (gate == "softplus" || gate == "qwen") {
    gate_activation_ = GateActivation::kSoftplus;
  } else if (gate == "sigmoid" || gate == "kda") {
    gate_activation_ = GateActivation::kSigmoid;
  } else {
    ORT_THROW("gate_activation must be one of: none, softplus (alias: qwen), sigmoid (alias: kda)");
  }

  gate_lower_bound_ = info.GetAttrOrDefault<float>("gate_lower_bound", -5.0f);
  ORT_ENFORCE(std::isfinite(gate_lower_bound_) && gate_lower_bound_ < 0.0f,
              "gate_lower_bound must be finite and negative");

  const std::string beta = info.GetAttrOrDefault<std::string>("beta_activation", "none");
  if (beta == "none") {
    beta_activation_ = BetaActivation::kNone;
  } else if (beta == "sigmoid") {
    beta_activation_ = BetaActivation::kSigmoid;
  } else {
    ORT_THROW("beta_activation must be one of: none, sigmoid");
  }

  scale_ = info.GetAttrOrDefault<float>("scale", 0.0f);
  const int64_t chunk_size = info.GetAttrOrDefault<int64_t>("chunk_size", 64);
  ORT_ENFORCE(chunk_size > 0 && chunk_size <= std::numeric_limits<int>::max(),
              "chunk_size must be positive and fit in int32, got ", chunk_size);
  chunk_size_ = static_cast<int>(chunk_size);
  const int64_t state_update_capacity =
      info.GetAttrOrDefault<int64_t>("state_update_capacity", 0);
  ORT_ENFORCE(state_update_capacity >= 0 && state_update_capacity <= 8,
              "state_update_capacity must be in [0, 8], got ", state_update_capacity);
  state_update_capacity_ = static_cast<int>(state_update_capacity);
  qk_l2_norm_ = info.GetAttrOrDefault<int64_t>("qk_l2_norm", 0) != 0;
}

template <typename T>
Status GatedDeltaNet<T>::Compute(OpKernelContext* context) const {
  const Tensor* query = context->Input<Tensor>(0);
  const Tensor* key = context->Input<Tensor>(1);
  const Tensor* value = context->Input<Tensor>(2);
  const Tensor* cu_seqlens = context->Input<Tensor>(3);
  const Tensor* decay = context->Input<Tensor>(4);
  const Tensor* beta = context->Input<Tensor>(5);
  const Tensor* initial_state = context->Input<Tensor>(6);
  const Tensor* a_log = context->Input<Tensor>(7);
  const Tensor* dt_bias = context->Input<Tensor>(8);
  const Tensor* capture_count = context->Input<Tensor>(9);
  const Tensor* state_update_active = context->Input<Tensor>(10);

  ORT_RETURN_IF_NOT(query != nullptr, "query is required");
  for (const Tensor* gate : {decay, beta}) {
    ORT_RETURN_IF_NOT(gate == nullptr || gate->IsDataType<float>() ||
                          gate->GetElementType() == query->GetElementType(),
                      "decay and beta must be float or match the query element type");
  }

  ORT_RETURN_IF_NOT((state_update_capacity_ > 0) == (capture_count != nullptr),
                    "capture_count must be present exactly when state_update_capacity is positive");

  const bool needs_decay =
      update_rule_ == UpdateRule::kGated || update_rule_ == UpdateRule::kGatedDelta;
  const bool needs_beta =
      update_rule_ == UpdateRule::kDelta || update_rule_ == UpdateRule::kGatedDelta;
  ORT_RETURN_IF_NOT(needs_decay == (decay != nullptr),
                    "decay input presence must match update_rule");
  ORT_RETURN_IF_NOT(needs_beta == (beta != nullptr),
                    "beta input presence must match update_rule");

  const bool is_packed_qkv = key == nullptr && value == nullptr;
  ORT_RETURN_IF_NOT(is_packed_qkv || (key != nullptr && value != nullptr),
                    "key and value must be both present or both absent");

  const auto& q_shape = query->Shape();
  const size_t qkv_rank = q_shape.NumDimensions();
  ORT_RETURN_IF_NOT(is_packed_qkv ? (qkv_rank == 2 || qkv_rank == 3) : (qkv_rank == 3 || qkv_rank == 4),
                    is_packed_qkv
                        ? "packed QKV must be rank 2 [total_tokens, packed_size] or rank 3 "
                          "[batch, sequence, packed_size]"
                        : "query, key and value must be rank 3 [total_tokens, num_heads, head_size] or "
                          "rank 4 [batch, sequence, num_heads, head_size]");

  const size_t token_dims = qkv_rank - (is_packed_qkv ? 1 : 2);
  const int64_t total_tokens = q_shape.SizeToDimension(token_dims);
  const int64_t max_int = std::numeric_limits<int>::max();
  ORT_RETURN_IF_NOT(total_tokens > 0 && total_tokens <= max_int,
                    "total_tokens must be positive and fit in int32");

  int num_heads_q = 0;
  int num_heads_k = 0;
  int num_heads_v = 0;
  int head_size_qk = 0;
  int head_size_v = 0;
  if (is_packed_qkv) {
    ORT_RETURN_IF_NOT(initial_state != nullptr,
                      "initial_state is required to derive packed QKV dimensions");
    const auto& state_shape = initial_state->Shape();
    ORT_RETURN_IF_NOT(state_shape.NumDimensions() == 4,
                      "initial_state must be rank 4 [batch, num_heads_v, head_size_v, head_size_qk]");
    ORT_RETURN_IF_NOT(state_shape[1] > 0 && state_shape[1] <= max_int &&
                          state_shape[2] > 0 && state_shape[2] <= max_int &&
                          state_shape[3] > 0 && state_shape[3] <= max_int,
                      "packed QKV head counts and head sizes must be positive and fit in int32");
    num_heads_v = static_cast<int>(state_shape[1]);
    head_size_v = static_cast<int>(state_shape[2]);
    head_size_qk = static_cast<int>(state_shape[3]);
    const int64_t packed_size = q_shape[token_dims];
    const int64_t value_size = static_cast<int64_t>(num_heads_v) * head_size_v;
    ORT_RETURN_IF_NOT(packed_size > value_size &&
                          (packed_size - value_size) % (2 * head_size_qk) == 0,
                      "packed QKV last dimension must be 2 * num_heads_q * head_size_qk + "
                      "num_heads_v * head_size_v");
    num_heads_q = static_cast<int>((packed_size - value_size) / (2 * head_size_qk));
    num_heads_k = num_heads_q;
  } else {
    const auto& k_shape = key->Shape();
    const auto& v_shape = value->Shape();
    ORT_RETURN_IF_NOT(k_shape.NumDimensions() == qkv_rank && v_shape.NumDimensions() == qkv_rank,
                      "query, key and value must have the same rank");
    ORT_RETURN_IF_NOT(k_shape.SizeToDimension(token_dims) == total_tokens &&
                          v_shape.SizeToDimension(token_dims) == total_tokens,
                      "query, key and value must agree on total_tokens");
    ORT_RETURN_IF_NOT(q_shape[token_dims] <= max_int && k_shape[token_dims] <= max_int &&
                          v_shape[token_dims] <= max_int && q_shape[token_dims + 1] <= max_int &&
                          v_shape[token_dims + 1] <= max_int,
                      "head counts and head sizes must fit in int32");
    num_heads_q = static_cast<int>(q_shape[token_dims]);
    num_heads_k = static_cast<int>(k_shape[token_dims]);
    num_heads_v = static_cast<int>(v_shape[token_dims]);
    head_size_qk = static_cast<int>(q_shape[token_dims + 1]);
    head_size_v = static_cast<int>(v_shape[token_dims + 1]);
    ORT_RETURN_IF_NOT(k_shape[token_dims + 1] == head_size_qk,
                      "key head_size must equal query head_size");
  }

  ORT_RETURN_IF_NOT(num_heads_q == num_heads_k,
                    "num_heads_q (", num_heads_q, ") must equal num_heads_k (", num_heads_k, ")");
  ORT_RETURN_IF_NOT(num_heads_v > 0 && num_heads_q > 0 && num_heads_v % num_heads_q == 0 &&
                        head_size_qk > 0 && head_size_v > 0,
                    "num_heads_v (", num_heads_v, ") must be a positive multiple of num_heads_q (",
                    num_heads_q, ")");

  if (initial_state != nullptr) {
    ORT_RETURN_IF_NOT(initial_state->Shape().NumDimensions() == 4,
                      "initial_state must be rank 4 [batch, num_heads_v, head_size_v, "
                      "head_size_qk]");
  }

  int64_t batch_dim = 1;
  if (cu_seqlens != nullptr) {
    ORT_RETURN_IF_NOT(token_dims == 1,
                      "cu_seqlens describes ragged packing, so query/key/value must be rank 3 (or rank 2 packed)");
    ORT_RETURN_IF_NOT(cu_seqlens->Shape().NumDimensions() == 1 && cu_seqlens->Shape()[0] >= 2,
                      "cu_seqlens must be rank 1 with at least 2 elements");
    batch_dim = cu_seqlens->Shape()[0] - 1;
  } else if (token_dims == 2) {
    batch_dim = q_shape[0];
  } else {
    ORT_RETURN_IF_NOT(initial_state != nullptr,
                      "cu_seqlens is absent, so initial_state is required to determine batch size");
    batch_dim = initial_state->Shape()[0];
  }
  ORT_RETURN_IF_NOT(batch_dim > 0 && batch_dim <= max_int,
                    "batch size must be positive and fit in int32");
  const int batch = static_cast<int>(batch_dim);
  if (cu_seqlens == nullptr && token_dims == 1) {
    ORT_RETURN_IF_NOT(total_tokens % batch == 0,
                      "total_tokens (", total_tokens, ") must be divisible by batch (", batch, ")");
  }

  if (initial_state != nullptr) {
    const auto& s = initial_state->Shape();
    ORT_RETURN_IF_NOT(s.NumDimensions() == 4 && s[0] == batch && s[1] == num_heads_v &&
                          s[2] == head_size_v && s[3] == head_size_qk,
                      "initial_state must be [batch, num_heads_v, head_size_v, head_size_qk] "
                      "(V-major)");
  }

  const bool decay_per_key_dim =
      decay != nullptr && decay->Shape().NumDimensions() == token_dims + 2;
  if (decay != nullptr) {
    const auto& d = decay->Shape();
    ORT_RETURN_IF_NOT(d.NumDimensions() == token_dims + 1 || decay_per_key_dim,
                      "decay must be [...tokens, num_heads_v] or "
                      "[...tokens, num_heads_v, head_size_qk]");
    ORT_RETURN_IF_NOT(d.SizeToDimension(token_dims) == total_tokens && d[token_dims] == num_heads_v,
                      "decay must be [...tokens, num_heads_v] or "
                      "[...tokens, num_heads_v, head_size_qk]");
    if (decay_per_key_dim) {
      ORT_RETURN_IF_NOT(d[token_dims + 1] == head_size_qk,
                        "per-key-dim decay last axis must be head_size_qk");
    }
  }
  if (beta != nullptr) {
    const auto& b = beta->Shape();
    ORT_RETURN_IF_NOT(b.NumDimensions() == token_dims + 1 &&
                          b.SizeToDimension(token_dims) == total_tokens &&
                          b[token_dims] == num_heads_v,
                      "beta must be [...tokens, num_heads_v]");
  }
  bool dt_bias_per_key_dim = false;
  if (gate_activation_ == GateActivation::kSoftplus ||
      gate_activation_ == GateActivation::kSigmoid) {
    ORT_RETURN_IF_NOT(a_log != nullptr && dt_bias != nullptr,
                      "gate_activation=softplus/sigmoid requires a_log and dt_bias");
    ORT_RETURN_IF_NOT(a_log->Shape().NumDimensions() == 1 && a_log->Shape()[0] == num_heads_v,
                      "a_log must be [num_heads_v]");
    const auto& dt_shape = dt_bias->Shape();
    const bool scalar_bias = dt_shape.NumDimensions() == 1 && dt_shape[0] == num_heads_v;
    const bool key_dim_bias =
        decay_per_key_dim &&
        ((dt_shape.NumDimensions() == 1 &&
          dt_shape[0] == static_cast<int64_t>(num_heads_v) * head_size_qk) ||
         (dt_shape.NumDimensions() == 2 && dt_shape[0] == num_heads_v &&
          dt_shape[1] == head_size_qk));
    ORT_RETURN_IF_NOT(scalar_bias || key_dim_bias,
                      "dt_bias must be [num_heads_v] or (for per-key-dim decay) "
                      "[num_heads_v, head_size_qk] / [num_heads_v * head_size_qk]");
    dt_bias_per_key_dim = key_dim_bias;
  } else {
    ORT_RETURN_IF_NOT(a_log == nullptr && dt_bias == nullptr,
                      "a_log and dt_bias require gate_activation=softplus or sigmoid");
  }

  if (capture_count != nullptr) {
    ORT_RETURN_IF_NOT(capture_count->Shape().NumDimensions() == 1 &&
                          capture_count->Shape()[0] == batch,
                      "capture_count must be [batch]");
  }
  if (state_update_active != nullptr) {
    ORT_RETURN_IF_NOT(state_update_active->Shape().NumDimensions() == 1 &&
                          state_update_active->Shape()[0] == 1,
                      "state_update_active must be [1]");
  }

  const int out_heads = std::max(num_heads_q, num_heads_v);
  TensorShapeVector out_dims(q_shape.GetDims().begin(), q_shape.GetDims().begin() + token_dims);
  out_dims.push_back(out_heads);
  out_dims.push_back(head_size_v);
  Tensor* output = context->Output(0, TensorShape(out_dims));

  TensorShape state_shape{batch, num_heads_v, head_size_v, head_size_qk};
  Tensor* final_state = context->Output(1, state_shape);

  const int64_t decay_capture_width =
      decay_per_key_dim ? static_cast<int64_t>(num_heads_v) * head_size_qk : num_heads_v;
  const int64_t state_update_width = static_cast<int64_t>(state_update_capacity_) *
                                     (decay_capture_width +
                                      static_cast<int64_t>(num_heads_k) * head_size_qk +
                                      static_cast<int64_t>(num_heads_v) * head_size_v);
  Tensor* state_update = context->Output(2, TensorShape{batch, state_update_width});

  const bool capture_active =
      state_update_capacity_ > 0 &&
      (state_update_active == nullptr || state_update_active->Data<int32_t>()[0] != 0);
  if (state_update != nullptr && state_update->SizeInBytes() > 0) {
    std::memset(state_update->MutableData<float>(), 0, state_update->SizeInBytes());
  }

  const float scale = scale_ != 0.0f ? scale_ : 1.0f / std::sqrt(static_cast<float>(head_size_qk));
  const float l2_eps = (gate_activation_ == GateActivation::kSigmoid) ? 1e-6f : 1e-12f;

  const T* q_base = query->Data<T>();
  const T* k_base = is_packed_qkv ? (q_base + static_cast<int64_t>(num_heads_q) * head_size_qk)
                                  : key->Data<T>();
  const T* v_base = is_packed_qkv
                        ? (k_base + static_cast<int64_t>(num_heads_k) * head_size_qk)
                        : value->Data<T>();
  const int64_t packed_stride =
      static_cast<int64_t>(num_heads_q + num_heads_k) * head_size_qk +
      static_cast<int64_t>(num_heads_v) * head_size_v;
  const int64_t q_stride = is_packed_qkv ? packed_stride : static_cast<int64_t>(num_heads_q) * head_size_qk;
  const int64_t k_stride = is_packed_qkv ? packed_stride : static_cast<int64_t>(num_heads_k) * head_size_qk;
  const int64_t v_in_stride = is_packed_qkv ? packed_stride : static_cast<int64_t>(num_heads_v) * head_size_v;
  const int64_t v_out_stride = static_cast<int64_t>(num_heads_v) * head_size_v;

  const int32_t* cu_seqlens_data = cu_seqlens != nullptr ? cu_seqlens->Data<int32_t>() : nullptr;
  if (cu_seqlens_data != nullptr) {
    ORT_RETURN_IF_NOT(cu_seqlens_data[0] == 0 && cu_seqlens_data[batch] == total_tokens,
                      "cu_seqlens must start at 0 and end at total_tokens");
    for (int b = 0; b < batch; ++b) {
      ORT_RETURN_IF_NOT(cu_seqlens_data[b] <= cu_seqlens_data[b + 1],
                        "cu_seqlens must be non-decreasing");
    }
  }
  const int32_t* capture_count_data =
      capture_active && capture_count != nullptr ? capture_count->Data<int32_t>() : nullptr;
  const float* init_state_data = initial_state != nullptr ? initial_state->Data<float>() : nullptr;
  const float* a_log_data = a_log != nullptr ? a_log->Data<float>() : nullptr;
  const float* dt_bias_data = dt_bias != nullptr ? dt_bias->Data<float>() : nullptr;
  T* out_data = output->MutableData<T>();
  float* final_state_data = final_state != nullptr ? final_state->MutableData<float>() : nullptr;
  float* state_update_data = state_update != nullptr ? state_update->MutableData<float>() : nullptr;

  const int64_t uniform_len = batch > 0 ? total_tokens / batch : 0;
  const bool needs_retrieval =
      update_rule_ == UpdateRule::kDelta || update_rule_ == UpdateRule::kGatedDelta;

  const int64_t total_tasks = static_cast<int64_t>(batch) * num_heads_v;
  concurrency::ThreadPool::TryBatchParallelFor(
      context->GetOperatorThreadPool(), onnxruntime::narrow<ptrdiff_t>(total_tasks),
      [&](ptrdiff_t task_idx) {
        const int b = static_cast<int>(task_idx / num_heads_v);
        const int hv = static_cast<int>(task_idx % num_heads_v);
        const int hq = hv * num_heads_q / num_heads_v;
        const int hk = hv * num_heads_k / num_heads_v;

        int64_t tok_base = 0;
        int64_t seq_len = 0;
        ResolveSequenceRange(cu_seqlens_data, b, total_tokens, uniform_len, &tok_base, &seq_len);

        const int64_t state_offset =
            (static_cast<int64_t>(b) * num_heads_v + hv) * head_size_v * head_size_qk;
        std::vector<float> S(static_cast<size_t>(head_size_v) * head_size_qk);
        if (init_state_data != nullptr) {
          std::memcpy(S.data(), init_state_data + state_offset, S.size() * sizeof(float));
        } else {
          std::fill(S.begin(), S.end(), 0.0f);
        }

        std::vector<float> q_vec(static_cast<size_t>(head_size_qk));
        std::vector<float> k_vec(static_cast<size_t>(head_size_qk));
        std::vector<float> g_vec(static_cast<size_t>(head_size_qk), 1.0f);
        std::vector<float> delta_vec(static_cast<size_t>(head_size_v));

        int captured = 0;
        if (capture_count_data != nullptr && state_update_capacity_ > 0) {
          captured = std::max(0, std::min({capture_count_data[b], state_update_capacity_,
                                           static_cast<int>(seq_len)}));
        }

        for (int64_t t = 0; t < seq_len; ++t) {
          const int64_t tok = tok_base + t;
          const T* q_ptr = q_base + tok * q_stride + static_cast<int64_t>(hq) * head_size_qk;
          const T* k_ptr = k_base + tok * k_stride + static_cast<int64_t>(hk) * head_size_qk;
          const T* v_ptr = v_base + tok * v_in_stride + static_cast<int64_t>(hv) * head_size_v;

          float q_norm_sq = 0.0f;
          float k_norm_sq = 0.0f;
          for (int d = 0; d < head_size_qk; ++d) {
            const float qv = static_cast<float>(q_ptr[d]);
            const float kv = static_cast<float>(k_ptr[d]);
            q_vec[d] = qv;
            k_vec[d] = kv;
            q_norm_sq += qv * qv;
            k_norm_sq += kv * kv;
          }
          if (qk_l2_norm_) {
            const float rq = 1.0f / std::sqrt(q_norm_sq + l2_eps);
            const float rk = 1.0f / std::sqrt(k_norm_sq + l2_eps);
            for (int d = 0; d < head_size_qk; ++d) {
              q_vec[d] *= rq;
              k_vec[d] *= rk;
            }
          }

          float decay_scalar = 1.0f;
          if (needs_decay) {
            if (decay_per_key_dim) {
              const int64_t g_base = (tok * num_heads_v + hv) * head_size_qk;
              const float a_exp = a_log_data != nullptr ? std::exp(a_log_data[hv]) : 1.0f;
              for (int d = 0; d < head_size_qk; ++d) {
                float g_val = LoadGateElement(decay, g_base + d);
                if (gate_activation_ == GateActivation::kSoftplus) {
                  const float bias = dt_bias_data != nullptr
                                         ? dt_bias_data[dt_bias_per_key_dim ? (hv * head_size_qk + d) : hv]
                                         : 0.0f;
                  g_val = -a_exp * SoftplusFloat(g_val + bias);
                } else if (gate_activation_ == GateActivation::kSigmoid) {
                  const float bias = dt_bias_data != nullptr
                                         ? dt_bias_data[dt_bias_per_key_dim ? (hv * head_size_qk + d) : hv]
                                         : 0.0f;
                  g_val = gate_lower_bound_ * SigmoidFloat(a_exp * (g_val + bias));
                }
                g_vec[d] = std::exp(g_val);
              }
            } else {
              float g_val = LoadGateElement(decay, tok * num_heads_v + hv);
              const float a_exp = a_log_data != nullptr ? std::exp(a_log_data[hv]) : 1.0f;
              const float bias = dt_bias_data != nullptr ? dt_bias_data[hv] : 0.0f;
              if (gate_activation_ == GateActivation::kSoftplus) {
                g_val = -a_exp * SoftplusFloat(g_val + bias);
              } else if (gate_activation_ == GateActivation::kSigmoid) {
                g_val = gate_lower_bound_ * SigmoidFloat(a_exp * (g_val + bias));
              }
              decay_scalar = std::exp(g_val);
            }
          }

          float beta_val = 1.0f;
          if (needs_beta) {
            beta_val = LoadGateElement(beta, tok * num_heads_v + hv);
            if (beta_activation_ == BetaActivation::kSigmoid) {
              beta_val = SigmoidFloat(beta_val);
            }
          }

          // State is V-major: S[v_idx * head_size_qk + k_idx]
          T* out_ptr = out_data + tok * v_out_stride + static_cast<int64_t>(hv) * head_size_v;
          for (int v_idx = 0; v_idx < head_size_v; ++v_idx) {
            float* s_row = S.data() + static_cast<size_t>(v_idx) * head_size_qk;
            float retrieved = 0.0f;
            if (needs_decay && decay_per_key_dim) {
              for (int k_idx = 0; k_idx < head_size_qk; ++k_idx) {
                s_row[k_idx] *= g_vec[k_idx];
                if (needs_retrieval) {
                  retrieved += s_row[k_idx] * k_vec[k_idx];
                }
              }
            } else {
              for (int k_idx = 0; k_idx < head_size_qk; ++k_idx) {
                s_row[k_idx] *= decay_scalar;
                if (needs_retrieval) {
                  retrieved += s_row[k_idx] * k_vec[k_idx];
                }
              }
            }
            const float vv = static_cast<float>(v_ptr[v_idx]);
            const float delta = beta_val * (vv - retrieved);
            delta_vec[v_idx] = delta;

            float out_acc = 0.0f;
            for (int k_idx = 0; k_idx < head_size_qk; ++k_idx) {
              s_row[k_idx] += k_vec[k_idx] * delta;
              out_acc += s_row[k_idx] * q_vec[k_idx];
            }
            out_ptr[v_idx] = static_cast<T>(scale * out_acc);
          }

          if (t < captured && state_update_data != nullptr) {
            float* row_base = state_update_data + static_cast<int64_t>(b) * state_update_width;
            const int64_t key_offset = static_cast<int64_t>(state_update_capacity_) * decay_capture_width;
            const int64_t delta_offset =
                key_offset + static_cast<int64_t>(state_update_capacity_) * num_heads_k * head_size_qk;
            if (decay_per_key_dim) {
              float* dst_decay =
                  row_base + (t * num_heads_v + hv) * head_size_qk;
              for (int d = 0; d < head_size_qk; ++d) {
                dst_decay[d] = needs_decay ? g_vec[d] : 1.0f;
              }
            } else {
              row_base[t * num_heads_v + hv] = decay_scalar;
            }
            if (hv == hk * (num_heads_v / num_heads_k)) {
              float* dst_key =
                  row_base + key_offset + (t * num_heads_k + hk) * head_size_qk;
              std::memcpy(dst_key, k_vec.data(), static_cast<size_t>(head_size_qk) * sizeof(float));
            }
            float* dst_delta =
                row_base + delta_offset + (t * num_heads_v + hv) * head_size_v;
            std::memcpy(dst_delta, delta_vec.data(), static_cast<size_t>(head_size_v) * sizeof(float));
          }
        }

        if (final_state_data != nullptr) {
          std::memcpy(final_state_data + state_offset, S.data(), S.size() * sizeof(float));
        }
      },
      0);

  return Status::OK();
}

template class GatedDeltaNet<float>;
template class GatedDeltaNet<MLFloat16>;
template class GatedDeltaNet<BFloat16>;

}  // namespace contrib
}  // namespace onnxruntime
