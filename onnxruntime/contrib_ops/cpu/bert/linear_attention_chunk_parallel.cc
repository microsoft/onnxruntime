#include "contrib_ops/cpu/bert/linear_attention_chunk_parallel.h"

#include <cmath>
#include <cstring>
#include <vector>

#include "core/util/math.h"
#include "core/providers/common.h"

using namespace ::onnxruntime::common;

namespace onnxruntime {
namespace contrib {

namespace {

inline float ToFloat(float v) { return v; }
inline float ToFloat(MLFloat16 v) { return v.ToFloat(); }
inline float ToFloat(BFloat16 v) { return v.ToFloat(); }

inline void StoreFloat(float val, float& out) { out = val; }
inline void StoreFloat(float val, MLFloat16& out) { out = MLFloat16(val); }
inline void StoreFloat(float val, BFloat16& out) { out = BFloat16(val); }

LinearAttentionUpdateRule ParseUpdateRule(const std::string& s) {
  if (s == "linear") return LinearAttentionUpdateRule::kLinear;
  if (s == "gated")  return LinearAttentionUpdateRule::kGated;
  if (s == "delta")  return LinearAttentionUpdateRule::kDelta;
  if (s == "gated_delta") return LinearAttentionUpdateRule::kGatedDelta;
  ORT_THROW("Unknown linear attention update_rule: ", s);
}

}

#define REGISTER_KERNEL_TYPED(T)                                           \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                           \
      LinearAttentionChunkParallel,                                        \
      kMSDomain,                                                           \
      1,                                                                   \
      T,                                                                   \
      kCpuExecutionProvider,                                               \
      (*KernelDefBuilder::Create())                                        \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),          \
      LinearAttentionChunkParallel<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

template <typename T>
LinearAttentionChunkParallel<T>::LinearAttentionChunkParallel(const OpKernelInfo& info)
    : OpKernel(info) {
  std::string rule_str = info.GetAttrOrDefault<std::string>("update_rule", "gated_delta");
  update_rule_ = ParseUpdateRule(rule_str);
  chunk_size_  = static_cast<int>(info.GetAttrOrDefault<int64_t>("chunk_size", 64));
  scale_       = info.GetAttrOrDefault<float>("scale", 0.0f);
}

template <typename T>
void LinearAttentionChunkParallel<T>::StepSingleHead(
    const float* q, const float* k, const float* v,
    float* state,
    const float* decay, float beta_val,
    float* output,
    int d_k, int d_v, float scale) const {

  std::vector<float> retrieved(d_v, 0.0f);
  if (update_rule_ == LinearAttentionUpdateRule::kDelta ||
      update_rule_ == LinearAttentionUpdateRule::kGatedDelta) {
    for (int i = 0; i < d_k; i++) {
      float gi = (update_rule_ == LinearAttentionUpdateRule::kGatedDelta) ? decay[i] : 1.0f;
      for (int j = 0; j < d_v; j++) {
        retrieved[j] += gi * state[i * d_v + j] * k[i];
      }
    }
  }

  std::fill(output, output + d_v, 0.0f);

  for (int i = 0; i < d_k; i++) {
    for (int j = 0; j < d_v; j++) {
      float s = state[i * d_v + j];
      float new_s = 0.0f;

      switch (update_rule_) {
        case LinearAttentionUpdateRule::kLinear:
          new_s = s + k[i] * v[j];
          break;
        case LinearAttentionUpdateRule::kGated:
          new_s = decay[i] * s + k[i] * v[j];
          break;
        case LinearAttentionUpdateRule::kDelta: {
          float delta = v[j] - retrieved[j];
          new_s = s + beta_val * k[i] * delta;
          break;
        }
        case LinearAttentionUpdateRule::kGatedDelta: {
          float delta = v[j] - retrieved[j];
          new_s = decay[i] * s + beta_val * k[i] * delta;
          break;
        }
      }

      state[i * d_v + j] = new_s;
      output[j] += q[i] * new_s;
    }
  }

  for (int j = 0; j < d_v; j++) {
    output[j] *= scale;
  }
}

template <typename T>
Status LinearAttentionChunkParallel<T>::Compute(OpKernelContext* context) const {
  const Tensor* query         = context->Input<Tensor>(0);
  const Tensor* key           = context->Input<Tensor>(1);
  const Tensor* value         = context->Input<Tensor>(2);
  const Tensor* initial_state = context->Input<Tensor>(3);  // optional
  const Tensor* decay         = context->Input<Tensor>(4);  // optional
  const Tensor* beta          = context->Input<Tensor>(5);  // optional

  ORT_RETURN_IF_NOT(query != nullptr, "query input is required");
  ORT_RETURN_IF_NOT(key   != nullptr, "key input is required");
  ORT_RETURN_IF_NOT(value != nullptr, "value input is required");

  const auto& q_shape = query->Shape();
  const auto& v_shape = value->Shape();

  ORT_RETURN_IF_NOT(q_shape.NumDimensions() == 4, "query must be 4D (B,H,T,d_k)");
  ORT_RETURN_IF_NOT(v_shape.NumDimensions() == 4, "value must be 4D (B,H,T,d_v)");

  const int batch_size = static_cast<int>(q_shape[0]);
  const int num_heads  = static_cast<int>(q_shape[1]);
  const int seq_len    = static_cast<int>(q_shape[2]);
  const int d_k        = static_cast<int>(q_shape[3]);
  const int d_v        = static_cast<int>(v_shape[3]);

  ORT_RETURN_IF_NOT(key->Shape()[2] == seq_len, "key sequence length must match query");
  ORT_RETURN_IF_NOT(v_shape[2]      == seq_len, "value sequence length must match query");

  if (initial_state != nullptr) {
    const auto& s = initial_state->Shape();
    ORT_RETURN_IF_NOT(s.NumDimensions() == 4 &&
                      s[0] == batch_size && s[1] == num_heads &&
                      s[2] == d_k        && s[3] == d_v,
                      "initial_state shape must be (B,H,d_k,d_v)");
  }

  const bool needs_decay = (update_rule_ == LinearAttentionUpdateRule::kGated ||
                             update_rule_ == LinearAttentionUpdateRule::kGatedDelta);
  const bool needs_beta  = (update_rule_ == LinearAttentionUpdateRule::kDelta ||
                             update_rule_ == LinearAttentionUpdateRule::kGatedDelta);

  ORT_RETURN_IF_NOT(!needs_decay || decay != nullptr,
                    "decay is required for gated/gated_delta update rules");
  ORT_RETURN_IF_NOT(!needs_beta  || beta  != nullptr,
                    "beta is required for delta/gated_delta update rules");

  bool decay_broadcasted = false;
  if (decay != nullptr) {
    ORT_RETURN_IF_NOT(decay->Shape().NumDimensions() == 4, "decay must be 4D");
    decay_broadcasted = (decay->Shape()[3] == d_k);
  }

  const float scale = (scale_ == 0.0f) ? (1.0f / sqrtf(static_cast<float>(d_k))) : scale_;

  Tensor* output      = context->Output(0, TensorShape({batch_size, num_heads, seq_len, d_v}));
  Tensor* final_state = context->Output(1, TensorShape({batch_size, num_heads, d_k, d_v}));

  const T* q_data   = query->Data<T>();
  const T* k_data   = key->Data<T>();
  const T* v_data   = value->Data<T>();
  T*       out_data = output->MutableData<T>();
  T*       fs_data  = final_state->MutableData<T>();

  const int state_elems = batch_size * num_heads * d_k * d_v;
  std::vector<float> state_f(state_elems, 0.0f);

  if (initial_state != nullptr) {
    const T* is_data = initial_state->Data<T>();
    for (int i = 0; i < state_elems; i++) {
      state_f[i] = ToFloat(is_data[i]);
    }
  }

  std::vector<float> q_f(d_k), k_f(d_k), v_f(d_v);
  std::vector<float> decay_f(d_k, 1.0f);
  std::vector<float> out_f(d_v);

  for (int t = 0; t < seq_len; t++) {
    for (int b = 0; b < batch_size; b++) {
      for (int h = 0; h < num_heads; h++) {
        const int bh = b * num_heads + h;

        for (int i = 0; i < d_k; i++) {
          q_f[i] = ToFloat(q_data[bh * seq_len * d_k + t * d_k + i]);
          k_f[i] = ToFloat(k_data[bh * seq_len * d_k + t * d_k + i]);
        }
        for (int j = 0; j < d_v; j++) {
          v_f[j] = ToFloat(v_data[bh * seq_len * d_v + t * d_v + j]);
        }

        if (decay != nullptr) {
          const T* decay_data = decay->Data<T>();
          if (decay_broadcasted) {
            for (int i = 0; i < d_k; i++) {
              decay_f[i] = expf(ToFloat(decay_data[bh * seq_len * d_k + t * d_k + i]));
            }
          } else {
            const float scalar = expf(ToFloat(decay_data[bh * seq_len + t]));
            std::fill(decay_f.begin(), decay_f.end(), scalar);
          }
        }

        float beta_val = 0.0f;
        if (beta != nullptr) {
          beta_val = ToFloat(beta->Data<T>()[bh * seq_len + t]);
        }

        StepSingleHead(
            q_f.data(), k_f.data(), v_f.data(),
            state_f.data() + bh * d_k * d_v,
            decay_f.data(), beta_val,
            out_f.data(),
            d_k, d_v, scale);

        for (int j = 0; j < d_v; j++) {
          StoreFloat(out_f[j], out_data[bh * seq_len * d_v + t * d_v + j]);
        }
      }
    }
  }

  for (int i = 0; i < state_elems; i++) {
    StoreFloat(state_f[i], fs_data[i]);
  }

  return Status::OK();
}

}
}
