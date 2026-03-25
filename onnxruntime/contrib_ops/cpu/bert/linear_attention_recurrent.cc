#include "contrib_ops/cpu/bert/linear_attention_recurrent.h"

#include <cmath>
#include <vector>

#include "core/util/math.h"
#include "core/mlas/inc/mlas.h"
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
  if (s == "gated") return LinearAttentionUpdateRule::kGated;
  if (s == "delta") return LinearAttentionUpdateRule::kDelta;
  if (s == "gated_delta") return LinearAttentionUpdateRule::kGatedDelta;
  ORT_THROW("Unknown linear attention update_rule: ", s);
}

}

#define REGISTER_KERNEL_TYPED(T)                                           \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                           \
      LinearAttentionRecurrent,                                            \
      kMSDomain,                                                           \
      1,                                                                   \
      T,                                                                   \
      kCpuExecutionProvider,                                               \
      (*KernelDefBuilder::Create())                                        \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),          \
      LinearAttentionRecurrent<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

template <typename T>
LinearAttentionRecurrent<T>::LinearAttentionRecurrent(const OpKernelInfo& info)
    : OpKernel(info) {
  std::string rule_str = info.GetAttrOrDefault<std::string>("update_rule", "gated_delta");
  update_rule_ = ParseUpdateRule(rule_str);
  scale_ = info.GetAttrOrDefault<float>("scale", 0.0f);
}

template <typename T>
void LinearAttentionRecurrent<T>::ComputeSingleHead(
    const float* q, const float* k, const float* v,
    float* state,
    const float* decay, float beta_val,
    float* output,
    int d_k, int d_v, float scale) const {

  // Step 1: Compute retrieved = (decay * S)^T k  (needed for delta modes)
  std::vector<float> retrieved(d_v, 0.0f);
  if (update_rule_ == LinearAttentionUpdateRule::kDelta ||
      update_rule_ == LinearAttentionUpdateRule::kGatedDelta) {
    for (int i = 0; i < d_k; i++) {
      // For gated_delta the state is decayed before retrieval (same as CUDA impl)
      float gi = (update_rule_ == LinearAttentionUpdateRule::kGatedDelta) ? decay[i] : 1.0f;
      for (int j = 0; j < d_v; j++) {
        retrieved[j] += gi * state[i * d_v + j] * k[i];
      }
    }
  }

  // Step 2 + 3: Update state and accumulate output in a single pass
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

  // Scale output
  for (int j = 0; j < d_v; j++) {
    output[j] *= scale;
  }
}

template <typename T>
Status LinearAttentionRecurrent<T>::Compute(OpKernelContext* context) const {
  const Tensor* query      = context->Input<Tensor>(0);
  const Tensor* key        = context->Input<Tensor>(1);
  const Tensor* value      = context->Input<Tensor>(2);
  const Tensor* past_state = context->Input<Tensor>(3);
  const Tensor* decay      = context->Input<Tensor>(4);  // optional
  const Tensor* beta       = context->Input<Tensor>(5);  // optional

  ORT_RETURN_IF_NOT(query      != nullptr, "query input is required");
  ORT_RETURN_IF_NOT(key        != nullptr, "key input is required");
  ORT_RETURN_IF_NOT(value      != nullptr, "value input is required");
  ORT_RETURN_IF_NOT(past_state != nullptr, "past_state input is required");

  const auto& q_shape = query->Shape();
  const auto& k_shape = key->Shape();
  const auto& v_shape = value->Shape();
  const auto& s_shape = past_state->Shape();

  ORT_RETURN_IF_NOT(q_shape.NumDimensions() == 4, "query must be 4D (B,H,1,d_k)");
  ORT_RETURN_IF_NOT(k_shape.NumDimensions() == 4, "key must be 4D (B,H,1,d_k)");
  ORT_RETURN_IF_NOT(v_shape.NumDimensions() == 4, "value must be 4D (B,H,1,d_v)");
  ORT_RETURN_IF_NOT(s_shape.NumDimensions() == 4, "past_state must be 4D (B,H,d_k,d_v)");

  ORT_RETURN_IF_NOT(q_shape[2] == 1, "query sequence length must be 1 (recurrent mode)");
  ORT_RETURN_IF_NOT(k_shape[2] == 1, "key sequence length must be 1 (recurrent mode)");
  ORT_RETURN_IF_NOT(v_shape[2] == 1, "value sequence length must be 1 (recurrent mode)");

  const int batch_size = static_cast<int>(q_shape[0]);
  const int num_heads  = static_cast<int>(q_shape[1]);
  const int d_k        = static_cast<int>(q_shape[3]);
  const int d_v        = static_cast<int>(v_shape[3]);

  ORT_RETURN_IF_NOT(s_shape[0] == batch_size && s_shape[1] == num_heads &&
                    s_shape[2] == d_k        && s_shape[3] == d_v,
                    "past_state shape must be (B,H,d_k,d_v)");

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

  float scale = (scale_ == 0.0f) ? (1.0f / sqrtf(static_cast<float>(d_k))) : scale_;

  Tensor* output        = context->Output(0, TensorShape({batch_size, num_heads, 1, d_v}));
  Tensor* present_state = context->Output(1, s_shape);

  const T* q_data     = query->Data<T>();
  const T* k_data     = key->Data<T>();
  const T* v_data     = value->Data<T>();
  const T* s_data     = past_state->Data<T>();
  T*       out_data   = output->MutableData<T>();
  T*       pstate_data = present_state->MutableData<T>();

  const int state_elems = batch_size * num_heads * d_k * d_v;
  std::vector<float> state_f(state_elems);
  for (int i = 0; i < state_elems; i++) {
    state_f[i] = ToFloat(s_data[i]);
  }

  for (int b = 0; b < batch_size; b++) {
    for (int h = 0; h < num_heads; h++) {
      const int bh = b * num_heads + h;

      std::vector<float> q_f(d_k), k_f(d_k), v_f(d_v);
      for (int i = 0; i < d_k; i++) {
        q_f[i] = ToFloat(q_data[bh * d_k + i]);
        k_f[i] = ToFloat(k_data[bh * d_k + i]);
      }
      for (int j = 0; j < d_v; j++) {
        v_f[j] = ToFloat(v_data[bh * d_v + j]);
      }

      std::vector<float> decay_f(d_k, 1.0f);
      if (decay != nullptr) {
        const T* decay_data = decay->Data<T>();
        if (decay_broadcasted) {
          for (int i = 0; i < d_k; i++) {
            decay_f[i] = expf(ToFloat(decay_data[bh * d_k + i]));
          }
        } else {
          const float scalar = expf(ToFloat(decay_data[bh]));
          std::fill(decay_f.begin(), decay_f.end(), scalar);
        }
      }

      float beta_val = 0.0f;
      if (beta != nullptr) {
        beta_val = ToFloat(beta->Data<T>()[bh]);
      }

      std::vector<float> out_f(d_v, 0.0f);
      ComputeSingleHead(
          q_f.data(), k_f.data(), v_f.data(),
          state_f.data() + bh * d_k * d_v,
          decay_f.data(), beta_val,
          out_f.data(),
          d_k, d_v, scale);

      for (int j = 0; j < d_v; j++) {
        StoreFloat(out_f[j], out_data[bh * d_v + j]);
      }
    }
  }

  for (int i = 0; i < state_elems; i++) {
    StoreFloat(state_f[i], pstate_data[i]);
  }

  return Status::OK();
}

}
}
