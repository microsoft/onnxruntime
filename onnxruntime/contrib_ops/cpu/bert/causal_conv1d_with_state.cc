#include "contrib_ops/cpu/bert/causal_conv1d_with_state.h"

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

inline float ApplySiLU(float x) {
  return x / (1.0f + expf(-x));
}

}

#define REGISTER_KERNEL_TYPED(T)                                           \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                           \
      CausalConv1DWithState,                                               \
      kMSDomain,                                                           \
      1,                                                                   \
      T,                                                                   \
      kCpuExecutionProvider,                                               \
      (*KernelDefBuilder::Create())                                        \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),          \
      CausalConv1DWithState<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

template <typename T>
CausalConv1DWithState<T>::CausalConv1DWithState(const OpKernelInfo& info)
    : OpKernel(info) {
  activation_str_ = info.GetAttrOrDefault<std::string>("activation", "silu");
  if (activation_str_ == "silu" || activation_str_ == "swish") {
    activation_ = CausalConv1DActivation::kSiLU;
  } else if (activation_str_ == "none") {
    activation_ = CausalConv1DActivation::kNone;
  } else {
    ORT_THROW("CausalConv1DWithState: unknown activation '", activation_str_, "'");
  }
}

template <typename T>
Status CausalConv1DWithState<T>::Compute(OpKernelContext* context) const {
  const Tensor* input      = context->Input<Tensor>(0);   // (B, D, T)
  const Tensor* weight     = context->Input<Tensor>(1);   // (D, 1, K)
  const Tensor* bias       = context->Input<Tensor>(2);   // (D,) optional
  const Tensor* conv_state = context->Input<Tensor>(3);   // (B, D, K-1) optional

  ORT_RETURN_IF_NOT(input  != nullptr, "input is required");
  ORT_RETURN_IF_NOT(weight != nullptr, "weight is required");

  const auto& in_shape  = input->Shape();
  const auto& wt_shape  = weight->Shape();

  ORT_RETURN_IF_NOT(in_shape.NumDimensions() == 3, "input must be 3D (B,D,T)");
  ORT_RETURN_IF_NOT(wt_shape.NumDimensions() == 3, "weight must be 3D (D,1,K)");

  const int batch_size  = static_cast<int>(in_shape[0]);
  const int channels    = static_cast<int>(in_shape[1]);
  const int seq_len     = static_cast<int>(in_shape[2]);
  const int kernel_size = static_cast<int>(wt_shape[2]);
  const int state_len   = kernel_size - 1;

  ORT_RETURN_IF_NOT(wt_shape[0] == channels, "weight dim 0 must equal channels");
  ORT_RETURN_IF_NOT(wt_shape[1] == 1,        "weight dim 1 must be 1 (depthwise)");
  ORT_RETURN_IF_NOT(kernel_size <= 32,        "kernel_size must be <= 32");

  if (bias != nullptr) {
    ORT_RETURN_IF_NOT(bias->Shape().NumDimensions() == 1, "bias must be 1D");
    ORT_RETURN_IF_NOT(bias->Shape()[0] == channels,       "bias length must equal channels");
  }

  if (conv_state != nullptr) {
    const auto& cs = conv_state->Shape();
    ORT_RETURN_IF_NOT(cs.NumDimensions() == 3,   "conv_state must be 3D (B,D,K-1)");
    ORT_RETURN_IF_NOT(cs[0] == batch_size,        "conv_state batch size must match input");
    ORT_RETURN_IF_NOT(cs[1] == channels,          "conv_state channels must match input");
    ORT_RETURN_IF_NOT(cs[2] == state_len,         "conv_state dim 2 must be K-1");
  }

  Tensor* output        = context->Output(0, TensorShape({batch_size, channels, seq_len}));
  Tensor* present_state = context->Output(1, TensorShape({batch_size, channels, state_len}));

  const T* in_data  = input->Data<T>();
  const T* wt_data  = weight->Data<T>();
  T*       out_data = output->MutableData<T>();
  T*       ps_data  = present_state->MutableData<T>();

  for (int b = 0; b < batch_size; b++) {
    for (int d = 0; d < channels; d++) {
      const int bd = b * channels + d;

      float w[32];
      for (int k = 0; k < kernel_size; k++) {
        w[k] = ToFloat(wt_data[d * kernel_size + k]);
      }

      float bias_val = (bias != nullptr) ? ToFloat(bias->Data<T>()[d]) : 0.0f;

      for (int t = 0; t < seq_len; t++) {
        float sum = bias_val;

        for (int k = 0; k < kernel_size; k++) {
          const int src_t = t - state_len + k;

          float input_val;
          if (src_t >= 0) {
            input_val = ToFloat(in_data[bd * seq_len + src_t]);
          } else {
            const int state_idx = state_len + src_t;
            if (conv_state != nullptr && state_idx >= 0) {
              input_val = ToFloat(conv_state->Data<T>()[bd * state_len + state_idx]);
            } else {
              input_val = 0.0f;
            }
          }

          sum += w[k] * input_val;
        }

        if (activation_ == CausalConv1DActivation::kSiLU) {
          sum = ApplySiLU(sum);
        }

        StoreFloat(sum, out_data[bd * seq_len + t]);
      }

      for (int k = 0; k < state_len; k++) {
        const int src_t = seq_len - state_len + k;
        float val;
        if (src_t >= 0) {
          val = ToFloat(in_data[bd * seq_len + src_t]);
        } else {
          const int state_idx = state_len + src_t;
          if (conv_state != nullptr && state_idx >= 0) {
            val = ToFloat(conv_state->Data<T>()[bd * state_len + state_idx]);
          } else {
            val = 0.0f;
          }
        }
        StoreFloat(val, ps_data[bd * state_len + k]);
      }
    }
  }

  return Status::OK();
}

}
}
