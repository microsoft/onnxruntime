// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/bert/causal_conv_with_state.h"

#include "core/framework/tensorprotoutils.h"
#include "core/common/safeint.h"
#include "core/platform/threadpool.h"

#include <cmath>
#include <cstring>
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
// MLAS usage: No MLAS kernels are used currently. The depthwise causal conv
// is implemented with scalar loops. Potential future optimization: use
// MlasConv1D or vectorized MLAS routines for the 1D convolution.
#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      CausalConvWithState,                                        \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCpuExecutionProvider,                                      \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      CausalConvWithState<T>);

REGISTER_KERNEL_TYPED(float)

template <typename T>
CausalConvWithState<T>::CausalConvWithState(const OpKernelInfo& info) : OpKernel(info) {
  int64_t ndim = info.GetAttrOrDefault<int64_t>("ndim", 1);
  ORT_ENFORCE(ndim == 1, "CPU CausalConvWithState only supports ndim=1");
  ndim_ = static_cast<int>(ndim);

  activation_ = info.GetAttrOrDefault<std::string>("activation", "none");
  ORT_ENFORCE(activation_ == "none" || activation_ == "silu" || activation_ == "swish",
              "activation must be one of: none, silu, swish");
}

namespace {

inline float ApplySilu(float x) {
  return x / (1.0f + std::exp(-x));
}

template <int K>
inline void ProcessChannelDecodeFixedK(
    const float* past_row,
    const float* input_val,
    const float* w,
    float bias_val,
    bool apply_silu,
    float* out_val,
    float* present_row) {
  constexpr int pad = K - 1;
  float sum = bias_val;
  if (past_row != nullptr) {
    for (int k = 0; k < pad; ++k) {
      sum += w[k] * past_row[k];
    }
  }
  sum += w[pad] * input_val[0];

  if (apply_silu) {
    sum = ApplySilu(sum);
  }
  out_val[0] = sum;

  if constexpr (pad > 0) {
    if (past_row != nullptr) {
      if constexpr (pad > 1) {
        std::memcpy(present_row, past_row + 1, static_cast<size_t>(pad - 1) * sizeof(float));
      }
    } else {
      if constexpr (pad > 1) {
        std::memset(present_row, 0, static_cast<size_t>(pad - 1) * sizeof(float));
      }
    }
    present_row[pad - 1] = input_val[0];
  }
}

// Decode fast-path: L=1, no padded buffer needed.
// The "visible window" for position 0 is [past_state(K-1 values), input(1 value)] = K values.
// Compute dot(weight, window), shift state left by 1, append new input.
void ProcessChannelDecode(
    const float* past_row,   // past_state for this (b,c): [K-1] or nullptr
    const float* input_val,  // &input[b,c,0] — single value
    const float* w,          // weight for this channel: [K]
    float bias_val,
    bool apply_silu,
    float* out_val,      // &output[b,c,0] — single value
    float* present_row,  // present_state for this (b,c): [K-1]
    int64_t K) {
  int64_t pad = K - 1;

  // Dot product over the window: [past_state..., input]
  float sum = bias_val;
  // First K-1 elements come from past_state
  if (past_row != nullptr) {
    for (int64_t k = 0; k < pad; ++k) {
      sum += w[k] * past_row[k];
    }
  }
  // Last element is the current input
  sum += w[pad] * input_val[0];

  if (apply_silu) {
    sum = ApplySilu(sum);
  }
  out_val[0] = sum;

  // Update present_state: shift past_state left by 1, append input
  if (pad > 0) {
    if (past_row != nullptr && pad > 1) {
      std::memcpy(present_row, past_row + 1, static_cast<size_t>(pad - 1) * sizeof(float));
    } else if (pad > 1) {
      std::memset(present_row, 0, static_cast<size_t>(pad - 1) * sizeof(float));
    }
    present_row[pad - 1] = input_val[0];
  }
}

// Prefill path: L>1, uses padded buffer for the convolution window.
void ProcessChannelPrefill(
    const float* past_row,  // past_state for this (b,c): [K-1] or nullptr
    const float* in_row,    // input for this (b,c): [L]
    const float* w,         // weight for this channel: [K]
    float bias_val,
    bool apply_silu,
    float* out_row,      // output for this (b,c): [L]
    float* present_row,  // present_state for this (b,c): [K-1]
    float* padded_row,   // scratch buffer: [K-1 + L]
    int64_t L,
    int64_t K) {
  int64_t pad = K - 1;
  int64_t padded_len = pad + L;

  // Build padded window: [past_state | input]
  if (past_row != nullptr) {
    std::memcpy(padded_row, past_row, static_cast<size_t>(pad) * sizeof(float));
  } else {
    std::memset(padded_row, 0, static_cast<size_t>(pad) * sizeof(float));
  }
  std::memcpy(padded_row + pad, in_row, static_cast<size_t>(L) * sizeof(float));

  // Depthwise 1D convolution
  for (int64_t l = 0; l < L; ++l) {
    float sum = bias_val;
    for (int64_t k = 0; k < K; ++k) {
      sum += w[k] * padded_row[l + k];
    }
    if (apply_silu) {
      sum = ApplySilu(sum);
    }
    out_row[l] = sum;
  }

  // Save present_state: last K-1 elements of (past_state | input)
  std::memcpy(present_row, padded_row + padded_len - pad, static_cast<size_t>(pad) * sizeof(float));
}

}  // anonymous namespace

template <typename T>
Status CausalConvWithState<T>::Compute(OpKernelContext* context) const {
  const Tensor* input_tensor = context->Input<Tensor>(0);
  const Tensor* weight_tensor = context->Input<Tensor>(1);
  const Tensor* bias_tensor = context->Input<Tensor>(2);        // optional
  const Tensor* past_state_tensor = context->Input<Tensor>(3);  // optional

  ORT_RETURN_IF_NOT(input_tensor != nullptr, "input is required");
  ORT_RETURN_IF_NOT(weight_tensor != nullptr, "weight is required");

  const auto& input_shape = input_tensor->Shape();
  const auto& weight_shape = weight_tensor->Shape();

  ORT_RETURN_IF_NOT(static_cast<int>(input_shape.NumDimensions()) == 2 + ndim_,
                    "input must have ", 2 + ndim_, " dimensions for ndim=", ndim_);
  ORT_RETURN_IF_NOT(static_cast<int>(weight_shape.NumDimensions()) == 2 + ndim_,
                    "weight must have ", 2 + ndim_, " dimensions for ndim=", ndim_);

  const int64_t batch_size = input_shape[0];
  const int64_t channels = input_shape[1];

  ORT_RETURN_IF_NOT(weight_shape[0] == channels, "weight channels must match input channels");
  ORT_RETURN_IF_NOT(weight_shape[1] == 1, "weight must be depthwise (group=1)");

  if (bias_tensor != nullptr) {
    ORT_RETURN_IF_NOT(bias_tensor->Shape().NumDimensions() == 1 &&
                          bias_tensor->Shape()[0] == channels,
                      "bias must be 1D with size C");
  }

  // ==== ndim=1 implementation: (B, C, L) with kernel (C, 1, K) ====
  if (ndim_ == 1) {
    const int64_t L = input_shape[2];
    const int64_t K = weight_shape[2];
    const int64_t pad = K - 1;

    if (past_state_tensor != nullptr) {
      const auto& ps_shape = past_state_tensor->Shape();
      ORT_RETURN_IF_NOT(ps_shape.NumDimensions() == 3 &&
                            ps_shape[0] == batch_size &&
                            ps_shape[1] == channels &&
                            ps_shape[2] == pad,
                        "past_state must be (B, C, K-1)");
    }

    // ==== Allocate outputs ====
    Tensor* output_tensor = context->Output(0, input_shape);
    float* output_data = output_tensor->MutableData<float>();

    TensorShape state_shape({batch_size, channels, pad});
    Tensor* present_state_tensor = context->Output(1, state_shape);
    float* present_data = present_state_tensor->MutableData<float>();

    const float* input_data = input_tensor->Data<float>();
    const float* weight_data = weight_tensor->Data<float>();
    const float* bias_data = bias_tensor ? bias_tensor->Data<float>() : nullptr;
    const float* past_data = past_state_tensor ? past_state_tensor->Data<float>() : nullptr;
    bool apply_silu = (activation_ == "silu" || activation_ == "swish");

    // ==== Thread-parallel over (batch, channel) pairs ====
    // Depthwise conv: each channel is fully independent.
    int64_t total_tasks = batch_size * channels;
    double cost_per_task = static_cast<double>(L * K);  // FLOPs per channel

    auto* tp = context->GetOperatorThreadPool();

    if (L == 1) {
      // ==== Decode fast-path: no padded buffer needed ====
      ThreadPool::TryParallelFor(
          tp,
          static_cast<std::ptrdiff_t>(total_tasks),
          cost_per_task,
          [&](std::ptrdiff_t first, std::ptrdiff_t last) {
            for (std::ptrdiff_t task = first; task < last; ++task) {
              int64_t b = task / channels;
              int64_t c = task % channels;

              const float* past_row = past_data
                                          ? past_data + (b * channels + c) * pad
                                          : nullptr;
              const float* input_val = input_data + (b * channels + c) * L;
              const float* w = weight_data + c * K;
              float bias_val = bias_data ? bias_data[c] : 0.0f;
              float* out_val = output_data + (b * channels + c) * L;
              float* present_row = present_data + (b * channels + c) * pad;
              switch (K) {
                case 2:
                  ProcessChannelDecodeFixedK<2>(past_row, input_val, w, bias_val, apply_silu,
                                                out_val, present_row);
                  break;
                case 3:
                  ProcessChannelDecodeFixedK<3>(past_row, input_val, w, bias_val, apply_silu,
                                                out_val, present_row);
                  break;
                case 4:
                  ProcessChannelDecodeFixedK<4>(past_row, input_val, w, bias_val, apply_silu,
                                                out_val, present_row);
                  break;
                case 5:
                  ProcessChannelDecodeFixedK<5>(past_row, input_val, w, bias_val, apply_silu,
                                                out_val, present_row);
                  break;
                default:
                  ProcessChannelDecode(past_row, input_val, w, bias_val, apply_silu,
                                       out_val, present_row, K);
                  break;
              }
            }
          });
    } else {
      // ==== Prefill path: uses per-thread scratch buffer ====
      ThreadPool::TryParallelFor(
          tp,
          static_cast<std::ptrdiff_t>(total_tasks),
          cost_per_task,
          [&](std::ptrdiff_t first, std::ptrdiff_t last) {
            // Per-thread scratch buffer for padded input
            std::vector<float> padded_buf(static_cast<size_t>(pad + L));

            for (std::ptrdiff_t task = first; task < last; ++task) {
              int64_t b = task / channels;
              int64_t c = task % channels;

              const float* past_row = past_data
                                          ? past_data + (b * channels + c) * pad
                                          : nullptr;
              const float* in_row = input_data + (b * channels + c) * L;
              const float* w = weight_data + c * K;
              float bias_val = bias_data ? bias_data[c] : 0.0f;
              float* out_row = output_data + (b * channels + c) * L;
              float* present_row = present_data + (b * channels + c) * pad;

              ProcessChannelPrefill(past_row, in_row, w, bias_val, apply_silu,
                                    out_row, present_row, padded_buf.data(), L, K);
            }
          });
    }

    return Status::OK();
  }

  // ==== ndim=2 or ndim=3: not yet implemented ====
  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                         "CausalConvWithState with ndim=", ndim_,
                         " is not yet implemented. "
                         "Currently only ndim=1 is supported.");
}

}  // namespace contrib
}  // namespace onnxruntime
