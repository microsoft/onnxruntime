// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/bert/causal_conv_with_state.h"
#include "contrib_ops/cpu/bert/causal_conv_with_state_helper.h"

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

  ORT_THROW_IF_ERROR(causal_conv_with_state_helper::ParseDilation(info, dilation_));
  ORT_THROW_IF_ERROR(causal_conv_with_state_helper::ParseChannelsLast(info, channels_last_));
  ORT_ENFORCE(!channels_last_ || ndim_ == 1, "channels_last requires ndim = 1");

  ORT_ENFORCE(info.GetAttrOrDefault<int64_t>("state_window", 0) == 0,
              "CPU CausalConvWithState does not support state_window > 0 (CUDA EP only)");
  state_window_ = 0;
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

// A channel's positions are contiguous in the channels-first layout (stride 1) and `channels`
// apart in the channels-last one, so every per-channel walk goes through these two helpers.
inline void GatherStrided(float* dst, const float* src, int64_t stride, int64_t count) {
  if (stride == 1) {
    std::memcpy(dst, src, static_cast<size_t>(count) * sizeof(float));
    return;
  }
  for (int64_t i = 0; i < count; ++i) {
    dst[i] = src[i * stride];
  }
}

inline void ScatterStrided(float* dst, int64_t stride, const float* src, int64_t count) {
  if (stride == 1) {
    std::memcpy(dst, src, static_cast<size_t>(count) * sizeof(float));
    return;
  }
  for (int64_t i = 0; i < count; ++i) {
    dst[i * stride] = src[i];
  }
}

// Decode fast-path: L=1, no padded buffer needed.
// The "visible window" is [past_state(pad values), input(1 value)] with pad = (K-1)*dilation.
// Tap k reads window position k*dilation, so the last tap is the current input. The state is then
// shifted left by one position and the new input appended.
void ProcessChannelDecode(
    const float* past_row,   // past_state for this (b,c): [pad] strided, or nullptr
    const float* input_val,  // input for this (b,c) — single value
    const float* w,          // weight for this channel: [K]
    float bias_val,
    bool apply_silu,
    float* out_val,      // output for this (b,c) — single value
    float* present_row,  // present_state for this (b,c): [pad] strided
    int64_t state_stride,
    int64_t K,
    int64_t dilation) {
  const int64_t pad = (K - 1) * dilation;

  // Dot product over the window: [past_state..., input]
  float sum = bias_val;
  // The first K-1 taps land in past_state, spaced `dilation` apart.
  if (past_row != nullptr) {
    for (int64_t k = 0; k < K - 1; ++k) {
      sum += w[k] * past_row[k * dilation * state_stride];
    }
  }
  // Last tap is the current input
  sum += w[K - 1] * input_val[0];

  if (apply_silu) {
    sum = ApplySilu(sum);
  }
  out_val[0] = sum;

  // Update present_state: shift past_state left by 1, append input. The copy runs forward from a
  // higher source index, so it stays correct even if past_state and present_state are the same
  // buffer.
  if (pad > 0) {
    for (int64_t s = 0; s < pad - 1; ++s) {
      present_row[s * state_stride] =
          (past_row != nullptr) ? past_row[(s + 1) * state_stride] : 0.0f;
    }
    present_row[(pad - 1) * state_stride] = input_val[0];
  }
}

// Prefill path: L>1, uses padded buffer for the convolution window.
void ProcessChannelPrefill(
    const float* past_row,  // past_state for this (b,c): [pad] strided, or nullptr
    const float* in_row,    // input for this (b,c): [L] strided
    const float* w,         // weight for this channel: [K]
    float bias_val,
    bool apply_silu,
    float* out_row,      // output for this (b,c): [L] strided
    float* present_row,  // present_state for this (b,c): [pad] strided
    float* padded_row,   // scratch buffer: [pad + L]
    float* out_buf,      // scratch buffer: [L], only used when act_stride != 1
    int64_t act_stride,
    int64_t state_stride,
    int64_t L,
    int64_t K,
    int64_t dilation) {
  const int64_t pad = (K - 1) * dilation;
  const int64_t padded_len = pad + L;

  // Build padded window: [past_state | input]
  if (past_row != nullptr) {
    GatherStrided(padded_row, past_row, state_stride, pad);
  } else {
    std::memset(padded_row, 0, static_cast<size_t>(pad) * sizeof(float));
  }
  GatherStrided(padded_row + pad, in_row, act_stride, L);

  // Depthwise 1D convolution. Tap k of output position l reads padded_row[l + k*dilation]; at
  // k = K-1 that is padded_row[l + pad], i.e. the current input position.
  // A contiguous output row is written in place; only a strided one needs the scratch round trip,
  // so the channels-first layout keeps the exact write pattern it had before strides existed.
  float* conv_dst = act_stride == 1 ? out_row : out_buf;
  for (int64_t l = 0; l < L; ++l) {
    float sum = bias_val;
    for (int64_t k = 0; k < K; ++k) {
      sum += w[k] * padded_row[l + k * dilation];
    }
    if (apply_silu) {
      sum = ApplySilu(sum);
    }
    conv_dst[l] = sum;
  }
  if (act_stride != 1) {
    ScatterStrided(out_row, act_stride, out_buf, L);
  }

  // Save present_state: last pad elements of (past_state | input)
  ScatterStrided(present_row, state_stride, padded_row + padded_len - pad, pad);
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

  if (channels_last_) {
    // (batch_size, sequence_length, d_1, ..., d_n): any number of trailing channel axes, so a
    // caller that keeps hyper-connections and hidden size separate needs no reshape.
    ORT_RETURN_IF_NOT(input_shape.NumDimensions() >= 3,
                      "input must have at least 3 dimensions when channels_last = 1");
  } else {
    ORT_RETURN_IF_NOT(static_cast<int>(input_shape.NumDimensions()) == 2 + ndim_,
                      "input must have ", 2 + ndim_, " dimensions for ndim=", ndim_);
  }
  ORT_RETURN_IF_NOT(static_cast<int>(weight_shape.NumDimensions()) == 2 + ndim_,
                    "weight must have ", 2 + ndim_, " dimensions for ndim=", ndim_);

  const int64_t batch_size = input_shape[0];
  const int64_t channels = channels_last_
                               ? input_shape.SizeFromDimension(2)
                               : input_shape[1];

  ORT_RETURN_IF_NOT(weight_shape[0] == channels, "weight channels must match input channels");
  ORT_RETURN_IF_NOT(weight_shape[1] == 1, "weight must be depthwise (group=1)");

  if (bias_tensor != nullptr) {
    ORT_RETURN_IF_NOT(bias_tensor->Shape().NumDimensions() == 1 &&
                          bias_tensor->Shape()[0] == channels,
                      "bias must be 1D with size C");
  }

  // ==== ndim=1 implementation: (B, C, L) with kernel (C, 1, K) ====
  if (ndim_ == 1) {
    const int64_t L = channels_last_ ? input_shape[1] : input_shape[2];
    const int64_t K = weight_shape[2];
    const int64_t dilation = dilation_;
    const int64_t pad = (K - 1) * dilation;

    // ==== Allocate outputs ====
    Tensor* output_tensor = context->Output(0, input_shape);
    float* output_data = output_tensor->MutableData<float>();

    // state_window_ is always 0 on CPU, so the state has no leading window axis.
    TensorShape state_shape;
    if (channels_last_) {
      ORT_RETURN_IF_ERROR(causal_conv_with_state_helper::CheckInputsChannelsLast(
          state_window_, input_shape, static_cast<int>(pad), past_state_tensor, state_shape,
          "CausalConvWithState"));
    } else {
      ORT_RETURN_IF_ERROR(causal_conv_with_state_helper::CheckInputs(
          state_window_, static_cast<int>(batch_size), static_cast<int>(channels),
          static_cast<int>(pad), past_state_tensor, state_shape, "CausalConvWithState"));
    }
    Tensor* present_state_tensor = context->Output(1, state_shape);
    float* present_data = present_state_tensor->MutableData<float>();

    const float* input_data = input_tensor->Data<float>();
    const float* weight_data = weight_tensor->Data<float>();
    const float* bias_data = bias_tensor ? bias_tensor->Data<float>() : nullptr;
    const float* past_data = past_state_tensor ? past_state_tensor->Data<float>() : nullptr;
    bool apply_silu = (activation_ == "silu" || activation_ == "swish");

    // Both layouts are dense, so one strided (batch, position, channel) view covers them.
    const auto act_layout = causal_conv_with_state_helper::MakeLayout(channels_last_, channels, L);
    const auto state_layout =
        causal_conv_with_state_helper::MakeLayout(channels_last_, channels, pad);

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

              const int64_t act_offset = act_layout.Offset(b, 0, c);
              const int64_t state_offset = state_layout.Offset(b, 0, c);

              const float* past_row = past_data ? past_data + state_offset : nullptr;
              const float* input_val = input_data + act_offset;
              const float* w = weight_data + c * K;
              float bias_val = bias_data ? bias_data[c] : 0.0f;
              float* out_val = output_data + act_offset;
              float* present_row = present_data + state_offset;
              // ProcessChannelDecodeFixedK assumes pad == K - 1 and contiguous state, so it only
              // applies to the undilated channels-first case.
              const int64_t fixed_k = (dilation == 1 && state_layout.pos_stride == 1) ? K : 0;
              switch (fixed_k) {
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
                                       out_val, present_row, state_layout.pos_stride, K, dilation);
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
            // Per-thread scratch buffers for the padded input window and, only when the output row
            // is strided, the contiguous convolution result that is then scattered into it.
            std::vector<float> padded_buf(static_cast<size_t>(pad + L));
            std::vector<float> out_buf(act_layout.pos_stride == 1 ? 0 : static_cast<size_t>(L));

            for (std::ptrdiff_t task = first; task < last; ++task) {
              int64_t b = task / channels;
              int64_t c = task % channels;

              const int64_t act_offset = act_layout.Offset(b, 0, c);
              const int64_t state_offset = state_layout.Offset(b, 0, c);

              const float* past_row = past_data ? past_data + state_offset : nullptr;
              const float* in_row = input_data + act_offset;
              const float* w = weight_data + c * K;
              float bias_val = bias_data ? bias_data[c] : 0.0f;
              float* out_row = output_data + act_offset;
              float* present_row = present_data + state_offset;

              ProcessChannelPrefill(past_row, in_row, w, bias_val, apply_silu,
                                    out_row, present_row, padded_buf.data(), out_buf.data(),
                                    act_layout.pos_stride, state_layout.pos_stride, L, K, dilation);
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
