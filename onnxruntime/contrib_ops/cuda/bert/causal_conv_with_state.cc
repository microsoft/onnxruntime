// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/causal_conv_with_state.h"
#include "contrib_ops/cuda/bert/causal_conv_with_state_impl.h"
#include "contrib_ops/cpu/bert/causal_conv_with_state_helper.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_type_conversion.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;  // CudaKernel, Stream, GetDeviceProp, ToCudaType

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      CausalConvWithState,                                        \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      CausalConvWithState<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

template <typename T>
CausalConvWithState<T>::CausalConvWithState(const OpKernelInfo& info) : CudaKernel(info) {
  int64_t ndim = info.GetAttrOrDefault<int64_t>("ndim", 1);
  ORT_ENFORCE(ndim == 1, "CUDA CausalConvWithState only supports ndim=1");
  ndim_ = static_cast<int>(ndim);

  activation_ = info.GetAttrOrDefault<std::string>("activation", "none");
  ORT_ENFORCE(activation_ == "none" || activation_ == "silu" || activation_ == "swish",
              "activation must be one of: none, silu, swish");

  ORT_THROW_IF_ERROR(causal_conv_with_state_helper::ParseDilation(info, dilation_));
  ORT_THROW_IF_ERROR(causal_conv_with_state_helper::ParseChannelsLast(info, channels_last_));
  ORT_ENFORCE(!channels_last_ || ndim_ == 1, "channels_last requires ndim = 1");

  // See LinearAttention: only the trailing per-position states are ever consumed, so a window
  // caps the allocation and the write traffic for long prompts. 0 keeps the plain single state.
  ORT_THROW_IF_ERROR(causal_conv_with_state_helper::ParseStateWindow(info, state_window_));
}

template <typename T>
Status CausalConvWithState<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input_tensor = context->Input<Tensor>(0);
  const Tensor* weight_tensor = context->Input<Tensor>(1);
  const Tensor* bias_tensor = context->Input<Tensor>(2);        // optional
  const Tensor* past_state_tensor = context->Input<Tensor>(3);  // optional

  ORT_RETURN_IF_NOT(input_tensor != nullptr, "input is required");
  ORT_RETURN_IF_NOT(weight_tensor != nullptr, "weight is required");

  const auto& input_shape = input_tensor->Shape();
  const auto& weight_shape = weight_tensor->Shape();

  // Validate input rank and weight rank
  if (channels_last_) {
    // (batch_size, sequence_length, d_1, ..., d_n): any number of trailing channel axes, so a
    // caller that keeps hyper-connections and hidden size separate needs no reshape.
    ORT_RETURN_IF_NOT(input_shape.NumDimensions() >= 3,
                      "input must have rank >= 3 (batch, length, ...channels) when "
                      "channels_last = 1, got rank ",
                      input_shape.NumDimensions());
  } else {
    ORT_RETURN_IF_NOT(input_shape.NumDimensions() == 3,
                      "input must be rank 3 (batch, channels, length), got rank ", input_shape.NumDimensions());
  }
  ORT_RETURN_IF_NOT(weight_shape.NumDimensions() == 3,
                    "weight must be rank 3 (channels, 1, kernel_size), got rank ", weight_shape.NumDimensions());

  const int batch_size = static_cast<int>(input_shape[0]);
  const int channels = static_cast<int>(channels_last_ ? input_shape.SizeFromDimension(2)
                                                       : input_shape[1]);
  const int L = static_cast<int>(channels_last_ ? input_shape[1] : input_shape[2]);
  const int K = static_cast<int>(weight_shape[2]);
  const int pad = (K - 1) * dilation_;

  ORT_RETURN_IF_NOT(L > 0, "input length must be positive, got ", L);

  // Validate weight shape compatibility
  ORT_RETURN_IF_NOT(weight_shape[0] == channels,
                    "weight[0] (", weight_shape[0], ") must match input channels (", channels, ")");
  ORT_RETURN_IF_NOT(weight_shape[1] == 1,
                    "weight[1] must be 1 for depthwise convolution, got ", weight_shape[1]);

  // Validate optional bias shape
  if (bias_tensor != nullptr) {
    const auto& bias_shape = bias_tensor->Shape();
    ORT_RETURN_IF_NOT(bias_shape.NumDimensions() == 1 && bias_shape[0] == channels,
                      "bias must have shape (", channels, "), got ", bias_shape.ToString());
  }

  // past_state / present_state are [B, C, pad], or [W, B, C, pad] when state_window_ = W > 0,
  // where pad = (K-1)*dilation.
  // Right-aligned: token t lands in slot t + W - L, so slot W-1 always holds the state after the
  // last token (and is the slot past_state is read from).
  const int state_slots = state_window_ > 0 ? state_window_ : 1;
  TensorShape state_shape;
  if (channels_last_) {
    ORT_RETURN_IF_ERROR(causal_conv_with_state_helper::CheckInputsChannelsLast(
        state_window_, input_shape, pad, past_state_tensor, state_shape, "CausalConvWithState"));
  } else {
    ORT_RETURN_IF_ERROR(causal_conv_with_state_helper::CheckInputs(
        state_window_, batch_size, channels, pad, past_state_tensor, state_shape, "CausalConvWithState"));
  }

  // Allocate outputs
  Tensor* output_tensor = context->Output(0, input_shape);
  Tensor* present_state_tensor = context->Output(1, state_shape);

  // The kernel writes slot t + W - L for every position t, i.e. slots max(0, W - L) .. W-1. When
  // the window is wider than the sequence, the leading W - L slots belong to positions before this
  // call and are deliberately left alone (that is what bounds the per-step work). present_state is
  // a freshly allocated output that never aliases past_state, so zero those slots to keep the
  // output fully defined instead of exposing uninitialized device memory.
  if (state_window_ > 0 && state_slots > L) {
    const size_t leading_bytes = static_cast<size_t>(state_slots - L) *
                                 (present_state_tensor->SizeInBytes() / static_cast<size_t>(state_slots));
    if (leading_bytes > 0) {
      CUDA_RETURN_IF_ERROR(cudaMemsetAsync(
          present_state_tensor->MutableDataRaw(), 0, leading_bytes, Stream(context)));
    }
  }

  bool apply_silu = (activation_ == "silu" || activation_ == "swish");

  typedef typename OrtToCudaType<T>::type CudaT;

  return LaunchCausalConvWithStateKernel<CudaT>(
      Stream(context),
      reinterpret_cast<const CudaT*>(input_tensor->Data<T>()),
      reinterpret_cast<const CudaT*>(weight_tensor->Data<T>()),
      bias_tensor ? reinterpret_cast<const CudaT*>(bias_tensor->Data<T>()) : nullptr,
      past_state_tensor ? reinterpret_cast<const CudaT*>(past_state_tensor->Data<T>()) : nullptr,
      reinterpret_cast<CudaT*>(output_tensor->MutableData<T>()),
      reinterpret_cast<CudaT*>(present_state_tensor->MutableData<T>()),
      batch_size,
      channels,
      L,
      K,
      dilation_,
      MakeCausalConvLayout(channels_last_, channels, L),
      MakeCausalConvLayout(channels_last_, channels, pad),
      apply_silu,
      GetDeviceProp().maxThreadsPerBlock,
      state_slots);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
