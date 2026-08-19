// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/varlen_causal_conv_with_state.h"
#include "contrib_ops/cpu/bert/causal_conv_with_state_helper.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_type_conversion.h"

#include <limits>

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;  // CudaKernel, Stream, GetDeviceProp, ToCudaType

#define REGISTER_KERNEL_TYPED(T)                                        \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                        \
      VarlenCausalConvWithState,                                        \
      kMSDomain,                                                        \
      1,                                                                \
      T,                                                                \
      kCudaExecutionProvider,                                           \
      (*KernelDefBuilder::Create())                                     \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())        \
          .TypeConstraint("M", DataTypeImpl::GetTensorType<int32_t>()), \
      VarlenCausalConvWithState<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

template <typename T>
VarlenCausalConvWithState<T>::VarlenCausalConvWithState(const OpKernelInfo& info) : CudaKernel(info) {
  int64_t ndim = info.GetAttrOrDefault<int64_t>("ndim", 1);
  ORT_ENFORCE(ndim == 1, "CUDA VarlenCausalConvWithState only supports ndim=1");
  ndim_ = static_cast<int>(ndim);

  activation_ = info.GetAttrOrDefault<std::string>("activation", "none");
  ORT_ENFORCE(activation_ == "none" || activation_ == "silu" || activation_ == "swish",
              "activation must be one of: none, silu, swish");

  ORT_THROW_IF_ERROR(causal_conv_with_state_helper::ParseStateWindow(info, state_window_));
}

template <typename T>
Status VarlenCausalConvWithState<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input_tensor = context->Input<Tensor>(0);
  const Tensor* weight_tensor = context->Input<Tensor>(1);
  const Tensor* cu_seqlens_tensor = context->Input<Tensor>(2);
  const Tensor* bias_tensor = context->Input<Tensor>(3);        // optional
  const Tensor* past_state_tensor = context->Input<Tensor>(4);  // optional

  ORT_RETURN_IF_NOT(input_tensor != nullptr, "input is required");
  ORT_RETURN_IF_NOT(weight_tensor != nullptr, "weight is required");
  ORT_RETURN_IF_NOT(cu_seqlens_tensor != nullptr, "cumulative_sequence_length input is required");

  const auto& input_shape = input_tensor->Shape();
  const auto& weight_shape = weight_tensor->Shape();

  ORT_RETURN_IF_NOT(input_shape.NumDimensions() == 2,
                    "input must be rank 2 (total_tokens, channels), got rank ", input_shape.NumDimensions());
  ORT_RETURN_IF_NOT(weight_shape.NumDimensions() == 3,
                    "weight must be rank 3 (channels, 1, kernel_size), got rank ", weight_shape.NumDimensions());

  const auto& cu_seqlens_shape = cu_seqlens_tensor->Shape();
  ORT_RETURN_IF_NOT(cu_seqlens_shape.NumDimensions() == 1,
                    "cumulative_sequence_length must be rank 1 (batch_size + 1), got rank ",
                    cu_seqlens_shape.NumDimensions());
  ORT_RETURN_IF_NOT(cu_seqlens_shape[0] >= 2,
                    "cumulative_sequence_length must have at least 2 elements (batch_size >= 1), got ",
                    cu_seqlens_shape[0]);
  // batch_size = cu_seqlens.Shape()[0] - 1, never derived from total_tokens: a ragged batch has
  // no fixed relationship between the two beyond total_tokens >= batch_size.
  const int64_t batch_size_64 = cu_seqlens_shape[0] - 1;
  ORT_RETURN_IF_NOT(batch_size_64 <= std::numeric_limits<int>::max(),
                    "batch size is too large for the CUDA kernel");
  const int batch_size = static_cast<int>(batch_size_64);

  const int64_t total_tokens_64 = input_shape[0];
  const int64_t channels_64 = input_shape[1];
  ORT_RETURN_IF_NOT(total_tokens_64 <= std::numeric_limits<int>::max() &&
                        channels_64 <= std::numeric_limits<int>::max(),
                    "input dimensions are too large for the CUDA kernel");
  ORT_RETURN_IF_NOT(total_tokens_64 >= batch_size_64,
                    "total_tokens must be at least batch_size because every sequence must contain a token");
  ORT_RETURN_IF_NOT(channels_64 > 0, "input channel dimension must be positive");
  const int channels = static_cast<int>(channels_64);

  ORT_RETURN_IF_NOT(weight_shape[0] == channels_64,
                    "weight[0] (", weight_shape[0], ") must match input channels (", channels_64, ")");
  ORT_RETURN_IF_NOT(weight_shape[1] == 1,
                    "weight[1] must be 1 for depthwise convolution, got ", weight_shape[1]);
  const int64_t kernel_size_64 = weight_shape[2];
  ORT_RETURN_IF_NOT(kernel_size_64 >= 1 && kernel_size_64 <= std::numeric_limits<int>::max(),
                    "weight last dim (kernel_size) must be positive, got ", kernel_size_64);
  const int kernel_size = static_cast<int>(kernel_size_64);
  const int pad = kernel_size - 1;

  if (bias_tensor != nullptr) {
    const auto& bias_shape = bias_tensor->Shape();
    ORT_RETURN_IF_NOT(bias_shape.NumDimensions() == 1 && bias_shape[0] == channels_64,
                      "bias must have shape (", channels_64, "), got ", bias_shape.ToString());
  }

  // past_state / present_state are [batch_size, C, K-1], or [W, batch_size, C, K-1] when
  // state_window_ = W > 0. batch_size here is the number of packed sequences (from cu_seqlens).
  const int state_slots = state_window_ > 0 ? state_window_ : 1;
  TensorShape state_shape;
  ORT_RETURN_IF_ERROR(causal_conv_with_state_helper::CheckInputs(
      state_window_, batch_size, channels, pad, past_state_tensor, state_shape, "VarlenCausalConvWithState"));

  Tensor* output_tensor = context->Output(0, input_shape);
  Tensor* present_state_tensor = context->Output(1, state_shape);

  // Every sequence's own length decides which window slots it writes (right-aligned: slot
  // t + W - L), so which slots stay untouched varies per sequence -- unlike the dense op, no
  // single contiguous prefix is safe to skip. Zero the whole fresh present_state buffer
  // unconditionally so every slot this call does not write is well-defined.
  if (present_state_tensor->SizeInBytes() > 0) {
    CUDA_RETURN_IF_ERROR(cudaMemsetAsync(
        present_state_tensor->MutableDataRaw(), 0, present_state_tensor->SizeInBytes(), Stream(context)));
  }

  bool apply_silu = (activation_ == "silu" || activation_ == "swish");

  // total_tokens == batch_size is a host-visible shape fact under the trusted offsets contract:
  // every sequence contributes at least one token, so this can only hold when every sequence
  // contributes exactly one. It selects the dedicated fast path without reading cu_seqlens.
  const bool all_ones = (total_tokens_64 == batch_size_64);

  typedef typename OrtToCudaType<T>::type CudaT;

  return LaunchVarlenCausalConvWithStateKernel<CudaT>(
      Stream(context),
      reinterpret_cast<const CudaT*>(input_tensor->Data<T>()),
      reinterpret_cast<const CudaT*>(weight_tensor->Data<T>()),
      bias_tensor ? reinterpret_cast<const CudaT*>(bias_tensor->Data<T>()) : nullptr,
      past_state_tensor ? reinterpret_cast<const CudaT*>(past_state_tensor->Data<T>()) : nullptr,
      reinterpret_cast<CudaT*>(output_tensor->MutableData<T>()),
      reinterpret_cast<CudaT*>(present_state_tensor->MutableData<T>()),
      cu_seqlens_tensor->Data<int32_t>(),
      batch_size,
      all_ones,
      channels,
      kernel_size,
      apply_silu,
      GetDeviceProp().maxThreadsPerBlock,
      state_slots);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
