// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/short_conv_with_state.h"
#include "contrib_ops/cuda/bert/short_conv_with_state_impl.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_type_conversion.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      ShortConvWithState,                                         \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      ShortConvWithState<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

#undef REGISTER_KERNEL_TYPED

template <typename T>
ShortConvWithState<T>::ShortConvWithState(const OpKernelInfo& info) : CudaKernel(info) {
  activation_ = info.GetAttrOrDefault<std::string>("activation", "silu");
  ORT_ENFORCE(activation_ == "none" || activation_ == "silu" || activation_ == "swish",
              "activation must be one of: none, silu, swish");
  dilation_ = info.GetAttrOrDefault<int64_t>("dilation", 1);
  ORT_ENFORCE(dilation_ >= 1, "dilation must be >= 1");
  epsilon_ = info.GetAttrOrDefault<float>("epsilon", 1.0e-5f);
}

template <typename T>
Status ShortConvWithState<T>::ComputeInternal(OpKernelContext* context) const {
  using CudaT = typename OrtToCudaType<T>::type;
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* past_state = context->Input<Tensor>(1);
  const Tensor* norm_scale = context->Input<Tensor>(2);
  const Tensor* weight = context->Input<Tensor>(3);
  const Tensor* bias = context->Input<Tensor>(4);

  const TensorShape& input_shape = input->Shape();
  const TensorShape& weight_shape = weight->Shape();
  const TensorShape& scale_shape = norm_scale->Shape();

  ORT_RETURN_IF_NOT(input_shape.NumDimensions() == 4,
                    "input must have shape (batch_size, sequence_length, hc_mult, hidden_size)");
  ORT_RETURN_IF_NOT(weight_shape.NumDimensions() == 3,
                    "weight must have shape (hc_mult * hidden_size, 1, kernel_size)");
  ORT_RETURN_IF_NOT(scale_shape.NumDimensions() == 2,
                    "norm_scale must have shape (hc_mult, hidden_size)");

  const int64_t batch_size = input_shape[0];
  const int64_t sequence_length = input_shape[1];
  const int64_t hc_mult = input_shape[2];
  const int64_t hidden_size = input_shape[3];
  const int64_t channels = hc_mult * hidden_size;
  const int64_t kernel_size = weight_shape[2];
  const int64_t state_len = dilation_ * (kernel_size - 1);

  ORT_RETURN_IF_NOT(weight_shape[0] == channels && weight_shape[1] == 1 && weight_shape[2] == kernel_size,
                    "weight must have shape (hc_mult * hidden_size, 1, kernel_size)");
  ORT_RETURN_IF_NOT(scale_shape[0] == hc_mult && scale_shape[1] == hidden_size,
                    "norm_scale shape must match input hc_mult and hidden_size");

  if (past_state != nullptr) {
    const TensorShape& past_shape = past_state->Shape();
    ORT_RETURN_IF_NOT(past_shape.NumDimensions() == 3 &&
                          past_shape[0] == batch_size &&
                          past_shape[1] == channels &&
                          past_shape[2] == state_len,
                      "past_state must have shape (batch_size, channels, dilation*(kernel_size-1))");
  }
  if (bias != nullptr) {
    ORT_RETURN_IF_NOT(bias->Shape().NumDimensions() == 1 && bias->Shape()[0] == channels,
                      "bias must have shape (hc_mult * hidden_size)");
  }

  Tensor* output = context->Output(0, input_shape);
  TensorShape present_shape({batch_size, channels, state_len});
  Tensor* present_state_out = context->Output(1, present_shape);

  // Scratch buffer for normalized values in type T: [B, S, C]
  // Using T (not float) ensures precision is consistent across chunk boundaries —
  // normed values within a chunk match the T-rounded values stored in present_state.
  const int64_t normed_count = batch_size * sequence_length * channels;
  auto normed_ws = GetScratchBuffer<CudaT>(static_cast<size_t>(normed_count), GetComputeStream(context));

  return LaunchShortConvWithStateKernel<CudaT>(
      Stream(context),
      reinterpret_cast<const CudaT*>(input->Data<T>()),
      past_state == nullptr ? nullptr : reinterpret_cast<const CudaT*>(past_state->Data<T>()),
      reinterpret_cast<const CudaT*>(norm_scale->Data<T>()),
      reinterpret_cast<const CudaT*>(weight->Data<T>()),
      bias == nullptr ? nullptr : reinterpret_cast<const CudaT*>(bias->Data<T>()),
      normed_ws.get(),
      reinterpret_cast<CudaT*>(output->MutableData<T>()),
      reinterpret_cast<CudaT*>(present_state_out->MutableData<T>()),
      batch_size,
      sequence_length,
      hc_mult,
      hidden_size,
      kernel_size,
      dilation_,
      state_len,
      epsilon_,
      activation_ == "silu" || activation_ == "swish");
}

template class ShortConvWithState<float>;
template class ShortConvWithState<MLFloat16>;
template class ShortConvWithState<BFloat16>;

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
