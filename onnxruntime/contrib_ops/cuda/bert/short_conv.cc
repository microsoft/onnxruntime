// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/short_conv.h"
#include "contrib_ops/cuda/bert/short_conv_impl.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_type_conversion.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      ShortConv,                                                  \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      ShortConv<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

#undef REGISTER_KERNEL_TYPED

template <typename T>
ShortConv<T>::ShortConv(const OpKernelInfo& info) : CudaKernel(info) {
  activation_ = info.GetAttrOrDefault<std::string>("activation", "silu");
  ORT_ENFORCE(activation_ == "none" || activation_ == "silu" || activation_ == "swish",
              "activation must be one of: none, silu, swish");
  dilation_ = info.GetAttrOrDefault<int64_t>("dilation", 1);
  ORT_ENFORCE(dilation_ >= 1, "dilation must be >= 1");
  epsilon_ = info.GetAttrOrDefault<float>("epsilon", 1.0e-5f);
}

template <typename T>
Status ShortConv<T>::ComputeInternal(OpKernelContext* context) const {
  using CudaT = typename OrtToCudaType<T>::type;
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* weight = context->Input<Tensor>(1);
  const Tensor* norm_scale = context->Input<Tensor>(2);
  const Tensor* bias = context->Input<Tensor>(3);
  const Tensor* past_state = context->Input<Tensor>(4);

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
  ORT_RETURN_IF_NOT(scale_shape[0] == hc_mult && scale_shape[1] == hidden_size,
                    "norm_scale shape must match input hc_mult and hidden_size");
  ORT_RETURN_IF_NOT(weight_shape[0] == channels && weight_shape[1] == 1,
                    "weight must have shape (hc_mult * hidden_size, 1, kernel_size)");
  if (bias != nullptr) {
    ORT_RETURN_IF_NOT(bias->Shape().NumDimensions() == 1 && bias->Shape()[0] == channels,
                      "bias must have shape (hc_mult * hidden_size)");
  }

  // The convolution receptive field reaches this many positions before the current token, so this is
  // exactly the amount of normed history that has to be carried between invocations.
  const int64_t state_length = (kernel_size - 1) * dilation_;
  const TensorShape state_shape({batch_size, state_length, hc_mult, hidden_size});
  if (past_state != nullptr) {
    ORT_RETURN_IF_NOT(past_state->Shape() == state_shape,
                      "past_state must have shape (batch_size, (kernel_size - 1) * dilation, hc_mult, hidden_size)");
  }

  Tensor* output = context->Output(0, input_shape);
  Tensor* present_state = context->Output(1, state_shape);
  if (input_shape.Size() == 0) {
    return Status::OK();
  }

  // Scratch buffer holding one inverse-RMS value per (batch, sequence, hc_mult) row so that the
  // reduction is not repeated for every output channel and convolution tap.
  const int64_t rows = batch_size * sequence_length * hc_mult;
  auto inv_rms = GetScratchBuffer<float>(static_cast<size_t>(rows), GetComputeStream(context));
  return LaunchShortConvKernel<CudaT>(
      Stream(context),
      reinterpret_cast<const CudaT*>(input->Data<T>()),
      reinterpret_cast<const CudaT*>(weight->Data<T>()),
      reinterpret_cast<const CudaT*>(norm_scale->Data<T>()),
      bias == nullptr ? nullptr : reinterpret_cast<const CudaT*>(bias->Data<T>()),
      past_state == nullptr ? nullptr : reinterpret_cast<const CudaT*>(past_state->Data<T>()),
      inv_rms.get(),
      reinterpret_cast<CudaT*>(output->MutableData<T>()),
      present_state == nullptr ? nullptr : reinterpret_cast<CudaT*>(present_state->MutableData<T>()),
      batch_size,
      sequence_length,
      hc_mult,
      hidden_size,
      kernel_size,
      dilation_,
      epsilon_,
      activation_ == "silu" || activation_ == "swish");
}

template class ShortConv<float>;
template class ShortConv<MLFloat16>;
template class ShortConv<BFloat16>;

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
