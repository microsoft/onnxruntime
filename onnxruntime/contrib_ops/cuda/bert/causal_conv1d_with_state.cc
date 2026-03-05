// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cuda/bert/causal_conv1d_with_state.h"
#include "contrib_ops/cuda/bert/causal_conv1d_with_state_impl.h"

using namespace onnxruntime::cuda;
using namespace ::onnxruntime::common;

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                      \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                      \
      CausalConv1DWithState,                                          \
      kMSDomain,                                                      \
      1,                                                              \
      T,                                                              \
      kCudaExecutionProvider,                                         \
      (*KernelDefBuilder::Create())                                   \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),     \
      CausalConv1DWithState<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

template <typename T>
CausalConv1DWithState<T>::CausalConv1DWithState(const OpKernelInfo& info)
    : CudaKernel(info) {
  activation_ = info.GetAttrOrDefault<std::string>("activation", "silu");
}

template <typename T>
Status CausalConv1DWithState<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* weight = context->Input<Tensor>(1);
  const Tensor* bias = context->Input<Tensor>(2);     // optional
  const Tensor* conv_state = context->Input<Tensor>(3);  // optional

  ORT_RETURN_IF_NOT(input != nullptr, "input is required");
  ORT_RETURN_IF_NOT(weight != nullptr, "weight is required");

  const auto& input_shape = input->Shape();
  const auto& weight_shape = weight->Shape();

  ORT_RETURN_IF_NOT(input_shape.NumDimensions() == 3, "input must be 3D (B,D,T)");
  ORT_RETURN_IF_NOT(weight_shape.NumDimensions() == 3, "weight must be 3D (D,1,K)");

  int batch_size = static_cast<int>(input_shape[0]);
  int channels = static_cast<int>(input_shape[1]);
  int seq_len = static_cast<int>(input_shape[2]);
  int kernel_size = static_cast<int>(weight_shape[2]);

  ORT_RETURN_IF_NOT(weight_shape[0] == channels, "weight dim 0 must equal number of channels");
  ORT_RETURN_IF_NOT(weight_shape[1] == 1, "weight dim 1 must be 1 (depthwise)");
  ORT_RETURN_IF_NOT(kernel_size <= 32, "kernel_size must be <= 32");

  if (bias != nullptr) {
    ORT_RETURN_IF_NOT(bias->Shape().NumDimensions() == 1, "bias must be 1D");
    ORT_RETURN_IF_NOT(bias->Shape()[0] == channels, "bias length must equal number of channels");
  }

  if (conv_state != nullptr) {
    const auto& state_shape = conv_state->Shape();
    ORT_RETURN_IF_NOT(state_shape.NumDimensions() == 3, "conv_state must be 3D (B,D,K-1)");
    ORT_RETURN_IF_NOT(state_shape[0] == batch_size, "conv_state batch size must match input");
    ORT_RETURN_IF_NOT(state_shape[1] == channels, "conv_state channels must match input");
    ORT_RETURN_IF_NOT(state_shape[2] == kernel_size - 1, "conv_state dim 2 must be K-1");
  }

  // Determine activation
  CausalConv1DActivation act = CausalConv1DActivation::kSiLU;
  if (activation_ == "none") {
    act = CausalConv1DActivation::kNone;
  } else if (activation_ == "silu" || activation_ == "swish") {
    act = CausalConv1DActivation::kSiLU;
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Unknown activation: ", activation_);
  }

  // Allocate outputs
  TensorShape output_shape({batch_size, channels, seq_len});
  Tensor* output = context->Output(0, output_shape);

  TensorShape present_state_shape({batch_size, channels, kernel_size - 1});
  Tensor* present_state = context->Output(1, present_state_shape);

  typedef typename ToCudaType<T>::MappedType CudaT;

  return LaunchCausalConv1DWithStateKernel<CudaT>(
      Stream(context),
      reinterpret_cast<const CudaT*>(input->Data<T>()),
      reinterpret_cast<const CudaT*>(weight->Data<T>()),
      bias != nullptr ? reinterpret_cast<const CudaT*>(bias->Data<T>()) : nullptr,
      conv_state != nullptr ? reinterpret_cast<const CudaT*>(conv_state->Data<T>()) : nullptr,
      reinterpret_cast<CudaT*>(output->MutableData<T>()),
      reinterpret_cast<CudaT*>(present_state->MutableData<T>()),
      act,
      batch_size,
      channels,
      seq_len,
      kernel_size);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
