// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/causal_conv_with_state.h"
#include "contrib_ops/cuda/bert/causal_conv_with_state_impl.h"
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

template <typename T>
CausalConvWithState<T>::CausalConvWithState(const OpKernelInfo& info) : CudaKernel(info) {
  int64_t ndim = info.GetAttrOrDefault<int64_t>("ndim", 1);
  ORT_ENFORCE(ndim == 1, "CUDA CausalConvWithState only supports ndim=1");
  ndim_ = static_cast<int>(ndim);

  activation_ = info.GetAttrOrDefault<std::string>("activation", "none");
  ORT_ENFORCE(activation_ == "none" || activation_ == "silu" || activation_ == "swish",
              "activation must be one of: none, silu, swish");
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
  ORT_RETURN_IF_NOT(input_shape.NumDimensions() == 3,
                    "input must be rank 3 (batch, channels, length), got rank ", input_shape.NumDimensions());
  ORT_RETURN_IF_NOT(weight_shape.NumDimensions() == 3,
                    "weight must be rank 3 (channels, 1, kernel_size), got rank ", weight_shape.NumDimensions());

  const int batch_size = static_cast<int>(input_shape[0]);
  const int channels = static_cast<int>(input_shape[1]);
  const int L = static_cast<int>(input_shape[2]);
  const int K = static_cast<int>(weight_shape[2]);
  const int pad = K - 1;

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

  // Validate optional past_state shape
  if (past_state_tensor != nullptr) {
    const auto& past_shape = past_state_tensor->Shape();
    ORT_RETURN_IF_NOT(past_shape.NumDimensions() == 3,
                      "past_state must be rank 3 (batch, channels, kernel_size-1), got rank ", past_shape.NumDimensions());
    ORT_RETURN_IF_NOT(past_shape[0] == batch_size && past_shape[1] == channels && past_shape[2] == pad,
                      "past_state shape mismatch: expected (", batch_size, ", ", channels, ", ", pad,
                      "), got (", past_shape[0], ", ", past_shape[1], ", ", past_shape[2], ")");
  }

  // Allocate outputs
  Tensor* output_tensor = context->Output(0, input_shape);
  TensorShape state_shape({batch_size, channels, pad});
  Tensor* present_state_tensor = context->Output(1, state_shape);

  // Note: no need to zero-initialize present_state — the kernel writes all
  // positions unconditionally.  When past_state is null, the kernel uses
  // zeros for the padding region internally.
  // Note: past_state pointer is passed to kernel; kernel reads it directly

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
      apply_silu,
      GetDeviceProp().maxThreadsPerBlock);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
