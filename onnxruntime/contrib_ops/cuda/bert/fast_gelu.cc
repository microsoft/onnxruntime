// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cudnn_common.h"
#include "fast_gelu.h"
#include "core/providers/cuda/tensor/gelu_impl.h"
#include "contrib_ops/cpu/bert/bias_gelu_helper.h"
#include "contrib_ops/cuda/bert/transformer_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      FastGelu,                                                   \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      FastGelu<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)
REGISTER_KERNEL_TYPED(double)

using namespace ONNX_NAMESPACE;

#ifdef BUILD_CUDA_EP_AS_PLUGIN
// PLUGIN BUILD ADAPTATION: bias_gelu_helper::CheckInputs lives in the CPU
// provider and cannot be linked into the plugin. Reimplement the same input
// validation (rank checks, bias shape matching) inline.
// Keep in sync with contrib_ops/cpu/bert/bias_gelu_helper.h.
static Status CheckInputsForPlugin(const OpKernelContext* context) {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* bias = context->Input<Tensor>(1);

  const auto& input_dims = input->Shape().GetDims();
  if (input_dims.size() < 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 0 is expected to have 1 or more dimensions, got ", input_dims.size());
  }

  if (nullptr != bias) {
    const auto& bias_dims = bias->Shape().GetDims();
    if (bias_dims.size() != 1) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 1 is expected to have 1 dimensions, got ", bias_dims.size());
    }
    if (bias_dims[0] != input_dims[input_dims.size() - 1]) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 1 dimension 0 should have same length as the last dimension of input 0");
    }
  }

  return Status::OK();
}
#endif

template <typename T>
FastGelu<T>::FastGelu(const OpKernelInfo& op_kernel_info) : CudaKernel(op_kernel_info) {
  const TransformerOptions* options = TransformerOptions::GetInstance();
  use_half2_ = !options->DisableHalf2();
}

template <typename T>
Status FastGelu<T>::ComputeInternal(OpKernelContext* context) const {
#ifdef BUILD_CUDA_EP_AS_PLUGIN
  ORT_RETURN_IF_ERROR(CheckInputsForPlugin(context));
#else
  ORT_RETURN_IF_ERROR(bias_gelu_helper::CheckInputs(context));
#endif

  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* bias = context->Input<Tensor>(1);
  Tensor* output = context->Output(0, input->Shape());

  int64_t input_length = input->Shape().Size();
  if (input_length == 0) {
    return Status::OK();
  }
  int64_t bias_length = (nullptr == bias) ? 0 : bias->Shape().Size();
  typedef typename ToCudaType<T>::MappedType CudaT;

  return LaunchFastGeluKernel<CudaT>(GetDeviceProp(),
                                     Stream(context),
                                     static_cast<int>(input_length),
                                     static_cast<int>(bias_length),
                                     reinterpret_cast<const CudaT*>(input->Data<T>()),
                                     (nullptr != bias) ? reinterpret_cast<const CudaT*>(bias->Data<T>()) : nullptr,
                                     reinterpret_cast<CudaT*>(output->MutableData<T>()),
                                     use_half2_);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
