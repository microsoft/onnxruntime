// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/cuda/cudnn_common.h"
#include "core/framework/tensorprotoutils.h"
#include "add_bias_gelu.h"
#include "add_bias_gelu_impl.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      AddBiasGelu,                                                \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      AddBiasGelu<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

using namespace ONNX_NAMESPACE;

template <typename T>
AddBiasGelu<T>::AddBiasGelu(const OpKernelInfo& op_kernel_info) : CudaKernel(op_kernel_info) {
}

template <typename T>
Status AddBiasGelu<T>::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* input = ctx->Input<Tensor>(0);
  const Tensor* bias = ctx->Input<Tensor>(1);
  Tensor* output = ctx->Output(0, input->Shape());

  const auto input_dims = input->Shape().GetDims();
  if (input_dims.size() != 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 0 is expected to have 3 dimensions, got ", input_dims.size());
  }

  const auto bias_dims = bias->Shape().GetDims();
  if (bias_dims.size() != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 1 is expected to have 1 dimensions, got ", bias_dims.size());
  }
  if (bias_dims[0] != input_dims[2]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 1 dimension 0 should have same length as dimension 1 of input 0");
  }

  int m = static_cast<int>(input_dims[0] * input_dims[1]);
  int n = static_cast<int>(input_dims[2]);
  
  if (n > 4 * 1024) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Dimension 1 of Input 0 is expected to be no more than 4048, got ", n);
  }
  
  if (m % 4 != 0 || n % 4 != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Dimensions 0 and 1 of Input 0 is expected to be divisible by 4");
  }

  bool is_ok = false;
  if (sizeof(T) == 4) {
    is_ok = LaunchAddBiasGeluKernel<T>(
        input->template Data<T>(),
        bias->template Data<T>(),
        output->template MutableData<T>(),
        m, n);
  } else {
    is_ok = LaunchAddBiasGeluKernel<half>(
        reinterpret_cast<const half*>(input->template Data<T>()),
        reinterpret_cast<const half*>(bias->template Data<T>()),
        reinterpret_cast<half*>(output->template MutableData<T>()),
        m, n);
  }

  if (!is_ok) {
    CUDA_CALL(cudaGetLastError());
    return Status(common::ONNXRUNTIME, common::FAIL);
  }

  return Status::OK();
}

}  //namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
