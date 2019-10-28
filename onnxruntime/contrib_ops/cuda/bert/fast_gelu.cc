// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/cuda/cudnn_common.h"
#include "core/framework/tensorprotoutils.h"
#include "fast_gelu.h"
#include "fast_gelu_impl.h"

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
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      FastGelu<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

using namespace ONNX_NAMESPACE;

template <typename T>
FastGelu<T>::FastGelu(const OpKernelInfo& op_kernel_info) : CudaKernel(op_kernel_info) {
}

template <typename T>
Status FastGelu<T>::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* input = ctx->Input<Tensor>(0);
  Tensor* output = ctx->Output(0, input->Shape());

  int element_count = 1;
  const auto input_dims = input->Shape().GetDims();
  for (size_t i = 0; i < input_dims.size(); i++) {
    element_count *= static_cast<int>(input_dims[i]);
  }
  
  bool is_ok = false;
  if (sizeof(T) == 4) {
    is_ok = computeGelu<T>(nullptr, element_count, input->template Data<T>(), output->template MutableData<T>());
  } else {
    is_ok = computeGelu<half>(nullptr, element_count, reinterpret_cast<const half*>(input->template Data<T>()), reinterpret_cast<half*>(output->template MutableData<T>()));
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
