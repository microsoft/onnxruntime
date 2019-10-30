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

  const auto input_dims = input->Shape().GetDims();
  if (input_dims.size() != 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 0 is expected to have 3 dimensions, got ", input_dims.size());
  }

  size_t num_inputs = OpKernel::Node().InputDefs().size();
  bool has_bias = (num_inputs == 2);

  const Tensor* bias = nullptr;
  if (has_bias) {
    bias = ctx->Input<Tensor>(1);
    const auto bias_dims = bias->Shape().GetDims();
    if (bias_dims.size() != 1) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 1 is expected to have 1 dimensions, got ", bias_dims.size());
    }
    if (bias_dims[0] != input_dims[2]) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 1 dimension 0 should have same length as dimension 1 of input 0");
    }
  }

  int m = static_cast<int>(input_dims[0] * input_dims[1]);
  int n = static_cast<int>(input_dims[2]);

  bool is_ok = false;
  if (sizeof(T) == 4) {
    is_ok = computeGelu<T>(nullptr, m, n, input->template Data<T>(), has_bias ? bias->template Data<T>() : nullptr, output->template MutableData<T>());
  } else {
    is_ok = computeGelu<half>(nullptr, m, n, reinterpret_cast<const half*>(input->template Data<T>()), has_bias ? reinterpret_cast<const half*>(bias->template Data<T>()): nullptr,  reinterpret_cast<half*>(output->template MutableData<T>()));
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
