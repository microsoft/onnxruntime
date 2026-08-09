// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/dsv4_swiglu.h"

#include "contrib_ops/cuda/math/dsv4_swiglu_impl.h"
#include "core/providers/cuda/cuda_common.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      DSV4SwiGLU,                                                 \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      DSV4SwiGLU<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

template <typename T>
DSV4SwiGLU<T>::DSV4SwiGLU(const OpKernelInfo& info) : CudaKernel(info) {
  limit_ = info.GetAttrOrDefault<float>("limit", 0.0f);
}

template <typename T>
Status DSV4SwiGLU<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* gate = context->Input<Tensor>(0);
  const Tensor* up = context->Input<Tensor>(1);

  if (up != nullptr) {
    if (gate->Shape() != up->Shape()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "gate is ", gate->Shape(),
                             " but up is ", up->Shape(), "; they must match.");
    }
    Tensor* output = context->Output(0, gate->Shape());
    const int64_t count = gate->Shape().Size();
    if (count == 0) return Status::OK();
    return LaunchDSV4SwiGLU<T>(Stream(context), count, 0, limit_, gate->Data<T>(), up->Data<T>(),
                               output->MutableData<T>());
  }

  // One projection produced both halves; slice it here rather than paying for a Split node.
  const TensorShape& in_shape = gate->Shape();
  const size_t rank = in_shape.NumDimensions();
  if (rank == 0 || in_shape[rank - 1] % 2 != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "with `up` omitted, gate is ", in_shape,
                           "; its last dimension must be a positive even number.");
  }
  const int64_t cols = in_shape[rank - 1] / 2;

  TensorShapeVector dims = in_shape.AsShapeVector();
  dims[rank - 1] = cols;
  Tensor* output = context->Output(0, TensorShape(dims));
  const int64_t count = output->Shape().Size();
  if (count == 0) return Status::OK();

  const T* data = gate->Data<T>();
  return LaunchDSV4SwiGLU<T>(Stream(context), count, cols, limit_, data, data + cols,
                             output->MutableData<T>());
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
