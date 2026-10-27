// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/swiglu.h"

#include "contrib_ops/cuda/math/swiglu_impl.h"
#include "core/providers/cuda/cuda_common.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      SwiGLU,                                                     \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      SwiGLU<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

template <typename T>
SwiGLU<T>::SwiGLU(const OpKernelInfo& info) : CudaKernel(info) {
  limit_ = info.GetAttrOrDefault<float>("limit", 0.0f);
  alpha_ = info.GetAttrOrDefault<float>("activation_alpha", 1.0f);
  beta_ = info.GetAttrOrDefault<float>("activation_beta", 0.0f);
  round_activation_ = info.GetAttrOrDefault<int64_t>("round_activation", 0) != 0;
}

template <typename T>
Status SwiGLU<T>::ComputeInternal(OpKernelContext* context) const {
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
    return LaunchSwiGLU<T>(Stream(context), count, 0, limit_, alpha_, beta_, round_activation_, gate->Data<T>(),
                           up->Data<T>(), output->MutableData<T>());
  }

  // One projection produced both halves; slice it here rather than paying for a Split node.
  const TensorShape& in_shape = gate->Shape();
  const size_t rank = in_shape.NumDimensions();
  if (rank == 0 || in_shape[rank - 1] % 2 != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "with `up` omitted, gate is ", in_shape,
                           "; it must have a last dimension, and that dimension must be even.");
  }
  const int64_t cols = in_shape[rank - 1] / 2;

  TensorShapeVector dims = in_shape.AsShapeVector();
  dims[rank - 1] = cols;
  Tensor* output = context->Output(0, TensorShape(dims));
  const int64_t count = output->Shape().Size();
  if (count == 0) return Status::OK();

  const T* data = gate->Data<T>();
  return LaunchSwiGLU<T>(Stream(context), count, cols, limit_, alpha_, beta_, round_activation_, data, data + cols,
                         output->MutableData<T>());
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
