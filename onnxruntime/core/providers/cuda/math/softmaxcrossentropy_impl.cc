// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "softmaxcrossentropy_impl.h"

namespace onnxruntime {
namespace cuda {
#define REGISTER_KERNEL_TYPED(Class, T, version)                                    \
  ONNX_OPERATOR_KERNEL_EX(                                                          \
      Class,                                                                        \
      kMSDomain,                                                                    \
      version,                                                                      \
      kCudaExecutionProvider,                                                       \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Class<T>);

template <typename T>
Status SoftmaxCrossEntropyGrad<T>::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor& dY = *ctx->Input<Tensor>(0);
  const Tensor& logits = *ctx->Input<Tensor>(1);
  const Tensor& lable = *ctx->Input<Tensor>(2);

  const TensorShape logits_shape{logits.Shape()};
  const TensorShape label_shape{lable.Shape()};

  ORT_ENFORCE(logits_shape.NumDimensions() == 2, "logits must be 2-dimensional");
  ORT_ENFORCE(label_shape == logits_shape, "The shape in logits and lable is not identical");

  Tensor* d_logits = ctx->Output(0, logits_shape);

  const T* logits_data = logits.template Data<T>();
  const T* labels_data = lable.template Data<T>();
  const T* dY_data = dY.template Data<T>();
  T* d_logits_data = d_logits->template MutableData<T>();

  ORT_RETURN_IF_ERROR(SoftMaxComputeHelper<T>(
      logits_data,
      logits_shape,
      d_logits_data,
      CudnnHandle(),
      1 /*axis default*/));

  SoftMaxCrossEntropyGradImpl(
      dY_data,        // Dy
      d_logits_data,  // pi
      labels_data,    // Label
      d_logits_data,  // gradient
      logits_shape.Size());

  return Status::OK();
}

template <typename T>
Status SoftmaxCrossEntropy<T>::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* X = ctx->Input<Tensor>(0);
  const TensorShape input_shape{X->Shape()};

  IAllocatorUniquePtr<T> temp_X = GetScratchBuffer<T>(/*input_count*/ input_shape.Size());

  // calculate softmax
  ORT_RETURN_IF_ERROR(SoftMaxComputeHelper<T>(X->template Data<T>(),
                                              input_shape,
                                              temp_X.get(),
                                              CudnnHandle(),
                                              1 /*axis default*/));

  // calculate  (label - log(softmax)) for each element
  SoftMaxCrossEntropyImpl(
      temp_X.get(),                               // softmax
      ctx->Input<Tensor>(1)->template Data<T>(),  // label
      temp_X.get(),                               // -(label * log(softmax))
      X->Shape().Size());

  // reduce to sum of elements in tensor
  const auto rank = input_shape.NumDimensions();

  if (rank > 8) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "cuDNN only supports up to 8-D tensors in reduction");
  }

  std::vector<int64_t> output_dims(rank, 0);
  for (int i = 0; i < rank; i++) {
    output_dims[i] = 1;
  }

  Tensor* Y = ctx->Output(0, TensorShape({}));
  //sum((label - log(softmax)) using Reduction
  ReduceKernelShared<T, T, CUDNN_REDUCE_TENSOR_NO_INDICES>(
      temp_X.get(),
      input_shape,
      Y->template MutableData<T>(),
      TensorShape({}),
      CUDNN_REDUCE_TENSOR_ADD,
      output_dims);

  return Status::OK();
}

#define SPECIALIZED_COMPUTE(Class, T, version) \
  REGISTER_KERNEL_TYPED(Class, T, version)     \
  template Status Class<T>::ComputeInternal(OpKernelContext* ctx) const;

SPECIALIZED_COMPUTE(SoftmaxCrossEntropy, float, 1)
SPECIALIZED_COMPUTE(SoftmaxCrossEntropyGrad, float, 1)


}  // namespace cuda
}  // namespace onnxruntime
