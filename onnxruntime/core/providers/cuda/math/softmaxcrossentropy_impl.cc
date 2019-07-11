// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "softmaxcrossentropy_impl.h"

namespace onnxruntime {
namespace cuda {
#define REGISTER_KERNEL_TYPED(Class, T, version)                                \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                \
      Class,                                                                    \
      kOnnxDomain,                                                                \
      version,                                                                  \
      T,                                                                        \
      kCudaExecutionProvider,                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Class<T>);

#define REGISTER_KERNEL_TYPED_TWO_TYPES(Class, T, Tin, version)                     \
  ONNX_OPERATOR_TWO_TYPED_KERNEL_EX(                                                \
      Class,                                                                        \
      kOnnxDomain,                                                                    \
      version,                                                                      \
      T, Tin,                                                                       \
      kCudaExecutionProvider,                                                       \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()).     \
                         TypeConstraint("Tin", DataTypeImpl::GetTensorType<Tin>()), \
      Class<T, Tin>);

template <typename T>
Status SoftmaxCrossEntropy<T>::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor& logit = *ctx->Input<Tensor>(0);
  const Tensor& label = *ctx->Input<Tensor>(1);

  const TensorShape logit_shape{logit.Shape()};
  const TensorShape label_shape{label.Shape()};

  ORT_ENFORCE(label_shape == logit_shape, "The shape in logits and labels is not identical");

  int64_t N = logit_shape.SizeToDimension(logit_shape.NumDimensions() - 1);
  int64_t D = logit_shape[logit_shape.NumDimensions() - 1];
  const TensorShape logit_reshape({N, D});

  IAllocatorUniquePtr<T> temp_X = GetScratchBuffer<T>(N * D);

  // calculate softmax
  ORT_RETURN_IF_ERROR(SoftMaxComputeHelper<T>(logit.template Data<T>(),
                                              logit_reshape,
                                              temp_X.get(),
                                              CudnnHandle(),
                                              1 /*axis default*/));

  // calculate  (label - log(softmax)) for each element
  SoftMaxCrossEntropyImpl(
      temp_X.get(),              // softmax
      label.template Data<T>(),  // label
      temp_X.get(),              // -(label * log(softmax))
      N * D);

  std::vector<int64_t> output_dims(2, 1);
  Tensor* Y = ctx->Output(0, TensorShape({}));
  // Sum((label - log(softmax)) using Reduction
  ReduceKernelShared<T, T, CUDNN_REDUCE_TENSOR_NO_INDICES>(
      temp_X.get(),
      logit_reshape,
      Y->template MutableData<T>(),
      TensorShape({}),
      CUDNN_REDUCE_TENSOR_ADD,
      output_dims);

  return Status::OK();
}

template <typename T>
Status SoftmaxCrossEntropyGrad<T>::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor& dY = *ctx->Input<Tensor>(0);
  const Tensor& logit = *ctx->Input<Tensor>(1);
  const Tensor& label = *ctx->Input<Tensor>(2);

  const TensorShape logit_shape{logit.Shape()};
  const TensorShape label_shape{label.Shape()};
  ORT_ENFORCE(label_shape == logit_shape, "The shape in logits and label is not identical");

  int64_t N = logit_shape.SizeToDimension(logit_shape.NumDimensions() - 1);
  int64_t D = logit_shape[logit_shape.NumDimensions() - 1];
  const TensorShape logit_reshape({N, D});

  Tensor* d_logits = ctx->Output(0, logit_shape);

  const T* logit_data = logit.template Data<T>();
  const T* label_data = label.template Data<T>();
  const T* dY_data = dY.template Data<T>();
  T* d_logits_data = d_logits->template MutableData<T>();

  ORT_RETURN_IF_ERROR(SoftMaxComputeHelper<T>(
      logit_data,
      logit_reshape,
      d_logits_data,
      CudnnHandle(),
      1 /*axis default*/));

  SoftMaxCrossEntropyGradImpl(
      dY_data,        // Dy
      d_logits_data,  // pi
      label_data,     // Label
      d_logits_data,  // gradient
      N * D);

  return Status::OK();
}

template <typename T, typename Tin>
Status SparseSoftmaxCrossEntropy<T, Tin>::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor& logit = *ctx->Input<Tensor>(0);
  const Tensor& label = *ctx->Input<Tensor>(1);

  const TensorShape logit_shape{logit.Shape()};
  const TensorShape label_shape{label.Shape()};
  ORT_ENFORCE(logit_shape.NumDimensions() == label_shape.NumDimensions() + 1,
              "logits_shape must be (1 + label_shape)");
  for (size_t i = 0; i < label_shape.NumDimensions(); i++) {
    ORT_ENFORCE(label_shape[i] == logit_shape[i], "The shape in logits and labels does not match");
  }

  int64_t N = label_shape.Size();
  int64_t D = logit_shape[logit_shape.NumDimensions() - 1];
  const TensorShape logit_reshape({N, D});
  const TensorShape label_reshape({N});

  IAllocatorUniquePtr<T> tmp_probability = GetScratchBuffer<T>(N * D);
  IAllocatorUniquePtr<T> tmp_loss_sample = GetScratchBuffer<T>(N);
  Tensor* total_loss = ctx->Output(0, TensorShape({}));

  const T* logit_data = logit.template Data<T>();
  const Tin* label_data = label.template Data<Tin>();
  T* total_loss_data = total_loss->template MutableData<T>();

  // calculate softmax
  ORT_RETURN_IF_ERROR(SoftMaxComputeHelper<T>(logit_data,
                                              logit_reshape,
                                              tmp_probability.get(),
                                              CudnnHandle(),
                                              1 /*axis default*/));

  // calculate  (label - log(softmax)) for each sample
  const T* weight_data = nullptr;
  if (OpKernel::Node().InputDefs().size() == 3) {
    const Tensor& weight = *ctx->Input<Tensor>(2);
    const TensorShape weight_shape{weight.Shape()};
    ORT_ENFORCE(weight_shape == label_shape, "The shape in weights and labels is different");
    weight_data = weight.template Data<T>();
  }
  SparseSoftmaxCrossEntropyImpl(tmp_probability.get(), label_data, weight_data, tmp_loss_sample.get(), N, D);

  // ReduceSum on loss_per_sample
  std::vector<int64_t> output_dims(1, 1);
  ReduceKernelShared<T, T, CUDNN_REDUCE_TENSOR_NO_INDICES>(
      tmp_loss_sample.get(),
      label_reshape,
      total_loss_data,
      TensorShape({}),
      CUDNN_REDUCE_TENSOR_ADD,
      output_dims);

  return Status::OK();
}

template <typename T, typename Tin>
Status SparseSoftmaxCrossEntropyGrad<T, Tin>::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor& dY = *ctx->Input<Tensor>(0);
  const Tensor& logit = *ctx->Input<Tensor>(1);
  const Tensor& label = *ctx->Input<Tensor>(2);

  const TensorShape logit_shape{logit.Shape()};
  const TensorShape label_shape{label.Shape()};
  ORT_ENFORCE(logit_shape.NumDimensions() == label_shape.NumDimensions() + 1,
              "logits_shape must be (1 + label_shape)");
  for (size_t i = 0; i < label_shape.NumDimensions(); i++) {
    ORT_ENFORCE(label_shape[i] == logit_shape[i], "The shape in logits and labels does not match");
  }

  int64_t N = label_shape.Size();
  int64_t D = logit_shape[logit_shape.NumDimensions() - 1];
  const TensorShape logit_reshape({N, D});

  Tensor* d_logit = ctx->Output(0, logit_shape);

  const T* dY_data = dY.template Data<T>();
  const T* logit_data = logit.template Data<T>();
  const Tin* label_data = label.template Data<Tin>();
  T* d_logit_data = d_logit->template MutableData<T>();

  ORT_RETURN_IF_ERROR(SoftMaxComputeHelper<T>(
      logit_data,
      logit_reshape,
      d_logit_data,
      CudnnHandle(),
      1 /*axis default*/));

  const T* weight_data = nullptr;
  if (OpKernel::Node().InputDefs().size() == 4) {
    const Tensor& weight = *ctx->Input<Tensor>(3);
    const TensorShape weight_shape{weight.Shape()};
    ORT_ENFORCE(weight_shape == label_shape, "The shape in weights and labels is different");
    weight_data = weight.template Data<T>();
  }
  SparseSoftmaxCrossEntropyGradImpl(dY_data, d_logit_data, label_data, weight_data, d_logit_data, N, D);

  return Status::OK();
}

#define SPECIALIZED_COMPUTE(Class, T, version) \
  REGISTER_KERNEL_TYPED(Class, T, version)     \
  template Status Class<T>::ComputeInternal(OpKernelContext* ctx) const;

SPECIALIZED_COMPUTE(SoftmaxCrossEntropy, float, 9)
SPECIALIZED_COMPUTE(SoftmaxCrossEntropyGrad, float, 9)

#define SPECIALIZED_COMPUTE_SPARSE(Class, T, Tin, version) \
  REGISTER_KERNEL_TYPED_TWO_TYPES(Class, T, Tin, version)  \
  template Status Class<T, Tin>::ComputeInternal(OpKernelContext* ctx) const;

// SPECIALIZED_COMPUTE_SPARSE(SparseSoftmaxCrossEntropy, float, int32_t, 9)
SPECIALIZED_COMPUTE_SPARSE(SparseSoftmaxCrossEntropy, float, int64_t, 9)
// SPECIALIZED_COMPUTE_SPARSE(SparseSoftmaxCrossEntropyGrad, float, int32_t, 9)
SPECIALIZED_COMPUTE_SPARSE(SparseSoftmaxCrossEntropyGrad, float, int64_t, 9)

}  // namespace cuda
}  // namespace onnxruntime
