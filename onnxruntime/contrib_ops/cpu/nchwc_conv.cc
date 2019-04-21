// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/op_kernel_context_internal.h"
#include "nchwc_conv.h"
#include "core/mlas/inc/mlas.h"

namespace onnxruntime {
namespace contrib {

ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    ReorderInput,
    1,
    float,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    ReorderInput<float>);

ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    ReorderOutput,
    1,
    float,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    ReorderOutput<float>);

ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    NchwcConv,
    1,
    float,
    KernelDefBuilder()
        .MayInplace(3, 0)
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    NchwcConv<float>);

template <typename T>
Status ReorderInput<T>::Compute(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& X_shape = X->Shape();
  Tensor* Y = context->Output(0, X_shape);
  MlasReorderInput(X_shape.GetDims().data(), X->template Data<T>(), Y->template MutableData<T>());
  return Status::OK();
}

template <typename T>
Status ReorderOutput<T>::Compute(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& X_shape = X->Shape();
  Tensor* Y = context->Output(0, X_shape);
  MlasReorderOutput(X_shape.GetDims().data(), X->template Data<T>(), Y->template MutableData<T>());
  return Status::OK();
}

template <typename T>
Status NchwcConv<T>::Compute(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  const Tensor* W = context->Input<Tensor>(1);
  const Tensor* B = context->Input<Tensor>(2);
  const Tensor* Sum = context->Input<Tensor>(3);
  const int64_t N = X->Shape()[0];
  const int64_t M = W->Shape()[0];
  ORT_RETURN_IF_ERROR(ValidateInputShape(X, W));

  std::vector<int64_t> kernel_shape;
  ORT_RETURN_IF_ERROR(ComputeKernelShape(W->Shape(), kernel_shape));

  std::vector<int64_t> pads(pads_);
  if (pads.empty()) {
    pads.resize(kernel_shape.size() * 2, 0);
  }
  std::vector<int64_t> dilations(dilations_);
  if (dilations.empty()) {
    dilations.resize(kernel_shape.size(), 1);
  }
  std::vector<int64_t> strides(strides_);
  if (strides.empty()) {
    strides.resize(kernel_shape.size(), 1);
  }

  std::vector<int64_t> Y_dims;
  Y_dims.insert(Y_dims.begin(), {N, M});
  TensorShape input_shape = X->Shape().Slice(2);
  ORT_RETURN_IF_ERROR(InferOutputShape(input_shape, kernel_shape, strides, dilations, &pads, &Y_dims));
  Tensor* Y = context->Output(0, Y_dims);

  MLAS_ACTIVATION Activation;
  if (activation_.empty()) {
    Activation.ActivationKind = MlasIdentityActivation;
  } else if (activation_ == "Relu") {
    Activation.ActivationKind = MlasReluActivation;
  } else if (activation_ == "LeakyRelu") {
    Activation.ActivationKind = MlasLeakyReluActivation;
    Activation.alpha = alpha_;
  } else if (activation_ == "Tanh") {
    Activation.ActivationKind = MlasTanhActivation;
  } else if (activation_ == "Sigmoid") {
    Activation.ActivationKind = MlasLogisticActivation;
  } else {
    ORT_NOT_IMPLEMENTED("Not implemented fused activation: ", activation_);
  }

  MlasConvNchwc(kernel_shape.size(),
                X->Shape().GetDims().data(),
                kernel_shape.data(),
                dilations.data(),
                pads.data(),
                strides.data(),
                Y_dims.data(),
                static_cast<size_t>(group_),
                X->template Data<float>(),
                W->template Data<float>(),
                B != nullptr ? B->template Data<float>() : nullptr,
                Y->template MutableData<float>(),
                &Activation,
                Sum == nullptr,
                const_cast<concurrency::ThreadPool*>(static_cast<OpKernelContextInternal*>(context)->GetOperatorThreadPool()));

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
