// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

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
  const int64_t N = X_shape[0];
  const int64_t C = X_shape[1];
  const int64_t H = X_shape[2];
  const int64_t W = X_shape[3];
  Tensor* Y = context->Output(0, {N, C, H, W});
  MLAS_CONV_PARAMETERS params = { };
  params.InputSize = H * W;
  // BUGBUG: ignores N...
  MlasConvReorderInput(&params, X->template Data<T>(), Y->template MutableData<T>(), static_cast<size_t>(C));
  return Status::OK();
}

template <typename T>
Status ReorderOutput<T>::Compute(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& X_shape = X->Shape();
  const int64_t N = X_shape[0];
  const int64_t C = X_shape[1];
  const int64_t H = X_shape[2];
  const int64_t W = X_shape[3];
  Tensor* Y = context->Output(0, {N, C, H, W});
  MLAS_CONV_PARAMETERS params = { };
  params.OutputSize = H * W;
  // BUGBUG: ignores N...
  MlasConvReorderOutput(&params, X->template Data<T>(), Y->template MutableData<T>(), static_cast<size_t>(C));
  return Status::OK();
}

template <typename T>
Status NchwcConv<T>::Compute(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  const Tensor* W = context->Input<Tensor>(1);
  const Tensor* B = context->Input<Tensor>(2);
  const Tensor* Sum = context->Input<Tensor>(3);
  const int64_t N = X->Shape()[0];
  const int64_t C = X->Shape()[1];
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
  Tensor* Y = context->Output(0, TensorShape(Y_dims));
  TensorShape output_shape = Y->Shape().Slice(2);

  const float* Xdata = X->template Data<float>();
  float* Ydata = Y->template MutableData<float>();

  const size_t kernel_rank = kernel_shape.size();

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

  MLAS_CONV_PARAMETERS Parameters;
  size_t WorkingBufferSize;
  MlasConvPrepare(&Parameters,
                  kernel_rank,
                  static_cast<size_t>(N),
                  static_cast<size_t>(group_),
                  static_cast<size_t>(C / group_),
                  input_shape.GetDims().data(),
                  kernel_shape.data(),
                  dilations.data(),
                  pads.data(),
                  strides.data(),
                  output_shape.GetDims().data(),
                  static_cast<size_t>(M / group_),
                  &Activation,
                  &WorkingBufferSize);

  MlasConvNchwc(&Parameters,
                Xdata,
                W->template Data<float>(),
                B != nullptr ? B->template Data<float>() : nullptr,
                Ydata,
                Sum == nullptr);

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
