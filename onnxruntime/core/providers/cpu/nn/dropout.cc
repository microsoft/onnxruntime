// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/nn/dropout.h"

namespace onnxruntime {

#ifdef ENABLE_TRAINING
ONNX_CPU_OPERATOR_KERNEL(
    Dropout,
    7,
    KernelDefBuilder().TypeConstraint("T", {DataTypeImpl::GetTensorType<MLFloat16>(), DataTypeImpl::GetTensorType<float>(), DataTypeImpl::GetTensorType<double>()}),
    Dropout);
#else
ONNX_CPU_OPERATOR_KERNEL(
    Dropout,
    7,
    KernelDefBuilder().TypeConstraint("T", {DataTypeImpl::GetTensorType<MLFloat16>(), DataTypeImpl::GetTensorType<float>(), DataTypeImpl::GetTensorType<double>()}),
    IdentityOp<true>);
#endif

Status Dropout::Compute(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& shape = X->Shape();
  Tensor* Y = context->Output(0, shape);
  const float* X_data = X->template Data<float>();
  float* Y_data = Y->template MutableData<float>();

  if (is_test_) {
    //If source and target pointers are not equal, we need to copy the data.
    if (Y_data != X_data) {
      memcpy(Y_data, X_data, shape.Size() * sizeof(float));
    }
  } else {
    float scale = 1.0f / keep_prob_;
    Tensor* mask = context->Output(1, shape);
    bool* mask_data = mask->template MutableData<bool>();

    // TODO: Compute is a const function, generator cannot be a private meber
    float seed = gsl::narrow_cast<float>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    std::default_random_engine generator{gsl::narrow_cast<uint32_t>(seed)};
    std::bernoulli_distribution distribution(keep_prob_);

    for (int i = 0; i < shape.Size(); ++i) {
      mask_data[i] = distribution(generator);
      Y_data[i] = X_data[i] * scale * mask_data[i];
    }
  }

  return Status::OK();
}

ONNX_CPU_OPERATOR_KERNEL(
    DrouputGrad,
    9,
    KernelDefBuilder().TypeConstraint("T", {DataTypeImpl::GetTensorType<MLFloat16>(), DataTypeImpl::GetTensorType<float>(), DataTypeImpl::GetTensorType<double>()}),
    DrouputGrad);

Status DrouputGrad::Compute(OpKernelContext* context) const {
  const Tensor* dY = context->Input<Tensor>(0);
  const TensorShape& shape = dY->Shape();
  Tensor* dX = context->Output(0, shape);

  const float* dY_data = dY->template Data<float>();
  float* dX_data = dX->template MutableData<float>();

  if (is_test_) {
    //If source and target pointers are not equal, we need to copy the data.
    if (dY_data != dX_data) {
      memcpy(dX_data, dY_data, shape.Size() * sizeof(float));
    }
  } else {
    const Tensor* mask = context->Input<Tensor>(1);
    const bool* mask_data = mask->template Data<bool>();

    float scale = 1.0f / keep_prob_;

    for (int i = 0; i < shape.Size(); ++i) {
      dX_data[i] = dY_data[i] * scale * mask_data[i];
    }
  }

  return Status::OK();
}
}  // namespace onnxruntime
