// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/ml/svmregressor.h"

namespace onnxruntime {
namespace ml {

ONNX_CPU_OPERATOR_ML_KERNEL(
    SVMRegressor,
    1,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    SVMRegressor<float>);

template <typename T>
SVMRegressor<T>::SVMRegressor(const OpKernelInfo& info)
    : OpKernel(info),
      SVMCommon<T>(info),
      vector_count_(info.GetAttrOrDefault<int64_t>("n_supports", 0)),
      support_vectors_(info.GetAttrsOrDefault<float>("support_vectors")),
      post_transform_(MakeTransform(info.GetAttrOrDefault<std::string>("post_transform", "NONE"))) {
  ORT_ENFORCE(info.GetAttrs<float>("rho", rho_).IsOK());
  ORT_ENFORCE(info.GetAttrs<float>("coefficients", coefficients_).IsOK());
  ORT_ENFORCE(!coefficients_.empty());

  int64_t onec = info.GetAttrOrDefault<int64_t>("one_class", 0);
  one_class_ = (onec != 0);

  if (vector_count_ > 0) {
    feature_count_ = support_vectors_.size() / vector_count_;  //length of each support vector
    mode_ = SVM_TYPE::SVM_SVC;
  } else {
    feature_count_ = coefficients_.size();
    mode_ = SVM_TYPE::SVM_LINEAR;
    set_kernel_type(KERNEL::LINEAR);
  }
}

template <typename T>
Status SVMRegressor<T>::Compute(OpKernelContext* ctx) const {
  const Tensor* X = ctx->Input<Tensor>(0);

  int64_t stride = X->Shape().NumDimensions() == 1 ? X->Shape()[0] : X->Shape()[1];
  int64_t N = X->Shape().NumDimensions() == 1 ? 1 : X->Shape()[0];

  Tensor* Y = ctx->Output(0, TensorShape({N, 1}));  // this op outputs for one target only
  const auto* x_data = X->template Data<T>();

  for (int64_t n = 0; n < N; n++) {  //for each example
    int64_t current_weight_0 = n * stride;
    std::vector<float> scores;

    float sum = 0.f;
    if (mode_ == SVM_TYPE::SVM_SVC) {
      for (int64_t j = 0; j < vector_count_; j++) {
        float val1 = kernel_dot(x_data, current_weight_0, support_vectors_, feature_count_ * j, feature_count_, get_kernel_type());
        float val2 = coefficients_[j];
        float val3 = val1 * val2;
        sum += val3;
      }
      sum += rho_[0];
    } else if (mode_ == SVM_TYPE::SVM_LINEAR) {  //liblinear
      sum = kernel_dot(x_data, current_weight_0, coefficients_, 0, feature_count_, get_kernel_type());
      sum += rho_[0];
    }
    if (one_class_ && sum > 0) {
      Y->template MutableData<float>()[n] = 1.f;
    } else if (one_class_) {
      Y->template MutableData<float>()[n] = -1.f;
    } else {
      Y->template MutableData<float>()[n] = sum;
    }
  }

  return Status::OK();
}

}  // namespace ml
}  // namespace onnxruntime
