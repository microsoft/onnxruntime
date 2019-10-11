// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/providers/cpu/ml/linearregressor.h"
#include "core/util/eigen_common_wrapper.h"

namespace onnxruntime {
namespace ml {

ONNX_OPERATOR_TYPED_KERNEL_EX(LinearRegressor, kMLDomain, 1, float, kCpuExecutionProvider,
                              KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                              LinearRegressor<float>);

ONNX_OPERATOR_TYPED_KERNEL_EX(LinearRegressor, kMLDomain, 1, double, kCpuExecutionProvider,
                              KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()),
                              LinearRegressor<double>);

template <typename T>
LinearRegressor<T>::LinearRegressor(const OpKernelInfo& info)
    : OpKernel(info), post_transform_(MakeTransform(info.GetAttrOrDefault<std::string>("post_transform", "NONE"))) {
  std::vector<float> c;
  c = info.GetAttrsOrDefault<float>("intercepts");
  intercepts_.resize(c.size());
  std::copy_n(c.data(), c.size(), intercepts_.data());
  ORT_ENFORCE(info.GetAttr<int64_t>("targets", &targets_).IsOK());
  c.clear();
  ORT_ENFORCE(info.GetAttrs<float>("coefficients", c).IsOK());
  coefficients_.resize(c.size());
  ORT_ENFORCE(c.size() % targets_ == 0);
  int64_t feature_size = static_cast<int64_t>(c.size()) / targets_;
  for (int64_t i = 0; i != targets_; ++i) {
    int64_t offset = i * feature_size;
    for (int64_t j = 0; j != feature_size; ++j) {
      coefficients_[j * targets_ + i] = c[offset + j];
    }
  }

  // A dirty hack to keep the code working as before
  if (targets_ == 1) {
    // In RS4, we implemented the PROBIT transform for single class cases,
    // but the outputted value is most likely NaN
    post_transform_ = POST_EVAL_TRANSFORM::NONE;
  } else if (post_transform_ == POST_EVAL_TRANSFORM::PROBIT) {
    // In RS4, we didn't implement the PROBIT transform for multiclass cases
    post_transform_ = POST_EVAL_TRANSFORM::NONE;
  }
}

}  // namespace ml
}  // namespace onnxruntime
