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
static T GetAttr(const OpKernelInfo& info, const std::string& name) {
  T value;
  ORT_ENFORCE(info.GetAttr<T>(name, &value).IsOK());
  return value;
}

template <typename T>
static Eigen::Tensor<T, 2, Eigen::RowMajor, Eigen::DenseIndex> GetCoefficients(const OpKernelInfo& info, int64_t targets_) {
  std::vector<float> c;
  ORT_ENFORCE(info.GetAttrs<float>("coefficients", c).IsOK());
  ORT_ENFORCE(c.size() % targets_ == 0);
  int64_t feature_size = static_cast<int64_t>(c.size()) / targets_;
  typename Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Unaligned> c_tensor(c.data(), targets_, feature_size);
  //Tranpose the data so that we can use math::MatMul instead of math::Gemm in the compute function
  Eigen::array<int, 2> shuffle{1, 0};
  return c_tensor.cast<T>().shuffle(shuffle);
}

template <typename T>
LinearRegressor<T>::LinearRegressor(const OpKernelInfo& info)
    : OpKernel(info),
      targets_(GetAttr<int64_t>(info, "targets")),
      coefficients_(GetCoefficients<T>(info, targets_)),
      post_transform_(MakeTransform(info.GetAttrOrDefault<std::string>("post_transform", "NONE"))) {
  {
    std::vector<float> c = info.GetAttrsOrDefault<float>("intercepts");
    intercepts_.resize(c.size());
    std::copy_n(c.data(), c.size(), intercepts_.data());
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
