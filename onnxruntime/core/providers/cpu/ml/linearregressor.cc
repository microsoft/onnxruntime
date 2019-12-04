// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/ml/linearregressor.h"

namespace onnxruntime {
namespace ml {

ONNX_CPU_OPERATOR_ML_KERNEL(
    LinearRegressor,
    1,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    LinearRegressor<float>);

template <typename T>
LinearRegressor<T>::LinearRegressor(const OpKernelInfo& info) : OpKernel(info),
                                                                intercepts_(info.GetAttrsOrDefault<float>("intercepts")),
                                                                post_transform_(MakeTransform(info.GetAttrOrDefault<std::string>("post_transform", "NONE"))) {
  ORT_ENFORCE(info.GetAttr<int64_t>("targets", &targets_).IsOK());
  ORT_ENFORCE(info.GetAttrs<float>("coefficients", coefficients_).IsOK());
}

template <>
Status LinearRegressor<float>::Compute(OpKernelContext* ctx) const {
  const auto* X = ctx->Input<Tensor>(0);
  if (X->Shape().NumDimensions() == 0) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                  "Input shape needs to be at least a single dimension.");
  }

  int64_t stride = X->Shape().NumDimensions() == 1 ? X->Shape()[0] : X->Shape()[1];
  int64_t N = X->Shape().NumDimensions() == 1 ? 1 : X->Shape()[0];
  Tensor* Y = ctx->Output(0, TensorShape({N, targets_}));
  const auto* Xdata = X->template Data<float>();
  int64_t yindex = 0;

  bool useIntercepts = intercepts_.size() == static_cast<size_t>(targets_);
  for (int64_t i = 0; i < N; i++)  //for each point
  {
    std::vector<float> scores;
    int64_t current_weight_0 = i * stride;
    for (int j = 0; j < targets_; j++)  //for each target
    {
      int64_t current_coeff_0 = j * stride;
      float weight = 0.f;
      for (int64_t k = 0; k < stride; k++)  //for each weight
      {
        weight = weight + Xdata[current_weight_0 + k] * coefficients_[current_coeff_0 + k];
      }
      if (useIntercepts) {
        weight = weight + intercepts_[j];
      }
      scores.push_back(weight);
    }
    ::onnxruntime::ml::write_scores(scores, post_transform_, yindex, Y, -1);
    yindex += scores.size();
  }
  return Status::OK();
}

}  // namespace ml
}  // namespace onnxruntime
