// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"
#include "ml_common.h"
#include "svmclassifier.h"

namespace onnxruntime {
namespace ml {

template <typename T>
class SVMRegressor final : public OpKernel, private SVMCommon {
  using SVMCommon::batched_kernel_dot;
  using SVMCommon::get_kernel_type;
  using SVMCommon::set_kernel_type;

 public:
  SVMRegressor(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;

 private:
  bool one_class_;
  ptrdiff_t feature_count_;
  ptrdiff_t vector_count_;
  std::vector<float> rho_;
  std::vector<float> coefficients_;
  std::vector<float> support_vectors_;
  POST_EVAL_TRANSFORM post_transform_;
  SVM_TYPE mode_;  // how are we computing SVM? 0=LibSVC, 1=LibLinear
};

}  // namespace ml
}  // namespace onnxruntime
