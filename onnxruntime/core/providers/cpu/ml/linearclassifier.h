// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"
#include "ml_common.h"

namespace onnxruntime {
namespace ml {

class LinearClassifier final : public OpKernel {
 public:
  LinearClassifier(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;

 private:
  void ComputeImpl(const gsl::span<const float> input, ptrdiff_t num_batches, ptrdiff_t num_features, ptrdiff_t num_targets,
                   const std::vector<float>& coefficients,
                   const std::vector<float>& intercepts,
                   Tensor& labels_output,
                   Tensor& scores_output,
                   POST_EVAL_TRANSFORM post_transform,
                   bool add_second_class,
                   concurrency::ThreadPool* threadpool) const;

  int64_t multi_class_;
  ptrdiff_t class_count_;
  POST_EVAL_TRANSFORM post_transform_;
  bool using_strings_;
  std::vector<float> coefficients_;
  std::vector<float> intercepts_;
  std::vector<std::string> classlabels_strings_;
  std::vector<int64_t> classlabels_ints_;
};

}  // namespace ml
}  // namespace onnxruntime
