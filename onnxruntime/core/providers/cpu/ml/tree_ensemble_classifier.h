// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "tree_ensemble_common.h"

namespace onnxruntime {
namespace ml {
template <typename T>
class TreeEnsembleClassifier final : public OpKernel {
  typedef T InputType;          // input type
  typedef float OutputType;     // output type
 public:
  explicit TreeEnsembleClassifier(const OpKernelInfo& info);
  common::Status Compute(OpKernelContext* context) const override;

 private:
  // Following pointer holds a pointer on one instance of 
  // detail::TreeEnsembleCommonClassifier<T, ThresholdType, float>
  // where TH is defined after accessing the attributes.
  std::unique_ptr<detail::TreeEnsembleCommonAttributes> p_tree_ensemble_;
};
}  // namespace ml
}  // namespace onnxruntime
