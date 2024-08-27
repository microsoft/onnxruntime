// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "tree_ensemble_common.h"

namespace onnxruntime {
namespace ml {
template <typename T>
class TreeEnsemble final : public OpKernel {
  typedef T InputType;       // input type
  typedef float OutputType;  // output type
 public:
  explicit TreeEnsemble(const OpKernelInfo& info);
  common::Status Compute(OpKernelContext* context) const override;
  Status GetRemovableAttributes(InlinedVector<std::string>& removable_attributes) const override;

 private:
  // Pointer on one instance of
  // detail::TreeEnsembleCommonV5<T, ThresholdType>
  // where ThresholdType is defined after accessing the attributes.
  std::unique_ptr<detail::TreeEnsembleCommonAttributes> p_tree_ensemble_;
};
}  // namespace ml
}  // namespace onnxruntime
