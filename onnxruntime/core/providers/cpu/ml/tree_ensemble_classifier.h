// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "tree_ensemble_common.h"

namespace onnxruntime {
namespace ml {
template <typename T>
class TreeEnsembleClassifier final : public OpKernel {
  typedef T TI;      // input type
  // typedef T TH;  // threshold type, double if T==double, float otherwise
  typedef float TO;  // output type
 public:
  explicit TreeEnsembleClassifier(const OpKernelInfo& info);
  ~TreeEnsembleClassifier();
  common::Status Compute(OpKernelContext* context) const override;

 private:
  // Following pointer holds a pointer on one instance of detail::TreeEnsembleCommonClassifier<T, TH>
  // where TH is defined after accessing the attributes.
  detail::TreeEnsembleCommonAttributes* p_tree_ensemble_;
};
}  // namespace ml
}  // namespace onnxruntime
