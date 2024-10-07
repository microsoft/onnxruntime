// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/ml/tree_ensemble.h"
#include "core/providers/cpu/ml/tree_ensemble_helper.h"
#include "core/common/inlined_containers_fwd.h"

namespace onnxruntime {
namespace ml {

ONNX_CPU_OPERATOR_TYPED_ML_KERNEL(
    TreeEnsemble,
    5,
    float,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()).MayInplace(0, 0),
    TreeEnsemble<float>);

ONNX_CPU_OPERATOR_TYPED_ML_KERNEL(
    TreeEnsemble,
    5,
    double,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()).MayInplace(0, 0),
    TreeEnsemble<double>);

template <typename T>
TreeEnsemble<T>::TreeEnsemble(const OpKernelInfo& info) : OpKernel(info) {
  if constexpr (std::is_same<T, double>::value) {
    p_tree_ensemble_ = std::make_unique<detail::TreeEnsembleCommonV5<T, double>>();
  } else {
    p_tree_ensemble_ = std::make_unique<detail::TreeEnsembleCommonV5<T, float>>();
  }
  ORT_THROW_IF_ERROR(p_tree_ensemble_->Init(info));
}

template <typename T>
Status TreeEnsemble<T>::GetRemovableAttributes(InlinedVector<std::string>& removable_attributes) const {
  InlinedVector<std::string> names{
      "leaf_targetids", "leaf_weights", "membership_values", "nodes_falseleafs",
      "nodes_falsenodeids", "nodes_featureids", "nodes_hitrates", "nodes_missing_value_tracks_true",
      "nodes_modes", "nodes_splits", "nodes_trueleafs", "nodes_truenodeids"};
  removable_attributes.swap(names);
  return Status::OK();
}

template <typename T>
common::Status TreeEnsemble<T>::Compute(OpKernelContext* context) const {
  const auto* X = context->Input<Tensor>(0);
  if (X == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
  if (X->Shape().NumDimensions() == 0) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                  "Input shape needs to be at least a single dimension.");
  }
  int64_t N = X->Shape().NumDimensions() == 1 ? 1 : X->Shape()[0];
  Tensor* Y = context->Output(0, {N, p_tree_ensemble_->get_target_or_class_count()});
  return p_tree_ensemble_->compute(context, X, Y, NULL);
}

}  // namespace ml
}  // namespace onnxruntime
