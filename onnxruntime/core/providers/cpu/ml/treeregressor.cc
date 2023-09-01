// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/ml/treeregressor.h"
#include "core/providers/cpu/ml/tree_ensemble_helper.h"
#include "core/common/inlined_containers_fwd.h"

namespace onnxruntime {
namespace ml {

ONNX_CPU_OPERATOR_VERSIONED_TYPED_ML_KERNEL(
    TreeEnsembleRegressor,
    1, 2,
    float,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()).MayInplace(0, 0),
    TreeEnsembleRegressor<float>);

ONNX_CPU_OPERATOR_VERSIONED_TYPED_ML_KERNEL(
    TreeEnsembleRegressor,
    1, 2,
    double,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()).MayInplace(0, 0),
    TreeEnsembleRegressor<double>);

ONNX_CPU_OPERATOR_TYPED_ML_KERNEL(
    TreeEnsembleRegressor,
    3,
    float,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()).MayInplace(0, 0),
    TreeEnsembleRegressor<float>);

ONNX_CPU_OPERATOR_TYPED_ML_KERNEL(
    TreeEnsembleRegressor,
    3,
    double,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()).MayInplace(0, 0),
    TreeEnsembleRegressor<double>);

template <typename T>
TreeEnsembleRegressor<T>::TreeEnsembleRegressor(const OpKernelInfo& info) : OpKernel(info) {
  if constexpr (std::is_same<T, double>::value) {
    p_tree_ensemble_ = std::make_unique<detail::TreeEnsembleCommon<T, double, OutputType>>();
  } else {
    p_tree_ensemble_ = std::make_unique<detail::TreeEnsembleCommon<T, float, OutputType>>();
  }
  ORT_THROW_IF_ERROR(p_tree_ensemble_->Init(info));
}

template <typename T>
Status TreeEnsembleRegressor<T>::GetRemovableAttributes(InlinedVector<std::string>& removable_attributes) const {
  InlinedVector<std::string> names {
    "base_values", "nodes_falsenodeids", "nodes_featureids", "nodes_hitrates",
        "nodes_missing_value_tracks_true", "nodes_modes", "nodes_nodeids", "nodes_treeids",
        "nodes_truenodeids", "nodes_values",
        "target_ids", "target_treeids", "target_nodeids",
        "target_weights"
#if !defined(ORT_MINIMAL_BUILD)
        "base_values_as_tensor",
        "nodes_hitrates_as_tensor", "nodes_values_as_tensor",
        "class_weights_as_tensor"
#endif
  };
  removable_attributes.swap(names);
  return Status::OK();
}

template <typename T>
common::Status TreeEnsembleRegressor<T>::Compute(OpKernelContext* context) const {
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
