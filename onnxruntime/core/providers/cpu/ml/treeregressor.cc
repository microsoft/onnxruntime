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

template <typename T, typename TH>
detail::TreeEnsembleCommonAttributes* InitializeTreeEnsembleCommon(const OpKernelInfo& info) {
  std::vector<TH> base_values_as_tensor, nodes_hitrates_as_tensor, nodes_values_as_tensor, target_weights_as_tensor;
  ORT_ENFORCE(GetVectorAttrsOrDefault(info, "base_values_as_tensor", base_values_as_tensor).IsOK());
  ORT_ENFORCE(GetVectorAttrsOrDefault(info, "nodes_hitrates_as_tensor", nodes_hitrates_as_tensor).IsOK());
  ORT_ENFORCE(GetVectorAttrsOrDefault(info, "nodes_values_as_tensor", nodes_values_as_tensor).IsOK());
  ORT_ENFORCE(GetVectorAttrsOrDefault(info, "target_weights_as_tensor", target_weights_as_tensor).IsOK());

  return new detail::TreeEnsembleCommon<T, TH>(
      80,
      50,
      info.GetAttrOrDefault<std::string>("aggregate_function", "SUM"),
      info.GetAttrsOrDefault<float>("base_values"),
      base_values_as_tensor,
      info.GetAttrOrDefault<int64_t>("n_targets", 0),
      info.GetAttrsOrDefault<int64_t>("nodes_falsenodeids"),
      info.GetAttrsOrDefault<int64_t>("nodes_featureids"),
      info.GetAttrsOrDefault<float>("nodes_hitrates"),
      nodes_hitrates_as_tensor,
      info.GetAttrsOrDefault<int64_t>("nodes_missing_value_tracks_true"),
      info.GetAttrsOrDefault<std::string>("nodes_modes"),
      info.GetAttrsOrDefault<int64_t>("nodes_nodeids"),
      info.GetAttrsOrDefault<int64_t>("nodes_treeids"),
      info.GetAttrsOrDefault<int64_t>("nodes_truenodeids"),
      info.GetAttrsOrDefault<float>("nodes_values"),
      nodes_values_as_tensor,
      info.GetAttrOrDefault<std::string>("post_transform", "NONE"),
      info.GetAttrsOrDefault<int64_t>("target_ids"),
      info.GetAttrsOrDefault<int64_t>("target_nodeids"),
      info.GetAttrsOrDefault<int64_t>("target_treeids"),
      info.GetAttrsOrDefault<float>("target_weights"),
      target_weights_as_tensor);
}


template <typename T>
TreeEnsembleRegressor<T>::TreeEnsembleRegressor(const OpKernelInfo& info) : OpKernel(info) {
  ORT_IF_CONSTEXPR(std::is_same<T, double>::value) {
    p_tree_ensemble_ = InitializeTreeEnsembleCommon<T, double>(info);
  }
  else {
    p_tree_ensemble_ = InitializeTreeEnsembleCommon<T, float>(info);
  }
} 

template <typename T>
TreeEnsembleRegressor<T>::~TreeEnsembleRegressor() {
  ORT_IF_CONSTEXPR(std::is_same<T, double>::value) {
    delete static_cast<detail::TreeEnsembleCommon<T, double>*>(p_tree_ensemble_);
  }
  else {
    delete static_cast<detail::TreeEnsembleCommon<T, float>*>(p_tree_ensemble_);
  }
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
  TO* y_data = Y->template MutableData<TO>();
  ORT_IF_CONSTEXPR(std::is_same<T, double>::value) {
    int64_t size = Y->Shape().Size();
    std::unique_ptr<double[]> y_data_th = std::make_unique<double[]>(size);
    double* tree_output = y_data_th.get();
    static_cast<detail::TreeEnsembleCommon<T, double>*>(p_tree_ensemble_)->compute(context, X, tree_output, NULL);
    for (int64_t i = 0; i < size; ++i) {
      y_data[i] = static_cast<TO>(tree_output[i]);
    }
  }
  else {
    static_cast<detail::TreeEnsembleCommon<T, float>*>(p_tree_ensemble_)->compute(context, X, y_data, NULL);
  }

  return Status::OK();
}

}  // namespace ml
}  // namespace onnxruntime
