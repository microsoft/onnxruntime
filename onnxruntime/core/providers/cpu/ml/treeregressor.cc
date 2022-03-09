// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/ml/treeregressor.h"
#include "core/providers/cpu/ml/tree_ensemble_helper.h"

namespace onnxruntime {
namespace ml {

ONNX_CPU_OPERATOR_TYPED_ML_KERNEL(
    TreeEnsembleRegressor,
    1,
    float,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()).MayInplace(0, 0),
    TreeEnsembleRegressor<float>);

ONNX_CPU_OPERATOR_TYPED_ML_KERNEL(
    TreeEnsembleRegressor,
    1,
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
    TreeEnsembleRegressor<double, double, float>);

template <typename T, typename TH, typename TO>
TreeEnsembleRegressor<T, TH, TO>::TreeEnsembleRegressor(const OpKernelInfo& info)
    : OpKernel(info),
      tree_ensemble_(
          80,
          50,
          info.GetAttrOrDefault<std::string>("aggregate_function", "SUM"),
          info.GetAttrsOrDefault<float>("base_values"),
          GetVectorAttrsOrDefault(info, "base_values_as_tensor", std::vector<TH>()),
          info.GetAttrOrDefault<int64_t>("n_targets", 0),
          info.GetAttrsOrDefault<int64_t>("nodes_falsenodeids"),
          info.GetAttrsOrDefault<int64_t>("nodes_featureids"),
          info.GetAttrsOrDefault<float>("nodes_hitrates"),
          GetVectorAttrsOrDefault(info, "nodes_hitrates_as_tensor", std::vector<TH>()),
          info.GetAttrsOrDefault<int64_t>("nodes_missing_value_tracks_true"),
          info.GetAttrsOrDefault<std::string>("nodes_modes"),
          info.GetAttrsOrDefault<int64_t>("nodes_nodeids"),
          info.GetAttrsOrDefault<int64_t>("nodes_treeids"),
          info.GetAttrsOrDefault<int64_t>("nodes_truenodeids"),
          info.GetAttrsOrDefault<float>("nodes_values"),
          GetVectorAttrsOrDefault(info, "nodes_values_as_tensor", std::vector<TH>()),
          info.GetAttrOrDefault<std::string>("post_transform", "NONE"),
          info.GetAttrsOrDefault<int64_t>("target_ids"),
          info.GetAttrsOrDefault<int64_t>("target_nodeids"),
          info.GetAttrsOrDefault<int64_t>("target_treeids"),
          info.GetAttrsOrDefault<float>("target_weights"),
          GetVectorAttrsOrDefault(info, "target_weights_as_tensor", std::vector<TH>())) {
}  // namespace ml

template <typename T, typename TH, typename TO>
common::Status TreeEnsembleRegressor<T, TH, TO>::Compute(OpKernelContext* context) const {
  const auto* X = context->Input<Tensor>(0);
  if (X == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
  if (X->Shape().NumDimensions() == 0) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                  "Input shape needs to be at least a single dimension.");
  }
  int64_t N = X->Shape().NumDimensions() == 1 ? 1 : X->Shape()[0];
  Tensor* Y = context->Output(0, {N, tree_ensemble_.n_targets_or_classes_});
  if (std::is_same<TH, TO>::value) {
    TH* y_data = Y->template MutableData<TH>();
    tree_ensemble_.compute(context, X, y_data, NULL);
  } else {
    std::vector<TH> y_data_th(Y->Shape().Size());
    tree_ensemble_.compute(context, X, y_data_th.data(), NULL);
    TO* y_data = Y->template MutableData<TO>();
    for (size_t i = 0; i < y_data_th.size(); ++i) {
      y_data[i] = static_cast<TO>(y_data_th[i]);
    }
  }

  return Status::OK();
}

}  // namespace ml
}  // namespace onnxruntime
