// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/ml/tree_ensemble_classifier.h"
#include "core/providers/cpu/ml/tree_ensemble_helper.h"
#include "core/common/inlined_containers_fwd.h"

using namespace ::onnxruntime::common;
using namespace std;
namespace onnxruntime {
namespace ml {

#define ADD_IN_TYPE_TREE_ENSEMBLE_CLASSIFIER_OP(in_type, value_type)                                                                                                                                          \
  ONNX_CPU_OPERATOR_TYPED_ML_KERNEL(                                                                                                                                                              \
      TreeEnsembleClassifier,                                                                                                                                                                     \
      1,                                                                                                                                                                                          \
      in_type,                                                                                                                                                                                    \
      KernelDefBuilder().TypeConstraint("T1", DataTypeImpl::GetTensorType<in_type>()).TypeConstraint("T2", {DataTypeImpl::GetTensorType<int64_t>(), DataTypeImpl::GetTensorType<std::string>()}), \
      TreeEnsembleClassifier<in_type, float, float>);                                                                                                                                                           \
  ONNX_CPU_OPERATOR_TYPED_ML_KERNEL(                                                                                                                                                              \
      TreeEnsembleClassifier,                                                                                                                                                                     \
      3,                                                                                                                                                                                          \
      in_type,                                                                                                                                                                                    \
      KernelDefBuilder().TypeConstraint("T1", DataTypeImpl::GetTensorType<in_type>()).TypeConstraint("T2", {DataTypeImpl::GetTensorType<int64_t>(), DataTypeImpl::GetTensorType<std::string>()}), \
      TreeEnsembleClassifier<in_type, value_type, float>);

ADD_IN_TYPE_TREE_ENSEMBLE_CLASSIFIER_OP(float, float);
ADD_IN_TYPE_TREE_ENSEMBLE_CLASSIFIER_OP(double, double);
ADD_IN_TYPE_TREE_ENSEMBLE_CLASSIFIER_OP(int64_t, float);
ADD_IN_TYPE_TREE_ENSEMBLE_CLASSIFIER_OP(int32_t, float);

template <typename T, typename TH, typename TO>
TreeEnsembleClassifier<T, TH, TO>::TreeEnsembleClassifier(const OpKernelInfo& info)
    : OpKernel(info),
      tree_ensemble_(
          80,
          50,
          info.GetAttrOrDefault<std::string>("aggregate_function", "SUM"),
          info.GetAttrsOrDefault<float>("base_values"),
          GetVectorAttrsOrDefault(info, "base_values_as_tensor", std::vector<TH>()),
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
          info.GetAttrsOrDefault<int64_t>("class_ids"),
          info.GetAttrsOrDefault<int64_t>("class_nodeids"),
          info.GetAttrsOrDefault<int64_t>("class_treeids"),
          info.GetAttrsOrDefault<float>("class_weights"),
          GetVectorAttrsOrDefault(info, "class_weights_as_tensor", std::vector<TH>()),
          info.GetAttrsOrDefault<std::string>("classlabels_strings"),
          info.GetAttrsOrDefault<int64_t>("classlabels_int64s")) {
}  // namespace ml

template <typename T, typename TH, typename TO>
common::Status TreeEnsembleClassifier<T, TH, TO>::Compute(OpKernelContext* context) const {
  const Tensor& X = *context->Input<Tensor>(0);
  auto x_dims = X.Shape().GetDims();
  if (x_dims.empty()) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "X dims is empty.");
  }

  int64_t N = x_dims.size() == 1 ? 1 : x_dims[0];
  Tensor* Y = context->Output(0, {N});
  Tensor* Z = context->Output(1, {N, tree_ensemble_.get_class_count()});

  ORT_IF_CONSTEXPR(std::is_same<TH, TO>::value) {
    TH* z_data = Z->template MutableData<TH>();
    tree_ensemble_.compute(context, &X, z_data, Y);
  } else {
    InlinedVector<TH> z_data_th(Z->Shape().Size());
    tree_ensemble_.compute(context, &X, z_data_th.data(), Y);
    TO* z_data = Z->template MutableData<TO>();
    for (size_t i = 0; i < z_data_th.size(); ++i) {
      z_data[i] = static_cast<TO>(z_data_th[i]);
    }
  }

  return Status::OK();
}

}  // namespace ml
}  // namespace onnxruntime
