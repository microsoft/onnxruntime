// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/ml/tree_ensemble_classifier.h"
#include "core/providers/cpu/ml/tree_ensemble_helper.h"
#include "core/common/inlined_containers_fwd.h"

using namespace ::onnxruntime::common;
using namespace std;
namespace onnxruntime {
namespace ml {

#define ADD_IN_TYPE_TREE_ENSEMBLE_CLASSIFIER_OP(in_type)                                                                                                                              \
  ONNX_CPU_OPERATOR_VERSIONED_TYPED_ML_KERNEL(                                                                                                                                                    \
      TreeEnsembleClassifier,                                                                                                                                                                     \
      1,  2,                                                                                                                                                                                      \
      in_type,                                                                                                                                                                                    \
      KernelDefBuilder().TypeConstraint("T1", DataTypeImpl::GetTensorType<in_type>()).TypeConstraint("T2", {DataTypeImpl::GetTensorType<int64_t>(), DataTypeImpl::GetTensorType<std::string>()}), \
      TreeEnsembleClassifier<in_type>);                                                                                                                                                           \
  ONNX_CPU_OPERATOR_TYPED_ML_KERNEL(                                                                                                                                                              \
      TreeEnsembleClassifier,                                                                                                                                                                     \
      3,                                                                                                                                                                                          \
      in_type,                                                                                                                                                                                    \
      KernelDefBuilder().TypeConstraint("T1", DataTypeImpl::GetTensorType<in_type>()).TypeConstraint("T2", {DataTypeImpl::GetTensorType<int64_t>(), DataTypeImpl::GetTensorType<std::string>()}), \
      TreeEnsembleClassifier<in_type>);

ADD_IN_TYPE_TREE_ENSEMBLE_CLASSIFIER_OP(float);
ADD_IN_TYPE_TREE_ENSEMBLE_CLASSIFIER_OP(double);
ADD_IN_TYPE_TREE_ENSEMBLE_CLASSIFIER_OP(int64_t);
ADD_IN_TYPE_TREE_ENSEMBLE_CLASSIFIER_OP(int32_t);


template <typename T, typename TH>
detail::TreeEnsembleCommonAttributes* InitializeTreeEnsembleCommonClassifier(const OpKernelInfo& info) {
  std::vector<TH> base_values_as_tensor, nodes_hitrates_as_tensor, nodes_values_as_tensor, class_weights_as_tensor;
  ORT_ENFORCE(GetVectorAttrsOrDefault(info, "base_values_as_tensor", base_values_as_tensor).IsOK());
  ORT_ENFORCE(GetVectorAttrsOrDefault(info, "nodes_hitrates_as_tensor", nodes_hitrates_as_tensor).IsOK());
  ORT_ENFORCE(GetVectorAttrsOrDefault(info, "nodes_values_as_tensor", nodes_values_as_tensor).IsOK());
  ORT_ENFORCE(GetVectorAttrsOrDefault(info, "class_weights_as_tensor", class_weights_as_tensor).IsOK());

  return new detail::TreeEnsembleCommonClassifier<T, TH>(
      80,
      50,
      info.GetAttrOrDefault<std::string>("aggregate_function", "SUM"),
      info.GetAttrsOrDefault<float>("base_values"),
      base_values_as_tensor,
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
      info.GetAttrsOrDefault<int64_t>("class_ids"),
      info.GetAttrsOrDefault<int64_t>("class_nodeids"),
      info.GetAttrsOrDefault<int64_t>("class_treeids"),
      info.GetAttrsOrDefault<float>("class_weights"),
      class_weights_as_tensor,
      info.GetAttrsOrDefault<std::string>("classlabels_strings"),
      info.GetAttrsOrDefault<int64_t>("classlabels_int64s"));
}

template <typename T>
TreeEnsembleClassifier<T>::TreeEnsembleClassifier(const OpKernelInfo& info) : OpKernel(info) {
  ORT_IF_CONSTEXPR(std::is_same<T, double>::value) {
    p_tree_ensemble_ = InitializeTreeEnsembleCommonClassifier<T, double>(info);
  }
  else {
    p_tree_ensemble_ = InitializeTreeEnsembleCommonClassifier<T, float>(info);
  }
}

template <typename T>
TreeEnsembleClassifier<T>::~TreeEnsembleClassifier() {
  ORT_IF_CONSTEXPR(std::is_same<T, double>::value) {
    delete static_cast<detail::TreeEnsembleCommonClassifier<T, double>*>(p_tree_ensemble_);
  }
  else {
    delete static_cast<detail::TreeEnsembleCommonClassifier<T, float>*>(p_tree_ensemble_);
  }
}

template <typename T>
common::Status TreeEnsembleClassifier<T>::Compute(OpKernelContext* context) const {
  const Tensor& X = *context->Input<Tensor>(0);
  auto x_dims = X.Shape().GetDims();
  if (x_dims.empty()) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "X dims is empty.");
  }

  int64_t N = x_dims.size() == 1 ? 1 : x_dims[0];
  Tensor* Y = context->Output(0, {N});
  Tensor* Z = context->Output(1, {N, p_tree_ensemble_->get_target_or_class_count()});

  ORT_IF_CONSTEXPR(std::is_same<T, double>::value) {
    int64_t size = Z->Shape().Size();
    std::unique_ptr<double[]> z_data_th = std::make_unique<double[]>(size);
    double* tree_output = z_data_th.get();
    static_cast<detail::TreeEnsembleCommonClassifier<T, double>*>(p_tree_ensemble_)->compute(context, &X, tree_output, Y);
    TO* z_data = Z->template MutableData<TO>();
    for (int64_t i = 0; i < size; ++i) {
      z_data[i] = static_cast<TO>(tree_output[i]);
    }
  } else {
    float* z_data = Z->template MutableData<float>();
    static_cast<detail::TreeEnsembleCommonClassifier<T, float>*>(p_tree_ensemble_)->compute(context, &X, z_data, Y);
  }

  return Status::OK();
}

}  // namespace ml
}  // namespace onnxruntime
