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


template <typename T>
TreeEnsembleClassifier<T>::TreeEnsembleClassifier(const OpKernelInfo& info) : OpKernel(info) {
  ORT_IF_CONSTEXPR(std::is_same<T, double>::value) {
    p_tree_ensemble_ = std::make_unique<detail::TreeEnsembleCommonClassifier<T, double, OutputType>>();
  }
  else {
    p_tree_ensemble_ = std::make_unique<detail::TreeEnsembleCommonClassifier<T, float, OutputType>>();
  }
  ORT_THROW_IF_ERROR(p_tree_ensemble_->Init(info));
}

template <typename T>
common::Status TreeEnsembleClassifier<T>::Compute(OpKernelContext* context) const {
  const Tensor& X = *context->Input<Tensor>(0);
  auto x_dims = X.Shape().GetDims();
  if (x_dims.empty()) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "X dims is empty.");
  }

  int64_t N = x_dims.size() == 1 ? 1 : x_dims[0];
  Tensor* label = context->Output(0, {N});  // int64_t
  Tensor* Z = context->Output(1, {N, p_tree_ensemble_->get_target_or_class_count()});  // TO
  return p_tree_ensemble_->compute(context, &X, Z, label);
}

}  // namespace ml
}  // namespace onnxruntime
