// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/ml/tree_ensemble_classifier.h"

/**
https://github.com/onnx/onnx/blob/master/onnx/defs/traditionalml/defs.cc
ONNX_OPERATOR_SCHEMA(TreeEnsembleClassifier)
.SetDomain("ai.onnx.ml")
.SetDoc(R"DOC(
Tree Ensemble classifier.  Returns the top class for each input in N.
All args with nodes_ are fields of a tuple of tree nodes, and
it is assumed they are the same length, and an index i will decode the
tuple across these inputs.  Each node id can appear only once
for each tree id.
All fields prefixed with class_ are tuples of votes at the leaves.
A leaf may have multiple votes, where each vote is weighted by
the associated class_weights index.
It is expected that either classlabels_strings or classlabels_int64s
will be passed and the class_ids are an index into this list.
Mode enum is BRANCH_LEQ, BRANCH_LT, BRANCH_GTE, BRANCH_GT, BRANCH_EQ, BRANCH_NEQ, LEAF
)DOC")
.Input(0, "X", "Input N,F", "T1")
.Output(0, "Y", "N, Top class for each point", "T2")
.Output(
1,
"Z",
"N,E the class score for each class, for each point",
"tensor(float)")
.TypeConstraint(
"T1",
{"tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)"},
" allowed types.")
.TypeConstraint(
"T2",
{"tensor(string)", "tensor(int64)"},
" allowed types.")
.Attr(
"nodes_treeids",
"tree id for this node",
AttributeProto::INTS,
OPTIONAL)
.Attr(
"nodes_nodeids",
"node id for this node, node ids may restart at zero for each tree (but not required).",
AttributeProto::INTS,
OPTIONAL)
.Attr(
"nodes_featureids",
"feature id for this node",
AttributeProto::INTS,
OPTIONAL)
.Attr(
"nodes_values",
"thresholds to do the splitting on for this node.",
AttributeProto::FLOATS,
OPTIONAL)
.Attr("nodes_hitrates", "", AttributeProto::FLOATS, OPTIONAL)
.Attr(
"nodes_modes",
"enum of behavior for this node 'BRANCH_LEQ', 'BRANCH_LT', 'BRANCH_GTE', 'BRANCH_GT', 'BRANCH_EQ', 'BRANCH_NEQ', 'LEAF'",
AttributeProto::STRINGS,
OPTIONAL)
.Attr(
"nodes_truenodeids",
"child node if expression is true",
AttributeProto::INTS,
OPTIONAL)
.Attr(
"nodes_falsenodeids",
"child node if expression is false",
AttributeProto::INTS,
OPTIONAL)
.Attr(
"nodes_missing_value_tracks_true",
"for each node, decide if the value is missing (nan) then use true branch, this field can be left unset and will assume false for all nodes",
AttributeProto::INTS,
OPTIONAL)
.Attr(
"class_treeids",
"tree that this node is in",
AttributeProto::INTS,
OPTIONAL)
.Attr(
"class_nodeids",
"node id that this weight is for",
AttributeProto::INTS,
OPTIONAL)
.Attr(
"class_ids",
"index of the class list that this weight is for",
AttributeProto::INTS,
OPTIONAL)
.Attr(
"class_weights",
"the weight for the class in class_id",
AttributeProto::FLOATS,
OPTIONAL)
.Attr(
"classlabels_strings",
"class labels if using string labels",
AttributeProto::STRINGS,
OPTIONAL)
.Attr(
"classlabels_int64s",
"class labels if using int labels",
AttributeProto::INTS,
OPTIONAL)
.Attr(
"post_transform",
"post eval transform for score, enum NONE, SOFTMAX, LOGISTIC, SOFTMAX_ZERO, PROBIT",
AttributeProto::STRING,
std::string("NONE"))
.Attr(
"base_values",
"base values for classification, added to final class score, size must be the same as classes or can be left unassigned (assumed 0)",
AttributeProto::FLOATS,
OPTIONAL);
*/
using namespace ::onnxruntime::common;
using namespace std;
namespace onnxruntime {
namespace ml {

#define ADD_IN_TYPE_TREE_ENSEMBLE_CLASSIFIER_OP(in_type)                                                                                                                                          \
  ONNX_CPU_OPERATOR_TYPED_ML_KERNEL(                                                                                                                                                              \
      TreeEnsembleClassifier,                                                                                                                                                                     \
      1,                                                                                                                                                                                          \
      in_type,                                                                                                                                                                                    \
      KernelDefBuilder().TypeConstraint("T1", DataTypeImpl::GetTensorType<in_type>()).TypeConstraint("T2", {DataTypeImpl::GetTensorType<int64_t>(), DataTypeImpl::GetTensorType<std::string>()}), \
      TreeEnsembleClassifier<in_type>);

ADD_IN_TYPE_TREE_ENSEMBLE_CLASSIFIER_OP(float);
ADD_IN_TYPE_TREE_ENSEMBLE_CLASSIFIER_OP(double);
ADD_IN_TYPE_TREE_ENSEMBLE_CLASSIFIER_OP(int64_t);
ADD_IN_TYPE_TREE_ENSEMBLE_CLASSIFIER_OP(int32_t);

template <typename T>
TreeEnsembleClassifier<T>::TreeEnsembleClassifier(const OpKernelInfo& info)
    : OpKernel(info),
      tree_ensemble_(
          100,
          50,
          info.GetAttrOrDefault<std::string>("aggregate_function", "SUM"),
          info.GetAttrsOrDefault<float>("base_values"),
          info.GetAttrsOrDefault<int64_t>("nodes_falsenodeids"),
          info.GetAttrsOrDefault<int64_t>("nodes_featureids"),
          info.GetAttrsOrDefault<float>("nodes_hitrates"),
          info.GetAttrsOrDefault<int64_t>("nodes_missing_value_tracks_true"),
          info.GetAttrsOrDefault<std::string>("nodes_modes"),
          info.GetAttrsOrDefault<int64_t>("nodes_nodeids"),
          info.GetAttrsOrDefault<int64_t>("nodes_treeids"),
          info.GetAttrsOrDefault<int64_t>("nodes_truenodeids"),
          info.GetAttrsOrDefault<float>("nodes_values"),
          info.GetAttrOrDefault<std::string>("post_transform", "NONE"),
          info.GetAttrsOrDefault<int64_t>("class_ids"),
          info.GetAttrsOrDefault<int64_t>("class_nodeids"),
          info.GetAttrsOrDefault<int64_t>("class_treeids"),
          info.GetAttrsOrDefault<float>("class_weights"),
          info.GetAttrsOrDefault<std::string>("classlabels_strings"),
          info.GetAttrsOrDefault<int64_t>("classlabels_int64s")) {
}  // namespace ml

template <typename T>
common::Status TreeEnsembleClassifier<T>::Compute(OpKernelContext* context) const {
  const Tensor& X = *context->Input<Tensor>(0);
  const std::vector<int64_t>& x_dims = X.Shape().GetDims();
  if (x_dims.empty()) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "X dims is empty.");
  }

  int64_t N = x_dims.size() == 1 ? 1 : x_dims[0];
  Tensor* Y = context->Output(0, {N});
  Tensor* Z = context->Output(1, {N, tree_ensemble_.get_class_count()});

  tree_ensemble_.compute(context, &X, Z, Y);
  return Status::OK();
}

}  // namespace ml
}  // namespace onnxruntime
