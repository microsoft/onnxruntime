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
      nodes_treeids_(info.GetAttrsOrDefault<int64_t>("nodes_treeids")),
      nodes_nodeids_(info.GetAttrsOrDefault<int64_t>("nodes_nodeids")),
      nodes_featureids_(info.GetAttrsOrDefault<int64_t>("nodes_featureids")),
      nodes_values_(info.GetAttrsOrDefault<float>("nodes_values")),
      nodes_hitrates_(info.GetAttrsOrDefault<float>("nodes_hitrates")),
      nodes_modes_names_(info.GetAttrsOrDefault<std::string>("nodes_modes")),
      nodes_truenodeids_(info.GetAttrsOrDefault<int64_t>("nodes_truenodeids")),
      nodes_falsenodeids_(info.GetAttrsOrDefault<int64_t>("nodes_falsenodeids")),
      missing_tracks_true_(info.GetAttrsOrDefault<int64_t>("nodes_missing_value_tracks_true")),
      class_nodeids_(info.GetAttrsOrDefault<int64_t>("class_nodeids")),
      class_treeids_(info.GetAttrsOrDefault<int64_t>("class_treeids")),
      class_ids_(info.GetAttrsOrDefault<int64_t>("class_ids")),
      class_weights_(info.GetAttrsOrDefault<float>("class_weights")),
      base_values_(info.GetAttrsOrDefault<float>("base_values")),
      classlabels_strings_(info.GetAttrsOrDefault<std::string>("classlabels_strings")),
      classlabels_int64s_(info.GetAttrsOrDefault<int64_t>("classlabels_int64s")),
      post_transform_(MakeTransform(info.GetAttrOrDefault<std::string>("post_transform", "NONE"))) {
  ORT_ENFORCE(!nodes_treeids_.empty());
  ORT_ENFORCE(class_nodeids_.size() == class_ids_.size());
  ORT_ENFORCE(class_nodeids_.size() == class_weights_.size());
  ORT_ENFORCE(nodes_nodeids_.size() == nodes_featureids_.size());
  ORT_ENFORCE(nodes_nodeids_.size() == nodes_modes_names_.size());
  ORT_ENFORCE(nodes_nodeids_.size() == nodes_values_.size());
  ORT_ENFORCE(nodes_nodeids_.size() == nodes_truenodeids_.size());
  ORT_ENFORCE(nodes_nodeids_.size() == nodes_falsenodeids_.size());
  ORT_ENFORCE((nodes_nodeids_.size() == nodes_hitrates_.size()) || (nodes_hitrates_.empty()));

  ORT_ENFORCE(classlabels_strings_.empty() ^ classlabels_int64s_.empty(),
              "Must provide classlabels_strings or classlabels_int64s but not both.");

  // in the absence of bool type supported by GetAttrs this ensure that we don't have any negative
  // values so that we can check for the truth condition without worrying about negative values.
  ORT_ENFORCE(std::all_of(
      std::begin(missing_tracks_true_),
      std::end(missing_tracks_true_), [](int64_t elem) { return elem >= 0; }));

  Initialize();
}

template <typename T>
void TreeEnsembleClassifier<T>::Initialize() {
  int64_t current_tree_id = 1234567891L;
  std::vector<int64_t> tree_offsets;
  weights_are_all_positive_ = true;

  for (int64_t i = 0, size_node_treeids = static_cast<int64_t>(nodes_treeids_.size());
       i < size_node_treeids;
       ++i) {
    if (nodes_treeids_[i] != current_tree_id) {
      tree_offsets.push_back(nodes_nodeids_[i]);
      current_tree_id = nodes_treeids_[i];
    }
    int64_t offset = tree_offsets[tree_offsets.size() - 1];
    nodes_nodeids_[i] = nodes_nodeids_[i] - offset;
    if (nodes_falsenodeids_[i] >= 0) {
      nodes_falsenodeids_[i] = nodes_falsenodeids_[i] - offset;
    }
    if (nodes_truenodeids_[i] >= 0) {
      nodes_truenodeids_[i] = nodes_truenodeids_[i] - offset;
    }
  }
  for (int64_t i = 0, size_class_nodeids = static_cast<int64_t>(class_nodeids_.size());
       i < size_class_nodeids;
       ++i) {
    int64_t offset = tree_offsets[class_treeids_[i]];
    class_nodeids_[i] = class_nodeids_[i] - offset;
    if (class_weights_[i] < 0) {
      weights_are_all_positive_ = false;
    }
  }

  nodes_modes_.reserve(nodes_modes_names_.size());
  for (size_t i = 0, end = nodes_modes_names_.size(); i < end; ++i) {
    nodes_modes_.push_back(MakeTreeNodeMode(nodes_modes_names_[i]));
  }

  // leafnode data, these are the votes that leaves do
  using LeafNodeData = std::tuple<int64_t, int64_t, int64_t, float>;
  for (size_t i = 0, end = class_nodeids_.size(); i < end; ++i) {
    leafnodedata_.push_back(std::make_tuple(class_treeids_[i], class_nodeids_[i], class_ids_[i], class_weights_[i]));
    weights_classes_.insert(class_ids_[i]);
  }
  std::sort(std::begin(leafnodedata_), std::end(leafnodedata_), [](LeafNodeData const& t1, LeafNodeData const& t2) {
    if (std::get<0>(t1) != std::get<0>(t2))
      return std::get<0>(t1) < std::get<0>(t2);

    return std::get<1>(t1) < std::get<1>(t2);
  });
  // make an index so we can find the leafnode data quickly when evaluating
  int64_t field0 = -1;
  int64_t field1 = -1;
  for (size_t i = 0, end = leafnodedata_.size(); i < end; ++i) {
    int64_t id0 = std::get<0>(leafnodedata_[i]);
    int64_t id1 = std::get<1>(leafnodedata_[i]);
    if (id0 != field0 || id1 != field1) {
      int64_t id = id0 * kOffset_ + id1;
      auto position = static_cast<int64_t>(i);
      auto p3 = std::make_pair(id, position);
      leafdata_map_.insert(p3);
      field0 = id;
      field1 = position;
    }
  }

  // treenode ids, some are roots_, and roots_ have no parents
  std::unordered_map<int64_t, int64_t> parents;  // holds count of all who point to you
  std::unordered_map<int64_t, int64_t> indices;
  // add all the nodes to a map, and the ones that have parents are not roots_
  std::unordered_map<int64_t, int64_t>::iterator it;
  for (size_t i = 0, end = nodes_treeids_.size(); i < end; ++i) {
    // make an index to look up later
    int64_t id = nodes_treeids_[i] * kOffset_ + nodes_nodeids_[i];
    auto position = static_cast<int64_t>(i);
    auto p3 = std::make_pair(id, position);
    indices.insert(p3);
    it = parents.find(id);
    if (it == parents.end()) {
      // start counter at 0
      auto b = (int64_t)0L;
      auto p1 = std::make_pair(id, b);
      parents.insert(p1);
    }
  }
  // all true nodes arent roots_
  for (size_t i = 0, end = nodes_truenodeids_.size(); i < end; ++i) {
    if (nodes_modes_[i] == NODE_MODE::LEAF) continue;
    // they must be in the same tree
    int64_t id = nodes_treeids_[i] * kOffset_ + nodes_truenodeids_[i];
    it = parents.find(id);
    ORT_ENFORCE(it != parents.end());
    it->second++;
  }
  // all false nodes arent roots_
  for (size_t i = 0, end = nodes_falsenodeids_.size(); i < end; ++i) {
    if (nodes_modes_[i] == NODE_MODE::LEAF) continue;
    // they must be in the same tree
    int64_t id = nodes_treeids_[i] * kOffset_ + nodes_falsenodeids_[i];
    it = parents.find(id);
    ORT_ENFORCE(it != parents.end());
    it->second++;
  }
  // find all the nodes that dont have other nodes pointing at them
  for (auto& parent : parents) {
    if (parent.second == 0) {
      int64_t id = parent.first;
      it = indices.find(id);
      roots_.push_back(it->second);
    }
  }
  class_count_ = !classlabels_strings_.empty() ? classlabels_strings_.size() : classlabels_int64s_.size();
  using_strings_ = !classlabels_strings_.empty();
  ORT_ENFORCE(base_values_.empty() ||
              base_values_.size() == static_cast<size_t>(class_count_) ||
              base_values_.size() == weights_classes_.size());
}

template <typename T>
common::Status TreeEnsembleClassifier<T>::Compute(OpKernelContext* context) const {
  const Tensor& X = *context->Input<Tensor>(0);
  const TensorShape& x_shape = X.Shape();
  vector<int64_t> x_dims = x_shape.GetDims();
  if (x_dims.empty()) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "X dims is empty.");
  }

  int64_t stride = x_dims.size() == 1 ? x_dims[0] : x_dims[1];  // TODO(task 495): how does this work in the case of 3D tensors?
  int64_t N = x_dims.size() == 1 ? 1 : x_dims[0];
  Tensor* Y = context->Output(0, TensorShape({N}));
  auto* Z = context->Output(1, TensorShape({N, class_count_}));

  int64_t zindex = 0;
  const T* x_data = X.template Data<T>();

  // for each class
  std::vector<float> scores;
  scores.reserve(class_count_);
  for (int64_t i = 0; i < N; ++i) {
    scores.clear();
    int64_t current_weight_0 = i * stride;
    std::map<int64_t, float> classes;
    // fill in base values, this might be empty but that is ok
    for (int64_t k = 0, end = static_cast<int64_t>(base_values_.size()); k < end; ++k) {
      auto p1 = std::make_pair<int64_t&, const float&>(k, base_values_[k]);
      classes.insert(p1);
    }
    // walk each tree from its root
    for (size_t j = 0, end = roots_.size(); j < end; ++j) {
      ORT_RETURN_IF_ERROR(ProcessTreeNode(classes, roots_[j], x_data, current_weight_0));
    }
    float maxweight = 0.f;
    int64_t maxclass = -1;
    // write top class
    int write_additional_scores = -1;
    if (class_count_ > 2) {
      for (auto& classe : classes) {
        if (maxclass == -1 || classe.second > maxweight) {
          maxclass = classe.first;
          maxweight = classe.second;
        }
      }
      if (using_strings_) {
        Y->template MutableData<std::string>()[i] = classlabels_strings_[maxclass];
      } else {
        Y->template MutableData<int64_t>()[i] = classlabels_int64s_[maxclass];
      }
    } else  // binary case
    {
      maxweight = !classes.empty() ? classes[0] : 0.f;  // only 1 class
      if (using_strings_) {
        auto* y_data = Y->template MutableData<std::string>();
        if (classlabels_strings_.size() == 2 &&
            weights_are_all_positive_ &&
            maxweight > 0.5 &&
            weights_classes_.size() == 1) {
          y_data[i] = classlabels_strings_[1];  // positive label
          write_additional_scores = 0;
        } else if (classlabels_strings_.size() == 2 &&
                   weights_are_all_positive_ &&
                   maxweight <= 0.5 &&
                   weights_classes_.size() == 1) {
          y_data[i] = classlabels_strings_[0];  // negative label
          write_additional_scores = 1;
        } else if (classlabels_strings_.size() == 2 &&
                   maxweight > 0 &&
                   !weights_are_all_positive_ && weights_classes_.size() == 1) {
          y_data[i] = classlabels_strings_[1];  // pos label
          write_additional_scores = 2;
        } else if (classlabels_strings_.size() == 2 &&
                   maxweight <= 0 &&
                   !weights_are_all_positive_ &&
                   weights_classes_.size() == 1) {
          y_data[i] = classlabels_strings_[0];  // neg label
          write_additional_scores = 3;
        } else if (maxweight > 0) {
          y_data[i] = "1";  // positive label
        } else {
          y_data[i] = "0";  // negative label
        }
      } else {
        auto* y_data = Y->template MutableData<int64_t>();
        if (classlabels_int64s_.size() == 2 &&
            weights_are_all_positive_ &&
            maxweight > 0.5 &&
            weights_classes_.size() == 1) {
          y_data[i] = classlabels_int64s_[1];  // positive label
          write_additional_scores = 0;
        } else if (classlabels_int64s_.size() == 2 &&
                   weights_are_all_positive_ &&
                   maxweight <= 0.5 &&
                   weights_classes_.size() == 1) {
          y_data[i] = classlabels_int64s_[0];  // negative label
          write_additional_scores = 1;
        } else if (classlabels_int64s_.size() == 2 &&
                   maxweight > 0 &&
                   !weights_are_all_positive_ &&
                   weights_classes_.size() == 1) {
          y_data[i] = classlabels_int64s_[1];  // pos label
          write_additional_scores = 2;
        } else if (classlabels_int64s_.size() == 2 &&
                   maxweight <= 0 &&
                   !weights_are_all_positive_ &&
                   weights_classes_.size() == 1) {
          y_data[i] = classlabels_int64s_[0];  // neg label
          write_additional_scores = 3;
        } else if (maxweight > 0) {
          y_data[i] = 1;  // positive label
        } else {
          y_data[i] = 0;  // negative label
        }
      }
    }
    // write float values, might not have all the classes in the output yet
    // for example a 10 class case where we only found 2 classes in the leaves
    if (weights_classes_.size() == static_cast<size_t>(class_count_)) {
      for (int64_t k = 0; k < class_count_; ++k) {
        auto it_classes = classes.find(k);
        if (it_classes != classes.end()) {
          scores.push_back(it_classes->second);
        } else {
          scores.push_back(0.f);
        }
      }
    } else {
      for (auto& classe : classes) {
        scores.push_back(classe.second);
      }
    }
    write_scores(scores, post_transform_, zindex, Z, write_additional_scores);
    zindex += scores.size();
  }  // for every batch
  return Status::OK();
}

template <typename T>
common::Status TreeEnsembleClassifier<T>::ProcessTreeNode(std::map<int64_t, float>& classes,
                                                          int64_t treeindex,
                                                          const T* x_data,
                                                          int64_t feature_base) const {
  // walk down tree to the leaf
  auto mode = static_cast<NODE_MODE>(nodes_modes_[treeindex]);
  int64_t loopcount = 0;
  int64_t root = treeindex;
  while (mode != NODE_MODE::LEAF) {
    T val = x_data[feature_base + nodes_featureids_[treeindex]];
    bool tracktrue = true;
    if (missing_tracks_true_.size() != nodes_truenodeids_.size()) {
      tracktrue = false;
    } else {
      tracktrue = missing_tracks_true_[treeindex] && std::isnan(static_cast<float>(val));
    }
    float threshold = nodes_values_[treeindex];
    switch (mode) {
      case NODE_MODE::BRANCH_LEQ:
        treeindex = val <= threshold || tracktrue ? nodes_truenodeids_[treeindex] : nodes_falsenodeids_[treeindex];
        break;
      case NODE_MODE::BRANCH_LT:
        treeindex = val < threshold || tracktrue ? nodes_truenodeids_[treeindex] : nodes_falsenodeids_[treeindex];
        break;
      case NODE_MODE::BRANCH_GTE:
        treeindex = val >= threshold || tracktrue ? nodes_truenodeids_[treeindex] : nodes_falsenodeids_[treeindex];
        break;
      case NODE_MODE::BRANCH_GT:
        treeindex = val > threshold || tracktrue ? nodes_truenodeids_[treeindex] : nodes_falsenodeids_[treeindex];
        break;
      case NODE_MODE::BRANCH_EQ:
        treeindex = val == threshold || tracktrue ? nodes_truenodeids_[treeindex] : nodes_falsenodeids_[treeindex];
        break;
      case NODE_MODE::BRANCH_NEQ:
        treeindex = val != threshold || tracktrue ? nodes_truenodeids_[treeindex] : nodes_falsenodeids_[treeindex];
        break;
      default: {
        std::ostringstream err_msg;
        err_msg << "Invalid mode of value: " << static_cast<std::underlying_type<NODE_MODE>::type>(mode);
        return Status(ONNXRUNTIME, INVALID_ARGUMENT, err_msg.str());
      }
    }
    ORT_ENFORCE(treeindex >= 0);
    treeindex = treeindex + root;
    mode = static_cast<NODE_MODE>(nodes_modes_[treeindex]);
    loopcount++;
    if (loopcount > kMaxTreeDepth_) break;
  }
  // should be at leaf
  int64_t id = nodes_treeids_[treeindex] * kOffset_ + nodes_nodeids_[treeindex];
  auto it_lp = leafdata_map_.find(id);
  if (it_lp == leafdata_map_.end()) {  // if not found, simply return
    return Status::OK();
  }
  int64_t index = it_lp->second;
  int64_t treeid = std::get<0>(leafnodedata_[index]);
  int64_t nodeid = std::get<1>(leafnodedata_[index]);
  while (treeid == nodes_treeids_[treeindex] && nodeid == nodes_nodeids_[treeindex]) {
    int64_t classid = std::get<2>(leafnodedata_[index]);
    float weight = std::get<3>(leafnodedata_[index]);
    std::map<int64_t, float>::iterator it_classes;
    it_classes = classes.find(classid);
    if (it_classes != classes.end()) {
      it_classes->second += weight;
    } else {
      auto p1 = std::make_pair(classid, weight);
      classes.insert(p1);
    }
    ++index;
    // some tree node will be last
    if (index >= static_cast<int64_t>(leafnodedata_.size())) {
      break;
    }
    treeid = std::get<0>(leafnodedata_[index]);
    nodeid = std::get<1>(leafnodedata_[index]);
  }
  return Status::OK();
}
}  // namespace ml
}  // namespace onnxruntime
