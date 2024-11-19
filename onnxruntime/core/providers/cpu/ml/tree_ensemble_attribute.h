// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/inlined_containers.h"
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "ml_common.h"
#include "tree_ensemble_helper.h"
#include <vector>

namespace onnxruntime {
namespace ml {
namespace detail {

inline bool _isnan_(float x) { return std::isnan(x); }
inline bool _isnan_(double x) { return std::isnan(x); }
inline bool _isnan_(int64_t) { return false; }
inline bool _isnan_(int32_t) { return false; }

template <typename ThresholdType>
struct TreeEnsembleAttributesV3 {
  TreeEnsembleAttributesV3() {}
  TreeEnsembleAttributesV3(const OpKernelInfo& info, bool classifier) {
#if !defined(ORT_MINIMAL_BUILD)
    ORT_THROW_IF_ERROR(GetVectorAttrsOrDefault(info, "base_values_as_tensor", base_values_as_tensor));
    ORT_THROW_IF_ERROR(GetVectorAttrsOrDefault(info, "nodes_hitrates_as_tensor", nodes_hitrates_as_tensor));
    ORT_THROW_IF_ERROR(GetVectorAttrsOrDefault(info, "nodes_values_as_tensor", nodes_values_as_tensor));
    if (classifier) {
      ORT_THROW_IF_ERROR(GetVectorAttrsOrDefault(info, "class_weights_as_tensor", target_class_weights_as_tensor));
    } else {
      ORT_THROW_IF_ERROR(GetVectorAttrsOrDefault(info, "target_weights_as_tensor", target_class_weights_as_tensor));
    }
#endif

    aggregate_function = info.GetAttrOrDefault<std::string>("aggregate_function", "SUM");
    base_values = info.GetAttrsOrDefault<float>("base_values");
    nodes_falsenodeids = info.GetAttrsOrDefault<int64_t>("nodes_falsenodeids");
    nodes_featureids = info.GetAttrsOrDefault<int64_t>("nodes_featureids");
    nodes_missing_value_tracks_true = info.GetAttrsOrDefault<int64_t>("nodes_missing_value_tracks_true");

    std::vector<std::string> nodes_modes_string = info.GetAttrsOrDefault<std::string>("nodes_modes");
    nodes_modes.reserve(nodes_modes_string.size());
    for (auto s : nodes_modes_string) {
      nodes_modes.emplace_back(MakeTreeNodeMode(s));
    }

    nodes_nodeids = info.GetAttrsOrDefault<int64_t>("nodes_nodeids");
    nodes_treeids = info.GetAttrsOrDefault<int64_t>("nodes_treeids");
    nodes_truenodeids = info.GetAttrsOrDefault<int64_t>("nodes_truenodeids");
    nodes_values = info.GetAttrsOrDefault<float>("nodes_values");
    post_transform = info.GetAttrOrDefault<std::string>("post_transform", "NONE");

    if (classifier) {
      target_class_ids = info.GetAttrsOrDefault<int64_t>("class_ids");
      target_class_nodeids = info.GetAttrsOrDefault<int64_t>("class_nodeids");
      target_class_treeids = info.GetAttrsOrDefault<int64_t>("class_treeids");
      target_class_weights = info.GetAttrsOrDefault<float>("class_weights");
      classlabels_strings = info.GetAttrsOrDefault<std::string>("classlabels_strings");
      classlabels_int64s = info.GetAttrsOrDefault<int64_t>("classlabels_int64s");
      n_targets_or_classes = classlabels_strings.empty() ? classlabels_int64s.size()
                                                         : classlabels_strings.size();
    } else {
      n_targets_or_classes = info.GetAttrOrDefault<int64_t>("n_targets", 0);
      target_class_ids = info.GetAttrsOrDefault<int64_t>("target_ids");
      target_class_nodeids = info.GetAttrsOrDefault<int64_t>("target_nodeids");
      target_class_treeids = info.GetAttrsOrDefault<int64_t>("target_treeids");
      target_class_weights = info.GetAttrsOrDefault<float>("target_weights");

      ORT_ENFORCE(n_targets_or_classes > 0);
      ORT_ENFORCE(nodes_falsenodeids.size() == nodes_featureids.size());
      ORT_ENFORCE(nodes_falsenodeids.size() == nodes_modes_string.size());
      ORT_ENFORCE(nodes_falsenodeids.size() == nodes_nodeids.size());
      ORT_ENFORCE(nodes_falsenodeids.size() == nodes_treeids.size());
      ORT_ENFORCE(nodes_falsenodeids.size() == nodes_truenodeids.size());
      ORT_ENFORCE(nodes_falsenodeids.size() == nodes_values.size() ||
                  nodes_falsenodeids.size() == nodes_values_as_tensor.size());
      ORT_ENFORCE(target_class_ids.size() == target_class_nodeids.size());
      ORT_ENFORCE(target_class_ids.size() == target_class_treeids.size());
      ORT_ENFORCE(target_class_weights.empty() || target_class_ids.size() == target_class_weights.size());
      ORT_ENFORCE(base_values.empty() || base_values_as_tensor.empty());
      ORT_ENFORCE(nodes_hitrates.empty() || nodes_hitrates_as_tensor.empty());
      ORT_ENFORCE(nodes_values.empty() || nodes_values_as_tensor.empty());
      ORT_ENFORCE(target_class_weights.empty() || target_class_weights_as_tensor.empty());
      ORT_ENFORCE(nodes_modes_string.size() < std::numeric_limits<uint32_t>::max());
    }
  }

  std::string aggregate_function;
  std::vector<float> base_values;
  std::vector<ThresholdType> base_values_as_tensor;
  int64_t n_targets_or_classes;
  std::vector<int64_t> nodes_falsenodeids;
  std::vector<int64_t> nodes_featureids;
  std::vector<float> nodes_hitrates;
  std::vector<ThresholdType> nodes_hitrates_as_tensor;
  std::vector<int64_t> nodes_missing_value_tracks_true;
  std::vector<NODE_MODE_ONNX> nodes_modes;
  std::vector<int64_t> nodes_nodeids;
  std::vector<int64_t> nodes_treeids;
  std::vector<int64_t> nodes_truenodeids;
  std::vector<float> nodes_values;
  std::vector<ThresholdType> nodes_values_as_tensor;
  std::string post_transform;
  std::vector<int64_t> target_class_ids;
  std::vector<int64_t> target_class_nodeids;
  std::vector<int64_t> target_class_treeids;
  std::vector<float> target_class_weights;
  std::vector<ThresholdType> target_class_weights_as_tensor;
  std::vector<std::string> classlabels_strings;
  std::vector<int64_t> classlabels_int64s;
  std::vector<int64_t> class_labels;
};

template <typename ThresholdType>
struct TreeEnsembleAttributesV5 {
  TreeEnsembleAttributesV5() {}
  TreeEnsembleAttributesV5(const OpKernelInfo& info) {
#if !defined(ORT_MINIMAL_BUILD)
    std::vector<uint8_t> nodes_modes_i;
    ORT_THROW_IF_ERROR(GetVectorAttrsOrDefault(info, "leaf_weights", leaf_weights));
    ORT_THROW_IF_ERROR(GetVectorAttrsOrDefault(info, "membership_values", membership_values));
    ORT_THROW_IF_ERROR(GetVectorAttrsOrDefault(info, "nodes_hitrates", nodes_hitrates));
    ORT_THROW_IF_ERROR(GetVectorAttrsOrDefault(info, "nodes_modes", nodes_modes_i));
    ORT_THROW_IF_ERROR(GetVectorAttrsOrDefault(info, "nodes_splits", nodes_splits));
    nodes_modes.reserve(nodes_modes.size());
    for (auto i : nodes_modes_i) {
      nodes_modes.push_back(static_cast<NODE_MODE_ONNX>(i));
    }
#else
    // GetVectorAttrsOrDefault is not part of the minimal build.
    // As a result, TreeEnsemble v5 cannot be available in this build.
    ORT_THROW("TreeEnsemble(ai.onnx.ml==5) is not supported with the minimal build.");
#endif

    aggregate_function = info.GetAttrOrDefault<int64_t>("aggregate_function", 1);
    leaf_targetids = info.GetAttrsOrDefault<int64_t>("leaf_targetids");
    n_targets = info.GetAttrOrDefault<int64_t>("n_targets", 0);
    nodes_falseleafs = info.GetAttrsOrDefault<int64_t>("nodes_falseleafs");
    nodes_falsenodeids = info.GetAttrsOrDefault<int64_t>("nodes_falsenodeids");
    nodes_featureids = info.GetAttrsOrDefault<int64_t>("nodes_featureids");
    nodes_missing_value_tracks_true = info.GetAttrsOrDefault<int64_t>("nodes_missing_value_tracks_true");
    nodes_trueleafs = info.GetAttrsOrDefault<int64_t>("nodes_trueleafs");
    nodes_truenodeids = info.GetAttrsOrDefault<int64_t>("nodes_truenodeids");
    post_transform = info.GetAttrOrDefault<int64_t>("post_transform", 0);
    tree_roots = info.GetAttrsOrDefault<int64_t>("tree_roots");
  }

  void convert_to_v3(TreeEnsembleAttributesV3<ThresholdType>& output) const {
    // Doing all transformations to get the old format.
    output.n_targets_or_classes = n_targets;
    output.aggregate_function = aggregateFunctionToString();
    output.post_transform = postTransformToString();
    std::vector<std::vector<ThresholdType>> membership_values_by_id;
    getMembershipValuesById(membership_values_by_id);
    transformInputAllTrees(output, membership_values_by_id);
  }

  int64_t aggregate_function;
  std::vector<int64_t> leaf_targetids;
  std::vector<ThresholdType> leaf_weights;
  std::vector<ThresholdType> membership_values;
  int64_t n_targets;
  std::vector<int64_t> nodes_falseleafs;
  std::vector<int64_t> nodes_falsenodeids;
  std::vector<int64_t> nodes_featureids;
  std::vector<ThresholdType> nodes_hitrates;
  std::vector<int64_t> nodes_missing_value_tracks_true;
  std::vector<NODE_MODE_ONNX> nodes_modes;
  std::vector<ThresholdType> nodes_splits;
  std::vector<int64_t> nodes_trueleafs;
  std::vector<int64_t> nodes_truenodeids;
  int64_t post_transform;
  std::vector<int64_t> tree_roots;

 private:
  // `membership_values` are seperated by NAN for different nodes
  // It is more convenient to preserve the values for each node in a vector
  // The vector would be empty for nodes that are not `BRANCH_MEMBER`
  void getMembershipValuesById(std::vector<std::vector<ThresholdType>>& membership_values_by_id) const {
    membership_values_by_id.clear();
    membership_values_by_id.reserve(nodes_modes.size());

    size_t curr_id = 0;
    for (const auto node_mode : nodes_modes) {
      membership_values_by_id.emplace_back();
      if (node_mode != NODE_MODE_ONNX::BRANCH_MEMBER) {
        continue;
      }

      while (curr_id < membership_values.size() && !_isnan_(membership_values[curr_id])) {
        membership_values_by_id.back().push_back(membership_values[curr_id++]);
      }
      curr_id++;
    }
  }

  std::string aggregateFunctionToString() const {
    switch (aggregate_function) {
      case static_cast<int64_t>(AGGREGATE_FUNCTION::AVERAGE):
        return "AVERAGE";
      case static_cast<int64_t>(AGGREGATE_FUNCTION::SUM):
        return "SUM";
      case static_cast<int64_t>(AGGREGATE_FUNCTION::MIN):
        return "MIN";
      case static_cast<int64_t>(AGGREGATE_FUNCTION::MAX):
        return "MAX";
      default:
        ORT_THROW("Unknown value for aggregate_function.");
    }
  }

  std::string postTransformToString() const {
    switch (post_transform) {
      case static_cast<int64_t>(POST_EVAL_TRANSFORM::NONE):
        return "NONE";
      case static_cast<int64_t>(POST_EVAL_TRANSFORM::SOFTMAX):
        return "SOFTMAX";
      case static_cast<int64_t>(POST_EVAL_TRANSFORM::LOGISTIC):
        return "LOGISTIC";
      case static_cast<int64_t>(POST_EVAL_TRANSFORM::SOFTMAX_ZERO):
        return "SOFTMAX_ZERO";
      case static_cast<int64_t>(POST_EVAL_TRANSFORM::PROBIT):
        return "PROBIT";
      default:
        ORT_THROW("Unknown value for post_transform.");
    }
  }

  int64_t transformInputOneTree(
      const size_t curr_id, const int64_t curr_treeid, const int64_t curr_nodeid, const size_t curr_membership_value_id,
      const bool is_leaf, std::vector<std::vector<ThresholdType>>& membership_values_by_id,
      TreeEnsembleAttributesV3<ThresholdType>& output) const {
    output.nodes_nodeids.push_back(curr_nodeid);
    output.nodes_treeids.push_back(curr_treeid);

    if (is_leaf) {
      output.nodes_modes.push_back(NODE_MODE_ONNX::LEAF);
      output.target_class_ids.push_back(leaf_targetids[curr_id]);
      output.target_class_nodeids.push_back(curr_nodeid);
      output.target_class_treeids.push_back(curr_treeid);
      output.target_class_weights_as_tensor.push_back(leaf_weights[curr_id]);

      // the below are irrelevant for a `LEAF`
      output.nodes_featureids.push_back(0);
      output.nodes_truenodeids.push_back(0);
      output.nodes_falsenodeids.push_back(0);
      output.nodes_values_as_tensor.push_back(0);
      if (!nodes_hitrates.empty()) {
        output.nodes_hitrates.push_back(0);
      }
      if (!nodes_missing_value_tracks_true.empty()) {
        output.nodes_missing_value_tracks_true.push_back(0);
      }

      return curr_nodeid;
    }

    output.nodes_featureids.push_back(nodes_featureids[curr_id]);
    if (!nodes_hitrates.empty()) {
      output.nodes_hitrates_as_tensor.push_back(nodes_hitrates[curr_id]);
    }
    if (!nodes_missing_value_tracks_true.empty()) {
      output.nodes_missing_value_tracks_true.push_back(nodes_missing_value_tracks_true[curr_id]);
    }

    // unroll `BRANCH_MEMBER` to a chain of `BRANCH_EQ`
    if (nodes_modes[curr_id] == NODE_MODE_ONNX::BRANCH_MEMBER) {
      output.nodes_modes.push_back(NODE_MODE_ONNX::BRANCH_EQ);
      output.nodes_values_as_tensor.push_back(membership_values_by_id[curr_id][curr_membership_value_id]);
    } else {
      output.nodes_modes.push_back(nodes_modes[curr_id]);
      output.nodes_values_as_tensor.push_back(nodes_splits[curr_id]);
    }

    size_t falsenodeid_id = output.nodes_falsenodeids.size();
    output.nodes_falsenodeids.push_back(0);  // change after pushing truenode subtree

    int64_t true_nodeid = curr_nodeid + 1;
    output.nodes_truenodeids.push_back(true_nodeid);
    true_nodeid = transformInputOneTree(onnxruntime::narrow<size_t>(nodes_truenodeids[curr_id]),
                                        curr_treeid, true_nodeid, 0U, nodes_trueleafs[curr_id] != 0,
                                        membership_values_by_id, output);

    int64_t false_nodeid = true_nodeid + 1;
    output.nodes_falsenodeids[falsenodeid_id] = false_nodeid;

    // if node is `BRANCH_MEMBER` we are unrolling the `membership_values` for that node
    // therefore if the value is not the last, the `falsenode_id` must be pointing to the "same" node with a different membership value
    // so in that case we are only moving the pointer for `membership_values`
    //
    // otherwise, the `falsenode_id` is pointing to the real falsenode subtree
    if (nodes_modes[curr_id] == NODE_MODE_ONNX::BRANCH_MEMBER &&
        curr_membership_value_id + 1 < membership_values_by_id[curr_id].size()) {
      false_nodeid = transformInputOneTree(curr_id, curr_treeid, false_nodeid, curr_membership_value_id + 1, false,
                                           membership_values_by_id, output);
    } else {
      false_nodeid = transformInputOneTree(onnxruntime::narrow<size_t>(nodes_falsenodeids[curr_id]),
                                           curr_treeid, false_nodeid, 0U, nodes_falseleafs[curr_id] != 0,
                                           membership_values_by_id, output);
    }
    return false_nodeid;
  }

  void transformInputAllTrees(TreeEnsembleAttributesV3<ThresholdType>& output,
                              std::vector<std::vector<ThresholdType>>& membership_values_by_id) const {
    int64_t curr_treeid = 0;
    for (const int64_t& tree_root : tree_roots) {
      size_t tree_root_size_t = onnxruntime::narrow<size_t>(tree_root);
      transformInputOneTree(tree_root_size_t, curr_treeid, 0, 0U,
                            nodes_falsenodeids[tree_root_size_t] == nodes_truenodeids[tree_root_size_t],
                            membership_values_by_id, output);
      curr_treeid++;
    }
  }
};

}  // namespace detail
}  // namespace ml
}  // namespace onnxruntime
