// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/inlined_containers.h"
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "ml_common.h"
#include "tree_ensemble_helper.h"
#include <unordered_map>
#include <stack>
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
  TreeEnsembleAttributesV3() : n_targets_or_classes(0) {}
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
  // For categorical features, this container stores set of members for every rule
  // MEMBERSHIP_BIGSET. The threshold or node value (stored in value_or_unique_weight)
  // is casted into an integer, an index, corresponding to the set in `bigsets`.
  std::vector<std::vector<ThresholdType>> bigsets;
};

template <typename ThresholdType>
struct TreeEnsembleAttributesV5 {
  TreeEnsembleAttributesV5() : aggregate_function(1), n_targets(0), post_transform(0) {}
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

  void transformInputOneTree(
      const size_t root_id,
      const int64_t curr_treeid,
      const int64_t root_nodeid,
      const bool root_is_leaf,
      std::vector<std::vector<ThresholdType>>& membership_values_by_id,
      TreeEnsembleAttributesV3<ThresholdType>& output) const {
    // This function switches from V5 to V3. It replaces membership values with a chain of `BRANCH_EQ` nodes
    // The inner implementation was recursive but it is now iterative to avoid stack overflow.
    // It relies on a stack to keep track of the current state.
    struct StackFrame {
      size_t curr_id;       // current node id being processed
      int64_t curr_nodeid;  // current node id in the tree
      bool is_leaf;         // is a leaf node
      int64_t placeholder_to_update;
    };

    std::stack<StackFrame> stack;
    std::unordered_map<int64_t, int64_t> false_nodeid_update_pos;
    int64_t last_curr_nodeid = root_nodeid;

    stack.push({root_id, root_nodeid, root_is_leaf, -1});

    while (!stack.empty()) {
      auto& frame = stack.top();

      const size_t curr_id = frame.curr_id;
      int64_t curr_nodeid = frame.curr_nodeid;
      const bool is_leaf = frame.is_leaf;
      const int64_t placeholder_to_update = frame.placeholder_to_update;

      stack.pop();
      if (curr_nodeid == -1) {
        curr_nodeid = last_curr_nodeid + 1;
      }
      last_curr_nodeid = curr_nodeid;

      if (placeholder_to_update != -1) {
        ORT_ENFORCE(output.nodes_falsenodeids[static_cast<size_t>(placeholder_to_update)] == -1,
                    "Placeholder for a false branch was already updated, placeholder=", placeholder_to_update,
                    ", curr_nodeid=", curr_nodeid);
        output.nodes_falsenodeids[static_cast<size_t>(placeholder_to_update)] = curr_nodeid;
      }

      if (is_leaf) {
        // leaf node
        output.nodes_nodeids.push_back(curr_nodeid);
        output.nodes_treeids.push_back(curr_treeid);
        output.nodes_modes.push_back(NODE_MODE_ONNX::LEAF);
        output.target_class_ids.push_back(leaf_targetids[curr_id]);
        output.target_class_nodeids.push_back(curr_nodeid);
        output.target_class_treeids.push_back(curr_treeid);
        output.target_class_weights_as_tensor.push_back(leaf_weights[curr_id]);

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
        // If it is a leaf node, then next one is a false branch.
      } else {
        int64_t before = -1;
        bool big_set = false;
        if (nodes_modes[curr_id] == NODE_MODE_ONNX::BRANCH_MEMBER) {
          if (membership_values_by_id[curr_id].size() < 31 &&
              *std::max_element(membership_values_by_id[curr_id].begin(), membership_values_by_id[curr_id].end()) <= static_cast<ThresholdType>(30) &&
              *std::min_element(membership_values_by_id[curr_id].begin(), membership_values_by_id[curr_id].end()) >= static_cast<ThresholdType>(0)) {
            // If the set of membership values is small enough, we can unroll it into a chain of `BRANCH_EQ` nodes.
            // Then when the tree is built, the chain of `BRANCH_EQ` nodes will be merged into a single `BRANCH_MEMBER` node.
            // This optimization was implemented before ai.onnx.ml<5 and is still valid.
            // However, for very big sets, unrolling the nodes is not efficient at all.
            // In that case, we change the `BRANCH_MEMBER` node into a `BRANCH_MEMBER_BIGSET` node.
            before = output.nodes_truenodeids.size();
            for (size_t i_member = 0; i_member < membership_values_by_id[curr_id].size() - 1; ++i_member) {
              auto member = membership_values_by_id[curr_id][i_member];
              output.nodes_nodeids.push_back(curr_nodeid);
              output.nodes_treeids.push_back(curr_treeid);
              output.nodes_featureids.push_back(nodes_featureids[curr_id]);
              if (!nodes_hitrates.empty()) {
                output.nodes_hitrates_as_tensor.push_back(nodes_hitrates[curr_id]);
              }
              if (!nodes_missing_value_tracks_true.empty()) {
                output.nodes_missing_value_tracks_true.push_back(nodes_missing_value_tracks_true[curr_id]);
              }

              output.nodes_modes.push_back(NODE_MODE_ONNX::BRANCH_EQ);
              output.nodes_values_as_tensor.push_back(member);

              output.nodes_falsenodeids.push_back(curr_nodeid + 1);  // false node is placed first
              output.nodes_truenodeids.push_back(-1);                // placeholder, we do not know yet the true node id but it should be the same
              ++curr_nodeid;
            }
          } else {
            // This a very big set, we change the `BRANCH_MEMBER` node into a `BRANCH_MEMBER_BIGSET` node.
            big_set = true;
          }
        }

        // not a leaf node, not a mode `BRANCH_MEMBER` or the last value of the set
        output.nodes_nodeids.push_back(curr_nodeid);
        output.nodes_treeids.push_back(curr_treeid);
        output.nodes_featureids.push_back(nodes_featureids[curr_id]);
        if (!nodes_hitrates.empty()) {
          output.nodes_hitrates_as_tensor.push_back(nodes_hitrates[curr_id]);
        }
        if (!nodes_missing_value_tracks_true.empty()) {
          output.nodes_missing_value_tracks_true.push_back(nodes_missing_value_tracks_true[curr_id]);
        }

        output.nodes_modes.push_back(nodes_modes[curr_id] == NODE_MODE_ONNX::BRANCH_MEMBER
                                         ? (big_set ? NODE_MODE_ONNX::BRANCH_MEMBER_BIGSET : NODE_MODE_ONNX::BRANCH_EQ)
                                         : nodes_modes[curr_id]);
        output.nodes_values_as_tensor.push_back(nodes_modes[curr_id] == NODE_MODE_ONNX::BRANCH_MEMBER
                                                    ? (big_set
                                                           ? output.bigsets.size()
                                                           : membership_values_by_id[curr_id][membership_values_by_id[curr_id].size() - 1])
                                                    : (nodes_modes[curr_id] == NODE_MODE_ONNX::BRANCH_MEMBER_BIGSET
                                                           ? output.bigsets.size()
                                                           : nodes_splits[curr_id]));

        output.nodes_falsenodeids.push_back(-1);              // false node is placed first
        output.nodes_truenodeids.push_back(curr_nodeid + 1);  // placeholder, we do not know yet the true node id

        if (before != -1) {
          // If we unrolled the `BRANCH_MEMBER` nodes, we need to update the true node ids of the previous nodes.
          for (size_t i = static_cast<size_t>(before); i < output.nodes_truenodeids.size() - 1; ++i) {
            output.nodes_truenodeids[i] = curr_nodeid + 1;
          }
        }

        if (big_set || nodes_modes[curr_id] == NODE_MODE_ONNX::BRANCH_MEMBER_BIGSET) {
          // If it is a big set, we need to store the membership values in a separate vector.
          output.bigsets.push_back(membership_values_by_id[curr_id]);
        }

        stack.push({onnxruntime::narrow<size_t>(nodes_falsenodeids[curr_id]), -1, nodes_falseleafs[curr_id] != 0, static_cast<int64_t>(output.nodes_falsenodeids.size()) - 1});
        stack.push({onnxruntime::narrow<size_t>(nodes_truenodeids[curr_id]), curr_nodeid + 1, nodes_trueleafs[curr_id] != 0, -1});
      }
    }
    // We check no placeholder was left behind.
    for (const auto& falsenodeid : output.nodes_falsenodeids) {
      ORT_ENFORCE(falsenodeid >= 0, "A placeholder for a false branch was not replaced, falsenodeid=", falsenodeid);
    }
  }

  void transformInputAllTrees(TreeEnsembleAttributesV3<ThresholdType>& output,
                              std::vector<std::vector<ThresholdType>>& membership_values_by_id) const {
    int64_t curr_treeid = 0;
    for (const int64_t& tree_root : tree_roots) {
      size_t tree_root_size_t = onnxruntime::narrow<size_t>(tree_root);
      transformInputOneTree(tree_root_size_t, curr_treeid, 0,
                            nodes_falsenodeids[tree_root_size_t] == nodes_truenodeids[tree_root_size_t],
                            membership_values_by_id, output);
      curr_treeid++;
    }
  }
};

}  // namespace detail
}  // namespace ml
}  // namespace onnxruntime
