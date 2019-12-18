// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "ml_common.h"

namespace onnxruntime {
namespace ml {
template <typename T>
class TreeEnsembleRegressor final : public OpKernel {
 public:
  explicit TreeEnsembleRegressor(const OpKernelInfo& info);
  common::Status Compute(OpKernelContext* context) const override;
  ~TreeEnsembleRegressor();

 private:
  struct TreeNodeElementId {
    int tree_id;
    int node_id;
    inline bool operator==(const TreeNodeElementId& xyz) const {
      return (tree_id == xyz.tree_id) && (node_id == xyz.node_id);
    }
    inline bool operator<(const TreeNodeElementId& xyz) const {
      return ((tree_id < xyz.tree_id) || (tree_id == xyz.tree_id && node_id < xyz.node_id));
    }
  };

  struct SparseValue {
    int64_t i;
    T value;
  };

  enum MissingTrack {
    NONE,
    TRUE,
    FALSE
  };

  struct TreeNodeElement {
    TreeNodeElementId id;
    int feature_id;
    T value;
    T hitrates;
    NODE_MODE mode;
    TreeNodeElement* truenode;
    TreeNodeElement* falsenode;
    MissingTrack missing_tracks;

    std::vector<SparseValue> weights;
  };

  std::vector<T> base_values_;
  int64_t n_targets_;
  POST_EVAL_TRANSFORM post_transform_;
  AGGREGATE_FUNCTION aggregate_function_;
  int64_t nbnodes_;
  TreeNodeElement* nodes_;
  std::vector<TreeNodeElement*> roots_;

  int64_t max_tree_depth_;
  int64_t nbtrees_;
  bool same_mode_;
  bool has_missing_tracks_;

  common::Status ProcessTreeNode(T* predictions, TreeNodeElement* root,
                                 const T* x_data,
                                 unsigned char* has_predictions) const;
};
}  // namespace ml
}  // namespace onnxruntime
