// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "ml_common.h"

namespace onnxruntime {
namespace ml {
template <typename T>
class TreeEnsembleClassifier final : public OpKernel {
 public:
  explicit TreeEnsembleClassifier(const OpKernelInfo& info);
  common::Status Compute(OpKernelContext* context) const override;

 private:
  void Initialize();
  common::Status ProcessTreeNode(std::map<int64_t, float>& classes,
                                 int64_t treeindex,
                                 const T* x_data,
                                 int64_t feature_base) const;

  std::vector<int64_t> nodes_treeids_;
  std::vector<int64_t> nodes_nodeids_;
  std::vector<int64_t> nodes_featureids_;
  std::vector<float> nodes_values_;
  std::vector<float> nodes_hitrates_;
  std::vector<std::string> nodes_modes_names_;
  std::vector<NODE_MODE> nodes_modes_;
  std::vector<int64_t> nodes_truenodeids_;
  std::vector<int64_t> nodes_falsenodeids_;
  std::vector<int64_t> missing_tracks_true_;  // no bool type

  std::vector<int64_t> class_nodeids_;
  std::vector<int64_t> class_treeids_;
  std::vector<int64_t> class_ids_;
  std::vector<float> class_weights_;
  int64_t class_count_;
  std::set<int64_t> weights_classes_;

  std::vector<float> base_values_;
  std::vector<std::string> classlabels_strings_;
  std::vector<int64_t> classlabels_int64s_;
  bool using_strings_;

  std::vector<std::tuple<int64_t, int64_t, int64_t, float>> leafnodedata_;
  std::unordered_map<int64_t, int64_t> leafdata_map_;
  std::vector<int64_t> roots_;
  const int64_t kOffset_ = 4000000000L;
  const int64_t kMaxTreeDepth_ = 1000;
  POST_EVAL_TRANSFORM post_transform_;
  bool weights_are_all_positive_;
};
}  // namespace ml
}  // namespace onnxruntime
