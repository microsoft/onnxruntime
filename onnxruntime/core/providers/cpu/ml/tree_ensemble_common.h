// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "tree_ensemble_aggregator.h"
#include "core/platform/ort_mutex.h"
#include "core/platform/threadpool.h"

namespace onnxruntime {
namespace ml {
namespace detail {

template <typename ITYPE, typename OTYPE>
class TreeEnsembleCommon {
 public:
  int64_t n_targets_or_classes_;

 protected:
  std::vector<OTYPE> base_values_;
  POST_EVAL_TRANSFORM post_transform_;
  AGGREGATE_FUNCTION aggregate_function_;
  int64_t n_nodes_;
  std::vector<TreeNodeElement<OTYPE>> nodes_;
  std::vector<TreeNodeElement<OTYPE>*> roots_;

  int64_t max_tree_depth_;
  int64_t n_trees_;
  bool same_mode_;
  bool has_missing_tracks_;
  int parallel_tree_;  // starts parallelizing the computing if n_tree >= parallel_tree_ and n_rows == 1
  int parallel_N_;     // starts parallelizing the computing if n_rows >= parallel_N_

 public:
  TreeEnsembleCommon(int parallel_tree,
                     int parallel_N,
                     const std::string& aggregate_function,
                     const std::vector<OTYPE>& base_values,
                     int64_t n_targets_or_classes,
                     const std::vector<int64_t>& nodes_falsenodeids,
                     const std::vector<int64_t>& nodes_featureids,
                     const std::vector<OTYPE>& nodes_hitrates,
                     const std::vector<int64_t>& nodes_missing_value_tracks_true,
                     const std::vector<std::string>& nodes_modes,
                     const std::vector<int64_t>& nodes_nodeids,
                     const std::vector<int64_t>& nodes_treeids,
                     const std::vector<int64_t>& nodes_truenodeids,
                     const std::vector<OTYPE>& nodes_values,
                     const std::string& post_transform,
                     const std::vector<int64_t>& target_class_ids,
                     const std::vector<int64_t>& target_class_nodeids,
                     const std::vector<int64_t>& target_class_treeids,
                     const std::vector<OTYPE>& target_class_weights);

  void compute(concurrency::ThreadPool* ttp, const Tensor* X, Tensor* Z, Tensor* label) const;

 protected:
  TreeNodeElement<OTYPE>* ProcessTreeNodeLeave(
      TreeNodeElement<OTYPE>* root, const ITYPE* x_data) const;

  template <typename AGG>
  void ComputeAgg(concurrency::ThreadPool* ttp, const Tensor* X, Tensor* Z, Tensor* label, const AGG& agg) const;
};

template <typename ITYPE, typename OTYPE>
TreeEnsembleCommon<ITYPE, OTYPE>::TreeEnsembleCommon(int parallel_tree, int parallel_N,
                                                     const std::string& aggregate_function,
                                                     const std::vector<OTYPE>& base_values,
                                                     int64_t n_targets_or_classes,
                                                     const std::vector<int64_t>& nodes_falsenodeids,
                                                     const std::vector<int64_t>& nodes_featureids,
                                                     const std::vector<OTYPE>& nodes_hitrates,
                                                     const std::vector<int64_t>& nodes_missing_value_tracks_true,
                                                     const std::vector<std::string>& nodes_modes,
                                                     const std::vector<int64_t>& nodes_nodeids,
                                                     const std::vector<int64_t>& nodes_treeids,
                                                     const std::vector<int64_t>& nodes_truenodeids,
                                                     const std::vector<OTYPE>& nodes_values,
                                                     const std::string& post_transform,
                                                     const std::vector<int64_t>& target_class_ids,
                                                     const std::vector<int64_t>& target_class_nodeids,
                                                     const std::vector<int64_t>& target_class_treeids,
                                                     const std::vector<OTYPE>& target_class_weights) {
  parallel_tree_ = parallel_tree;
  parallel_N_ = parallel_N;

  ORT_ENFORCE(n_targets_or_classes > 0);
  ORT_ENFORCE(nodes_falsenodeids.size() == nodes_featureids.size());
  ORT_ENFORCE(nodes_falsenodeids.size() == nodes_modes.size());
  ORT_ENFORCE(nodes_falsenodeids.size() == nodes_nodeids.size());
  ORT_ENFORCE(nodes_falsenodeids.size() == nodes_treeids.size());
  ORT_ENFORCE(nodes_falsenodeids.size() == nodes_truenodeids.size());
  ORT_ENFORCE(nodes_falsenodeids.size() == nodes_values.size());
  ORT_ENFORCE(target_class_ids.size() == target_class_nodeids.size());
  ORT_ENFORCE(target_class_ids.size() == target_class_treeids.size());
  ORT_ENFORCE(target_class_ids.size() == target_class_treeids.size());

  aggregate_function_ = MakeAggregateFunction(aggregate_function);
  post_transform_ = MakeTransform(post_transform);
  base_values_ = base_values;
  n_targets_or_classes_ = n_targets_or_classes;
  max_tree_depth_ = 1000;

  // additional members
  std::vector<NODE_MODE> cmodes(nodes_modes.size());
  same_mode_ = true;
  int fpos = -1;
  for (size_t i = 0; i < nodes_modes.size(); ++i) {
    cmodes[i] = MakeTreeNodeMode(nodes_modes[i]);
    if (cmodes[i] == NODE_MODE::LEAF)
      continue;
    if (fpos == -1) {
      fpos = static_cast<int>(i);
      continue;
    }
    if (cmodes[i] != cmodes[fpos])
      same_mode_ = false;
  }

  // filling nodes

  n_nodes_ = nodes_treeids.size();
  nodes_.resize(n_nodes_);
  roots_.clear();
  std::map<TreeNodeElementId, TreeNodeElement<OTYPE>*> idi;
  size_t i;

  for (i = 0; i < nodes_treeids.size(); ++i) {
    TreeNodeElement<OTYPE>& node = nodes_[i];
    node.id.tree_id = static_cast<int>(nodes_treeids[i]);
    node.id.node_id = static_cast<int>(nodes_nodeids[i]);
    node.feature_id = static_cast<int>(nodes_featureids[i]);
    node.value = nodes_values[i];
    node.hitrates = i < nodes_hitrates.size() ? nodes_hitrates[i] : -1;
    node.mode = cmodes[i];
    node.is_not_leaf = node.mode != NODE_MODE::LEAF;
    node.truenode = nullptr;   // nodes_truenodeids[i];
    node.falsenode = nullptr;  // nodes_falsenodeids[i];
    node.missing_tracks = i < static_cast<size_t>(nodes_missing_value_tracks_true.size())
                              ? (nodes_missing_value_tracks_true[i] == 1
                                     ? MissingTrack::kTrue
                                     : MissingTrack::kFalse)
                              : MissingTrack::kNone;
    node.is_missing_track_true = node.missing_tracks == MissingTrack::kTrue;
    if (idi.find(node.id) != idi.end()) {
      ORT_THROW("Node ", node.id.node_id, " in tree ", node.id.tree_id, " is already there.");
    }
    idi.insert(std::pair<TreeNodeElementId, TreeNodeElement<OTYPE>*>(node.id, &node));
  }

  TreeNodeElementId coor;
  for (auto it = nodes_.begin(); it != nodes_.end(); ++it, ++i) {
    if (!it->is_not_leaf)
      continue;
    i = std::distance(nodes_.begin(), it);
    coor.tree_id = it->id.tree_id;
    coor.node_id = static_cast<int>(nodes_truenodeids[i]);

    auto found = idi.find(coor);
    if (found == idi.end()) {
      ORT_THROW("Unable to find node ", coor.tree_id, "-", coor.node_id, " (truenode).");
    }
    if (coor.node_id >= 0 && coor.node_id < n_nodes_) {
      it->truenode = found->second;
      if ((it->truenode->id.tree_id != it->id.tree_id) ||
          (it->truenode->id.node_id == it->id.node_id)) {
        ORT_THROW("One falsenode is pointing either to itself, either to another tree.");
      }
    } else
      it->truenode = nullptr;

    coor.node_id = static_cast<int>(nodes_falsenodeids[i]);
    found = idi.find(coor);
    if (found == idi.end()) {
      ORT_THROW("Unable to find node ", coor.tree_id, "-", coor.node_id, " (falsenode).");
    }
    if (coor.node_id >= 0 && coor.node_id < n_nodes_) {
      it->falsenode = found->second;
      if ((it->falsenode->id.tree_id != it->id.tree_id) ||
          (it->falsenode->id.node_id == it->id.node_id)) {
        ORT_THROW("One falsenode is pointing either to itself, either to another tree.");
      }
    } else
      it->falsenode = nullptr;
  }

  int64_t previous = -1;
  for (i = 0; i < static_cast<size_t>(n_nodes_); ++i) {
    if ((previous == -1) || (previous != nodes_[i].id.tree_id))
      roots_.push_back(&(nodes_[i]));
    previous = nodes_[i].id.tree_id;
  }

  TreeNodeElementId ind;
  SparseValue<OTYPE> w;
  for (i = 0; i < target_class_nodeids.size(); i++) {
    ind.tree_id = static_cast<int>(target_class_treeids[i]);
    ind.node_id = static_cast<int>(target_class_nodeids[i]);
    if (idi.find(ind) == idi.end()) {
      ORT_THROW("Unable to find node ", coor.tree_id, "-", coor.node_id, " (weights).");
    }
    w.i = target_class_ids[i];
    w.value = target_class_weights[i];
    idi[ind]->weights.push_back(w);
  }

  n_trees_ = roots_.size();
  has_missing_tracks_ = false;
  for (auto itm = nodes_missing_value_tracks_true.begin();
       itm != nodes_missing_value_tracks_true.end(); ++itm) {
    if (*itm) {
      has_missing_tracks_ = true;
      break;
    }
  }
}

template <typename ITYPE, typename OTYPE>
void TreeEnsembleCommon<ITYPE, OTYPE>::compute(concurrency::ThreadPool* ttp, const Tensor* X, Tensor* Z,
                                               Tensor* label) const {
  switch (aggregate_function_) {
    case AGGREGATE_FUNCTION::AVERAGE:
      ComputeAgg(
          ttp, X, Z, label,
          TreeAggregatorAverage<ITYPE, OTYPE>(
              roots_.size(), n_targets_or_classes_,
              post_transform_, base_values_));
      return;
    case AGGREGATE_FUNCTION::SUM:
      ComputeAgg(
          ttp, X, Z, label,
          TreeAggregatorSum<ITYPE, OTYPE>(
              roots_.size(), n_targets_or_classes_,
              post_transform_, base_values_));
      return;
    case AGGREGATE_FUNCTION::MIN:
      ComputeAgg(
          ttp, X, Z, label,
          TreeAggregatorMin<ITYPE, OTYPE>(
              roots_.size(), n_targets_or_classes_,
              post_transform_, base_values_));
      return;
    case AGGREGATE_FUNCTION::MAX:
      ComputeAgg(
          ttp, X, Z, label,
          TreeAggregatorMax<ITYPE, OTYPE>(
              roots_.size(), n_targets_or_classes_,
              post_transform_, base_values_));
      return;
    default:
      ORT_THROW("Unknown aggregation function in TreeEnsemble.");
  }
}

template <typename ITYPE, typename OTYPE>
template <typename AGG>
void TreeEnsembleCommon<ITYPE, OTYPE>::ComputeAgg(concurrency::ThreadPool* ttp, const Tensor* X, Tensor* Z,
                                                  Tensor* label, const AGG& agg) const {
  int64_t stride = X->Shape().NumDimensions() == 1 ? X->Shape()[0] : X->Shape()[1];
  int64_t N = X->Shape().NumDimensions() == 1 ? 1 : X->Shape()[0];

  const ITYPE* x_data = X->template Data<ITYPE>();
  OTYPE* z_data = Z->template MutableData<OTYPE>();
  int64_t* label_data = label == nullptr ? nullptr : label->template MutableData<int64_t>();

  if (n_targets_or_classes_ == 1) {
    if (N == 1) {
      ScoreValue<OTYPE> score = {0, 0};
      if (n_trees_ <= parallel_tree_) {
        for (int64_t j = 0; j < n_trees_; ++j) {
          agg.ProcessTreeNodePrediction1(score, *ProcessTreeNodeLeave(roots_[j], x_data));
        }
      } else {
        std::vector<ScoreValue<OTYPE>> scores_t(n_trees_, {0, 0});
        concurrency::ThreadPool::TryBatchParallelFor(
            ttp,
            SafeInt<int32_t>(n_trees_),
            [this, &scores_t, &agg, x_data](ptrdiff_t j) {
              agg.ProcessTreeNodePrediction1(scores_t[j], *ProcessTreeNodeLeave(roots_[j], x_data));
            },
            0);

        for (auto it = scores_t.cbegin(); it != scores_t.cend(); ++it) {
          agg.MergePrediction1(score, *it);
        }
      }

      agg.FinalizeScores1(z_data, score, label_data);
    } else {
      if (N <= parallel_N_) {
        ScoreValue<OTYPE> score;
        size_t j;

        for (int64_t i = 0; i < N; ++i) {
          score = {0, 0};
          for (j = 0; j < static_cast<size_t>(n_trees_); ++j) {
            agg.ProcessTreeNodePrediction1(score, *ProcessTreeNodeLeave(roots_[j], x_data + i * stride));
          }

          agg.FinalizeScores1(z_data + i * n_targets_or_classes_, score,
                              label_data == nullptr ? nullptr : (label_data + i));
        }
      } else {
        concurrency::ThreadPool::TryBatchParallelFor(
            ttp,
            SafeInt<int32_t>(N),
            [this, &agg, x_data, z_data, stride, label_data](ptrdiff_t i) {
              ScoreValue<OTYPE> score = {0, 0};
              for (size_t j = 0; j < static_cast<size_t>(n_trees_); ++j) {
                agg.ProcessTreeNodePrediction1(score, *ProcessTreeNodeLeave(roots_[j], x_data + i * stride));
              }

              agg.FinalizeScores1(z_data + i * n_targets_or_classes_, score,
                                  label_data == nullptr ? nullptr : (label_data + i));
            },
            0);
      }
    }
  } else {
    if (N == 1) {
      std::vector<ScoreValue<OTYPE>> scores(n_targets_or_classes_, {0, 0});
      if (n_trees_ <= parallel_tree_) {
        for (int64_t j = 0; j < n_trees_; ++j) {
          agg.ProcessTreeNodePrediction(scores, *ProcessTreeNodeLeave(roots_[j], x_data));
        }
      } else {
        // split the work into one block per thread so we can re-use the 'private_scores' vector as much as possible
        // TODO: Refine the number of threads used
        auto num_threads = std::min<int32_t>(concurrency::ThreadPool::DegreeOfParallelism(ttp), SafeInt<int32_t>(n_trees_));
        OrtMutex merge_mutex;
        concurrency::ThreadPool::TrySimpleParallelFor(
            ttp,
            num_threads,
            [this, &agg, &scores, &merge_mutex, num_threads, x_data](ptrdiff_t batch_num) {
              std::vector<ScoreValue<OTYPE>> private_scores(n_targets_or_classes_, {0, 0});
              auto work = concurrency::ThreadPool::PartitionWork(batch_num, num_threads, n_trees_);
              for (auto j = work.start; j < work.end; ++j) {
                agg.ProcessTreeNodePrediction(private_scores, *ProcessTreeNodeLeave(roots_[j], x_data));
              }

              std::lock_guard<OrtMutex> lock(merge_mutex);
              agg.MergePrediction(scores, private_scores);
            });
      }

      agg.FinalizeScores(scores, z_data, -1, label_data);
    } else {
      if (N <= parallel_N_) {
        std::vector<ScoreValue<OTYPE>> scores(n_targets_or_classes_);
        size_t j;

        for (int64_t i = 0; i < N; ++i) {
          std::fill(scores.begin(), scores.end(), ScoreValue<OTYPE>({0, 0}));
          for (j = 0; j < roots_.size(); ++j) {
            agg.ProcessTreeNodePrediction(scores, *ProcessTreeNodeLeave(roots_[j], x_data + i * stride));
          }

          agg.FinalizeScores(scores, z_data + i * n_targets_or_classes_, -1,
                             label_data == nullptr ? nullptr : (label_data + i));
        }
      } else {
        // split the work into one block per thread so we can re-use the 'scores' vector as much as possible
        // TODO: Refine the number of threads used.
        auto num_threads = std::min<int32_t>(concurrency::ThreadPool::DegreeOfParallelism(ttp), SafeInt<int32_t>(N));
        concurrency::ThreadPool::TrySimpleParallelFor(
            ttp,
            num_threads,
            [this, &agg, num_threads, x_data, z_data, label_data, N, stride](ptrdiff_t batch_num) {
              size_t j;
              std::vector<ScoreValue<OTYPE>> scores(n_targets_or_classes_);
              auto work = concurrency::ThreadPool::PartitionWork(batch_num, num_threads, N);

              for (auto i = work.start; i < work.end; ++i) {
                std::fill(scores.begin(), scores.end(), ScoreValue<OTYPE>({0, 0}));
                for (j = 0; j < roots_.size(); ++j) {
                  agg.ProcessTreeNodePrediction(scores, *ProcessTreeNodeLeave(roots_[j], x_data + i * stride));
                }

                agg.FinalizeScores(scores,
                                   z_data + i * n_targets_or_classes_, -1,
                                   label_data == nullptr ? nullptr : (label_data + i));
              }
            });
      }
    }
  }
}  // namespace detail

#define TREE_FIND_VALUE(CMP)                                         \
  if (has_missing_tracks_) {                                         \
    while (root->is_not_leaf) {                                      \
      val = x_data[root->feature_id];                                \
      root = (val CMP root->value ||                                 \
              (root->is_missing_track_true && _isnan_(val)))         \
                 ? root->truenode                                    \
                 : root->falsenode;                                  \
    }                                                                \
  } else {                                                           \
    while (root->is_not_leaf) {                                      \
      val = x_data[root->feature_id];                                \
      root = val CMP root->value ? root->truenode : root->falsenode; \
    }                                                                \
  }

inline bool _isnan_(float x) { return std::isnan(x); }
inline bool _isnan_(double x) { return std::isnan(x); }
inline bool _isnan_(int64_t) { return false; }
inline bool _isnan_(int32_t) { return false; }

template <typename ITYPE, typename OTYPE>
TreeNodeElement<OTYPE>*
TreeEnsembleCommon<ITYPE, OTYPE>::ProcessTreeNodeLeave(
    TreeNodeElement<OTYPE>* root, const ITYPE* x_data) const {
  ITYPE val;
  if (same_mode_) {
    switch (root->mode) {
      case NODE_MODE::BRANCH_LEQ:
        if (has_missing_tracks_) {
          while (root->is_not_leaf) {
            val = x_data[root->feature_id];
            root = (val <= root->value ||
                    (root->is_missing_track_true && _isnan_(val)))
                       ? root->truenode
                       : root->falsenode;
          }
        } else {
          while (root->is_not_leaf) {
            val = x_data[root->feature_id];
            root = val <= root->value ? root->truenode : root->falsenode;
          }
        }
        break;
      case NODE_MODE::BRANCH_LT:
        TREE_FIND_VALUE(<)
        break;
      case NODE_MODE::BRANCH_GTE:
        TREE_FIND_VALUE(>=)
        break;
      case NODE_MODE::BRANCH_GT:
        TREE_FIND_VALUE(>)
        break;
      case NODE_MODE::BRANCH_EQ:
        TREE_FIND_VALUE(==)
        break;
      case NODE_MODE::BRANCH_NEQ:
        TREE_FIND_VALUE(!=)
        break;
      case NODE_MODE::LEAF:
        break;
    }
  } else {  // Different rules to compare to node thresholds.
    OTYPE threshold;
    while (root->is_not_leaf) {
      val = x_data[root->feature_id];
      threshold = root->value;
      switch (root->mode) {
        case NODE_MODE::BRANCH_LEQ:
          root = val <= threshold || (root->is_missing_track_true && _isnan_(val))
                     ? root->truenode
                     : root->falsenode;
          break;
        case NODE_MODE::BRANCH_LT:
          root = val < threshold || (root->is_missing_track_true && _isnan_(val))
                     ? root->truenode
                     : root->falsenode;
          break;
        case NODE_MODE::BRANCH_GTE:
          root = val >= threshold || (root->is_missing_track_true && _isnan_(val))
                     ? root->truenode
                     : root->falsenode;
          break;
        case NODE_MODE::BRANCH_GT:
          root = val > threshold || (root->is_missing_track_true && _isnan_(val))
                     ? root->truenode
                     : root->falsenode;
          break;
        case NODE_MODE::BRANCH_EQ:
          root = val == threshold || (root->is_missing_track_true && _isnan_(val))
                     ? root->truenode
                     : root->falsenode;
          break;
        case NODE_MODE::BRANCH_NEQ:
          root = val != threshold || (root->is_missing_track_true && _isnan_(val))
                     ? root->truenode
                     : root->falsenode;
          break;
        case NODE_MODE::LEAF:
          break;
      }
    }
  }
  return root;
}

template <typename ITYPE, typename OTYPE>
class TreeEnsembleCommonClassifier : TreeEnsembleCommon<ITYPE, OTYPE> {
 private:
  bool weights_are_all_positive_;
  bool binary_case_;
  std::vector<std::string> classlabels_strings_;
  std::vector<int64_t> classlabels_int64s_;
  std::vector<int64_t> class_labels_;

 public:
  TreeEnsembleCommonClassifier(int parallel_tree,
                               int parallel_N,
                               const std::string& aggregate_function,
                               const std::vector<OTYPE>& base_values,
                               const std::vector<int64_t>& nodes_falsenodeids,
                               const std::vector<int64_t>& nodes_featureids,
                               const std::vector<OTYPE>& nodes_hitrates,
                               const std::vector<int64_t>& nodes_missing_value_tracks_true,
                               const std::vector<std::string>& nodes_modes,
                               const std::vector<int64_t>& nodes_nodeids,
                               const std::vector<int64_t>& nodes_treeids,
                               const std::vector<int64_t>& nodes_truenodeids,
                               const std::vector<OTYPE>& nodes_values,
                               const std::string& post_transform,
                               const std::vector<int64_t>& class_ids,
                               const std::vector<int64_t>& class_nodeids,
                               const std::vector<int64_t>& class_treeids,
                               const std::vector<OTYPE>& class_weights,
                               const std::vector<std::string>& classlabels_strings,
                               const std::vector<int64_t>& classlabels_int64s);

  int64_t get_class_count() const { return this->n_targets_or_classes_; }

  void compute(concurrency::ThreadPool* ttp, const Tensor* X, Tensor* Z, Tensor* label) const;
};

template <typename ITYPE, typename OTYPE>
TreeEnsembleCommonClassifier<ITYPE, OTYPE>::TreeEnsembleCommonClassifier(
    int parallel_tree,
    int parallel_N,
    const std::string& aggregate_function,
    const std::vector<OTYPE>& base_values,
    const std::vector<int64_t>& nodes_falsenodeids,
    const std::vector<int64_t>& nodes_featureids,
    const std::vector<OTYPE>& nodes_hitrates,
    const std::vector<int64_t>& nodes_missing_value_tracks_true,
    const std::vector<std::string>& nodes_modes,
    const std::vector<int64_t>& nodes_nodeids,
    const std::vector<int64_t>& nodes_treeids,
    const std::vector<int64_t>& nodes_truenodeids,
    const std::vector<OTYPE>& nodes_values,
    const std::string& post_transform,
    const std::vector<int64_t>& class_ids,
    const std::vector<int64_t>& class_nodeids,
    const std::vector<int64_t>& class_treeids,
    const std::vector<OTYPE>& class_weights,
    const std::vector<std::string>& classlabels_strings,
    const std::vector<int64_t>& classlabels_int64s)
    : TreeEnsembleCommon<ITYPE, OTYPE>(parallel_tree,
                                       parallel_N,
                                       aggregate_function,
                                       base_values,
                                       classlabels_strings.size() == 0 ? classlabels_int64s.size()
                                                                       : classlabels_strings.size(),
                                       nodes_falsenodeids,
                                       nodes_featureids,
                                       nodes_hitrates,
                                       nodes_missing_value_tracks_true,
                                       nodes_modes,
                                       nodes_nodeids,
                                       nodes_treeids,
                                       nodes_truenodeids,
                                       nodes_values,
                                       post_transform,
                                       class_ids,
                                       class_nodeids,
                                       class_treeids,
                                       class_weights) {
  classlabels_strings_ = classlabels_strings;
  classlabels_int64s_ = classlabels_int64s;

  std::set<int64_t> weights_classes;
  weights_are_all_positive_ = true;
  for (size_t i = 0, end = class_ids.size(); i < end; ++i) {
    weights_classes.insert(class_ids[i]);
    if (weights_are_all_positive_ && class_weights[i] < 0)
      weights_are_all_positive_ = false;
  }
  binary_case_ = this->n_targets_or_classes_ == 2 && weights_classes.size() == 1;
  if (classlabels_strings_.size() > 0) {
    class_labels_.resize(classlabels_strings_.size());
    for (size_t i = 0; i < classlabels_strings_.size(); ++i)
      class_labels_[i] = i;
  }
}

template <typename ITYPE, typename OTYPE>
void TreeEnsembleCommonClassifier<ITYPE, OTYPE>::compute(concurrency::ThreadPool* ttp, const Tensor* X, Tensor* Z,
                                                         Tensor* label) const {
  if (classlabels_strings_.size() == 0) {
    this->ComputeAgg(
        ttp, X, Z, label,
        TreeAggregatorClassifier<ITYPE, OTYPE>(
            this->roots_.size(), this->n_targets_or_classes_,
            this->post_transform_, this->base_values_,
            classlabels_int64s_, binary_case_,
            weights_are_all_positive_));
  } else {
    int64_t N = X->Shape().NumDimensions() == 1 ? 1 : X->Shape()[0];
    std::shared_ptr<IAllocator> allocator = std::make_shared<CPUAllocator>();
    Tensor label_int64(DataTypeImpl::GetType<int64_t>(), TensorShape({N}), allocator);
    this->ComputeAgg(
        ttp, X, Z, &label_int64,
        TreeAggregatorClassifier<ITYPE, OTYPE>(
            this->roots_.size(), this->n_targets_or_classes_,
            this->post_transform_, this->base_values_,
            class_labels_, binary_case_,
            weights_are_all_positive_));
    const int64_t* plabel = label_int64.template Data<int64_t>();
    std::string* labels = label->template MutableData<std::string>();
    for (size_t i = 0; i < (size_t)N; ++i)
      labels[i] = classlabels_strings_[plabel[i]];
  }
}

}  // namespace detail
}  // namespace ml
}  // namespace onnxruntime
