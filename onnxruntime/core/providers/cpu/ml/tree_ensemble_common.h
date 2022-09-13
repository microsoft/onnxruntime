// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "tree_ensemble_aggregator.h"
#include "core/platform/ort_mutex.h"
#include "core/platform/threadpool.h"
#include "tree_ensemble_helper.h"

namespace onnxruntime {
namespace ml {
namespace detail {

class TreeEnsembleCommonAttributes {
 public:
  int64_t get_target_or_class_count() const { return this->n_targets_or_classes_; }
  virtual Status Init(const OpKernelInfo&) = 0;
  virtual Status compute(OpKernelContext*, const Tensor*, Tensor*, Tensor*) const = 0;
  virtual ~TreeEnsembleCommonAttributes() {}

 protected:
  int64_t n_targets_or_classes_;
  POST_EVAL_TRANSFORM post_transform_;
  AGGREGATE_FUNCTION aggregate_function_;
  int64_t n_nodes_;
  int64_t max_tree_depth_;
  int64_t max_feature_id_;
  int64_t n_trees_;
  bool same_mode_;
  bool has_missing_tracks_;
  int parallel_tree_;  // starts parallelizing the computing if n_tree >= parallel_tree_ and n_rows == 1
  int parallel_N_;     // starts parallelizing the computing if n_rows >= parallel_N_
};

// TI: input type
// TH: tree type (types of the node values and targets)
// TO: output type, usually float
template <typename InputType, typename ThresholdType, typename OutputType>
class TreeEnsembleCommon : public TreeEnsembleCommonAttributes {
 protected:
  std::vector<ThresholdType> base_values_;
  std::vector<TreeNodeElement<ThresholdType>> nodes_;
  std::vector<TreeNodeElement<ThresholdType>*> roots_;

 public:
  TreeEnsembleCommon() {}

  virtual Status Init(const OpKernelInfo& info);
  virtual Status compute(OpKernelContext* ctx, const Tensor* X, Tensor* Y, Tensor* label) const;

  Status Init(int parallel_tree,
              int parallel_N,
              const std::string& aggregate_function,
              const std::vector<float>& base_values,
              const std::vector<ThresholdType>& base_values_as_tensor,
              int64_t n_targets_or_classes,
              const std::vector<int64_t>& nodes_falsenodeids,
              const std::vector<int64_t>& nodes_featureids,
              const std::vector<float>& nodes_hitrates,
              const std::vector<ThresholdType>& nodes_hitrates_as_tensor,
              const std::vector<int64_t>& nodes_missing_value_tracks_true,
              const std::vector<std::string>& nodes_modes,
              const std::vector<int64_t>& nodes_nodeids,
              const std::vector<int64_t>& nodes_treeids,
              const std::vector<int64_t>& nodes_truenodeids,
              const std::vector<float>& nodes_values,
              const std::vector<ThresholdType>& nodes_values_as_tensor,
              const std::string& post_transform,
              const std::vector<int64_t>& target_class_ids,
              const std::vector<int64_t>& target_class_nodeids,
              const std::vector<int64_t>& target_class_treeids,
              const std::vector<float>& target_class_weights,
              const std::vector<ThresholdType>& target_class_weights_as_tensor);

 protected:
  TreeNodeElement<ThresholdType>* ProcessTreeNodeLeave(TreeNodeElement<ThresholdType>* root,
                                                       const InputType* x_data) const;

  template <typename AGG>
  void ComputeAgg(concurrency::ThreadPool* ttp, const Tensor* X, Tensor* Y, Tensor* label, const AGG& agg) const;
};

template <typename InputType, typename ThresholdType, typename OutputType>
Status TreeEnsembleCommon<InputType, ThresholdType, OutputType>::Init(const OpKernelInfo& info) {
  std::vector<ThresholdType> base_values_as_tensor, nodes_hitrates_as_tensor,
                             nodes_values_as_tensor, target_weights_as_tensor;
#if !defined(ORT_MINIMAL_BUILD)
  ORT_THROW_IF_ERROR(GetVectorAttrsOrDefault(info, "base_values_as_tensor", base_values_as_tensor));
  ORT_THROW_IF_ERROR(GetVectorAttrsOrDefault(info, "nodes_hitrates_as_tensor", nodes_hitrates_as_tensor));
  ORT_THROW_IF_ERROR(GetVectorAttrsOrDefault(info, "nodes_values_as_tensor", nodes_values_as_tensor));
  ORT_THROW_IF_ERROR(GetVectorAttrsOrDefault(info, "target_weights_as_tensor", target_weights_as_tensor));
#endif

  return Init(
      80,
      50,
      info.GetAttrOrDefault<std::string>("aggregate_function", "SUM"),
      info.GetAttrsOrDefault<float>("base_values"),
      base_values_as_tensor,
      info.GetAttrOrDefault<int64_t>("n_targets", 0),
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
      info.GetAttrsOrDefault<int64_t>("target_ids"),
      info.GetAttrsOrDefault<int64_t>("target_nodeids"),
      info.GetAttrsOrDefault<int64_t>("target_treeids"),
      info.GetAttrsOrDefault<float>("target_weights"),
      target_weights_as_tensor);
}

template <typename InputType, typename ThresholdType, typename OutputType>
Status TreeEnsembleCommon<InputType, ThresholdType, OutputType>::Init(int parallel_tree, int parallel_N,
                                            const std::string& aggregate_function,
                                            const std::vector<float>& base_values,
                                            const std::vector<ThresholdType>& base_values_as_tensor,
                                            int64_t n_targets_or_classes,
                                            const std::vector<int64_t>& nodes_falsenodeids,
                                            const std::vector<int64_t>& nodes_featureids,
                                            const std::vector<float>& nodes_hitrates,
                                            const std::vector<ThresholdType>& nodes_hitrates_as_tensor,
                                            const std::vector<int64_t>& nodes_missing_value_tracks_true,
                                            const std::vector<std::string>& nodes_modes,
                                            const std::vector<int64_t>& nodes_nodeids,
                                            const std::vector<int64_t>& nodes_treeids,
                                            const std::vector<int64_t>& nodes_truenodeids,
                                            const std::vector<float>& nodes_values,
                                            const std::vector<ThresholdType>& nodes_values_as_tensor,
                                            const std::string& post_transform,
                                            const std::vector<int64_t>& target_class_ids,
                                            const std::vector<int64_t>& target_class_nodeids,
                                            const std::vector<int64_t>& target_class_treeids,
                                            const std::vector<float>& target_class_weights,
                                            const std::vector<ThresholdType>& target_class_weights_as_tensor) {
  parallel_tree_ = parallel_tree;
  parallel_N_ = parallel_N;

  ORT_ENFORCE(n_targets_or_classes > 0);
  ORT_ENFORCE(nodes_falsenodeids.size() == nodes_featureids.size());
  ORT_ENFORCE(nodes_falsenodeids.size() == nodes_modes.size());
  ORT_ENFORCE(nodes_falsenodeids.size() == nodes_nodeids.size());
  ORT_ENFORCE(nodes_falsenodeids.size() == nodes_treeids.size());
  ORT_ENFORCE(nodes_falsenodeids.size() == nodes_truenodeids.size());
  ORT_ENFORCE(nodes_falsenodeids.size() == nodes_values.size() ||
              nodes_falsenodeids.size() == nodes_values_as_tensor.size());
  ORT_ENFORCE(target_class_ids.size() == target_class_nodeids.size());
  ORT_ENFORCE(target_class_ids.size() == target_class_treeids.size());
  ORT_ENFORCE(target_class_ids.size() == target_class_treeids.size());
  ORT_ENFORCE(base_values.empty() || base_values_as_tensor.empty());
  ORT_ENFORCE(nodes_hitrates.empty() || nodes_hitrates_as_tensor.empty());
  ORT_ENFORCE(nodes_values.empty() || nodes_values_as_tensor.empty());
  ORT_ENFORCE(target_class_weights.empty() || target_class_weights_as_tensor.empty());

  aggregate_function_ = MakeAggregateFunction(aggregate_function);
  post_transform_ = MakeTransform(post_transform);
  if (!base_values_as_tensor.empty()) {
    ORT_ENFORCE(base_values.empty());
    base_values_ = base_values_as_tensor;
  } else {
    base_values_.reserve(base_values.size());
    for (size_t i = 0, limit = base_values.size(); i < limit; ++i) {
      base_values_.push_back(static_cast<ThresholdType>(base_values[i]));
    }
  }
  n_targets_or_classes_ = n_targets_or_classes;
  max_tree_depth_ = 1000;

  // additional members
  size_t i, limit;
  std::vector<NODE_MODE> cmodes(nodes_modes.size());
  same_mode_ = true;
  int fpos = -1;
  for (i = 0, limit = nodes_modes.size(); i < limit; ++i) {
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
  std::unordered_map<TreeNodeElementId, TreeNodeElement<ThresholdType>*, TreeNodeElementId::hash_fn> idi;
  max_feature_id_ = 0;

  for (i = 0, limit = nodes_treeids.size(); i < limit; ++i) {
    TreeNodeElement<ThresholdType>& node = nodes_[i];
    node.id.tree_id = static_cast<int>(nodes_treeids[i]);
    node.id.node_id = static_cast<int>(nodes_nodeids[i]);
    node.feature_id = static_cast<int>(nodes_featureids[i]);
    if (node.feature_id > max_feature_id_) {
      max_feature_id_ = node.feature_id;
    }
    if (nodes_values_as_tensor.empty()) {
      node.value = static_cast<ThresholdType>(nodes_values[i]);
    } else {
      node.value = nodes_values_as_tensor[i];
    }
    if (nodes_hitrates_as_tensor.empty()) {
      node.hitrates = static_cast<ThresholdType>(i < nodes_hitrates.size() ? nodes_hitrates[i] : -1);
    } else {
      node.hitrates = i < nodes_hitrates_as_tensor.size() ? nodes_hitrates_as_tensor[i] : -1;
    }
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
    idi.insert(std::pair<TreeNodeElementId, TreeNodeElement<ThresholdType>*>(node.id, &node));
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
  SparseValue<ThresholdType> w;
  for (i = 0, limit = target_class_nodeids.size(); i < limit; i++) {
    ind.tree_id = static_cast<int>(target_class_treeids[i]);
    ind.node_id = static_cast<int>(target_class_nodeids[i]);
    if (idi.find(ind) == idi.end()) {
      ORT_THROW("Unable to find node ", coor.tree_id, "-", coor.node_id, " (weights).");
    }
    w.i = target_class_ids[i];
    if (target_class_weights_as_tensor.empty()) {
      w.value = static_cast<ThresholdType>(target_class_weights[i]);
    } else {
      w.value = target_class_weights_as_tensor[i];
    }
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
  return Status::OK();
}

template <typename InputType, typename ThresholdType, typename OutputType>
Status TreeEnsembleCommon<InputType, ThresholdType, OutputType>::compute(OpKernelContext* ctx,
                                                                         const Tensor* X,
                                                                         Tensor* Y,
                                                                         Tensor* label) const {
  switch (aggregate_function_) {
    case AGGREGATE_FUNCTION::AVERAGE:
      ComputeAgg(
          ctx->GetOperatorThreadPool(), X, Y, label,
          TreeAggregatorAverage<InputType, ThresholdType, OutputType>(
              roots_.size(), n_targets_or_classes_,
              post_transform_, base_values_));
      return Status::OK();
    case AGGREGATE_FUNCTION::SUM:
      ComputeAgg(
          ctx->GetOperatorThreadPool(), X, Y, label,
          TreeAggregatorSum<InputType, ThresholdType, OutputType>(
              roots_.size(), n_targets_or_classes_,
              post_transform_, base_values_));
      return Status::OK();
    case AGGREGATE_FUNCTION::MIN:
      ComputeAgg(
          ctx->GetOperatorThreadPool(), X, Y, label,
          TreeAggregatorMin<InputType, ThresholdType, OutputType>(
              roots_.size(), n_targets_or_classes_,
              post_transform_, base_values_));
      return Status::OK();
    case AGGREGATE_FUNCTION::MAX:
      ComputeAgg(
          ctx->GetOperatorThreadPool(), X, Y, label,
          TreeAggregatorMax<InputType, ThresholdType, OutputType>(
              roots_.size(), n_targets_or_classes_,
              post_transform_, base_values_));
      return Status::OK();
    default:
      ORT_THROW("Unknown aggregation function in TreeEnsemble.");
  }
}

template <typename InputType, typename ThresholdType, typename OutputType>
template <typename AGG>
void TreeEnsembleCommon<InputType, ThresholdType, OutputType>::ComputeAgg(concurrency::ThreadPool* ttp,
                                                                          const Tensor* X, Tensor* Z,
                                                                          Tensor* label, const AGG& agg) const {
  if (X->Shape().NumDimensions() > 2) {
    ORT_THROW("TreeEnsemble only works on 1D, 2D tensors.");
  }
  int64_t stride = X->Shape().NumDimensions() == 1 ? X->Shape()[0] : X->Shape()[1];
  int64_t N = X->Shape().NumDimensions() == 1 ? 1 : X->Shape()[0];
  int64_t C = X->Shape().NumDimensions() == 2 ? X->Shape()[1] : 1;
  if (max_feature_id_ >= C) {
    ORT_THROW("One path in the graph requests feature ", max_feature_id_, " but input tensor has ", C, " features.");
  }
  OutputType* z_data = Z->MutableData<OutputType>();

  const InputType* x_data = X->Data<InputType>();
  int64_t* label_data = label == nullptr ? nullptr : label->MutableData<int64_t>();
  auto max_num_threads = concurrency::ThreadPool::DegreeOfParallelism(ttp);

  if (n_targets_or_classes_ == 1) {
    if (N == 1) {
      ScoreValue<ThresholdType> score = {0, 0};
      if (n_trees_ <= parallel_tree_) { /* section A: 1 output, 1 row and not enough trees to parallelize */
        for (int64_t j = 0; j < n_trees_; ++j) {
          agg.ProcessTreeNodePrediction1(score, *ProcessTreeNodeLeave(roots_[j], x_data));
        }
      } else { /* section B: 1 output, 1 row and enough trees to parallelize */
        std::vector<ScoreValue<ThresholdType>> scores(n_trees_, {0, 0});
        concurrency::ThreadPool::TryBatchParallelFor(
            ttp,
            SafeInt<int32_t>(n_trees_),
            [this, &scores, &agg, x_data](ptrdiff_t j) {
              agg.ProcessTreeNodePrediction1(scores[j], *ProcessTreeNodeLeave(roots_[j], x_data));
            },
            0);

        for (auto it = scores.cbegin(); it != scores.cend(); ++it) {
          agg.MergePrediction1(score, *it);
        }
      }
      agg.FinalizeScores1(z_data, score, label_data);
    } else if (N <= parallel_N_) { /* section C: 1 output, 2+ rows but not enough rows to parallelize */
      ScoreValue<ThresholdType> score;
      size_t j;

      for (int64_t i = 0; i < N; ++i) {
        score = {0, 0};
        for (j = 0; j < static_cast<size_t>(n_trees_); ++j) {
          agg.ProcessTreeNodePrediction1(score, *ProcessTreeNodeLeave(roots_[j], x_data + i * stride));
        }

        agg.FinalizeScores1(z_data + i, score,
                            label_data == nullptr ? nullptr : (label_data + i));
      }
    } else if (n_trees_ > max_num_threads) { /* section D: 1 output, 2+ rows and enough trees to parallelize */
      auto num_threads = std::min<int32_t>(max_num_threads, SafeInt<int32_t>(n_trees_));
      std::vector<ScoreValue<ThresholdType>> scores(num_threads * N);
      concurrency::ThreadPool::TrySimpleParallelFor(
          ttp,
          num_threads,
          [this, &agg, &scores, num_threads, x_data, N, stride](ptrdiff_t batch_num) {
            auto work = concurrency::ThreadPool::PartitionWork(batch_num, num_threads, this->n_trees_);
            for (int64_t i = 0; i < N; ++i) {
              scores[batch_num * N + i] = {0, 0};
            }
            for (auto j = work.start; j < work.end; ++j) {
              for (int64_t i = 0; i < N; ++i) {
                agg.ProcessTreeNodePrediction1(scores[batch_num * N + i],
                                               *ProcessTreeNodeLeave(roots_[j], x_data + i * stride));
              }
            }
          });

      concurrency::ThreadPool::TrySimpleParallelFor(
          ttp,
          num_threads,
          [&agg, &scores, num_threads, label_data, z_data, N](ptrdiff_t batch_num) {
            auto work = concurrency::ThreadPool::PartitionWork(batch_num, num_threads, N);
            for (auto i = work.start; i < work.end; ++i) {
              for (int64_t j = 1; j < num_threads; ++j) {
                agg.MergePrediction1(scores[i], scores[j * N + i]);
              }
              agg.FinalizeScores1(z_data + i, scores[i],
                                  label_data == nullptr ? nullptr : (label_data + i));
            }
          });
    } else { /* section E: 1 output, 2+ rows, parallelization by rows */
      concurrency::ThreadPool::TryBatchParallelFor(
          ttp,
          SafeInt<int32_t>(N),
          [this, &agg, x_data, z_data, stride, label_data](ptrdiff_t i) {
            ScoreValue<ThresholdType> score = {0, 0};
            for (size_t j = 0; j < static_cast<size_t>(n_trees_); ++j) {
              agg.ProcessTreeNodePrediction1(score, *ProcessTreeNodeLeave(roots_[j], x_data + i * stride));
            }

            agg.FinalizeScores1(z_data + i, score,
                                label_data == nullptr ? nullptr : (label_data + i));
          },
          0);
    }
  } else {
    if (N == 1) {                       /* section A2: 2+ outputs, 1 row, not enough trees to parallelize */
      if (n_trees_ <= parallel_tree_) { /* section A2 */
        InlinedVector<ScoreValue<ThresholdType>> scores(n_targets_or_classes_, {0, 0});
        for (int64_t j = 0; j < n_trees_; ++j) {
          agg.ProcessTreeNodePrediction(scores, *ProcessTreeNodeLeave(roots_[j], x_data));
        }
        agg.FinalizeScores(scores, z_data, -1, label_data);
      } else { /* section B2: 2+ outputs, 1 row, enough trees to parallelize */
        auto num_threads = std::min<int32_t>(max_num_threads, SafeInt<int32_t>(n_trees_));
        std::vector<InlinedVector<ScoreValue<ThresholdType>>> scores(num_threads);
        concurrency::ThreadPool::TrySimpleParallelFor(
            ttp,
            num_threads,
            [this, &agg, &scores, num_threads, x_data](ptrdiff_t batch_num) {
              scores[batch_num].resize(n_targets_or_classes_, {0, 0});
              auto work = concurrency::ThreadPool::PartitionWork(batch_num, num_threads, n_trees_);
              for (auto j = work.start; j < work.end; ++j) {
                agg.ProcessTreeNodePrediction(scores[batch_num], *ProcessTreeNodeLeave(roots_[j], x_data));
              }
            });
        for (size_t i = 1, limit = scores.size(); i < limit; ++i) {
          agg.MergePrediction(scores[0], scores[i]);
        }
        agg.FinalizeScores(scores[0], z_data, -1, label_data);
      }
    } else if (N <= parallel_N_) { /* section C2: 2+ outputs, 2+ rows, not enough rows to parallelize */
      InlinedVector<ScoreValue<ThresholdType>> scores(n_targets_or_classes_);
      size_t j, limit;

      for (int64_t i = 0; i < N; ++i) {
        std::fill(scores.begin(), scores.end(), ScoreValue<ThresholdType>({0, 0}));
        for (j = 0, limit = roots_.size(); j < limit; ++j) {
          agg.ProcessTreeNodePrediction(scores, *ProcessTreeNodeLeave(roots_[j], x_data + i * stride));
        }

        agg.FinalizeScores(scores, z_data + i * n_targets_or_classes_, -1,
                           label_data == nullptr ? nullptr : (label_data + i));
      }
    } else if (n_trees_ >= max_num_threads) { /* section: D2: 2+ outputs, 2+ rows, enough trees to parallelize*/
      auto num_threads = std::min<int32_t>(max_num_threads, SafeInt<int32_t>(n_trees_));
      std::vector<InlinedVector<ScoreValue<ThresholdType>>> scores(num_threads * N);
      concurrency::ThreadPool::TrySimpleParallelFor(
          ttp,
          num_threads,
          [this, &agg, &scores, num_threads, x_data, N, stride](ptrdiff_t batch_num) {
            auto work = concurrency::ThreadPool::PartitionWork(batch_num, num_threads, this->n_trees_);
            for (int64_t i = 0; i < N; ++i) {
              scores[batch_num * N + i].resize(n_targets_or_classes_, {0, 0});
            }
            for (auto j = work.start; j < work.end; ++j) {
              for (int64_t i = 0; i < N; ++i) {
                agg.ProcessTreeNodePrediction(scores[batch_num * N + i],
                                              *ProcessTreeNodeLeave(roots_[j], x_data + i * stride));
              }
            }
          });

      concurrency::ThreadPool::TrySimpleParallelFor(
          ttp,
          num_threads,
          [this, &agg, &scores, num_threads, label_data, z_data, N](ptrdiff_t batch_num) {
            auto work = concurrency::ThreadPool::PartitionWork(batch_num, num_threads, N);
            for (auto i = work.start; i < work.end; ++i) {
              for (int64_t j = 1; j < num_threads; ++j) {
                agg.MergePrediction(scores[i], scores[j * N + i]);
              }
              agg.FinalizeScores(scores[i], z_data + i * this->n_targets_or_classes_, -1,
                                 label_data == nullptr ? nullptr : (label_data + i));
            }
          });
    } else { /* section E2: 2+ outputs, 2+ rows, parallelization by rows */
      auto num_threads = std::min<int32_t>(max_num_threads, SafeInt<int32_t>(N));
      concurrency::ThreadPool::TrySimpleParallelFor(
          ttp,
          num_threads,
          [this, &agg, num_threads, x_data, z_data, label_data, N, stride](ptrdiff_t batch_num) {
            size_t j, limit;
            InlinedVector<ScoreValue<ThresholdType>> scores(n_targets_or_classes_);
            auto work = concurrency::ThreadPool::PartitionWork(batch_num, num_threads, N);

            for (auto i = work.start; i < work.end; ++i) {
              std::fill(scores.begin(), scores.end(), ScoreValue<ThresholdType>({0, 0}));
              for (j = 0, limit = roots_.size(); j < limit; ++j) {
                agg.ProcessTreeNodePrediction(scores, *ProcessTreeNodeLeave(roots_[j], x_data + i * stride));
              }

              agg.FinalizeScores(scores,
                                 z_data + i * n_targets_or_classes_, -1,
                                 label_data == nullptr ? nullptr : (label_data + i));
            }
          });
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

template <typename InputType, typename ThresholdType, typename OutputType>
TreeNodeElement<ThresholdType>*
TreeEnsembleCommon<InputType, ThresholdType, OutputType>::ProcessTreeNodeLeave(
    TreeNodeElement<ThresholdType>* root, const InputType* x_data) const {
  InputType val;
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
    ThresholdType threshold;
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

// TI: input type
// TH: threshold type, double if T==double, float otherwise
// TO: output type
template <typename InputType, typename ThresholdType, typename OutputType>
class TreeEnsembleCommonClassifier : public TreeEnsembleCommon<InputType, ThresholdType, OutputType> {
 private:
  bool weights_are_all_positive_;
  bool binary_case_;
  std::vector<std::string> classlabels_strings_;
  std::vector<int64_t> classlabels_int64s_;
  std::vector<int64_t> class_labels_;

 public:

  virtual Status Init(const OpKernelInfo& info);
  virtual Status compute(OpKernelContext* ctx, const Tensor* X, Tensor* Z, Tensor* label) const;

  Status Init(int parallel_tree,
              int parallel_N,
              const std::string& aggregate_function,
              const std::vector<float>& base_values,
              const std::vector<ThresholdType>& base_values_as_tensor,
              const std::vector<int64_t>& nodes_falsenodeids,
              const std::vector<int64_t>& nodes_featureids,
              const std::vector<float>& nodes_hitrates,
              const std::vector<ThresholdType>& nodes_hitrates_as_tensor,
              const std::vector<int64_t>& nodes_missing_value_tracks_true,
              const std::vector<std::string>& nodes_modes,
              const std::vector<int64_t>& nodes_nodeids,
              const std::vector<int64_t>& nodes_treeids,
              const std::vector<int64_t>& nodes_truenodeids,
              const std::vector<float>& nodes_values,
              const std::vector<ThresholdType>& nodes_values_as_tensor,
              const std::string& post_transform,
              const std::vector<int64_t>& class_ids,
              const std::vector<int64_t>& class_nodeids,
              const std::vector<int64_t>& class_treeids,
              const std::vector<float>& class_weights,
              const std::vector<ThresholdType>& class_weights_as_tensor,
              const std::vector<std::string>& classlabels_strings,
              const std::vector<int64_t>& classlabels_int64s);
};


template <typename InputType, typename ThresholdType, typename OutputType>
Status TreeEnsembleCommonClassifier<InputType, ThresholdType, OutputType>::Init(const OpKernelInfo& info) {
  std::vector<ThresholdType> base_values_as_tensor, nodes_hitrates_as_tensor,
                             nodes_values_as_tensor, class_weights_as_tensor;
#if !defined(ORT_MINIMAL_BUILD)
  ORT_THROW_IF_ERROR(GetVectorAttrsOrDefault(info, "base_values_as_tensor", base_values_as_tensor));
  ORT_THROW_IF_ERROR(GetVectorAttrsOrDefault(info, "nodes_hitrates_as_tensor", nodes_hitrates_as_tensor));
  ORT_THROW_IF_ERROR(GetVectorAttrsOrDefault(info, "nodes_values_as_tensor", nodes_values_as_tensor));
  ORT_THROW_IF_ERROR(GetVectorAttrsOrDefault(info, "class_weights_as_tensor", class_weights_as_tensor));
#endif

  return Init(
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

template <typename InputType, typename ThresholdType, typename OutputType>
Status TreeEnsembleCommonClassifier<InputType, ThresholdType, OutputType>::Init(int parallel_tree,
                                                      int parallel_N,
                                                      const std::string& aggregate_function,
                                                      const std::vector<float>& base_values,
                                                      const std::vector<ThresholdType>& base_values_as_tensor,
                                                      const std::vector<int64_t>& nodes_falsenodeids,
                                                      const std::vector<int64_t>& nodes_featureids,
                                                      const std::vector<float>& nodes_hitrates,
                                                      const std::vector<ThresholdType>& nodes_hitrates_as_tensor,
                                                      const std::vector<int64_t>& nodes_missing_value_tracks_true,
                                                      const std::vector<std::string>& nodes_modes,
                                                      const std::vector<int64_t>& nodes_nodeids,
                                                      const std::vector<int64_t>& nodes_treeids,
                                                      const std::vector<int64_t>& nodes_truenodeids,
                                                      const std::vector<float>& nodes_values,
                                                      const std::vector<ThresholdType>& nodes_values_as_tensor,
                                                      const std::string& post_transform,
                                                      const std::vector<int64_t>& class_ids,
                                                      const std::vector<int64_t>& class_nodeids,
                                                      const std::vector<int64_t>& class_treeids,
                                                      const std::vector<float>& class_weights,
                                                      const std::vector<ThresholdType>& class_weights_as_tensor,
                                                      const std::vector<std::string>& classlabels_strings,
                                                      const std::vector<int64_t>& classlabels_int64s) {
  auto status = TreeEnsembleCommon<InputType, ThresholdType, OutputType>::Init(parallel_tree,
                                                     parallel_N,
                                                     aggregate_function,
                                                     base_values,
                                                     base_values_as_tensor,
                                                     classlabels_strings.empty() ? classlabels_int64s.size()
                                                                                 : classlabels_strings.size(),
                                                     nodes_falsenodeids,
                                                     nodes_featureids,
                                                     nodes_hitrates,
                                                     nodes_hitrates_as_tensor,
                                                     nodes_missing_value_tracks_true,
                                                     nodes_modes,
                                                     nodes_nodeids,
                                                     nodes_treeids,
                                                     nodes_truenodeids,
                                                     nodes_values,
                                                     nodes_values_as_tensor,
                                                     post_transform,
                                                     class_ids,
                                                     class_nodeids,
                                                     class_treeids,
                                                     class_weights,
                                                     class_weights_as_tensor);
  ORT_RETURN_IF_ERROR(status);

  classlabels_strings_ = classlabels_strings;
  classlabels_int64s_ = classlabels_int64s;

  InlinedHashSet<int64_t> weights_classes;
  weights_classes.reserve(class_ids.size());
  weights_are_all_positive_ = true;
  for (size_t i = 0, end = class_ids.size(); i < end; ++i) {
    weights_classes.insert(class_ids[i]);
    if (weights_are_all_positive_ && (!class_weights.empty() ? class_weights[i] : class_weights_as_tensor[i]) < 0)
      weights_are_all_positive_ = false;
  }
  binary_case_ = this->n_targets_or_classes_ == 2 && weights_classes.size() == 1;
  if (!classlabels_strings_.empty()) {
    class_labels_.reserve(classlabels_strings_.size());
    for (size_t i = 0, end = classlabels_strings_.size(); i < end; ++i)
      class_labels_.push_back(i);
  }
  return Status::OK();
}

template <typename InputType, typename ThresholdType, typename OutputType>
Status TreeEnsembleCommonClassifier<InputType, ThresholdType, OutputType>::compute(OpKernelContext* ctx,
                                                                                   const Tensor* X,
                                                                                   Tensor* Z,
                                                                                   Tensor* label) const {
  if (classlabels_strings_.empty()) {
    this->ComputeAgg(
        ctx->GetOperatorThreadPool(), X, Z, label,
        TreeAggregatorClassifier<InputType, ThresholdType, OutputType>(
            this->roots_.size(), this->n_targets_or_classes_,
            this->post_transform_, this->base_values_,
            classlabels_int64s_, binary_case_,
            weights_are_all_positive_));
  } else {
    int64_t N = X->Shape().NumDimensions() == 1 ? 1 : X->Shape()[0];
    AllocatorPtr alloc;
    ORT_THROW_IF_ERROR(ctx->GetTempSpaceAllocator(&alloc));
    Tensor label_int64(DataTypeImpl::GetType<int64_t>(), TensorShape({N}), std::move(alloc));
    this->ComputeAgg(
        ctx->GetOperatorThreadPool(), X, Z, &label_int64,
        TreeAggregatorClassifier<InputType, ThresholdType, OutputType>(
            this->roots_.size(), this->n_targets_or_classes_,
            this->post_transform_, this->base_values_,
            class_labels_, binary_case_,
            weights_are_all_positive_));
    const int64_t* plabel = label_int64.Data<int64_t>();
    std::string* labels = label->MutableData<std::string>();
    for (size_t i = 0; i < (size_t)N; ++i)
      labels[i] = classlabels_strings_[plabel[i]];
  }
  return Status::OK();
}

}  // namespace detail
}  // namespace ml
}  // namespace onnxruntime
