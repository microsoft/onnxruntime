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
  int parallel_tree_;    // starts parallelizing the computing by trees if n_tree >= parallel_tree_
  int parallel_tree_N_;  // batch size if parallelizing by trees
  int parallel_N_;       // starts parallelizing the computing by rows if n_rows <= parallel_N_
};

// TI: input type
// TH: tree type (types of the node values and targets)
// TO: output type, usually float
template <typename InputType, typename ThresholdType, typename OutputType>
class TreeEnsembleCommon : public TreeEnsembleCommonAttributes {
 protected:
  std::vector<ThresholdType> base_values_;
  std::vector<TreeNodeElement<ThresholdType>> nodes_;
  // Type of weights should be a vector of OutputType. Onnx specifications says it must be float.
  // Lightgbm requires a double to do the summation of all trees predictions. That's why
  // `ThresholdType` is used as well for output type (double as well for lightgbm) and not `OutputType`.
  std::vector<SparseValue<ThresholdType>> weights_;
  std::vector<TreeNodeElement<ThresholdType>*> roots_;

 public:
  TreeEnsembleCommon() {}

  virtual Status Init(const OpKernelInfo& info);
  virtual Status compute(OpKernelContext* ctx, const Tensor* X, Tensor* Y, Tensor* label) const;

  Status Init(int parallel_tree,
              int parallel_tree_N,
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
      128,
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
Status TreeEnsembleCommon<InputType, ThresholdType, OutputType>::Init(
    int parallel_tree,
    int parallel_tree_N,
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
    const std::vector<ThresholdType>& target_class_weights_as_tensor) {
  parallel_tree_ = parallel_tree;
  parallel_tree_N_ = parallel_tree_N;
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
  ORT_ENFORCE(nodes_modes.size() < std::numeric_limits<uint32_t>::max());

  // additional members
  size_t limit;
  uint32_t i;
  InlinedVector<NODE_MODE> cmodes;
  cmodes.reserve(nodes_modes.size());
  same_mode_ = true;
  int fpos = -1;
  for (i = 0, limit = nodes_modes.size(); i < limit; ++i) {
    cmodes.push_back(MakeTreeNodeMode(nodes_modes[i]));
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
  limit = static_cast<size_t>(n_nodes_);
  InlinedVector<TreeNodeElementId> node_tree_ids;
  node_tree_ids.reserve(limit);
  nodes_.clear();
  nodes_.reserve(limit);
  roots_.clear();
  std::unordered_map<TreeNodeElementId, uint32_t, TreeNodeElementId::hash_fn> idi;
  idi.reserve(limit);
  max_feature_id_ = 0;

  for (i = 0; i < limit; ++i) {
    TreeNodeElementId node_tree_id{static_cast<int>(nodes_treeids[i]),
                                   static_cast<int>(nodes_nodeids[i])};
    TreeNodeElement<ThresholdType> node;
    node.feature_id = static_cast<int>(nodes_featureids[i]);
    if (node.feature_id > max_feature_id_) {
      max_feature_id_ = node.feature_id;
    }
    node.value_or_unique_weight = nodes_values_as_tensor.empty()
                                      ? static_cast<ThresholdType>(nodes_values[i])
                                      : nodes_values_as_tensor[i];

    /* hitrates is not used for inference, they are ignored.
    if (nodes_hitrates_as_tensor.empty()) {
      node.hitrates = static_cast<ThresholdType>(i < nodes_hitrates.size() ? nodes_hitrates[i] : -1);
    } else {
      node.hitrates = i < nodes_hitrates_as_tensor.size() ? nodes_hitrates_as_tensor[i] : -1;
    } */

    node.flags = static_cast<uint8_t>(cmodes[i]);
    node.truenode_inc_or_first_weight = 0;  // nodes_truenodeids[i] if not a leaf
    node.falsenode_inc_or_n_weights = 0;    // nodes_falsenodeids[i] if not a leaf

    if (i < static_cast<size_t>(nodes_missing_value_tracks_true.size()) && nodes_missing_value_tracks_true[i] == 1) {
      node.flags |= static_cast<uint8_t>(MissingTrack::kTrue);
    }
    auto p = idi.insert(std::pair<TreeNodeElementId, uint32_t>(node_tree_id, i));
    if (!p.second) {
      ORT_THROW("Node ", node_tree_id.node_id, " in tree ", node_tree_id.tree_id, " is already there.");
    }
    nodes_.emplace_back(node);
    node_tree_ids.emplace_back(node_tree_id);
  }

  InlinedVector<int64_t> truenode_ids, falsenode_ids;
  truenode_ids.reserve(limit);
  falsenode_ids.reserve(limit);
  TreeNodeElementId coor;
  i = 0;
  for (auto it = nodes_.begin(); it != nodes_.end(); ++it, ++i) {
    if (!it->is_not_leaf()) {
      truenode_ids.push_back(0);
      falsenode_ids.push_back(0);
      continue;
    }

    TreeNodeElementId& node_tree_id = node_tree_ids[i];
    coor.tree_id = node_tree_id.tree_id;
    coor.node_id = static_cast<int>(nodes_truenodeids[i]);
    ORT_ENFORCE((coor.node_id >= 0 && coor.node_id < n_nodes_));

    auto found = idi.find(coor);
    if (found == idi.end()) {
      ORT_THROW("Unable to find node ", coor.tree_id, "-", coor.node_id, " (truenode).");
    }
    if (found->second == truenode_ids.size()) {
      ORT_THROW("A node cannot point to itself: ", coor.tree_id, "-", node_tree_id.node_id, " (truenode).");
    }
    truenode_ids.emplace_back(found->second);

    coor.node_id = static_cast<int>(nodes_falsenodeids[i]);
    ORT_ENFORCE((coor.node_id >= 0 && coor.node_id < n_nodes_));
    found = idi.find(coor);
    if (found == idi.end()) {
      ORT_THROW("Unable to find node ", coor.tree_id, "-", coor.node_id, " (falsenode).");
    }
    if (found->second == falsenode_ids.size()) {
      ORT_THROW("A node cannot point to itself: ", coor.tree_id, "-", node_tree_id.node_id, " (falsenode).");
    }
    falsenode_ids.emplace_back(found->second);
    // We could also check that truenode_ids[truenode_ids.size() - 1] != falsenode_ids[falsenode_ids.size() - 1]).
    // It is valid but no training algorithm would produce a tree where left and right nodes are the same.
  }

  // sort targets
  InlinedVector<std::pair<TreeNodeElementId, uint32_t>> indices;
  indices.reserve(target_class_nodeids.size());
  for (i = 0, limit = target_class_nodeids.size(); i < limit; i++) {
    indices.emplace_back(std::pair<TreeNodeElementId, uint32_t>(
        TreeNodeElementId{target_class_treeids[i], target_class_nodeids[i]},
        i));
  }
  std::sort(indices.begin(), indices.end());

  // Initialize the leaves.
  TreeNodeElementId ind;
  SparseValue<ThresholdType> w;
  size_t indi;
  for (indi = 0, limit = target_class_nodeids.size(); indi < limit; ++indi) {
    ind = indices[indi].first;
    i = indices[indi].second;
    auto found = idi.find(ind);
    if (found == idi.end()) {
      ORT_THROW("Unable to find node ", ind.tree_id, "-", ind.node_id, " (weights).");
    }

    TreeNodeElement<ThresholdType>& leaf = nodes_[found->second];
    if (leaf.is_not_leaf()) {
      // An exception should be raised in that case. But this case may happen in
      // models converted with an old version of onnxmltools. These weights are ignored.
      // ORT_THROW("Node ", ind.tree_id, "-", ind.node_id, " is not a leaf.");
      continue;
    }

    w.i = target_class_ids[i];
    w.value = target_class_weights_as_tensor.empty()
                  ? static_cast<ThresholdType>(target_class_weights[i])
                  : target_class_weights_as_tensor[i];
    if (leaf.falsenode_inc_or_n_weights == 0) {
      leaf.truenode_inc_or_first_weight = static_cast<int32_t>(weights_.size());
      leaf.value_or_unique_weight = w.value;
    }
    ++leaf.falsenode_inc_or_n_weights;
    weights_.push_back(w);
  }

  // Initialize all the nodes but the leaves.
  int64_t previous = -1;
  for (i = 0, limit = static_cast<uint32_t>(n_nodes_); i < limit; ++i) {
    if ((previous == -1) || (previous != node_tree_ids[i].tree_id))
      roots_.push_back(&(nodes_[idi[node_tree_ids[i]]]));
    previous = node_tree_ids[i].tree_id;
    if (!nodes_[i].is_not_leaf()) {
      if (nodes_[i].falsenode_inc_or_n_weights == 0) {
        ORT_THROW("Target is missing for leaf ", ind.tree_id, "-", ind.node_id, ".");
      }
      continue;
    }
    ORT_ENFORCE(truenode_ids[i] != i);  // That would mean the left node is itself, leading to an infinite loop.
    nodes_[i].truenode_inc_or_first_weight = static_cast<int32_t>(truenode_ids[i] - i);
    ORT_ENFORCE(falsenode_ids[i] != i);  // That would mean the right node is itself, leading to an infinite loop.
    nodes_[i].falsenode_inc_or_n_weights = static_cast<int32_t>(falsenode_ids[i] - i);
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
      if (n_trees_ <= parallel_tree_ || max_num_threads == 1) { /* section A: 1 output, 1 row and not enough trees to parallelize */
        for (int64_t j = 0; j < n_trees_; ++j) {
          agg.ProcessTreeNodePrediction1(score, *ProcessTreeNodeLeave(roots_[onnxruntime::narrow<size_t>(j)], x_data));
        }
      } else { /* section B: 1 output, 1 row and enough trees to parallelize */
        std::vector<ScoreValue<ThresholdType>> scores(onnxruntime::narrow<size_t>(n_trees_), {0, 0});
        concurrency::ThreadPool::TryBatchParallelFor(
            ttp,
            SafeInt<int32_t>(n_trees_),
            [this, &scores, &agg, x_data](ptrdiff_t j) {
              agg.ProcessTreeNodePrediction1(scores[j], *ProcessTreeNodeLeave(roots_[j], x_data));
            },
            max_num_threads);

        for (auto it = scores.cbegin(); it != scores.cend(); ++it) {
          agg.MergePrediction1(score, *it);
        }
      }
      agg.FinalizeScores1(z_data, score, label_data);
    } else if (N <= parallel_N_ || max_num_threads == 1) { /* section C: 1 output, 2+ rows but not enough rows to parallelize */
      // Not enough data to parallelize but the computation is split into batches of 128 rows,
      // and then loop on trees to evaluate every tree on this batch.
      // This change was introduced by PR: https://github.com/microsoft/onnxruntime/pull/13835.
      // The input tensor (2D) is stored in a contiguous array. Therefore, it is faster
      // to loop on tree first and inside that loop evaluate a tree on the input tensor (inner loop).
      // The processor is faster when it has to move chunks of a contiguous array (branching).
      // However, if the input tensor is too big, the data does not hold on caches (L1, L2, L3).
      // In that case, looping first on tree or on data is almost the same. That's why the first loop
      // split into batch so that every batch holds on caches, then loop on trees and finally loop
      // on the batch rows.
      std::vector<ScoreValue<ThresholdType>> scores(parallel_tree_N_);
      size_t j;
      int64_t i, batch, batch_end;

      for (batch = 0; batch < N; batch += parallel_tree_N_) {
        batch_end = std::min(N, batch + parallel_tree_N_);
        for (i = batch; i < batch_end; ++i) {
          scores[SafeInt<ptrdiff_t>(i - batch)] = {0, 0};
        }
        for (j = 0; j < static_cast<size_t>(n_trees_); ++j) {
          for (i = batch; i < batch_end; ++i) {
            agg.ProcessTreeNodePrediction1(scores[SafeInt<ptrdiff_t>(i - batch)], *ProcessTreeNodeLeave(roots_[j], x_data + i * stride));
          }
        }
        for (i = batch; i < batch_end; ++i) {
          agg.FinalizeScores1(z_data + i, scores[SafeInt<ptrdiff_t>(i - batch)],
                              label_data == nullptr ? nullptr : (label_data + i));
        }
      }
    } else if (n_trees_ > max_num_threads) { /* section D: 1 output, 2+ rows and enough trees to parallelize */
      auto num_threads = std::min<int32_t>(max_num_threads, SafeInt<int32_t>(n_trees_));
      std::vector<ScoreValue<ThresholdType>> scores(SafeInt<size_t>(num_threads) * N);
      int64_t end_n, begin_n = 0;
      while (begin_n < N) {
        end_n = std::min(N, begin_n + parallel_tree_N_);
        concurrency::ThreadPool::TrySimpleParallelFor(
            ttp,
            num_threads,
            [this, &agg, &scores, num_threads, x_data, N, begin_n, end_n, stride](ptrdiff_t batch_num) {
              auto work = concurrency::ThreadPool::PartitionWork(batch_num, num_threads, onnxruntime::narrow<size_t>(this->n_trees_));
              for (int64_t i = begin_n; i < end_n; ++i) {
                scores[batch_num * SafeInt<ptrdiff_t>(N) + i] = {0, 0};
              }
              for (auto j = work.start; j < work.end; ++j) {
                for (int64_t i = begin_n; i < end_n; ++i) {
                  agg.ProcessTreeNodePrediction1(scores[batch_num * SafeInt<ptrdiff_t>(N) + i],
                                                 *ProcessTreeNodeLeave(roots_[j], x_data + i * stride));
                }
              }
            });
        begin_n = end_n;
      }
      concurrency::ThreadPool::TrySimpleParallelFor(
          ttp,
          num_threads,
          [&agg, &scores, num_threads, label_data, z_data, N](ptrdiff_t batch_num) {
            auto work = concurrency::ThreadPool::PartitionWork(batch_num, num_threads, onnxruntime::narrow<size_t>(N));
            for (auto i = work.start; i < work.end; ++i) {
              for (int64_t j = 1; j < num_threads; ++j) {
                agg.MergePrediction1(scores[i], scores[j * SafeInt<ptrdiff_t>(N) + i]);
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
          max_num_threads);
    }
  } else {
    if (N == 1) {                                               /* section A2: 2+ outputs, 1 row, not enough trees to parallelize */
      if (n_trees_ <= parallel_tree_ || max_num_threads == 1) { /* section A2 */
        InlinedVector<ScoreValue<ThresholdType>> scores(onnxruntime::narrow<size_t>(n_targets_or_classes_), {0, 0});
        for (int64_t j = 0; j < n_trees_; ++j) {
          agg.ProcessTreeNodePrediction(scores, *ProcessTreeNodeLeave(roots_[onnxruntime::narrow<size_t>(j)], x_data), weights_);
        }
        agg.FinalizeScores(scores, z_data, -1, label_data);
      } else { /* section B2: 2+ outputs, 1 row, enough trees to parallelize */
        auto num_threads = std::min<int32_t>(max_num_threads, SafeInt<int32_t>(n_trees_));
        std::vector<InlinedVector<ScoreValue<ThresholdType>>> scores(num_threads);
        concurrency::ThreadPool::TrySimpleParallelFor(
            ttp,
            num_threads,
            [this, &agg, &scores, num_threads, x_data](ptrdiff_t batch_num) {
              scores[batch_num].resize(onnxruntime::narrow<size_t>(n_targets_or_classes_), {0, 0});
              auto work = concurrency::ThreadPool::PartitionWork(batch_num, num_threads, onnxruntime::narrow<size_t>(n_trees_));
              for (auto j = work.start; j < work.end; ++j) {
                agg.ProcessTreeNodePrediction(scores[batch_num], *ProcessTreeNodeLeave(roots_[j], x_data), weights_);
              }
            });
        for (size_t i = 1, limit = scores.size(); i < limit; ++i) {
          agg.MergePrediction(scores[0], scores[i]);
        }
        agg.FinalizeScores(scores[0], z_data, -1, label_data);
      }
    } else if (N <= parallel_N_ || max_num_threads == 1) { /* section C2: 2+ outputs, 2+ rows, not enough rows to parallelize */
      std::vector<InlinedVector<ScoreValue<ThresholdType>>> scores(parallel_tree_N_);
      size_t j, limit;
      int64_t i, batch, batch_end;
      batch_end = std::min(N, static_cast<int64_t>(parallel_tree_N_));
      for (i = 0; i < batch_end; ++i) {
        scores[SafeInt<ptrdiff_t>(i)].resize(onnxruntime::narrow<size_t>(n_targets_or_classes_));
      }
      for (batch = 0; batch < N; batch += parallel_tree_N_) {
        batch_end = std::min(N, batch + parallel_tree_N_);
        for (i = batch; i < batch_end; ++i) {
          std::fill(scores[SafeInt<ptrdiff_t>(i - batch)].begin(), scores[SafeInt<ptrdiff_t>(i - batch)].end(), ScoreValue<ThresholdType>({0, 0}));
        }
        for (j = 0, limit = roots_.size(); j < limit; ++j) {
          for (i = batch; i < batch_end; ++i) {
            agg.ProcessTreeNodePrediction(scores[SafeInt<ptrdiff_t>(i - batch)], *ProcessTreeNodeLeave(roots_[j], x_data + i * stride), weights_);
          }
        }
        for (i = batch; i < batch_end; ++i) {
          agg.FinalizeScores(scores[SafeInt<ptrdiff_t>(i - batch)], z_data + i * n_targets_or_classes_, -1,
                             label_data == nullptr ? nullptr : (label_data + i));
        }
      }

    } else if (n_trees_ >= max_num_threads) { /* section: D2: 2+ outputs, 2+ rows, enough trees to parallelize*/
      auto num_threads = std::min<int32_t>(max_num_threads, SafeInt<int32_t>(n_trees_));
      std::vector<InlinedVector<ScoreValue<ThresholdType>>> scores(SafeInt<size_t>(num_threads) * N);
      int64_t end_n, begin_n = 0;
      while (begin_n < N) {
        end_n = std::min(N, begin_n + parallel_tree_N_);
        concurrency::ThreadPool::TrySimpleParallelFor(
            ttp,
            num_threads,
            [this, &agg, &scores, num_threads, x_data, N, stride, begin_n, end_n](ptrdiff_t batch_num) {
              auto work = concurrency::ThreadPool::PartitionWork(batch_num, num_threads, onnxruntime::narrow<size_t>(this->n_trees_));
              for (int64_t i = begin_n; i < end_n; ++i) {
                scores[batch_num * SafeInt<ptrdiff_t>(N) + i].resize(onnxruntime::narrow<size_t>(n_targets_or_classes_), {0, 0});
              }
              for (auto j = work.start; j < work.end; ++j) {
                for (int64_t i = begin_n; i < end_n; ++i) {
                  agg.ProcessTreeNodePrediction(scores[batch_num * SafeInt<ptrdiff_t>(N) + i],
                                                *ProcessTreeNodeLeave(roots_[j], x_data + i * stride), weights_);
                }
              }
            });
        begin_n = end_n;
      }
      concurrency::ThreadPool::TrySimpleParallelFor(
          ttp,
          num_threads,
          [this, &agg, &scores, num_threads, label_data, z_data, N](ptrdiff_t batch_num) {
            auto work = concurrency::ThreadPool::PartitionWork(batch_num, onnxruntime::narrow<ptrdiff_t>(num_threads), onnxruntime::narrow<ptrdiff_t>(N));
            for (auto i = work.start; i < work.end; ++i) {
              for (int64_t j = 1; j < num_threads; ++j) {
                agg.MergePrediction(scores[i], scores[j * SafeInt<ptrdiff_t>(N) + i]);
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
            InlinedVector<ScoreValue<ThresholdType>> scores(onnxruntime::narrow<size_t>(n_targets_or_classes_));
            auto work = concurrency::ThreadPool::PartitionWork(batch_num, onnxruntime::narrow<ptrdiff_t>(num_threads), onnxruntime::narrow<ptrdiff_t>(N));

            for (auto i = work.start; i < work.end; ++i) {
              std::fill(scores.begin(), scores.end(), ScoreValue<ThresholdType>({0, 0}));
              for (j = 0, limit = roots_.size(); j < limit; ++j) {
                agg.ProcessTreeNodePrediction(scores, *ProcessTreeNodeLeave(roots_[j], x_data + i * stride), weights_);
              }

              agg.FinalizeScores(scores,
                                 z_data + i * n_targets_or_classes_, -1,
                                 label_data == nullptr ? nullptr : (label_data + i));
            }
          });
    }
  }
}  // namespace detail

#define TREE_FIND_VALUE(CMP)                                    \
  if (has_missing_tracks_) {                                    \
    while (root->is_not_leaf()) {                               \
      val = x_data[root->feature_id];                           \
      root += (val CMP root->value_or_unique_weight ||          \
               (root->is_missing_track_true() && _isnan_(val))) \
                  ? root->truenode_inc_or_first_weight          \
                  : root->falsenode_inc_or_n_weights;           \
    }                                                           \
  } else {                                                      \
    while (root->is_not_leaf()) {                               \
      val = x_data[root->feature_id];                           \
      root += val CMP root->value_or_unique_weight              \
                  ? root->truenode_inc_or_first_weight          \
                  : root->falsenode_inc_or_n_weights;           \
    }                                                           \
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
    switch (root->mode()) {
      case NODE_MODE::BRANCH_LEQ:
        if (has_missing_tracks_) {
          while (root->is_not_leaf()) {
            val = x_data[root->feature_id];
            root += (val <= root->value_or_unique_weight ||
                     (root->is_missing_track_true() && _isnan_(val)))
                        ? root->truenode_inc_or_first_weight
                        : root->falsenode_inc_or_n_weights;
          }
        } else {
          while (root->is_not_leaf()) {
            val = x_data[root->feature_id];
            root += val <= root->value_or_unique_weight ? root->truenode_inc_or_first_weight : root->falsenode_inc_or_n_weights;
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
    while (root->is_not_leaf()) {
      val = x_data[root->feature_id];
      threshold = root->value_or_unique_weight;
      switch (root->mode()) {
        case NODE_MODE::BRANCH_LEQ:
          root += val <= threshold || (root->is_missing_track_true() && _isnan_(val))
                      ? root->truenode_inc_or_first_weight
                      : root->falsenode_inc_or_n_weights;
          break;
        case NODE_MODE::BRANCH_LT:
          root += val < threshold || (root->is_missing_track_true() && _isnan_(val))
                      ? root->truenode_inc_or_first_weight
                      : root->falsenode_inc_or_n_weights;
          break;
        case NODE_MODE::BRANCH_GTE:
          root += val >= threshold || (root->is_missing_track_true() && _isnan_(val))
                      ? root->truenode_inc_or_first_weight
                      : root->falsenode_inc_or_n_weights;
          break;
        case NODE_MODE::BRANCH_GT:
          root += val > threshold || (root->is_missing_track_true() && _isnan_(val))
                      ? root->truenode_inc_or_first_weight
                      : root->falsenode_inc_or_n_weights;
          break;
        case NODE_MODE::BRANCH_EQ:
          root += val == threshold || (root->is_missing_track_true() && _isnan_(val))
                      ? root->truenode_inc_or_first_weight
                      : root->falsenode_inc_or_n_weights;
          break;
        case NODE_MODE::BRANCH_NEQ:
          root += val != threshold || (root->is_missing_track_true() && _isnan_(val))
                      ? root->truenode_inc_or_first_weight
                      : root->falsenode_inc_or_n_weights;
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
              int parallel_tree_N,
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
      128,
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
Status TreeEnsembleCommonClassifier<InputType, ThresholdType, OutputType>::Init(
    int parallel_tree,
    int parallel_tree_N,
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
  auto status = TreeEnsembleCommon<InputType, ThresholdType, OutputType>::Init(
      parallel_tree,
      parallel_tree_N,
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
      labels[i] = classlabels_strings_[onnxruntime::narrow<size_t>(plabel[i])];
  }
  return Status::OK();
}

}  // namespace detail
}  // namespace ml
}  // namespace onnxruntime
