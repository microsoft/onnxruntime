// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/inlined_containers.h"
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "ml_common.h"
#include <math.h>

namespace onnxruntime {
namespace ml {
namespace detail {

struct TreeNodeElementId {
  int64_t tree_id;
  int64_t node_id;
  bool operator==(const TreeNodeElementId& xyz) const {
    return (tree_id == xyz.tree_id) && (node_id == xyz.node_id);
  }
  bool operator<(const TreeNodeElementId& xyz) const {
    return ((tree_id < xyz.tree_id) || (tree_id == xyz.tree_id && node_id < xyz.node_id));
  }
  struct hash_fn {
    std::size_t operator()(const TreeNodeElementId& key) const {
      return static_cast<std::size_t>(static_cast<uint64_t>(key.tree_id) << 32 | static_cast<uint64_t>(key.node_id));
    }
  };
};

template <typename T>
struct SparseValue {
  int64_t i;
  T value;
};

template <typename T>
struct ScoreValue {
  T score;
  unsigned char has_score;
  operator T() const { return has_score ? score : 0; }
  T operator-() { return has_score ? -score : 0; }
  T operator*(float val) { return has_score ? score * static_cast<T>(val) : 0; }
  T operator*(double val) { return has_score ? score * static_cast<T>(val) : 0; }
  ScoreValue<T>& operator=(ScoreValue<T> v) {
    this->score = v.score;
    this->has_score = v.has_score;
    return *this;
  }
  ScoreValue<T>& operator=(float v) {
    this->score = static_cast<T>(v);
    this->has_score = 1;
    return *this;
  }
  ScoreValue<T>& operator=(double v) {
    this->score = static_cast<T>(v);
    this->has_score = 1;
    return *this;
  }
};

enum MissingTrack : uint8_t {
  kTrue = 16,
  kFalse = 0
};

template <typename T>
struct TreeNodeElement;

template <typename T>
union PtrOrWeight {
  TreeNodeElement<T>* ptr;
  struct WeightData {
    int32_t weight;
    int32_t n_weights;
  } weight_data;
};

template <typename T>
struct TreeNodeElement {
  int feature_id;

  // Stores the node threshold or the weights if the tree has one target.
  T value_or_unique_weight;

  // The onnx specification says hitrates is used to store information about the node,
  // but this information is not used for inference.
  // T hitrates;

  // PtrOrWeight acts as a tagged union, with the "tag" being whether the node is a leaf or not (see `is_not_leaf`).

  // If it is not a leaf, it is a pointer to the true child node when traversing the decision tree. The false branch is
  // always 1 position away from the TreeNodeElement in practice in `TreeEnsembleCommon::nodes_` so it is not stored.

  // If it is a leaf, it contains `weight` and `n_weights` attributes which are used to indicate the position of the
  // weight in array `TreeEnsembleCommon::weights_`. If the number of targets or classes is one, the weight is also
  // stored in `value_or_unique_weight`.
  PtrOrWeight<T> truenode_or_weight;
  uint8_t flags;

  inline NODE_MODE mode() const { return NODE_MODE(flags & 0xF); }
  inline bool is_not_leaf() const { return !(flags & NODE_MODE::LEAF); }
  inline bool is_missing_track_true() const { return flags & MissingTrack::kTrue; }
};

template <typename InputType, typename ThresholdType, typename OutputType>
class TreeAggregator {
 protected:
  size_t n_trees_;
  int64_t n_targets_or_classes_;
  POST_EVAL_TRANSFORM post_transform_;
  const std::vector<ThresholdType>& base_values_;
  ThresholdType origin_;
  bool use_base_values_;

 public:
  TreeAggregator(size_t n_trees,
                 const int64_t& n_targets_or_classes,
                 POST_EVAL_TRANSFORM post_transform,
                 const std::vector<ThresholdType>& base_values) : n_trees_(n_trees), n_targets_or_classes_(n_targets_or_classes), post_transform_(post_transform), base_values_(base_values) {
    origin_ = base_values_.size() == 1 ? base_values_[0] : 0;
    use_base_values_ = base_values_.size() == static_cast<size_t>(n_targets_or_classes_);
  }

  // 1 output

  void ProcessTreeNodePrediction1(ScoreValue<ThresholdType>& /*prediction*/,
                                  const TreeNodeElement<ThresholdType>& /*root*/) const {}

  void MergePrediction1(ScoreValue<ThresholdType>& /*prediction*/, ScoreValue<ThresholdType>& /*prediction2*/) const {}

  void FinalizeScores1(OutputType* Z, ScoreValue<ThresholdType>& prediction, int64_t* /*Y*/) const {
    prediction.score = prediction.has_score ? (prediction.score + origin_) : origin_;
    *Z = this->post_transform_ == POST_EVAL_TRANSFORM::PROBIT
             ? static_cast<OutputType>(ComputeProbit(static_cast<float>(prediction.score)))
             : static_cast<OutputType>(prediction.score);
  }

  // N outputs

  void ProcessTreeNodePrediction(InlinedVector<ScoreValue<ThresholdType>>& /*predictions*/,
                                 const TreeNodeElement<ThresholdType>& /*root*/,
                                 gsl::span<const SparseValue<ThresholdType>> /*weights*/) const {}

  void MergePrediction(InlinedVector<ScoreValue<ThresholdType>>& /*predictions*/,
                       const InlinedVector<ScoreValue<ThresholdType>>& /*predictions2*/) const {}

  void FinalizeScores(InlinedVector<ScoreValue<ThresholdType>>& predictions,
                      OutputType* Z, int add_second_class, int64_t*) const {
    ORT_ENFORCE(predictions.size() == (size_t)n_targets_or_classes_);
    ThresholdType val;
    auto it = predictions.begin();
    for (size_t jt = 0; jt < onnxruntime::narrow<size_t>(n_targets_or_classes_); ++jt, ++it) {
      val = use_base_values_ ? base_values_[jt] : 0.f;
      val += it->has_score ? it->score : 0;
      it->score = val;
    }
    write_scores(predictions, post_transform_, Z, add_second_class);
  }
};

/////////////
// regression
/////////////

template <typename InputType, typename ThresholdType, typename OutputType>
class TreeAggregatorSum : public TreeAggregator<InputType, ThresholdType, OutputType> {
 public:
  TreeAggregatorSum(size_t n_trees,
                    const int64_t& n_targets_or_classes,
                    POST_EVAL_TRANSFORM post_transform,
                    const std::vector<ThresholdType>& base_values) : TreeAggregator<InputType, ThresholdType, OutputType>(n_trees,
                                                                                                                          n_targets_or_classes, post_transform, base_values) {}

  // 1 output

  void ProcessTreeNodePrediction1(ScoreValue<ThresholdType>& prediction,
                                  const TreeNodeElement<ThresholdType>& root) const {
    prediction.score += root.value_or_unique_weight;
  }

  void MergePrediction1(ScoreValue<ThresholdType>& prediction,
                        const ScoreValue<ThresholdType>& prediction2) const {
    prediction.score += prediction2.score;
  }

  void FinalizeScores1(OutputType* Z, ScoreValue<ThresholdType>& prediction, int64_t* /*Y*/) const {
    prediction.score += this->origin_;
    *Z = this->post_transform_ == POST_EVAL_TRANSFORM::PROBIT
             ? static_cast<OutputType>(ComputeProbit(static_cast<float>(prediction.score)))
             : static_cast<OutputType>(prediction.score);
  }

  // N outputs

  void ProcessTreeNodePrediction(InlinedVector<ScoreValue<ThresholdType>>& predictions,
                                 const TreeNodeElement<ThresholdType>& root,
                                 gsl::span<const SparseValue<ThresholdType>> weights) const {
    auto it = weights.begin() + root.truenode_or_weight.weight_data.weight;
    for (int32_t i = 0; i < root.truenode_or_weight.weight_data.n_weights; ++i, ++it) {
      ORT_ENFORCE(it->i < (int64_t)predictions.size());
      predictions[onnxruntime::narrow<size_t>(it->i)].score += it->value;
      predictions[onnxruntime::narrow<size_t>(it->i)].has_score = 1;
    }
  }

  void MergePrediction(InlinedVector<ScoreValue<ThresholdType>>& predictions,
                       const InlinedVector<ScoreValue<ThresholdType>>& predictions2) const {
    ORT_ENFORCE(predictions.size() == predictions2.size());
    for (size_t i = 0; i < predictions.size(); ++i) {
      if (predictions2[i].has_score) {
        predictions[i].score += predictions2[i].score;
        predictions[i].has_score = 1;
      }
    }
  }

  void FinalizeScores(InlinedVector<ScoreValue<ThresholdType>>& predictions,
                      OutputType* Z, int add_second_class, int64_t*) const {
    auto it = predictions.begin();
    if (this->use_base_values_) {
      auto it2 = this->base_values_.cbegin();
      for (; it != predictions.end(); ++it, ++it2)
        it->score = it->score + *it2;
    }
    write_scores(predictions, this->post_transform_, Z, add_second_class);
  }
};

template <typename InputType, typename ThresholdType, typename OutputType>
class TreeAggregatorAverage : public TreeAggregatorSum<InputType, ThresholdType, OutputType> {
 public:
  TreeAggregatorAverage(size_t n_trees,
                        const int64_t& n_targets_or_classes,
                        POST_EVAL_TRANSFORM post_transform,
                        const std::vector<ThresholdType>& base_values) : TreeAggregatorSum<InputType, ThresholdType, OutputType>(n_trees,
                                                                                                                                 n_targets_or_classes,
                                                                                                                                 post_transform,
                                                                                                                                 base_values) {}

  void FinalizeScores1(OutputType* Z, ScoreValue<ThresholdType>& prediction, int64_t* /*Y*/) const {
    prediction.score /= this->n_trees_;
    prediction.score += this->origin_;
    *Z = this->post_transform_ == POST_EVAL_TRANSFORM::PROBIT
             ? static_cast<OutputType>(ComputeProbit(static_cast<float>(prediction.score)))
             : static_cast<OutputType>(prediction.score);
  }

  void FinalizeScores(InlinedVector<ScoreValue<ThresholdType>>& predictions,
                      OutputType* Z, int add_second_class, int64_t*) const {
    if (this->use_base_values_) {
      ORT_ENFORCE(this->base_values_.size() == predictions.size());
      auto it = predictions.begin();
      auto it2 = this->base_values_.cbegin();
      for (; it != predictions.end(); ++it, ++it2)
        it->score = it->score / this->n_trees_ + *it2;
    } else {
      auto it = predictions.begin();
      for (; it != predictions.end(); ++it)
        it->score /= this->n_trees_;
    }
    write_scores(predictions, this->post_transform_, Z, add_second_class);
  }
};

template <typename InputType, typename ThresholdType, typename OutputType>
class TreeAggregatorMin : public TreeAggregator<InputType, ThresholdType, OutputType> {
 public:
  TreeAggregatorMin(size_t n_trees,
                    const int64_t& n_targets_or_classes,
                    POST_EVAL_TRANSFORM post_transform,
                    const std::vector<ThresholdType>& base_values) : TreeAggregator<InputType, ThresholdType, OutputType>(n_trees,
                                                                                                                          n_targets_or_classes,
                                                                                                                          post_transform,
                                                                                                                          base_values) {}

  // 1 output

  void ProcessTreeNodePrediction1(ScoreValue<ThresholdType>& prediction,
                                  const TreeNodeElement<ThresholdType>& root) const {
    prediction.score = (!(prediction.has_score) || root.value_or_unique_weight < prediction.score)
                           ? root.value_or_unique_weight
                           : prediction.score;
    prediction.has_score = 1;
  }

  void MergePrediction1(ScoreValue<ThresholdType>& prediction,
                        const ScoreValue<ThresholdType>& prediction2) const {
    if (prediction2.has_score) {
      prediction.score = prediction.has_score && (prediction.score < prediction2.score)
                             ? prediction.score
                             : prediction2.score;
      prediction.has_score = 1;
    }
  }

  // N outputs

  void ProcessTreeNodePrediction(InlinedVector<ScoreValue<ThresholdType>>& predictions,
                                 const TreeNodeElement<ThresholdType>& root,
                                 gsl::span<const SparseValue<ThresholdType>> weights) const {
    auto it = weights.begin() + root.truenode_or_weight.weight_data.weight;
    for (int32_t i = 0; i < root.truenode_or_weight.weight_data.n_weights; ++i, ++it) {
      predictions[onnxruntime::narrow<size_t>(it->i)].score =
          (!predictions[onnxruntime::narrow<size_t>(it->i)].has_score || it->value < predictions[onnxruntime::narrow<size_t>(it->i)].score)
              ? it->value
              : predictions[onnxruntime::narrow<size_t>(it->i)].score;
      predictions[onnxruntime::narrow<size_t>(it->i)].has_score = 1;
    }
  }

  void MergePrediction(InlinedVector<ScoreValue<ThresholdType>>& predictions,
                       const InlinedVector<ScoreValue<ThresholdType>>& predictions2) const {
    ORT_ENFORCE(predictions.size() == predictions2.size());
    for (size_t i = 0; i < predictions.size(); ++i) {
      if (predictions2[i].has_score) {
        predictions[i].score = predictions[i].has_score && (predictions[i].score < predictions2[i].score)
                                   ? predictions[i].score
                                   : predictions2[i].score;
        predictions[i].has_score = 1;
      }
    }
  }
};

template <typename InputType, typename ThresholdType, typename OutputType>
class TreeAggregatorMax : public TreeAggregator<InputType, ThresholdType, OutputType> {
 public:
  TreeAggregatorMax<InputType, ThresholdType, OutputType>(size_t n_trees,
                                                          const int64_t& n_targets_or_classes,
                                                          POST_EVAL_TRANSFORM post_transform,
                                                          const std::vector<ThresholdType>& base_values) : TreeAggregator<InputType, ThresholdType, OutputType>(n_trees, n_targets_or_classes,
                                                                                                                                                                post_transform, base_values) {}

  // 1 output

  void ProcessTreeNodePrediction1(ScoreValue<ThresholdType>& prediction,
                                  const TreeNodeElement<ThresholdType>& root) const {
    prediction.score = (!(prediction.has_score) || root.value_or_unique_weight > prediction.score)
                           ? root.value_or_unique_weight
                           : prediction.score;
    prediction.has_score = 1;
  }

  void MergePrediction1(ScoreValue<ThresholdType>& prediction, const ScoreValue<ThresholdType>& prediction2) const {
    if (prediction2.has_score) {
      prediction.score = prediction.has_score && (prediction.score > prediction2.score)
                             ? prediction.score
                             : prediction2.score;
      prediction.has_score = 1;
    }
  }

  // N outputs

  void ProcessTreeNodePrediction(InlinedVector<ScoreValue<ThresholdType>>& predictions,
                                 const TreeNodeElement<ThresholdType>& root,
                                 gsl::span<const SparseValue<ThresholdType>> weights) const {
    auto it = weights.begin() + root.truenode_or_weight.weight_data.weight;
    for (int32_t i = 0; i < root.truenode_or_weight.weight_data.n_weights; ++i, ++it) {
      predictions[onnxruntime::narrow<size_t>(it->i)].score =
          (!predictions[onnxruntime::narrow<size_t>(it->i)].has_score || it->value > predictions[onnxruntime::narrow<size_t>(it->i)].score)
              ? it->value
              : predictions[onnxruntime::narrow<size_t>(it->i)].score;
      predictions[onnxruntime::narrow<size_t>(it->i)].has_score = 1;
    }
  }

  void MergePrediction(InlinedVector<ScoreValue<ThresholdType>>& predictions,
                       const InlinedVector<ScoreValue<ThresholdType>>& predictions2) const {
    ORT_ENFORCE(predictions.size() == predictions2.size());
    for (size_t i = 0; i < predictions.size(); ++i) {
      if (predictions2[i].has_score) {
        predictions[i].score = predictions[i].has_score && (predictions[i].score > predictions2[i].score)
                                   ? predictions[i].score
                                   : predictions2[i].score;
        predictions[i].has_score = 1;
      }
    }
  }
};

/////////////////
// classification
/////////////////

template <typename InputType, typename ThresholdType, typename OutputType>
class TreeAggregatorClassifier : public TreeAggregatorSum<InputType, ThresholdType, OutputType> {
 private:
  const std::vector<int64_t>& class_labels_;
  bool binary_case_;
  bool weights_are_all_positive_;
  int64_t positive_label_;
  int64_t negative_label_;

 public:
  TreeAggregatorClassifier(size_t n_trees,
                           const int64_t& n_targets_or_classes,
                           POST_EVAL_TRANSFORM post_transform,
                           const std::vector<ThresholdType>& base_values,
                           const std::vector<int64_t>& class_labels,
                           bool binary_case,
                           bool weights_are_all_positive,
                           int64_t positive_label = 1,
                           int64_t negative_label = 0) : TreeAggregatorSum<InputType, ThresholdType, OutputType>(n_trees, n_targets_or_classes,
                                                                                                                 post_transform, base_values),
                                                         class_labels_(class_labels),
                                                         binary_case_(binary_case),
                                                         weights_are_all_positive_(weights_are_all_positive),
                                                         positive_label_(positive_label),
                                                         negative_label_(negative_label) {}

  void get_max_weight(const InlinedVector<ScoreValue<ThresholdType>>& classes, int64_t& maxclass,
                      ThresholdType& maxweight) const {
    maxclass = -1;
    maxweight = 0;
    for (auto it = classes.cbegin(); it != classes.cend(); ++it) {
      if (it->has_score && (maxclass == -1 || it->score > maxweight)) {
        maxclass = (int64_t)(it - classes.cbegin());
        maxweight = it->score;
      }
    }
  }

  int64_t _set_score_binary(int& write_additional_scores,
                            const InlinedVector<ScoreValue<ThresholdType>>& classes) const {
    ORT_ENFORCE(classes.size() == 2 || classes.size() == 1);
    return (classes.size() == 2 && classes[1].has_score)
               ? _set_score_binary(write_additional_scores, classes[0].score,
                                   classes[0].has_score, classes[1].score, classes[1].has_score)
               : _set_score_binary(write_additional_scores, classes[0].score, classes[0].has_score, 0, 0);
  }

  int64_t _set_score_binary(int& write_additional_scores, ThresholdType score0, unsigned char has_score0,
                            ThresholdType score1, unsigned char has_score1) const {
    ThresholdType pos_weight = has_score1 ? score1 : (has_score0 ? score0 : 0);  // only 1 class
    if (binary_case_) {
      if (weights_are_all_positive_) {
        if (pos_weight > 0.5) {
          write_additional_scores = 0;
          return class_labels_[1];  // positive label
        } else {
          write_additional_scores = 1;
          return class_labels_[0];  // negative label
        }
      } else {
        if (pos_weight > 0) {
          write_additional_scores = 2;
          return class_labels_[1];  // positive label
        } else {
          write_additional_scores = 3;
          return class_labels_[0];  // negative label
        }
      }
    }
    return (pos_weight > 0)
               ? positive_label_   // positive label
               : negative_label_;  // negative label
  }

  // 1 output

  void FinalizeScores1(OutputType* Z, ScoreValue<ThresholdType>& prediction, int64_t* Y) const {
    InlinedVector<ThresholdType> scores(2);
    unsigned char has_scores[2] = {1, 0};

    int write_additional_scores = -1;
    if (this->base_values_.size() == 2) {
      // add base_values
      prediction.score += this->base_values_[1];
      scores[1] = prediction.score;
      scores[0] = -scores[1];
      // has_score = true;
      has_scores[1] = 1;
      *Y = _set_score_binary(write_additional_scores, scores[0], has_scores[0], scores[1], has_scores[1]);
    } else if (this->base_values_.size() == 1) {
      // ONNX is vague about two classes and only one base_values.
      prediction.score += this->base_values_[0];
      scores[0] = prediction.score;
      scores.pop_back();
      *Y = _set_score_binary(write_additional_scores, scores[0], has_scores[0], 0, 0);
    } else if (this->base_values_.empty()) {
      scores[0] = prediction.score;
      scores.pop_back();
      *Y = _set_score_binary(write_additional_scores, scores[0], has_scores[0], 0, 0);
    } else {
      scores[0] = prediction.score;
      scores.pop_back();
      *Y = _set_score_binary(write_additional_scores, scores[0], has_scores[0], 0, 0);
    }

    write_scores(scores, this->post_transform_, Z, write_additional_scores);
  }

  // N outputs

  void FinalizeScores(InlinedVector<ScoreValue<ThresholdType>>& predictions,
                      OutputType* Z, int /*add_second_class*/, int64_t* Y = 0) const {
    ThresholdType maxweight = 0;
    int64_t maxclass = -1;

    int write_additional_scores = -1;
    if (this->n_targets_or_classes_ > 2) {
      // add base values
      for (size_t k = 0; k < this->base_values_.size(); ++k) {
        if (!predictions[k].has_score) {
          predictions[k].has_score = 1;
          predictions[k].score = this->base_values_[k];
        } else {
          predictions[k].score += this->base_values_[k];
        }
      }
      get_max_weight(predictions, maxclass, maxweight);
      *Y = class_labels_[onnxruntime::narrow<size_t>(maxclass)];
    } else {  // binary case
      ORT_ENFORCE(predictions.size() == 2);
      if (this->base_values_.size() == 2) {
        // add base values
        if (predictions[1].has_score) {
          // base_value_[0] is not used.
          // It assumes base_value[0] == base_value[1] in this case.
          // The specification does not forbid it but does not
          // say what the output should be in that case.
          predictions[1].score = this->base_values_[1] + predictions[0].score;
          predictions[0].score = -predictions[1].score;
          predictions[1].has_score = 1;
        } else {
          // binary as multiclass
          predictions[1].score += this->base_values_[1];
          predictions[0].score += this->base_values_[0];
        }
      } else if (this->base_values_.size() == 1) {
        // ONNX is vague about two classes and only one base_values.
        predictions[0].score += this->base_values_[0];
        if (!predictions[1].has_score)
          predictions.pop_back();
      } else if (this->base_values_.empty()) {
        write_additional_scores = 3;
        if (!predictions[1].has_score)
          predictions.pop_back();
      }

      *Y = _set_score_binary(write_additional_scores, predictions);
    }
    write_scores(predictions, this->post_transform_, Z, write_additional_scores);
    if (predictions.size() == 1)
      predictions.resize(2);
  }
};

}  // namespace detail
}  // namespace ml
}  // namespace onnxruntime
