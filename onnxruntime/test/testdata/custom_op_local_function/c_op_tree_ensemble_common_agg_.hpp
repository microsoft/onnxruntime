#pragma once

// Inspired from
// https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/tree_ensemble_regressor.cc.

#include "c_op_common_parameters.h"
#include "c_op_math.h"
#include "c_op_status.h"
#include "onnx_extended_helpers.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <thread>
#include <vector>

namespace onnx_c_ops {

struct TreeNodeElementId {
  int64_t tree_id;
  int64_t node_id;
  bool operator==(const TreeNodeElementId &xyz) const {
    return (tree_id == xyz.tree_id) && (node_id == xyz.node_id);
  }
  bool operator<(const TreeNodeElementId &xyz) const {
    return ((tree_id < xyz.tree_id) || (tree_id == xyz.tree_id && node_id < xyz.node_id));
  }
  struct hash_fn {
    std::size_t operator()(const TreeNodeElementId &key) const {
      return static_cast<std::size_t>(static_cast<uint64_t>(key.tree_id) << 32 |
                                      static_cast<uint64_t>(key.node_id));
    }
  };
};

template <typename T> struct SparseValue {
  int64_t i;
  T value;
};

template <typename T> struct ScoreValue {
  T score;
  unsigned char has_score;
  inline ScoreValue() {
    score = static_cast<T>(0);
    has_score = 1;
  }
  inline ScoreValue(float t) {
    score = static_cast<T>(t);
    has_score = 1;
  }
  inline ScoreValue(int t) {
    score = static_cast<T>(t);
    has_score = 1;
  }
  inline ScoreValue(double t) {
    score = static_cast<T>(t);
    has_score = 1;
  }
  inline ScoreValue(float t, unsigned char h) {
    score = static_cast<T>(t);
    has_score = h;
  }
  inline ScoreValue(int t, unsigned char h) {
    score = static_cast<T>(t);
    has_score = h;
  }
  inline ScoreValue(double t, unsigned char h) {
    score = static_cast<T>(t);
    has_score = h;
  }
  inline operator T() const { return has_score ? score : 0; }

  inline ScoreValue<T> &operator=(const ScoreValue<T> &v) {
    this->score = v.score;
    this->has_score = v.has_score;
    return *this;
  }
  inline ScoreValue<T> &operator=(int v) {
    this->score = static_cast<T>(v);
    this->has_score = 1;
    return *this;
  }
  inline ScoreValue<T> &operator=(float v) {
    this->score = static_cast<T>(v);
    this->has_score = 1;
    return *this;
  }
  inline ScoreValue<T> &operator=(double v) {
    this->score = static_cast<T>(v);
    this->has_score = 1;
    return *this;
  }

  inline ScoreValue<T> &operator+=(int v) {
    this->score += static_cast<T>(v);
    this->has_score = 1;
    return *this;
  }
  inline ScoreValue<T> &operator+=(float v) {
    this->score += static_cast<T>(v);
    this->has_score = 1;
    return *this;
  }
  inline ScoreValue<T> &operator+=(double v) {
    this->score += static_cast<T>(v);
    this->has_score = 1;
    return *this;
  }

  inline ScoreValue<T> &operator/=(int v) {
    this->score /= static_cast<T>(v);
    this->has_score = 1;
    return *this;
  }
  inline ScoreValue<T> &operator/=(float v) {
    this->score /= static_cast<T>(v);
    this->has_score = 1;
    return *this;
  }
  inline ScoreValue<T> &operator/=(double v) {
    this->score /= static_cast<T>(v);
    this->has_score = 1;
    return *this;
  }

  inline ScoreValue<T> &operator*=(int v) {
    this->score *= static_cast<T>(v);
    this->has_score = 1;
    return *this;
  }
  inline ScoreValue<T> &operator*=(float v) {
    this->score *= static_cast<T>(v);
    this->has_score = 1;
    return *this;
  }
  inline ScoreValue<T> &operator*=(double v) {
    this->score *= static_cast<T>(v);
    this->has_score = 1;
    return *this;
  }
};

enum MissingTrack : uint8_t { kTrue = 16, kFalse = 0 };

template <typename T> struct TreeNodeElement {
  int feature_id;

  // Stores the node threshold or the weights if the tree has one target.
  T value_or_unique_weight;

  // onnx specification says hitrates is used to store information about the
  // node, but this information is not used for inference. T hitrates;

  // True node, false node are obtained by computing `this +
  // truenode_inc_or_first_weight`, `this + falsenode_inc_or_n_weights` if the
  // node is not a leaf. In case of a leaf, these attributes are used to
  // indicate the position of the weight in array
  // `TreeEnsembleCommon::weights_`. If the number of targets or classes is one,
  // the weight is also stored in `value_or_unique_weight`.
  // This implementation assumes a tree has less than 2^31 nodes,
  // and the total number of leave in the set of trees is below 2^31.
  // A node cannot point to itself.
  int32_t truenode_inc_or_first_weight;
  // In case of a leaf, the following attribute indicates the number of weights
  // in array `TreeEnsembleCommon::weights_`. If not a leaf, it indicates
  // `this + falsenode_inc_or_n_weights` is the false node.
  // A node cannot point to itself.
  int32_t falsenode_inc_or_n_weights;
  uint8_t flags;

  inline NODE_MODE mode() const { return NODE_MODE(flags & 0xF); }
  inline bool is_not_leaf() const { return !(flags & NODE_MODE::LEAF); }
  inline bool is_missing_track_true() const { return flags & MissingTrack::kTrue; }
};

enum MissingTrack3 : uint8_t { kTrue0 = 16, kTrue1 = 32, kTrue2 = 64, kChildren3 = 128 };

template <typename T> struct TreeNodeElement3 {
  // This structure is equivalent to 3 nodes TreeNodeElement.
  // It allows to save (11*4+4)/((4*4+1)*3)=48/51 ~ 5% reduction.
  T thresholds[4];
  int32_t node_id[4];
  int feature_id[3];
  uint32_t flags;

  inline NODE_MODE mode() const { return NODE_MODE(flags & 0xF); }
  inline bool is_not_leaf() const { return !(flags & NODE_MODE::LEAF); }
  inline bool is_missing_track_true0() const { return flags & MissingTrack3::kTrue0; }
  inline bool is_missing_track_true1() const { return flags & MissingTrack3::kTrue1; }
  inline bool is_missing_track_true2() const { return flags & MissingTrack3::kTrue2; }
  inline bool children_are_tree_element3() const { return flags & MissingTrack3::kChildren3; }
};

template <typename InputType, typename ThresholdType, typename OutputType>
class TreeAggregator {
protected:
  std::size_t n_trees_;
  int64_t n_targets_or_classes_;
  POST_EVAL_TRANSFORM post_transform_;
  const std::vector<ThresholdType> &base_values_;
  ThresholdType origin_;
  bool use_base_values_;
  OutputType bias_;

public:
  TreeAggregator(std::size_t n_trees, const int64_t &n_targets_or_classes,
                 POST_EVAL_TRANSFORM post_transform,
                 const std::vector<ThresholdType> &base_values, OutputType bias)
      : n_trees_(n_trees), n_targets_or_classes_(n_targets_or_classes),
        post_transform_(post_transform), base_values_(base_values), bias_(bias) {
    origin_ = base_values_.size() == 1 ? base_values_[0] : 0;
    use_base_values_ = base_values_.size() == static_cast<std::size_t>(n_targets_or_classes_);
  }

  // 1 output

  void ProcessTreeNodePrediction1(ScoreValue<ThresholdType> & /*prediction*/,
                                  const TreeNodeElement<ThresholdType> & /*root*/) const {}

  void MergePrediction1(ScoreValue<ThresholdType> & /*prediction*/,
                        ScoreValue<ThresholdType> & /*prediction2*/) const {}

  void FinalizeScores1(OutputType *Z, ScoreValue<ThresholdType> &prediction,
                       int64_t * /*Y*/) const {
    prediction.score =
        prediction.has_score ? (prediction.score + origin_ + bias_) : origin_ + bias_;
    *Z = this->post_transform_ == POST_EVAL_TRANSFORM::PROBIT
             ? static_cast<OutputType>(ComputeProbit(static_cast<float>(prediction.score)))
             : static_cast<OutputType>(prediction.score);
  }

  // N outputs

  void ProcessTreeNodePrediction(
      InlinedVector<ScoreValue<ThresholdType>> & /*predictions*/,
      const TreeNodeElement<ThresholdType> & /*root*/,
      const InlinedVector<SparseValue<ThresholdType>> & /*weights*/) const {}

  void
  MergePrediction(InlinedVector<ScoreValue<ThresholdType>> & /*predictions*/,
                  const InlinedVector<ScoreValue<ThresholdType>> & /*predictions2*/) const {}

  void FinalizeScores(InlinedVector<ScoreValue<ThresholdType>> &predictions, OutputType *Z,
                      int add_second_class, int64_t *) const {
    EXT_ENFORCE(predictions.size() == (std::size_t)n_targets_or_classes_);
    ThresholdType val;
    auto it = predictions.begin();
    for (std::size_t jt = 0; jt < static_cast<std::size_t>(n_targets_or_classes_); ++jt, ++it) {
      val = use_base_values_ ? base_values_[jt] : 0.f;
      val += it->has_score ? it->score : 0;
      it->score = val + bias_;
    }
    write_scores(predictions, post_transform_, Z, add_second_class);
  }

  const char *kind() const { return "NONE"; }
};

/////////////
// regression
/////////////

template <typename InputType, typename ThresholdType, typename OutputType>
class TreeAggregatorSum : public TreeAggregator<InputType, ThresholdType, OutputType> {
public:
  TreeAggregatorSum(std::size_t n_trees, const int64_t &n_targets_or_classes,
                    POST_EVAL_TRANSFORM post_transform,
                    const std::vector<ThresholdType> &base_values, OutputType bias)
      : TreeAggregator<InputType, ThresholdType, OutputType>(
            n_trees, n_targets_or_classes, post_transform, base_values, bias) {}

  // 1 output

  void ProcessTreeNodePrediction1(ScoreValue<ThresholdType> &prediction,
                                  const TreeNodeElement<ThresholdType> &root) const {
    prediction.score += root.value_or_unique_weight;
  }

  void MergePrediction1(ScoreValue<ThresholdType> &prediction,
                        const ScoreValue<ThresholdType> &prediction2) const {
    prediction.score += prediction2.score;
  }

  void FinalizeScores1(OutputType *Z, ScoreValue<ThresholdType> &prediction,
                       int64_t * /*Y*/) const {
    prediction.score += this->origin_ + this->bias_ * this->n_trees_;
    *Z = this->post_transform_ == POST_EVAL_TRANSFORM::PROBIT
             ? static_cast<OutputType>(ComputeProbit(static_cast<float>(prediction.score)))
             : static_cast<OutputType>(prediction.score);
  }

  // N outputs

  void
  ProcessTreeNodePrediction(InlinedVector<ScoreValue<ThresholdType>> &predictions,
                            const TreeNodeElement<ThresholdType> &root,
                            const InlinedVector<SparseValue<ThresholdType>> &weights) const {
    auto it = weights.begin() + root.truenode_inc_or_first_weight;
    for (int32_t i = 0; i < root.falsenode_inc_or_n_weights; ++i, ++it) {
      // EXT_ENFORCE(it->i < (int64_t)predictions.size());
      predictions[static_cast<std::size_t>(it->i)].score += it->value;
      predictions[static_cast<std::size_t>(it->i)].has_score = 1;
    }
  }

  void MergePrediction(InlinedVector<ScoreValue<ThresholdType>> &predictions,
                       const InlinedVector<ScoreValue<ThresholdType>> &predictions2) const {
    EXT_ENFORCE(predictions.size() == predictions2.size());
    for (std::size_t i = 0; i < predictions.size(); ++i) {
      if (predictions2[i].has_score) {
        predictions[i].score += predictions2[i].score;
        predictions[i].has_score = 1;
      }
    }
  }

  void FinalizeScores(InlinedVector<ScoreValue<ThresholdType>> &predictions, OutputType *Z,
                      int add_second_class, int64_t *) const {
    auto it = predictions.begin();
    if (this->use_base_values_) {
      auto it2 = this->base_values_.cbegin();
      OutputType bias = this->bias_ * this->n_trees_;
      for (; it != predictions.end(); ++it, ++it2)
        it->score = it->score + *it2 + bias;
    } else if (this->bias_ != 0) {
      OutputType bias = this->bias_ * this->n_trees_;
      for (; it != predictions.end(); ++it)
        it->score += bias;
    }
    write_scores(predictions, this->post_transform_, Z, add_second_class);
  }

  const char *kind() const { return "SUM"; }
};

template <typename InputType, typename ThresholdType, typename OutputType>
class TreeAggregatorAverage : public TreeAggregatorSum<InputType, ThresholdType, OutputType> {
public:
  TreeAggregatorAverage(std::size_t n_trees, const int64_t &n_targets_or_classes,
                        POST_EVAL_TRANSFORM post_transform,
                        const std::vector<ThresholdType> &base_values, OutputType bias)
      : TreeAggregatorSum<InputType, ThresholdType, OutputType>(
            n_trees, n_targets_or_classes, post_transform, base_values, bias) {}

  void FinalizeScores1(OutputType *Z, ScoreValue<ThresholdType> &prediction,
                       int64_t * /*Y*/) const {
    prediction.score /= this->n_trees_;
    prediction.score += this->origin_ + this->bias_;
    *Z = this->post_transform_ == POST_EVAL_TRANSFORM::PROBIT
             ? static_cast<OutputType>(ComputeProbit(static_cast<float>(prediction.score)))
             : static_cast<OutputType>(prediction.score);
  }

  void FinalizeScores(InlinedVector<ScoreValue<ThresholdType>> &predictions, OutputType *Z,
                      int add_second_class, int64_t *) const {
    if (this->use_base_values_) {
      EXT_ENFORCE(this->base_values_.size() == predictions.size());
      auto it = predictions.begin();
      auto it2 = this->base_values_.cbegin();
      for (; it != predictions.end(); ++it, ++it2) {
        it->score = it->score / this->n_trees_ + *it2 + this->bias_;
      }
    } else {
      auto it = predictions.begin();
      for (; it != predictions.end(); ++it) {
        it->score /= this->n_trees_;
        it->score += this->bias_;
      }
    }
    write_scores(predictions, this->post_transform_, Z, add_second_class);
  }

  const char *kind() const { return "AVERAGE"; }
};

template <typename InputType, typename ThresholdType, typename OutputType>
class TreeAggregatorMin : public TreeAggregator<InputType, ThresholdType, OutputType> {
public:
  TreeAggregatorMin(std::size_t n_trees, const int64_t &n_targets_or_classes,
                    POST_EVAL_TRANSFORM post_transform,
                    const std::vector<ThresholdType> &base_values, OutputType bias)
      : TreeAggregator<InputType, ThresholdType, OutputType>(
            n_trees, n_targets_or_classes, post_transform, base_values, bias) {
    EXT_ENFORCE(bias == 0);
  }

  // 1 output

  void ProcessTreeNodePrediction1(ScoreValue<ThresholdType> &prediction,
                                  const TreeNodeElement<ThresholdType> &root) const {
    prediction.score =
        (!(prediction.has_score) || root.value_or_unique_weight < prediction.score)
            ? root.value_or_unique_weight
            : prediction.score;
    prediction.has_score = 1;
  }

  void MergePrediction1(ScoreValue<ThresholdType> &prediction,
                        const ScoreValue<ThresholdType> &prediction2) const {
    if (prediction2.has_score) {
      prediction.score = prediction.has_score && (prediction.score < prediction2.score)
                             ? prediction.score
                             : prediction2.score;
      prediction.has_score = 1;
    }
  }

  // N outputs

  void
  ProcessTreeNodePrediction(InlinedVector<ScoreValue<ThresholdType>> &predictions,
                            const TreeNodeElement<ThresholdType> &root,
                            const InlinedVector<SparseValue<ThresholdType>> &weights) const {
    auto it = weights.begin() + root.truenode_inc_or_first_weight;
    for (int32_t i = 0; i < root.falsenode_inc_or_n_weights; ++i, ++it) {
      predictions[static_cast<std::size_t>(it->i)].score =
          (!predictions[static_cast<std::size_t>(it->i)].has_score ||
           it->value < predictions[static_cast<std::size_t>(it->i)].score)
              ? it->value
              : predictions[static_cast<std::size_t>(it->i)].score;
      predictions[static_cast<std::size_t>(it->i)].has_score = 1;
    }
  }

  void MergePrediction(InlinedVector<ScoreValue<ThresholdType>> &predictions,
                       const InlinedVector<ScoreValue<ThresholdType>> &predictions2) const {
    EXT_ENFORCE(predictions.size() == predictions2.size());
    for (std::size_t i = 0; i < predictions.size(); ++i) {
      if (predictions2[i].has_score) {
        predictions[i].score =
            predictions[i].has_score && (predictions[i].score < predictions2[i].score)
                ? predictions[i].score
                : predictions2[i].score;
        predictions[i].has_score = 1;
      }
    }
  }

  const char *kind() const { return "MIN"; }
};

template <typename InputType, typename ThresholdType, typename OutputType>
class TreeAggregatorMax : public TreeAggregator<InputType, ThresholdType, OutputType> {
public:
  TreeAggregatorMax(std::size_t n_trees, const int64_t &n_targets_or_classes,
                    POST_EVAL_TRANSFORM post_transform,
                    const std::vector<ThresholdType> &base_values, OutputType bias)
      : TreeAggregator<InputType, ThresholdType, OutputType>(
            n_trees, n_targets_or_classes, post_transform, base_values, bias) {
    EXT_ENFORCE(bias == 0);
  }

  // 1 output

  void ProcessTreeNodePrediction1(ScoreValue<ThresholdType> &prediction,
                                  const TreeNodeElement<ThresholdType> &root) const {
    prediction.score =
        (!(prediction.has_score) || root.value_or_unique_weight > prediction.score)
            ? root.value_or_unique_weight
            : prediction.score;
    prediction.has_score = 1;
  }

  void MergePrediction1(ScoreValue<ThresholdType> &prediction,
                        const ScoreValue<ThresholdType> &prediction2) const {
    if (prediction2.has_score) {
      prediction.score = prediction.has_score && (prediction.score > prediction2.score)
                             ? prediction.score
                             : prediction2.score;
      prediction.has_score = 1;
    }
  }

  // N outputs

  void
  ProcessTreeNodePrediction(InlinedVector<ScoreValue<ThresholdType>> &predictions,
                            const TreeNodeElement<ThresholdType> &root,
                            const InlinedVector<SparseValue<ThresholdType>> &weights) const {
    auto it = weights.begin() + root.truenode_inc_or_first_weight;
    for (int32_t i = 0; i < root.falsenode_inc_or_n_weights; ++i, ++it) {
      predictions[static_cast<std::size_t>(it->i)].score =
          (!predictions[static_cast<std::size_t>(it->i)].has_score ||
           it->value > predictions[static_cast<std::size_t>(it->i)].score)
              ? it->value
              : predictions[static_cast<std::size_t>(it->i)].score;
      predictions[static_cast<std::size_t>(it->i)].has_score = 1;
    }
  }

  void MergePrediction(InlinedVector<ScoreValue<ThresholdType>> &predictions,
                       const InlinedVector<ScoreValue<ThresholdType>> &predictions2) const {
    EXT_ENFORCE(predictions.size() == predictions2.size());
    for (std::size_t i = 0; i < predictions.size(); ++i) {
      if (predictions2[i].has_score) {
        predictions[i].score =
            predictions[i].has_score && (predictions[i].score > predictions2[i].score)
                ? predictions[i].score
                : predictions2[i].score;
        predictions[i].has_score = 1;
      }
    }
  }

  const char *kind() const { return "MAX"; }
};

/////////////////
// classification
/////////////////

template <typename InputType, typename ThresholdType, typename OutputType>
class TreeAggregatorClassifier
    : public TreeAggregatorSum<InputType, ThresholdType, OutputType> {
private:
  bool binary_case_;
  bool weights_are_all_positive_;
  int64_t positive_label_;
  int64_t negative_label_;

public:
  TreeAggregatorClassifier(std::size_t n_trees, const int64_t &n_targets_or_classes,
                           POST_EVAL_TRANSFORM post_transform,
                           const std::vector<ThresholdType> &base_values, OutputType bias,
                           bool binary_case, bool weights_are_all_positive,
                           int64_t positive_label = 1, int64_t negative_label = 0)
      : TreeAggregatorSum<InputType, ThresholdType, OutputType>(
            n_trees, n_targets_or_classes, post_transform, base_values, bias),
        binary_case_(binary_case), weights_are_all_positive_(weights_are_all_positive),
        positive_label_(positive_label), negative_label_(negative_label) {
    EXT_ENFORCE(bias == 0);
  }

  void get_max_weight(const InlinedVector<ScoreValue<ThresholdType>> &classes,
                      int64_t &maxclass, ThresholdType &maxweight) const {
    maxclass = -1;
    maxweight = 0;
    for (auto it = classes.cbegin(); it != classes.cend(); ++it) {
      if (it->has_score && (maxclass == -1 || it->score > maxweight)) {
        maxclass = (int64_t)(it - classes.cbegin());
        maxweight = it->score;
      }
    }
  }

  int64_t _set_score_binary(int &write_additional_scores,
                            const InlinedVector<ScoreValue<ThresholdType>> &classes) const {
    EXT_ENFORCE(classes.size() == 2 || classes.size() == 1);
    return (classes.size() == 2 && classes[1].has_score)
               ? _set_score_binary(write_additional_scores, classes[0].score,
                                   classes[0].has_score, classes[1].score, classes[1].has_score)
               : _set_score_binary(write_additional_scores, classes[0].score,
                                   classes[0].has_score, 0, 0);
  }

  int64_t _set_score_binary(int &write_additional_scores, ThresholdType score0,
                            unsigned char has_score0, ThresholdType score1,
                            unsigned char has_score1) const {
    ThresholdType pos_weight = has_score1 ? score1 : (has_score0 ? score0 : 0); // only 1 class
    if (binary_case_) {
      if (weights_are_all_positive_) {
        if (pos_weight > 0.5) {
          write_additional_scores = 0;
          return 1; // positive label
        } else {
          write_additional_scores = 1;
          return 0; // negative label
        }
      } else {
        if (pos_weight > 0) {
          write_additional_scores = 2;
          return 1; // positive label
        } else {
          write_additional_scores = 3;
          return 0; // negative label
        }
      }
    }
    return (pos_weight > 0) ? positive_label_  // positive label
                            : negative_label_; // negative label
  }

  // 1 output

  void FinalizeScores1(OutputType *Z, ScoreValue<ThresholdType> &prediction, int64_t *Y) const {
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
      *Y = _set_score_binary(write_additional_scores, scores[0], has_scores[0], scores[1],
                             has_scores[1]);
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

  void FinalizeScores(InlinedVector<ScoreValue<ThresholdType>> &predictions, OutputType *Z,
                      int /*add_second_class*/, int64_t *Y = 0) const {
    EXT_ENFORCE(Y != nullptr)
    ThresholdType maxweight = 0;
    int64_t maxclass = -1;

    int write_additional_scores = -1;
    if (this->n_targets_or_classes_ > 2) {
      // add base values
      for (std::size_t k = 0; k < this->base_values_.size(); ++k) {
        if (!predictions[k].has_score) {
          predictions[k].has_score = 1;
          predictions[k].score = this->base_values_[k];
        } else {
          predictions[k].score += this->base_values_[k];
        }
      }
      get_max_weight(predictions, maxclass, maxweight);
      *Y = maxclass;
    } else { // binary case
      EXT_ENFORCE(predictions.size() == 2);
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

  const char *kind() const { return "CLASSIFICATION"; }
};

} // namespace onnx_c_ops
