// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "ml_common.h"
#include <math.h>

namespace onnxruntime {
namespace ml {

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

template <typename T>
struct SparseValue {
  int64_t i;
  T value;
};

enum MissingTrack {
  NONE,
  TRUE,
  FALSE
};

template <typename T>
struct TreeNodeElement {
  TreeNodeElementId id;
  int feature_id;
  T value;
  T hitrates;
  NODE_MODE mode;
  TreeNodeElement<T>* truenode;
  TreeNodeElement<T>* falsenode;
  MissingTrack missing_tracks;
  std::vector<SparseValue<T>> weights;

  bool is_not_leave;
  bool is_missing_track_true;
};

template <typename ITYPE, typename OTYPE>
class _Aggregator {
 protected:
  size_t n_trees_;
  int64_t n_targets_or_classes_;
  POST_EVAL_TRANSFORM post_transform_;
  const std::vector<OTYPE>* base_values_;
  OTYPE origin_;
  bool use_base_values_;

 public:
  inline _Aggregator(size_t n_trees,
                     const int64_t& n_targets_or_classes,
                     POST_EVAL_TRANSFORM post_transform,
                     const std::vector<OTYPE>* base_values) : n_trees_(n_trees), n_targets_or_classes_(n_targets_or_classes), post_transform_(post_transform), base_values_(base_values) {
    origin_ = base_values_->size() == 1 ? (*base_values_)[0] : 0.f;
    use_base_values_ = base_values_->size() == static_cast<size_t>(n_targets_or_classes_);
  }

  const char* name() const { return "_Aggregator"; }

  // 1 output

  inline void ProcessTreeNodePrediction1(OTYPE* /*predictions*/, TreeNodeElement<OTYPE>* /*rooOTYPE*/,
                                         unsigned char* /*has_predictions*/) const {}

  inline void MergePrediction1(OTYPE* /*predictions*/, unsigned char* /*has_predictions*/,
                               OTYPE* /*predictions2*/, unsigned char* /*has_predictions2*/) const {}

  inline void FinalizeScores1(OTYPE* Z, OTYPE& val,
                              unsigned char& has_scores,
                              int64_t*) const {
    val = has_scores ? (val + origin_) : origin_;
    *Z = this->post_transform_ == POST_EVAL_TRANSFORM::PROBIT ? static_cast<OTYPE>(ComputeProbit(static_cast<float>(val))) : val;
  }

  // N outputs

  void ProcessTreeNodePrediction(OTYPE* /*predictions*/, TreeNodeElement<OTYPE>* /*rooOTYPE*/,
                                 unsigned char* /*has_predictions*/) const {}

  void MergePrediction(int64_t /*n*/, OTYPE* /*predictions*/, unsigned char* /*has_predictions*/,
                       OTYPE* /*predictions2*/, unsigned char* /*has_predictions2*/) const {}

  void FinalizeScores(std::vector<OTYPE>& scores,
                      std::vector<unsigned char>& has_scores,
                      OTYPE* Z, int add_second_class,
                      int64_t*) const {
    OTYPE val;
    for (int64_t jt = 0; jt < n_targets_or_classes_; ++jt) {
      val = use_base_values_ ? (*base_values_)[jt] : 0.f;
      val += has_scores[jt] ? scores[jt] : 0;
      scores[jt] = val;
    }
    this->write_scores(scores, post_transform_, Z, add_second_class);
  }

  inline void ComputeSoftmax(std::vector<OTYPE>& values) const {
    // compute exp with negative number to be numerically stable
    OTYPE v_max = -std::numeric_limits<OTYPE>::max();
    for (OTYPE value : values) {
      if (value > v_max)
        v_max = value;
    }
    OTYPE this_sum = 0.f;
    for (OTYPE& value : values) {
      value = std::exp(value - v_max);
      this_sum += value;
    }
    for (OTYPE& value : values)
      value /= this_sum;
  }

  inline void ComputeSoftmaxZero(std::vector<OTYPE>& values) const {
    // compute exp with negative number to be numerically stable
    OTYPE v_max = -std::numeric_limits<OTYPE>::max();
    for (OTYPE value : values) {
      if (value > v_max)
        v_max = value;
    }
    OTYPE exp_neg_v_max = std::exp(-v_max);
    OTYPE this_sum = 0.f;
    for (OTYPE& value : values) {
      if (value > 0.0000001f || value < -0.0000001f) {
        value = std::exp(value - v_max);
        this_sum += value;
      } else {
        value *= exp_neg_v_max;
      }
    }
    for (OTYPE& value : values)
      value /= this_sum;
  }

  void write_scores(std::vector<OTYPE>& scores, POST_EVAL_TRANSFORM post_transform,
                    OTYPE* Z, int add_second_class) const {
    if (scores.size() >= 2) {
      switch (post_transform) {
        case POST_EVAL_TRANSFORM::PROBIT:
          for (auto it = scores.cbegin(); it != scores.cend(); ++it, ++Z)
            *Z = static_cast<OTYPE>(ComputeProbit(static_cast<float>(*it)));
          break;
        case POST_EVAL_TRANSFORM::LOGISTIC:
          for (auto it = scores.cbegin(); it != scores.cend(); ++it, ++Z)
            *Z = static_cast<OTYPE>(ComputeLogistic(static_cast<float>(*it)));
          break;
        case POST_EVAL_TRANSFORM::SOFTMAX:
          this->ComputeSoftmax(scores);
          memcpy(Z, scores.data(), scores.size() * sizeof(OTYPE));
          break;
        case POST_EVAL_TRANSFORM::SOFTMAX_ZERO:
          this->ComputeSoftmaxZero(scores);
          memcpy(Z, scores.data(), scores.size() * sizeof(OTYPE));
          break;
        default:
        case POST_EVAL_TRANSFORM::NONE:
          memcpy(Z, scores.data(), scores.size() * sizeof(OTYPE));
          break;
      }
    } else if (scores.size() == 1) {  //binary case
      if (post_transform == POST_EVAL_TRANSFORM::PROBIT) {
        scores[0] = static_cast<OTYPE>(ComputeProbit(static_cast<float>(scores[0])));
        *Z = scores[0];
      } else {
        switch (add_second_class) {
          case 0:  //0=all positive weights, winning class is positive
            scores.push_back(scores[0]);
            scores[0] = 1.f - scores[0];  //put opposite score in positive slot
            break;
          case 1:  //1 = all positive weights, winning class is negative
            scores.push_back(scores[0]);
            scores[0] = 1.f - scores[0];  //put opposite score in positive slot
            break;
          case 2:
          case 3:  //2 = mixed weights, winning class is positive
            if (post_transform == POST_EVAL_TRANSFORM::LOGISTIC) {
              scores.push_back(static_cast<OTYPE>(ComputeLogistic(static_cast<float>(scores[0]))));
              scores[0] = static_cast<OTYPE>(ComputeLogistic(static_cast<float>(-scores[0])));
            } else {
              scores.push_back(scores[0]);
              scores[0] = -scores[0];
            }
            break;
        }
        *Z = scores[0];
        *(Z + 1) = scores[1];
      }
    }
  }
};

/////////////
// regression
/////////////

template <typename ITYPE, typename OTYPE>
class _AggregatorSum : public _Aggregator<ITYPE, OTYPE> {
  // has_score is not used.
 public:
  inline _AggregatorSum(size_t n_trees,
                        const int64_t& n_targets_or_classes,
                        POST_EVAL_TRANSFORM post_transform,
                        const std::vector<OTYPE>* base_values) : _Aggregator<ITYPE, OTYPE>(n_trees, n_targets_or_classes,
                                                                                           post_transform, base_values) {}

  const char* name() const { return "_AggregatorSum"; }

  // 1 output

  inline void ProcessTreeNodePrediction1(OTYPE* predictions,
                                         TreeNodeElement<OTYPE>* root,
                                         unsigned char*) const {
    *predictions += root->weights[0].value;
  }

  inline void MergePrediction1(OTYPE* predictions, unsigned char*,
                               const OTYPE* predictions2, const unsigned char*) const {
    *predictions += *predictions2;
  }

  inline void FinalizeScores1(OTYPE* Z, OTYPE& val,
                              unsigned char&,
                              int64_t*) const {
    val += this->origin_;
    *Z = this->post_transform_ == POST_EVAL_TRANSFORM::PROBIT ? static_cast<OTYPE>(ComputeProbit(static_cast<float>(val))) : val;
  }

  // N outputs

  void ProcessTreeNodePrediction(OTYPE* predictions, TreeNodeElement<OTYPE>* root,
                                 unsigned char* has_predictions) const {
    for (auto it = root->weights.begin(); it != root->weights.end(); ++it) {
      predictions[it->i] += it->value;
      has_predictions[it->i] = 1;
    }
  }

  void MergePrediction(int64_t n, OTYPE* predictions, unsigned char* has_predictions,
                       const OTYPE* predictions2, const unsigned char* has_predictions2) const {
    for (int64_t i = 0; i < n; ++i) {
      if (has_predictions2[i]) {
        predictions[i] += predictions2[i];
        has_predictions[i] = 1;
      }
    }
  }

  void FinalizeScores(std::vector<OTYPE>& scores,
                      std::vector<unsigned char>&,
                      OTYPE* Z, int add_second_class,
                      int64_t*) const {
    if (this->use_base_values_) {
      auto it = scores.begin();
      auto it2 = this->base_values_->cbegin();
      for (; it != scores.end(); ++it, ++it2)
        *it += *it2;
    }
    this->write_scores(scores, this->post_transform_, Z, add_second_class);
  }
};

template <typename ITYPE, typename OTYPE>
class _AggregatorAverage : public _AggregatorSum<ITYPE, OTYPE> {
 public:
  inline _AggregatorAverage(size_t n_trees,
                            const int64_t& n_targets_or_classes,
                            POST_EVAL_TRANSFORM post_transform,
                            const std::vector<OTYPE>* base_values) : _AggregatorSum<ITYPE, OTYPE>(n_trees, n_targets_or_classes,
                                                                                                  post_transform, base_values) {}

  const char* name() const { return "_AggregatorAverage"; }

  inline void FinalizeScores1(OTYPE* Z, OTYPE& val,
                              unsigned char&,
                              int64_t*) const {
    val /= this->n_trees_;
    val += this->origin_;
    *Z = this->post_transform_ == POST_EVAL_TRANSFORM::PROBIT ? static_cast<OTYPE>(ComputeProbit(static_cast<float>(val))) : val;
  }

  void FinalizeScores(std::vector<OTYPE>& scores,
                      std::vector<unsigned char>&,
                      OTYPE* Z, int add_second_class,
                      int64_t*) const {
    if (this->use_base_values_) {
      auto it = scores.begin();
      auto it2 = this->base_values_->cbegin();
      for (; it != scores.end(); ++it, ++it2)
        *it = *it / this->n_trees_ + *it2;
    } else {
      auto it = scores.begin();
      for (; it != scores.end(); ++it)
        *it /= this->n_trees_;
    }
    this->write_scores(scores, this->post_transform_, Z, add_second_class);
  }
};

template <typename ITYPE, typename OTYPE>
class _AggregatorMin : public _Aggregator<ITYPE, OTYPE> {
 public:
  inline _AggregatorMin(size_t n_trees,
                        const int64_t& n_targets_or_classes,
                        POST_EVAL_TRANSFORM post_transform,
                        const std::vector<OTYPE>* base_values) : _Aggregator<ITYPE, OTYPE>(n_trees, n_targets_or_classes,
                                                                                           post_transform, base_values) {}

  const char* name() const { return "_AggregatorMin"; }

  // 1 output

  inline void ProcessTreeNodePrediction1(OTYPE* predictions, TreeNodeElement<OTYPE>* root,
                                         unsigned char* has_predictions) const {
    *predictions = (!(*has_predictions) || root->weights[0].value < *predictions)
                       ? root->weights[0].value
                       : *predictions;
    *has_predictions = 1;
  }

  inline void MergePrediction1(OTYPE* predictions, unsigned char* has_predictions,
                               const OTYPE* predictions2, const unsigned char* has_predictions2) const {
    if (*has_predictions2) {
      *predictions = *has_predictions && (*predictions < *predictions2)
                         ? *predictions
                         : *predictions2;
      *has_predictions = 1;
    }
  }

  // N outputs

  void ProcessTreeNodePrediction(OTYPE* predictions, TreeNodeElement<OTYPE>* root,
                                 unsigned char* has_predictions) const {
    for (auto it = root->weights.begin(); it != root->weights.end(); ++it) {
      predictions[it->i] = (!has_predictions[it->i] || it->value < predictions[it->i])
                               ? it->value
                               : predictions[it->i];
      has_predictions[it->i] = 1;
    }
  }

  void MergePrediction(int64_t n, OTYPE* predictions, unsigned char* has_predictions,
                       const OTYPE* predictions2, const unsigned char* has_predictions2) const {
    for (int64_t i = 0; i < n; ++i) {
      if (has_predictions2[i]) {
        predictions[i] = has_predictions[i] && (predictions[i] < predictions2[i])
                             ? predictions[i]
                             : predictions2[i];
        has_predictions[i] = 1;
      }
    }
  }
};

template <typename ITYPE, typename OTYPE>
class _AggregatorMax : public _Aggregator<ITYPE, OTYPE> {
 public:
  inline _AggregatorMax<ITYPE, OTYPE>(size_t n_trees,
                                      const int64_t& n_targets_or_classes,
                                      POST_EVAL_TRANSFORM post_transform,
                                      const std::vector<OTYPE>* base_values) : _Aggregator<ITYPE, OTYPE>(n_trees, n_targets_or_classes,
                                                                                                         post_transform, base_values) {}

  const char* name() const { return "_AggregatorMax"; }

  // 1 output

  inline void ProcessTreeNodePrediction1(OTYPE* predictions, TreeNodeElement<OTYPE>* root,
                                         unsigned char* has_predictions) const {
    *predictions = (!(*has_predictions) || root->weights[0].value > *predictions)
                       ? root->weights[0].value
                       : *predictions;
    *has_predictions = 1;
  }

  inline void MergePrediction1(OTYPE* predictions, unsigned char* has_predictions,
                               const OTYPE* predictions2, const unsigned char* has_predictions2) const {
    if (*has_predictions2) {
      *predictions = *has_predictions && (*predictions > *predictions2)
                         ? *predictions
                         : *predictions2;
      *has_predictions = 1;
    }
  }

  // N outputs

  void ProcessTreeNodePrediction(OTYPE* predictions, TreeNodeElement<OTYPE>* root,
                                 unsigned char* has_predictions) const {
    for (auto it = root->weights.begin(); it != root->weights.end(); ++it) {
      predictions[it->i] = (!has_predictions[it->i] || it->value > predictions[it->i])
                               ? it->value
                               : predictions[it->i];
      has_predictions[it->i] = 1;
    }
  }

  void MergePrediction(int64_t n, OTYPE* predictions, unsigned char* has_predictions,
                       OTYPE* predictions2, unsigned char* has_predictions2) const {
    for (int64_t i = 0; i < n; ++i) {
      if (has_predictions2[i]) {
        predictions[i] = has_predictions[i] && (predictions[i] > predictions2[i])
                             ? predictions[i]
                             : predictions2[i];
        has_predictions[i] = 1;
      }
    }
  }
};

/////////////////
// classification
/////////////////

template <typename ITYPE, typename OTYPE>
class _AggregatorClassifier : public _AggregatorSum<ITYPE, OTYPE> {
 private:
  const std::vector<int64_t>* class_labels_;
  bool binary_case_;
  bool weights_are_all_positive_;
  int64_t positive_label_;
  int64_t negative_label_;

 public:
  inline _AggregatorClassifier(size_t n_trees,
                               const int64_t& n_targets_or_classes,
                               POST_EVAL_TRANSFORM post_transform,
                               const std::vector<OTYPE>* base_values,
                               const std::vector<int64_t>* class_labels,
                               bool binary_case,
                               bool weights_are_all_positive,
                               int64_t positive_label = 1,
                               int64_t negative_label = 0) : _AggregatorSum<ITYPE, OTYPE>(n_trees, n_targets_or_classes,
                                                                                          post_transform, base_values),
                                                             class_labels_(class_labels),
                                                             binary_case_(binary_case),
                                                             weights_are_all_positive_(weights_are_all_positive),
                                                             positive_label_(positive_label),
                                                             negative_label_(negative_label) {}

  const char* name() const { return "_AggregatorClassifier"; }

  void get_max_weight(const std::vector<OTYPE>& classes,
                      const std::vector<unsigned char>& has_scores,
                      int64_t& maxclass, OTYPE& maxweight) const {
    maxclass = -1;
    maxweight = 0;
    typename std::vector<OTYPE>::const_iterator it;
    typename std::vector<unsigned char>::const_iterator itb;
    for (it = classes.begin(), itb = has_scores.begin();
         it != classes.end(); ++it, ++itb) {
      if (*itb && (maxclass == -1 || *it > maxweight)) {
        maxclass = (int64_t)(it - classes.begin());
        maxweight = *it;
      }
    }
  }

  inline int64_t _set_score_binary(int& write_additional_scores,
                                   const OTYPE* classes,
                                   const unsigned char* has_scores) const {
    OTYPE pos_weight = has_scores[1]
                           ? classes[1]
                           : (has_scores[0] ? classes[0] : 0);  // only 1 class
    if (binary_case_) {
      if (weights_are_all_positive_) {
        if (pos_weight > 0.5) {
          write_additional_scores = 0;
          return (*class_labels_)[1];  // positive label
        } else {
          write_additional_scores = 1;
          return (*class_labels_)[0];  // negative label
        }
      } else {
        if (pos_weight > 0) {
          write_additional_scores = 2;
          return (*class_labels_)[1];  // positive label
        } else {
          write_additional_scores = 3;
          return (*class_labels_)[0];  // negative label
        }
      }
    }
    return (pos_weight > 0)
               ? positive_label_   // positive label
               : negative_label_;  // negative label
  }

  // 1 output

  inline void FinalizeScores1(OTYPE* Z, OTYPE& val,
                              unsigned char& /*has_score*/,
                              int64_t* Y = 0) const {
    std::vector<OTYPE> scores(2);
    unsigned char has_scores[2] = {1, 0};

    int write_additional_scores = -1;
    if (this->base_values_->size() == 2) {
      // add base values
      scores[1] = (*(this->base_values_))[1] + val;
      scores[0] = -scores[1];
      //has_score = true;
      has_scores[1] = 1;
    } else if (this->base_values_->size() == 1) {
      // ONNX is vague about two classes and only one base_values.
      scores[0] = val + (*(this->base_values_))[0];
      //if (!has_scores[1])
      //scores.pop_back();
      scores[0] = val;
    } else if (this->base_values_->size() == 0) {
      //if (!has_score)
      //  scores.pop_back();
      scores[0] = val;
    }

    *Y = _set_score_binary(write_additional_scores, &(scores[0]), has_scores);
    this->write_scores(scores, this->post_transform_, Z, write_additional_scores);
  }

  // N outputs

  void FinalizeScores(std::vector<OTYPE>& scores,
                      std::vector<unsigned char>& has_scores,
                      OTYPE* Z, int /*add_second_class*/,
                      int64_t* Y = 0) const {
    OTYPE maxweight = 0;
    int64_t maxclass = -1;

    int write_additional_scores = -1;
    if (this->n_targets_or_classes_ > 2) {
      // add base values
      for (int64_t k = 0, end = static_cast<int64_t>(this->base_values_->size()); k < end; ++k) {
        if (!has_scores[k]) {
          has_scores[k] = true;
          scores[k] = (*(this->base_values_))[k];
        } else {
          scores[k] += (*(this->base_values_))[k];
        }
      }
      get_max_weight(scores, has_scores, maxclass, maxweight);
      *Y = (*class_labels_)[maxclass];
    } else {  // binary case
      if (this->base_values_->size() == 2) {
        // add base values
        if (has_scores[1]) {
          // base_value_[0] is not used.
          // It assumes base_value[0] == base_value[1] in this case.
          // The specification does not forbid it but does not
          // say what the output should be in that case.
          scores[1] = (*(this->base_values_))[1] + scores[0];
          scores[0] = -scores[1];
          has_scores[1] = true;
        } else {
          // binary as multiclass
          scores[1] += (*(this->base_values_))[1];
          scores[0] += (*(this->base_values_))[0];
        }
      } else if (this->base_values_->size() == 1) {
        // ONNX is vague about two classes and only one base_values.
        scores[0] += (*(this->base_values_))[0];
        if (!has_scores[1])
          scores.pop_back();
      } else if (this->base_values_->size() == 0) {
        if (!has_scores[1])
          scores.pop_back();
      }

      *Y = _set_score_binary(write_additional_scores, &(scores[0]), &(has_scores[0]));
    }

    this->write_scores(scores, this->post_transform_, Z, write_additional_scores);
  }
};

}  // namespace ml
}  // namespace onnxruntime
