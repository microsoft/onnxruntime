// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "tree_ensemble_aggregator.h"

namespace onnxruntime {
namespace ml {

template <typename ITYPE, typename OTYPE>
class TreeEnsembleCommon {
 public:
  int64_t n_targets_or_classes_;

 protected:
  std::vector<OTYPE> base_values_;
  POST_EVAL_TRANSFORM post_transform_;
  AGGREGATE_FUNCTION aggregate_function_;
  int64_t n_nodes_;
  TreeNodeElement<OTYPE>* nodes_;
  std::vector<TreeNodeElement<OTYPE>*> roots_;

  int64_t max_tree_depth_;
  int64_t n_trees_;
  bool same_mode_;
  bool has_missing_tracks_;
  int omp_tree_;
  int omp_N_;

 public:
  TreeEnsembleCommon(int omp_tree, int omp_N);
  ~TreeEnsembleCommon();

  void init(
      const std::string& aggregate_function,
      std::vector<OTYPE> base_values,
      int64_t n_targets_or_classes,
      std::vector<int64_t> nodes_falsenodeids,
      std::vector<int64_t> nodes_featureids,
      std::vector<OTYPE> nodes_hitrates,
      std::vector<int64_t> nodes_missing_value_tracks_true,
      const std::vector<std::string>& nodes_modes,
      std::vector<int64_t> nodes_nodeids,
      std::vector<int64_t> nodes_treeids,
      std::vector<int64_t> nodes_truenodeids,
      std::vector<OTYPE> nodes_values,
      const std::string& post_transform,
      std::vector<int64_t> target_class_ids,
      std::vector<int64_t> target_class_nodeids,
      std::vector<int64_t> target_class_treeids,
      std::vector<OTYPE> target_class_weights);

  std::string runtime_options();

  int omp_get_max_threads();

  void compute(const Tensor* X, Tensor* Z, Tensor* label) const;

 protected:
  void init_c(
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

  TreeNodeElement<OTYPE>* ProcessTreeNodeLeave(
      TreeNodeElement<OTYPE>* root, const ITYPE* x_data) const;

  template <typename AGG>
  void compute_agg(const Tensor* X, Tensor* Z, Tensor* label, const AGG& agg) const;
};

template <typename ITYPE, typename OTYPE>
TreeEnsembleCommon<ITYPE, OTYPE>::TreeEnsembleCommon(int omp_tree, int omp_N) {
  omp_tree_ = omp_tree;
  omp_N_ = omp_N;
  nodes_ = NULL;
}

template <typename ITYPE, typename OTYPE>
TreeEnsembleCommon<ITYPE, OTYPE>::~TreeEnsembleCommon() {
  if (nodes_ != NULL)
    delete[] nodes_;
}

template <typename ITYPE, typename OTYPE>
std::string TreeEnsembleCommon<ITYPE, OTYPE>::runtime_options() {
  std::string res;
#ifdef USE_OPENMP
  res += "OPENMP";
#endif
  return res;
}

template <typename ITYPE, typename OTYPE>
int TreeEnsembleCommon<ITYPE, OTYPE>::omp_get_max_threads() {
#if USE_OPENMP
  return ::omp_get_max_threads();
#else
  return 1;
#endif
}

template <typename ITYPE, typename OTYPE>
void TreeEnsembleCommon<ITYPE, OTYPE>::init(
    const std::string& aggregate_function,
    std::vector<OTYPE> base_values,
    int64_t n_targets_or_classes,
    std::vector<int64_t> nodes_falsenodeids,
    std::vector<int64_t> nodes_featureids,
    std::vector<OTYPE> nodes_hitrates,
    std::vector<int64_t> nodes_missing_value_tracks_true,
    const std::vector<std::string>& nodes_modes,
    std::vector<int64_t> nodes_nodeids,
    std::vector<int64_t> nodes_treeids,
    std::vector<int64_t> nodes_truenodeids,
    std::vector<OTYPE> nodes_values,
    const std::string& post_transform,
    std::vector<int64_t> target_class_ids,
    std::vector<int64_t> target_class_nodeids,
    std::vector<int64_t> target_class_treeids,
    std::vector<OTYPE> target_class_weights) {
  init_c(aggregate_function, base_values, n_targets_or_classes,
         nodes_falsenodeids, nodes_featureids, nodes_hitrates,
         nodes_missing_value_tracks_true, nodes_modes,
         nodes_nodeids, nodes_treeids, nodes_truenodeids,
         nodes_values, post_transform, target_class_ids,
         target_class_nodeids, target_class_treeids,
         target_class_weights);
}

template <typename ITYPE, typename OTYPE>
void TreeEnsembleCommon<ITYPE, OTYPE>::init_c(
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
  aggregate_function_ = ::onnxruntime::ml::MakeAggregateFunction(aggregate_function);
  post_transform_ = ::onnxruntime::ml::MakeTransform(post_transform);
  base_values_ = base_values;
  n_targets_or_classes_ = n_targets_or_classes;
  max_tree_depth_ = 1000;

  // additional members
  std::vector<NODE_MODE> cmodes(nodes_modes.size());
  same_mode_ = true;
  int fpos = -1;
  for (size_t i = 0; i < nodes_modes.size(); ++i) {
    cmodes[i] = ::onnxruntime::ml::MakeTreeNodeMode(nodes_modes[i]);
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
  nodes_ = new TreeNodeElement<OTYPE>[static_cast<int>(n_nodes_)];
  roots_.clear();
  std::map<TreeNodeElementId, TreeNodeElement<OTYPE>*> idi;
  size_t i;

  for (i = 0; i < nodes_treeids.size(); ++i) {
    TreeNodeElement<OTYPE>* node = nodes_ + i;
    node->id.tree_id = static_cast<int>(nodes_treeids[i]);
    node->id.node_id = static_cast<int>(nodes_nodeids[i]);
    node->feature_id = static_cast<int>(nodes_featureids[i]);
    node->value = nodes_values[i];
    node->hitrates = i < nodes_hitrates.size() ? nodes_hitrates[i] : -1;
    node->mode = cmodes[i];
    node->is_not_leave = node->mode != NODE_MODE::LEAF;
    node->truenode = NULL;   // nodes_truenodeids[i];
    node->falsenode = NULL;  // nodes_falsenodeids[i];
    node->missing_tracks = i < static_cast<size_t>(nodes_missing_value_tracks_true.size())
                               ? (nodes_missing_value_tracks_true[i] == 1
                                      ? MissingTrack::TRUE
                                      : MissingTrack::FALSE)
                               : MissingTrack::NONE;
    node->is_missing_track_true = node->missing_tracks == MissingTrack::TRUE;
    if (idi.find(node->id) != idi.end()) {
      ORT_THROW("Node ", node->id.node_id, " in tree ", node->id.tree_id, " is already there.");
    }
    idi.insert(std::pair<TreeNodeElementId, TreeNodeElement<OTYPE>*>(node->id, node));
  }

  TreeNodeElementId coor;
  TreeNodeElement<OTYPE>* it;
  for (i = 0; i < static_cast<size_t>(n_nodes_); ++i) {
    it = nodes_ + i;
    if (!it->is_not_leave)
      continue;
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
      it->truenode = NULL;

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
      it->falsenode = NULL;
  }

  int64_t previous = -1;
  for (i = 0; i < static_cast<size_t>(n_nodes_); ++i) {
    if ((previous == -1) || (previous != nodes_[i].id.tree_id))
      roots_.push_back(nodes_ + i);
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
void TreeEnsembleCommon<ITYPE, OTYPE>::compute(const Tensor* X, Tensor* Z, Tensor* label) const {
  switch (aggregate_function_) {
    case AGGREGATE_FUNCTION::AVERAGE:
      compute_agg(
          X, Z, label,
          _AggregatorAverage<ITYPE, OTYPE>(
              roots_.size(), n_targets_or_classes_,
              post_transform_, &(base_values_)));
      return;
    case AGGREGATE_FUNCTION::SUM:
      compute_agg(
          X, Z, label,
          _AggregatorSum<ITYPE, OTYPE>(
              roots_.size(), n_targets_or_classes_,
              post_transform_, &(base_values_)));
      return;
    case AGGREGATE_FUNCTION::MIN:
      compute_agg(
          X, Z, label,
          _AggregatorMin<ITYPE, OTYPE>(
              roots_.size(), n_targets_or_classes_,
              post_transform_, &(base_values_)));
      return;
    case AGGREGATE_FUNCTION::MAX:
      compute_agg(
          X, Z, label,
          _AggregatorMax<ITYPE, OTYPE>(
              roots_.size(), n_targets_or_classes_,
              post_transform_, &(base_values_)));
      return;
  }
  ORT_THROW("Unknown aggregation function in TreeEnsemble.");
}

template <typename ITYPE, typename OTYPE>
template <typename AGG>
void TreeEnsembleCommon<ITYPE, OTYPE>::compute_agg(const Tensor* X, Tensor* Z, Tensor* label, const AGG& agg) const {
  int64_t stride = X->Shape().NumDimensions() == 1 ? X->Shape()[0] : X->Shape()[1];
  int64_t N = X->Shape().NumDimensions() == 1 ? 1 : X->Shape()[0];

  const ITYPE* x_data = X->template Data<ITYPE>();
  OTYPE* z_data = Z->template MutableData<OTYPE>();
  int64_t* label_data = label == NULL ? NULL : label->template MutableData<int64_t>();

  if (n_targets_or_classes_ == 1) {
    if (N == 1) {
      OTYPE scores = 0;
      unsigned char has_scores = 0;
      if (n_trees_ <= omp_tree_) {
        for (int64_t j = 0; j < n_trees_; ++j)
          agg.ProcessTreeNodePrediction1(
              &scores,
              ProcessTreeNodeLeave(roots_[j], x_data),
              &has_scores);
      } else {
        std::vector<OTYPE> scores_t(n_trees_, 0);
        std::vector<unsigned char> has_scores_t(n_trees_, 0);
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
        for (int64_t j = 0; j < n_trees_; ++j) {
          agg.ProcessTreeNodePrediction1(
              &(scores_t[j]),
              ProcessTreeNodeLeave(roots_[j], x_data),
              &(has_scores_t[j]));
        }
        auto it = scores_t.cbegin();
        auto it2 = has_scores_t.cbegin();
        for (; it != scores_t.cend(); ++it, ++it2)
          agg.MergePrediction1(&scores, &has_scores, &(*it), &(*it2));
      }

      agg.FinalizeScores1(z_data, scores, has_scores, label_data);
    } else {
      if (N <= omp_N_) {
        OTYPE scores;
        unsigned char has_scores;
        size_t j;

        for (int64_t i = 0; i < N; ++i) {
          scores = 0;
          has_scores = 0;
          for (j = 0; j < static_cast<size_t>(n_trees_); ++j)
            agg.ProcessTreeNodePrediction1(
                &scores,
                ProcessTreeNodeLeave(roots_[j], x_data + i * stride),
                &has_scores);
          agg.FinalizeScores1(z_data + i * n_targets_or_classes_, scores, has_scores,
                              label_data == NULL ? NULL : (label_data + i));
        }
      } else {
        OTYPE scores;
        unsigned char has_scores;
        size_t j;

#ifdef USE_OPENMP
#pragma omp parallel for private(j, scores, has_scores)
#endif
        for (int64_t i = 0; i < N; ++i) {
          scores = 0;
          has_scores = 0;
          for (j = 0; j < static_cast<size_t>(n_trees_); ++j)
            agg.ProcessTreeNodePrediction1(
                &scores,
                ProcessTreeNodeLeave(roots_[j], x_data + i * stride),
                &has_scores);
          agg.FinalizeScores1(z_data + i * n_targets_or_classes_,
                              scores, has_scores,
                              label_data == NULL ? NULL : (label_data + i));
        }
      }
    }
  } else {
    if (N == 1) {
      std::vector<OTYPE> scores(n_targets_or_classes_, 0);
      std::vector<unsigned char> has_scores(n_targets_or_classes_, 0);

      if (n_trees_ <= omp_tree_) {
        for (int64_t j = 0; j < n_trees_; ++j)
          agg.ProcessTreeNodePrediction(
              scores.data(),
              ProcessTreeNodeLeave(roots_[j], x_data),
              has_scores.data());
        agg.FinalizeScores(scores, has_scores, z_data, -1, label_data);
      } else {
#ifdef USE_OPENMP
#pragma omp parallel
#endif
        {
          std::vector<OTYPE> private_scores(n_targets_or_classes_, 0);
          std::vector<unsigned char> private_has_scores(n_targets_or_classes_, 0);
#ifdef USE_OPENMP
#pragma omp for
#endif
          for (int64_t j = 0; j < n_trees_; ++j) {
            agg.ProcessTreeNodePrediction(
                private_scores.data(),
                ProcessTreeNodeLeave(roots_[j], x_data),
                private_has_scores.data());
          }

#ifdef USE_OPENMP
#pragma omp critical
#endif
          agg.MergePrediction(n_targets_or_classes_,
                              &(scores[0]), &(has_scores[0]),
                              private_scores.data(), private_has_scores.data());
        }

        agg.FinalizeScores(scores, has_scores, z_data, -1, label_data);
      }
    } else {
      if (N <= omp_N_) {
        std::vector<OTYPE> scores(n_targets_or_classes_);
        std::vector<unsigned char> has_scores(n_targets_or_classes_);
        size_t j;

        for (int64_t i = 0; i < N; ++i) {
          std::fill(scores.begin(), scores.end(), static_cast<OTYPE>(0));
          std::fill(has_scores.begin(), has_scores.end(), static_cast<unsigned char>(0));
          for (j = 0; j < roots_.size(); ++j)
            agg.ProcessTreeNodePrediction(
                scores.data(),
                ProcessTreeNodeLeave(roots_[j], x_data + i * stride),
                has_scores.data());
          agg.FinalizeScores(scores, has_scores,
                             z_data + i * n_targets_or_classes_, -1,
                             label_data == NULL ? NULL : (label_data + i));
        }
      } else {
#ifdef USE_OPENMP
#pragma omp parallel
#endif
        {
          std::vector<OTYPE> scores(n_targets_or_classes_);
          std::vector<unsigned char> has_scores(n_targets_or_classes_);
          size_t j;

#ifdef USE_OPENMP
#pragma omp for
#endif
          for (int64_t i = 0; i < N; ++i) {
            std::fill(scores.begin(), scores.end(), static_cast<OTYPE>(0));
            std::fill(has_scores.begin(), has_scores.end(), static_cast<unsigned char>(0));
            for (j = 0; j < roots_.size(); ++j)
              agg.ProcessTreeNodePrediction(
                  scores.data(),
                  ProcessTreeNodeLeave(roots_[j], x_data + i * stride),
                  has_scores.data());
            agg.FinalizeScores(scores, has_scores,
                               z_data + i * n_targets_or_classes_, -1,
                               label_data == NULL ? NULL : (label_data + i));
          }
        }
      }
    }
  }
}

#define TREE_FIND_VALUE(CMP)                                         \
  if (has_missing_tracks_) {                                         \
    while (root->is_not_leave) {                                     \
      val = x_data[root->feature_id];                                \
      root = (val CMP root->value ||                                 \
              (root->is_missing_track_true && _isnan_(val)))         \
                 ? root->truenode                                    \
                 : root->falsenode;                                  \
    }                                                                \
  } else {                                                           \
    while (root->is_not_leave) {                                     \
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
          while (root->is_not_leave) {
            val = x_data[root->feature_id];
            root = (val <= root->value ||
                    (root->is_missing_track_true && _isnan_(val)))
                       ? root->truenode
                       : root->falsenode;
          }
        } else {
          while (root->is_not_leave) {
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
    while (root->is_not_leave) {
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
 protected:
  bool weights_are_all_positive_;
  bool binary_case_;
  int64_t class_count_;
  std::vector<std::string> classlabels_strings_;
  std::vector<int64_t> classlabels_int64s_;
  std::vector<int64_t> class_labels_;

 public:
  TreeEnsembleCommonClassifier(int omp_tree, int omp_N) : TreeEnsembleCommon<ITYPE, OTYPE>(omp_tree, omp_N) {}

  inline int64_t get_class_count() const { return class_count_; }

  void init(
      const std::string& aggregate_function,
      std::vector<OTYPE> base_values,
      std::vector<int64_t> nodes_falsenodeids,
      std::vector<int64_t> nodes_featureids,
      std::vector<OTYPE> nodes_hitrates,
      std::vector<int64_t> nodes_missing_value_tracks_true,
      const std::vector<std::string>& nodes_modes,
      std::vector<int64_t> nodes_nodeids,
      std::vector<int64_t> nodes_treeids,
      std::vector<int64_t> nodes_truenodeids,
      std::vector<OTYPE> nodes_values,
      const std::string& post_transform,
      std::vector<int64_t> class_ids,
      std::vector<int64_t> class_nodeids,
      std::vector<int64_t> class_treeids,
      std::vector<OTYPE> class_weights,
      std::vector<std::string> classlabels_strings,
      std::vector<int64_t> classlabels_int64s);

  void compute(const Tensor* X, Tensor* Z, Tensor* label) const;
};

template <typename ITYPE, typename OTYPE>
void TreeEnsembleCommonClassifier<ITYPE, OTYPE>::init(
    const std::string& aggregate_function,
    std::vector<OTYPE> base_values,
    std::vector<int64_t> nodes_falsenodeids,
    std::vector<int64_t> nodes_featureids,
    std::vector<OTYPE> nodes_hitrates,
    std::vector<int64_t> nodes_missing_value_tracks_true,
    const std::vector<std::string>& nodes_modes,
    std::vector<int64_t> nodes_nodeids,
    std::vector<int64_t> nodes_treeids,
    std::vector<int64_t> nodes_truenodeids,
    std::vector<OTYPE> nodes_values,
    const std::string& post_transform,
    std::vector<int64_t> class_ids,
    std::vector<int64_t> class_nodeids,
    std::vector<int64_t> class_treeids,
    std::vector<OTYPE> class_weights,
    std::vector<std::string> classlabels_strings,
    std::vector<int64_t> classlabels_int64s) {
  classlabels_strings_ = classlabels_strings;
  classlabels_int64s_ = classlabels_int64s;
  class_count_ = classlabels_strings_.size() == 0 ? classlabels_int64s_.size() : classlabels_strings_.size();
  if (class_count_ == 0) {
    ORT_THROW("classlabels_int64s_ and classlabels_strings_ cannot be both empty.");
  }

  TreeEnsembleCommon<ITYPE, OTYPE>::init(
      aggregate_function, base_values, class_count_, nodes_falsenodeids,
      nodes_featureids, nodes_hitrates, nodes_missing_value_tracks_true,
      nodes_modes, nodes_nodeids, nodes_treeids, nodes_truenodeids,
      nodes_values, post_transform, class_ids, class_nodeids,
      class_treeids, class_weights);

  std::set<int64_t> weights_classes;
  weights_are_all_positive_ = true;
  for (size_t i = 0, end = class_ids.size(); i < end; ++i) {
    weights_classes.insert(class_ids[i]);
    if (class_weights[i] < 0)
      weights_are_all_positive_ = false;
  }
  binary_case_ = class_count_ == 2 && weights_classes.size() == 1;
  if (classlabels_strings_.size() > 0) {
    class_labels_.resize(classlabels_strings_.size());
    for (size_t i = 0; i < classlabels_strings_.size(); ++i)
      class_labels_[i] = i;
  }
}

template <typename ITYPE, typename OTYPE>
void TreeEnsembleCommonClassifier<ITYPE, OTYPE>::compute(const Tensor* X, Tensor* Z, Tensor* label) const {
  if (classlabels_strings_.size() == 0) {
    this->compute_agg(
        X, Z, label,
        _AggregatorClassifier<ITYPE, OTYPE>(
            this->roots_.size(), class_count_,
            this->post_transform_, &(this->base_values_),
            &(classlabels_int64s_), binary_case_,
            weights_are_all_positive_));
  } else {
    int64_t N = X->Shape().NumDimensions() == 1 ? 1 : X->Shape()[0];
    std::shared_ptr<IAllocator> allocator = std::make_shared<CPUAllocator>();
    Tensor label_int64(DataTypeImpl::GetType<int64_t>(), TensorShape({N}), allocator);
    this->compute_agg(
        X, Z, &label_int64,
        _AggregatorClassifier<ITYPE, OTYPE>(
            this->roots_.size(), class_count_,
            this->post_transform_, &(this->base_values_),
            &(class_labels_), binary_case_,
            weights_are_all_positive_));
    const int64_t* plabel = label_int64.template Data<int64_t>();
    for (size_t i = 0; i < (size_t)N; ++i)
      label->template MutableData<std::string>()[i] = classlabels_strings_[plabel[i]];
  }
}

}  // namespace ml
}  // namespace onnxruntime
