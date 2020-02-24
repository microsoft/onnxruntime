// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/ml/treeregressor.h"

namespace onnxruntime {
namespace ml {

ONNX_CPU_OPERATOR_TYPED_ML_KERNEL(
    TreeEnsembleRegressor,
    1,
    float,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()).MayInplace(0, 0),
    TreeEnsembleRegressor<float>);

ONNX_CPU_OPERATOR_TYPED_ML_KERNEL(
    TreeEnsembleRegressor,
    1,
    double,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()).MayInplace(0, 0),
    TreeEnsembleRegressor<double>);

template <typename T>
TreeEnsembleRegressor<T>::TreeEnsembleRegressor(const OpKernelInfo& info)
    : OpKernel(info),
      base_values_(info.GetAttrsOrDefault<float>("base_values")),
      post_transform_(::onnxruntime::ml::MakeTransform(info.GetAttrOrDefault<std::string>("post_transform", "NONE"))),
      aggregate_function_(::onnxruntime::ml::MakeAggregateFunction(info.GetAttrOrDefault<std::string>("aggregate_function", "SUM"))) {
  ORT_ENFORCE(info.GetAttr<int64_t>("n_targets", &n_targets_).IsOK());

  std::vector<int64_t> nodes_treeids_(info.GetAttrsOrDefault<int64_t>("nodes_treeids"));
  std::vector<int64_t> nodes_nodeids_(info.GetAttrsOrDefault<int64_t>("nodes_nodeids"));
  std::vector<int64_t> nodes_featureids_(info.GetAttrsOrDefault<int64_t>("nodes_featureids"));
  // GetAttrsOrDefault for double not available.
  std::vector<float> nodes_values_(info.GetAttrsOrDefault<float>("nodes_values"));
  std::vector<float> nodes_hitrates_(info.GetAttrsOrDefault<float>("nodes_hitrates"));
  std::vector<int64_t> nodes_truenodeids_(info.GetAttrsOrDefault<int64_t>("nodes_truenodeids"));
  std::vector<int64_t> nodes_falsenodeids_(info.GetAttrsOrDefault<int64_t>("nodes_falsenodeids"));
  std::vector<int64_t> missing_tracks_true_(info.GetAttrsOrDefault<int64_t>("nodes_missing_value_tracks_true"));
  std::vector<int64_t> target_nodeids_(info.GetAttrsOrDefault<int64_t>("target_nodeids"));
  std::vector<int64_t> target_treeids_(info.GetAttrsOrDefault<int64_t>("target_treeids"));
  std::vector<int64_t> target_ids_(info.GetAttrsOrDefault<int64_t>("target_ids"));
  std::vector<float> target_weights_(info.GetAttrsOrDefault<float>("target_weights"));

  //update nodeids to start at 0
  ORT_ENFORCE(!nodes_treeids_.empty());
  std::vector<NODE_MODE> nodes_modes_;
  std::vector<std::string> modes = info.GetAttrsOrDefault<std::string>("nodes_modes");

  for (const auto& mode : modes) {
    nodes_modes_.push_back(::onnxruntime::ml::MakeTreeNodeMode(mode));
  }

  size_t nodes_id_size = nodes_nodeids_.size();
  ORT_ENFORCE(target_nodeids_.size() == target_ids_.size());
  ORT_ENFORCE(target_nodeids_.size() == target_weights_.size());
  ORT_ENFORCE(nodes_id_size == nodes_featureids_.size());
  ORT_ENFORCE(nodes_id_size == nodes_values_.size());
  ORT_ENFORCE(nodes_id_size == nodes_modes_.size());
  ORT_ENFORCE(nodes_id_size == nodes_truenodeids_.size());
  ORT_ENFORCE(nodes_id_size == nodes_falsenodeids_.size());
  ORT_ENFORCE((nodes_id_size == nodes_hitrates_.size()) || (nodes_hitrates_.empty()));
  ORT_ENFORCE(base_values_.empty() || base_values_.size() == static_cast<size_t>(n_targets_));

  max_tree_depth_ = 1000;

  // Filling nodes_.
  nbnodes_ = nodes_treeids_.size();
  nodes_ = new TreeNodeElement[(int)nbnodes_];
  roots_.clear();
  std::map<TreeNodeElementId, TreeNodeElement*> idi;
  size_t i;

  for (i = 0; i < nodes_treeids_.size(); ++i) {
    TreeNodeElement* node = nodes_ + i;
    node->id.tree_id = (int)nodes_treeids_[i];
    node->id.node_id = (int)nodes_nodeids_[i];
    node->feature_id = (int)nodes_featureids_[i];
    node->value = nodes_values_[i];
    node->hitrates = i < nodes_hitrates_.size() ? nodes_hitrates_[i] : -1;
    node->mode = nodes_modes_[i];
    node->truenode = NULL;   // nodes_truenodeids_[i];
    node->falsenode = NULL;  // nodes_falsenodeids_[i];
    node->missing_tracks = i < (size_t)missing_tracks_true_.size()
                               ? (missing_tracks_true_[i] == 1
                                      ? MissingTrack::TRUE
                                      : MissingTrack::FALSE)
                               : MissingTrack::NONE;
    ORT_ENFORCE(idi.find(node->id) == idi.end());
    idi.insert(std::pair<TreeNodeElementId, TreeNodeElement*>(node->id, node));
  }

  TreeNodeElementId coor;
  TreeNodeElement* it;
  for (i = 0; i < (size_t)nbnodes_; ++i) {
    it = nodes_ + i;
    if (it->mode == NODE_MODE::LEAF)
      continue;
    coor.tree_id = it->id.tree_id;
    coor.node_id = (int)nodes_truenodeids_[i];

    auto found = idi.find(coor);
    ORT_ENFORCE(found != idi.end());
    if (coor.node_id >= 0 && coor.node_id < nbnodes_) {
      it->truenode = found->second;
      ORT_ENFORCE(!((it->truenode->id.tree_id != it->id.tree_id) ||
                    (it->truenode->id.node_id == it->id.node_id)));
    } else
      it->truenode = NULL;

    coor.node_id = (int)nodes_falsenodeids_[i];
    found = idi.find(coor);
    ORT_ENFORCE(found != idi.end());
    if (coor.node_id >= 0 && coor.node_id < nbnodes_) {
      it->falsenode = found->second;
      ORT_ENFORCE(!((it->falsenode->id.tree_id != it->id.tree_id) ||
                    (it->falsenode->id.node_id == it->id.node_id)));
    } else
      it->falsenode = NULL;
  }

  int64_t previous = -1;
  for (i = 0; i < (size_t)nbnodes_; ++i) {
    if ((previous == -1) || (previous != nodes_[i].id.tree_id))
      roots_.push_back(nodes_ + i);
    previous = nodes_[i].id.tree_id;
  }

  TreeNodeElementId ind;
  SparseValue w;
  for (i = 0; i < target_nodeids_.size(); i++) {
    ind.tree_id = (int)target_treeids_[i];
    ind.node_id = (int)target_nodeids_[i];
    ORT_ENFORCE(idi.find(ind) != idi.end());
    w.i = target_ids_[i];
    w.value = target_weights_[i];
    idi[ind]->weights.push_back(w);
  }

  nbtrees_ = roots_.size();
  has_missing_tracks_ = missing_tracks_true_.size() == nodes_truenodeids_.size();
}

template <typename T>
TreeEnsembleRegressor<T>::~TreeEnsembleRegressor() {
  delete[] nodes_;
}

#define TREE_FIND_VALUE(CMP)                                         \
  if (has_missing_tracks_) {                                         \
    while (root->mode != NODE_MODE::LEAF && loopcount >= 0) {        \
      val = x_data[root->feature_id];                                \
      root = (val CMP root->value ||                                 \
              (root->missing_tracks == MissingTrack::TRUE &&         \
               std::isnan(static_cast<T>(val))))                     \
                 ? root->truenode                                    \
                 : root->falsenode;                                  \
      --loopcount;                                                   \
    }                                                                \
  } else {                                                           \
    while (root->mode != NODE_MODE::LEAF && loopcount >= 0) {        \
      val = x_data[root->feature_id];                                \
      root = val CMP root->value ? root->truenode : root->falsenode; \
      --loopcount;                                                   \
    }                                                                \
  }

template <typename T>
common::Status TreeEnsembleRegressor<T>::ProcessTreeNode(float* predictions, TreeNodeElement* root,
                                                         const T* x_data,
                                                         unsigned char* has_predictions) const {
  bool tracktrue;
  T val;
  if (same_mode_) {
    int64_t loopcount = max_tree_depth_;
    switch (root->mode) {
      case NODE_MODE::BRANCH_LEQ:
        TREE_FIND_VALUE(<=)
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
      default: {
        ORT_THROW("Invalid mode of value: ", static_cast<std::underlying_type<NODE_MODE>::type>(root->mode));
      }
    }
  } else {  // Different rules to compare to node thresholds.
    int64_t loopcount = 0;
    T threshold;
    while ((root->mode != NODE_MODE::LEAF) && (loopcount <= max_tree_depth_)) {
      val = x_data[root->feature_id];
      tracktrue = root->missing_tracks == MissingTrack::TRUE &&
                  std::isnan(static_cast<T>(val));
      threshold = root->value;
      switch (root->mode) {
        case NODE_MODE::BRANCH_LEQ:
          root = val <= threshold || tracktrue
                     ? root->truenode
                     : root->falsenode;
          break;
        case NODE_MODE::BRANCH_LT:
          root = val < threshold || tracktrue
                     ? root->truenode
                     : root->falsenode;
          break;
        case NODE_MODE::BRANCH_GTE:
          root = val >= threshold || tracktrue
                     ? root->truenode
                     : root->falsenode;
          break;
        case NODE_MODE::BRANCH_GT:
          root = val > threshold || tracktrue
                     ? root->truenode
                     : root->falsenode;
          break;
        case NODE_MODE::BRANCH_EQ:
          root = val == threshold || tracktrue
                     ? root->truenode
                     : root->falsenode;
          break;
        case NODE_MODE::BRANCH_NEQ:
          root = val != threshold || tracktrue
                     ? root->truenode
                     : root->falsenode;
          break;
        default: {
          ORT_THROW("Invalid mode of value: ", static_cast<std::underlying_type<NODE_MODE>::type>(root->mode));
        }
      }
      ++loopcount;
    }
  }

  //should be at leaf
  switch (aggregate_function_) {
    case AGGREGATE_FUNCTION::AVERAGE:
    case AGGREGATE_FUNCTION::SUM:
      for (auto it = root->weights.begin(); it != root->weights.end(); ++it) {
        predictions[it->i] += it->value;
        has_predictions[it->i] = 1;
      }
      break;
    case AGGREGATE_FUNCTION::MIN:
      for (auto it = root->weights.begin(); it != root->weights.end(); ++it) {
        predictions[it->i] = (!has_predictions[it->i] || it->value < predictions[it->i])
                                 ? it->value
                                 : predictions[it->i];
        has_predictions[it->i] = 1;
      }
      break;
    case AGGREGATE_FUNCTION::MAX:
      for (auto it = root->weights.begin(); it != root->weights.end(); ++it) {
        predictions[it->i] = (!has_predictions[it->i] || it->value > predictions[it->i])
                                 ? it->value
                                 : predictions[it->i];
        has_predictions[it->i] = 1;
      }
      break;
  }
  return common::Status::OK();
}

template <typename T>
common::Status TreeEnsembleRegressor<T>::Compute(OpKernelContext* context) const {
  const auto* X = context->Input<Tensor>(0);
  if (X == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
  if (X->Shape().NumDimensions() == 0) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                  "Input shape needs to be at least a single dimension.");
  }

  int64_t stride = X->Shape().NumDimensions() == 1 ? X->Shape()[0] : X->Shape()[1];
  int64_t N = X->Shape().NumDimensions() == 1 ? 1 : X->Shape()[0];
  Tensor* Y = context->Output(0, TensorShape({N, n_targets_}));

  const auto* x_data = X->template Data<T>();

  if (n_targets_ == 1) {
    float origin = base_values_.size() == 1 ? base_values_[0] : 0.f;
    if (N == 1) {
      float scores = 0;
      unsigned char has_scores = 0;

      if (nbtrees_ >= 2 && (aggregate_function_ == AGGREGATE_FUNCTION::AVERAGE ||
                            aggregate_function_ == AGGREGATE_FUNCTION::SUM)) {
#ifdef USE_OPENMP
#pragma omp parallel for reduction(|                         \
                                   : has_scores) reduction(+ \
                                                           : scores)
#endif
        for (int64_t j = 0; j < nbtrees_; ++j)
          ProcessTreeNode(&scores, roots_[j], x_data, &has_scores);
      } else {
        for (int64_t j = 0; j < nbtrees_; ++j)
          ProcessTreeNode(&scores, roots_[j], x_data, &has_scores);
      }

      float val = has_scores
                      ? (aggregate_function_ == AGGREGATE_FUNCTION::AVERAGE
                             ? scores / roots_.size()
                             : scores) +
                            origin
                      : origin;
      *((float*)(Y->template MutableData<T>())) = (post_transform_ == POST_EVAL_TRANSFORM::PROBIT)
                                                      ? ComputeProbit(val)
                                                      : val;
    } else {
      float scores;
      unsigned char has_scores;
      float val;
#ifdef USE_OPENMP
#pragma omp parallel for private(scores, has_scores, val)
#endif
      for (int64_t i = 0; i < N; ++i) {
        scores = 0;
        has_scores = 0;

        for (size_t j = 0; j < (size_t)nbtrees_; ++j)
          ProcessTreeNode(&scores, roots_[j], x_data + i * stride, &has_scores);

        val = has_scores
                  ? (aggregate_function_ == AGGREGATE_FUNCTION::AVERAGE
                         ? scores / roots_.size()
                         : scores) +
                        origin
                  : origin;
        *((float*)(Y->template MutableData<T>()) + i) = (post_transform_ == POST_EVAL_TRANSFORM::PROBIT)
                                                            ? ComputeProbit(val)
                                                            : val;
      }
    }
  } else {
    if (N == 1) {
      std::vector<float> scores(n_targets_, (T)0);
      std::vector<unsigned char> has_scores(n_targets_, 0);

#ifdef USE_OPENMP
#pragma omp parallel
#endif
      {
        std::vector<float> private_scores(scores);
        std::vector<unsigned char> private_has_scores(has_scores);
#ifdef USE_OPENMP
#pragma omp for
#endif
        for (int64_t j = 0; j < nbtrees_; ++j)
          ProcessTreeNode(private_scores.data(), roots_[j], x_data, private_has_scores.data());

        switch (aggregate_function_) {
          case AGGREGATE_FUNCTION::AVERAGE:
          case AGGREGATE_FUNCTION::SUM:
#ifdef USE_OPENMP
#pragma omp critical
#endif
            for (int64_t n = 0; n < n_targets_; ++n) {
              if (private_has_scores[n]) {
                has_scores[n] = private_has_scores[n];
                scores[n] += private_scores[n];
              }
            }
            break;
          case AGGREGATE_FUNCTION::MIN:
#ifdef USE_OPENMP
#pragma omp critical
#endif
            for (int64_t n = 0; n < n_targets_; ++n) {
              if (private_has_scores[n]) {
                scores[n] = has_scores[n] && (private_scores[n] > scores[n]) ? scores[n] : private_scores[n];
                has_scores[n] = private_has_scores[n];
              }
            }
            break;
          case AGGREGATE_FUNCTION::MAX:
#ifdef USE_OPENMP
#pragma omp critical
#endif
            for (int64_t n = 0; n < n_targets_; ++n) {
              if (private_has_scores[n]) {
                scores[n] = has_scores[n] && (private_scores[n] < scores[n]) ? scores[n] : private_scores[n];
                has_scores[n] = private_has_scores[n];
              }
            }
            break;
        }
      }

      std::vector<float> outputs(n_targets_);
      float val;
      for (int64_t j = 0; j < n_targets_; ++j) {
        //reweight scores based on number of voters
        val = base_values_.size() == (size_t)n_targets_ ? base_values_[j] : 0.f;
        val = (has_scores[j])
                  ? val + (aggregate_function_ == AGGREGATE_FUNCTION::AVERAGE
                               ? scores[j] / roots_.size()
                               : scores[j])
                  : val;
        outputs[j] = val;
      }
      write_scores(outputs, post_transform_, 0, Y, -1);

    } else {
#ifdef USE_OPENMP
#pragma omp parallel
#endif
      {
        std::vector<float> scores(n_targets_, (T)0);
        std::vector<float> outputs(n_targets_);
        std::vector<unsigned char> has_scores(n_targets_, 0);
        int64_t current_weight_0;
        float val;
        size_t j;
        int64_t jt;

#ifdef USE_OPENMP
#pragma omp parallel for
#endif
        for (int64_t i = 0; i < N; ++i) {
          current_weight_0 = i * stride;
          std::fill(scores.begin(), scores.end(), 0.f);
          std::fill(outputs.begin(), outputs.end(), 0.f);
          memset(has_scores.data(), 0, has_scores.size());

          for (j = 0; j < roots_.size(); ++j)
            ProcessTreeNode(scores.data(), roots_[j], x_data + current_weight_0,
                            has_scores.data());

          for (jt = 0; jt < n_targets_; ++jt) {
            val = base_values_.size() == (size_t)n_targets_ ? base_values_[jt] : 0.f;
            val = (has_scores[jt])
                      ? val + (aggregate_function_ == AGGREGATE_FUNCTION::AVERAGE
                                   ? scores[jt] / roots_.size()
                                   : scores[jt])
                      : val;
            outputs[jt] = val;
          }
          write_scores(outputs, post_transform_, i * n_targets_, Y, -1);
        }
      }
    }
  }

  return Status::OK();
}

}  // namespace ml
}  // namespace onnxruntime
