#pragma once
// Implements TreeEnsembleCommonAttributes, TreeEnsembleCommon.

// Inspired from
// https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/tree_ensemble_regressor.cc.

#include "c_op_tree_ensemble_common_agg_.hpp"
#include "c_op_allocation.h"
#include "c_op_common_parallel.hpp"
#include "sparse_tensor.h"
#include "onnx_extended_helpers.h"

#include <deque>
#include <limits>
#include <map>
#include <unordered_map>

// uncomment the following line to debug the computation if needed
// #define DEBUG_CHECK
// #define DEBUG_STEP

#if defined(DEBUG_CHECK)
// #define DEBUG_PRINT(...)
#define DEBUG_PRINT(...)                                                                       \
  printf(                                                                                      \
      "%s",                                                                                    \
      MakeString("*", __FILE__, ":", __LINE__, ":", MakeString(__VA_ARGS__), "\n").c_str());
#define DEBUG_INDEX(index, total, ...)                                                         \
  {                                                                                            \
    if (index >= total)                                                                        \
      throw std::runtime_error(                                                                \
          MakeString("*", __FILE__, ":", __LINE__, ":", MakeString(__VA_ARGS__), "\n")         \
              .c_str());                                                                       \
  }
#else
#define DEBUG_PRINT(...)
#define DEBUG_INDEX(index, total, ...)
#endif

#if defined(DEBUG_STEP)
#define DEBUG_PRINT_STEP(msg) printf("%s\n", msg);
#else
#define DEBUG_PRINT_STEP(msg)
#endif

// https://cims.nyu.edu/~stadler/hpc17/material/ompLec.pdf
// http://amestoy.perso.enseeiht.fr/COURS/CoursMulticoreProgrammingButtari.pdf

using namespace onnx_extended_helpers;

namespace onnx_c_ops {

template <class Tp> struct TreeAlloc {
  typedef Tp value_type;

  TreeAlloc() = default;
  template <class T> TreeAlloc(const TreeAlloc<T> &) {}

  Tp *allocate(std::size_t n) {
    n *= sizeof(Tp);
    Tp *p = (Tp *)AllocatorDefaultAlloc(n);
    return p;
  }

  void deallocate(Tp *p, std::size_t) { AllocatorDefaultFree(p); }
};

enum FeatureRepresentation { NONE, DENSE, SPARSE };

template <typename T> struct FeatureAccessor {
  typedef T ValueType;
  const T *data;
  int64_t n_rows;
  int64_t n_features;
  inline FeatureAccessor(const T *ptr, int64_t n, int64_t c)
      : data(ptr), n_rows(n), n_features(c) {}
  static inline FeatureRepresentation FeatureType() { return FeatureRepresentation::NONE; }
};

template <typename T> struct DenseFeatureAccessor : public FeatureAccessor<T> {
  struct RowAccessor {
    const T *ptr;
    inline T get(int64_t col) const { return ptr[col]; }
    static inline FeatureRepresentation FeatureType() { return FeatureRepresentation::DENSE; }
  };

  inline DenseFeatureAccessor(const T *ptr, int64_t n, int64_t c)
      : FeatureAccessor<T>(ptr, n, c) {}

  inline RowAccessor get(int64_t row) const {
    return RowAccessor{this->data + row * this->n_features};
  }

  static inline FeatureRepresentation FeatureType() { return FeatureRepresentation::DENSE; }
};

template <typename T> struct SparseFeatureAccessor : public FeatureAccessor<T> {
  const onnx_sparse::sparse_struct *sp;
  const uint32_t *indices;
  std::vector<uint32_t> row_indices;
  std::vector<uint32_t> element_indices;

  struct RowAccessor {
    const T *values;
    const uint32_t *root;
    const uint32_t *begin;
    const uint32_t *end;

    inline T get(int64_t col) const {
      auto it = std::lower_bound(begin, end, static_cast<uint32_t>(col));
      return (it != end && col == *it) ? values[it - root]
                                       : std::numeric_limits<T>::quiet_NaN();
    }

    static inline FeatureRepresentation FeatureType() { return FeatureRepresentation::SPARSE; }
  };

  inline SparseFeatureAccessor(const T *ptr, int64_t n, int64_t c)
      : FeatureAccessor<T>(ptr, n, c) {
    sp = (const onnx_sparse::sparse_struct *)ptr;
    if (sp->n_dims == 2) {
      this->n_rows = sp->shape[0];
      this->n_features = sp->shape[1];
    } else if (sp->n_dims == 1) {
      this->n_rows = sp->shape[0];
      this->n_features = 1;
    } else {
      this->n_rows = 1;
      for (uint32_t i = 0; i < sp->n_dims - 1; ++i)
        this->n_rows *= sp->shape[i];
      this->n_features = sp->shape[sp->n_dims - 1];
    }
    ((onnx_sparse::sparse_struct *)ptr)->csr(row_indices, element_indices);
  }

  inline RowAccessor get(int64_t row) const {
    return RowAccessor{sp->values(), &element_indices[0],
                       element_indices.data() + row_indices[row],
                       element_indices.data() + row_indices[row + 1]};
  }

  static inline FeatureRepresentation FeatureType() { return FeatureRepresentation::SPARSE; }
};

class TreeEnsembleCommonAttributes {
public:
  TreeEnsembleCommonAttributes() {
    parallel_tree_ = 80;
    parallel_tree_N_ = 128;
    parallel_N_ = 50;
    batch_size_tree_ = 1;
    batch_size_rows_ = 1;
    use_node3_ = 0;
  }

  int64_t get_target_or_class_count() const { return this->n_targets_or_classes_; }

  void set(int parallel_tree, int parallel_tree_N, int parallel_N, int batch_size_tree,
           int batch_size_rows, int use_node3) {
    if (parallel_tree >= 0)
      parallel_tree_ = parallel_tree;
    if (parallel_tree_N >= 0)
      parallel_tree_N_ = parallel_tree_N;
    if (parallel_N >= 0)
      parallel_N_ = parallel_N;
    if (batch_size_tree >= 0)
      batch_size_tree_ = batch_size_tree;
    if (batch_size_rows >= 0)
      batch_size_rows_ = batch_size_rows;
    if (use_node3 >= 0)
      use_node3_ = use_node3;
  }

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
  int parallel_tree_;   // starts parallelizing the computing by trees if n_tree
                        // >= parallel_tree_
  int parallel_tree_N_; // batch size if parallelizing by trees
  int parallel_N_;      // starts parallelizing the computing by rows if n_rows <=
                        // parallel_N_
  int batch_size_tree_;
  int batch_size_rows_;
  int use_node3_;
};

template <typename FeatureType, typename ThresholdType, typename OutputType>
class TreeEnsembleCommon : public TreeEnsembleCommonAttributes {
public:
  typedef typename FeatureType::ValueType InputType;

protected:
  std::vector<ThresholdType> base_values_;
  std::vector<TreeNodeElement<ThresholdType>, TreeAlloc<TreeNodeElement<ThresholdType>>> nodes_;
  // Type of weights should be a vector of OutputType. Onnx specifications says
  // it must be float. Lightgbm requires a double to do the summation of all
  // trees predictions. That's why `ThresholdType` is used as well for output
  // type (double as well for lightgbm) and not `OutputType`.
  std::vector<SparseValue<ThresholdType>> weights_;
  std::vector<TreeNodeElement<ThresholdType> *> roots_;

  // optimisation
  std::vector<TreeNodeElement3<ThresholdType>, TreeAlloc<TreeNodeElement3<ThresholdType>>>
      nodes3_;
  std::vector<TreeNodeElement3<ThresholdType> *> roots3_;
  OutputType bias_;

public:
  TreeEnsembleCommon() {}

  Status Init(const std::string &aggregate_function,                       // 3
              const std::vector<ThresholdType> &base_values,               // 4
              int64_t n_targets_or_classes,                                // 5
              const std::vector<int64_t> &nodes_falsenodeids,              // 6
              const std::vector<int64_t> &nodes_featureids,                // 7
              const std::vector<ThresholdType> &nodes_hitrates,            // 8
              const std::vector<int64_t> &nodes_missing_value_tracks_true, // 9
              const std::vector<std::string> &nodes_modes,                 // 10
              const std::vector<int64_t> &nodes_nodeids,                   // 11
              const std::vector<int64_t> &nodes_treeids,                   // 12
              const std::vector<int64_t> &nodes_truenodeids,               // 13
              const std::vector<ThresholdType> &nodes_values,              // 14
              const std::string &post_transform,                           // 15
              const std::vector<int64_t> &target_class_ids,                // 16
              const std::vector<int64_t> &target_class_nodeids,            // 17
              const std::vector<int64_t> &target_class_treeids,            // 18
              const std::vector<ThresholdType> &target_class_weights,      // 19
              bool is_classifier);

  Status Compute(int64_t n_rows, int64_t n_features, const InputType *X, OutputType *Y,
                 int64_t *label) const;

  int omp_get_max_threads() const;
  int64_t get_sizeof() const;

protected:
  void ConvertTreeIntoTree3();
  int ConvertTreeNodeElementIntoTreeNodeElement3(std::size_t root_id,
                                                 InlinedVector<std::size_t> &to_remove);

  const TreeNodeElement<ThresholdType> *
  ProcessTreeNodeLeave(std::size_t root_id, const typename FeatureType::RowAccessor &row) const;
  const TreeNodeElement<ThresholdType> *
  ProcessTreeNodeLeave3(std::size_t root_id,
                        const typename FeatureType::RowAccessor &row) const;

  template <typename AGG>
  void ComputeAgg(const FeatureType &data, OutputType *Y, int64_t *labels,
                  const AGG &agg) const;
};

template <typename FeatureType, typename ThresholdType, typename OutputType>
int64_t TreeEnsembleCommon<FeatureType, ThresholdType, OutputType>::get_sizeof() const {
  int64_t res = 0;
  res += base_values_.size() * sizeof(ThresholdType);
  res += nodes_.size() * sizeof(TreeNodeElement<ThresholdType>);
  res += weights_.size() * sizeof(SparseValue<ThresholdType>);
  res += roots_.size() * sizeof(TreeNodeElement<ThresholdType> *);
  res += nodes3_.size() * sizeof(TreeNodeElement3<ThresholdType>);
  res += roots3_.size() * sizeof(TreeNodeElement3<ThresholdType> *);
  return res;
}

template <typename FeatureType, typename ThresholdType, typename OutputType>
int TreeEnsembleCommon<FeatureType, ThresholdType, OutputType>::omp_get_max_threads() const {
  return ::omp_get_max_threads();
}

template <typename FeatureType, typename ThresholdType, typename OutputType>
Status TreeEnsembleCommon<FeatureType, ThresholdType, OutputType>::Init(
    const std::string &aggregate_function, const std::vector<ThresholdType> &base_values,
    int64_t n_targets_or_classes, const std::vector<int64_t> &nodes_falsenodeids,
    const std::vector<int64_t> &nodes_featureids,
    const std::vector<ThresholdType> & /* nodes_hitrates */,
    const std::vector<int64_t> &nodes_missing_value_tracks_true,
    const std::vector<std::string> &nodes_modes, const std::vector<int64_t> &nodes_nodeids,
    const std::vector<int64_t> &nodes_treeids, const std::vector<int64_t> &nodes_truenodeids,
    const std::vector<ThresholdType> &nodes_values, const std::string &post_transform,
    const std::vector<int64_t> &target_class_ids,
    const std::vector<int64_t> &target_class_nodeids,
    const std::vector<int64_t> &target_class_treeids,
    const std::vector<ThresholdType> &target_class_weights, bool is_classifier) {

  DEBUG_PRINT("Init:Check")
  EXT_ENFORCE(n_targets_or_classes > 0);
  EXT_ENFORCE(nodes_falsenodeids.size() == nodes_featureids.size());
  EXT_ENFORCE(nodes_falsenodeids.size() == nodes_modes.size(),
              "nodes_falsenodeids.size()=", (uint64_t)nodes_falsenodeids.size(),
              " nodes_modes.size()=", (int64_t)nodes_modes.size());
  EXT_ENFORCE(nodes_falsenodeids.size() == nodes_nodeids.size());
  EXT_ENFORCE(nodes_falsenodeids.size() == nodes_treeids.size());
  EXT_ENFORCE(nodes_falsenodeids.size() == nodes_truenodeids.size());
  EXT_ENFORCE(nodes_falsenodeids.size() == nodes_values.size());
  EXT_ENFORCE(target_class_ids.size() == target_class_nodeids.size());
  EXT_ENFORCE(target_class_ids.size() == target_class_treeids.size());
  EXT_ENFORCE(target_class_ids.size() == target_class_treeids.size());
  EXT_ENFORCE(target_class_weights.size() > 0);

  aggregate_function_ = to_AGGREGATE_FUNCTION(aggregate_function);
  post_transform_ = to_POST_EVAL_TRANSFORM(post_transform);
  base_values_.reserve(base_values.size());
  for (std::size_t i = 0, limit = base_values.size(); i < limit; ++i) {
    base_values_.push_back(static_cast<ThresholdType>(base_values[i]));
  }
  n_targets_or_classes_ = n_targets_or_classes;
  max_tree_depth_ = 1000;
  EXT_ENFORCE(nodes_modes.size() < std::numeric_limits<uint32_t>::max());

  // additional members
  std::size_t limit;
  uint32_t i;
  InlinedVector<NODE_MODE> cmodes;
  cmodes.reserve(nodes_modes.size());
  same_mode_ = true;
  int fpos = -1;
  for (i = 0, limit = nodes_modes.size(); i < limit; ++i) {
    cmodes.push_back(to_NODE_MODE(nodes_modes[i]));
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
  limit = static_cast<std::size_t>(n_nodes_);
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
    node.value_or_unique_weight = nodes_values[i];

    /* hitrates is not used for inference, they are ignored.
    node.hitrates = nodes_hitrates[i];
    */

    node.flags = static_cast<uint8_t>(cmodes[i]);
    node.truenode_inc_or_first_weight = 0; // nodes_truenodeids[i] if not a leaf
    node.falsenode_inc_or_n_weights = 0;   // nodes_falsenodeids[i] if not a leaf

    if (i < static_cast<std::size_t>(nodes_missing_value_tracks_true.size()) &&
        nodes_missing_value_tracks_true[i] == 1) {
      node.flags |= static_cast<uint8_t>(MissingTrack::kTrue);
    }
    auto p = idi.insert(std::pair<TreeNodeElementId, uint32_t>(node_tree_id, i));
    if (!p.second) {
      EXT_THROW("Node ", node_tree_id.node_id, " in tree ", node_tree_id.tree_id,
                " is already there.");
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

    TreeNodeElementId &node_tree_id = node_tree_ids[i];
    coor.tree_id = node_tree_id.tree_id;
    coor.node_id = static_cast<int>(nodes_truenodeids[i]);
    EXT_ENFORCE((coor.node_id >= 0 && coor.node_id < n_nodes_));

    auto found = idi.find(coor);
    if (found == idi.end()) {
      EXT_THROW("Unable to find node ", coor.tree_id, "-", coor.node_id, " (truenode).");
    }
    if (found->second == truenode_ids.size()) {
      EXT_THROW("A node cannot point to itself: ", coor.tree_id, "-", node_tree_id.node_id,
                " (truenode).");
    }
    truenode_ids.emplace_back(found->second);

    coor.node_id = static_cast<int>(nodes_falsenodeids[i]);
    EXT_ENFORCE((coor.node_id >= 0 && coor.node_id < n_nodes_));
    found = idi.find(coor);
    if (found == idi.end()) {
      EXT_THROW("Unable to find node ", coor.tree_id, "-", coor.node_id, " (falsenode).");
    }
    if (found->second == falsenode_ids.size()) {
      EXT_THROW("A node cannot point to itself: ", coor.tree_id, "-", node_tree_id.node_id,
                " (falsenode).");
    }
    falsenode_ids.emplace_back(found->second);
    // We could also check that truenode_ids[truenode_ids.size() - 1] !=
    // falsenode_ids[falsenode_ids.size() - 1]). It is valid but no training
    // algorithm would produce a tree where left and right nodes are the same.
  }

  // sort targets
  InlinedVector<std::pair<TreeNodeElementId, uint32_t>> indices;
  indices.reserve(target_class_nodeids.size());
  for (i = 0, limit = target_class_nodeids.size(); i < limit; i++) {
    indices.emplace_back(std::pair<TreeNodeElementId, uint32_t>(
        TreeNodeElementId{target_class_treeids[i], target_class_nodeids[i]}, i));
  }
  std::sort(indices.begin(), indices.end());

  // bias estimations
  bias_ = static_cast<OutputType>(0);
  if (!is_classifier) {
    switch (aggregate_function_) {
    case AGGREGATE_FUNCTION::AVERAGE:
    case AGGREGATE_FUNCTION::SUM: {
      for (std::size_t wi = 0; wi < target_class_weights.size(); ++wi)
        bias_ += target_class_weights[wi];
      bias_ /= static_cast<OutputType>(target_class_weights.size());
    } break;
    default:
      break;
    }
  }

  // Initialize the leaves.
  TreeNodeElementId ind;
  SparseValue<ThresholdType> w;
  std::size_t indi;
  for (indi = 0, limit = target_class_nodeids.size(); indi < limit; ++indi) {
    ind = indices[indi].first;
    i = indices[indi].second;
    auto found = idi.find(ind);
    if (found == idi.end()) {
      EXT_THROW("Unable to find node ", ind.tree_id, "-", ind.node_id, " (weights).");
    }

    TreeNodeElement<ThresholdType> &leaf = nodes_[found->second];
    if (leaf.is_not_leaf()) {
      // An exception should be raised in that case. But this case may happen
      // in models converted with an old version of onnxmltools. These weights
      // are ignored. EXT_THROW("Node ", ind.tree_id, "-", ind.node_id, " is
      // not a leaf.");
      continue;
    }

    w.i = target_class_ids[i];
    w.value = target_class_weights[i] - bias_;
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
        EXT_THROW("Target is missing for leaf ", ind.tree_id, "-", ind.node_id, ".");
      }
      continue;
    }
    EXT_ENFORCE(truenode_ids[i] != i); // That would mean the left node is
                                       // itself, leading to an infinite loop.
    nodes_[i].truenode_inc_or_first_weight = static_cast<int32_t>(truenode_ids[i] - i);
    EXT_ENFORCE(falsenode_ids[i] != i); // That would mean the right node is
                                        // itself, leading to an infinite loop.
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

  if (use_node3_) {
    // Use optimized implementation with bigger nodes.
    DEBUG_PRINT("Init:Tree3")
    ConvertTreeIntoTree3();
  }
  DEBUG_PRINT("Init:End")
  return Status::OK();
}

template <typename FeatureType, typename ThresholdType, typename OutputType>
void TreeEnsembleCommon<FeatureType, ThresholdType, OutputType>::ConvertTreeIntoTree3() {
  DEBUG_PRINT("ConvertTreeIntoTree3")
  roots3_.clear();
  nodes3_.clear();
  if (!same_mode_ || (nodes_.size() >= (static_cast<std::size_t>(2) << 30))) {
    // Not applicable in that case.
    return;
  }
  InlinedVector<int> root3_ids;
  root3_ids.reserve(roots_.size());
  InlinedVector<std::size_t> to_remove;
  to_remove.reserve(nodes_.size());
  for (std::size_t root_id = 0; root_id < roots_.size(); ++root_id) {
    auto root3_id = ConvertTreeNodeElementIntoTreeNodeElement3(root_id, to_remove);
    root3_ids.push_back(root3_id);
  }

  if (to_remove.size() == 0) {
    // No improvment.
    return;
  }
  // TODO: We should rename and remove node_.

  // Captures the pointer on the root3.
  roots3_.reserve(root3_ids.size());
  for (auto it = root3_ids.begin(); it != root3_ids.end(); ++it) {
    roots3_.push_back(*it >= 0 ? &nodes3_[*it] : nullptr);
  }
}

template <typename FeatureType, typename ThresholdType, typename OutputType>
int TreeEnsembleCommon<FeatureType, ThresholdType, OutputType>::
    ConvertTreeNodeElementIntoTreeNodeElement3(std::size_t root_id,
                                               InlinedVector<std::size_t> &to_remove) {
  std::vector<std::size_t> removed_nodes;
  TreeNodeElement<ThresholdType> *node, *true_node, *false_node;
  std::deque<std::pair<std::size_t, TreeNodeElement<ThresholdType> *>> stack;
  std::unordered_map<std::size_t, std::size_t> map_node_to_node3;
  std::pair<std::size_t, TreeNodeElement<ThresholdType> *> pair;
  std::size_t last_node3 = nodes3_.size();
  nodes3_.reserve(nodes_.size() / 3);
  stack.push_back(std::pair<std::size_t, TreeNodeElement<ThresholdType> *>(
      roots_[root_id] - &(nodes_[0]), roots_[root_id]));
  while (!stack.empty()) {
    pair = stack.front();
    stack.pop_front();
    // EXT_ENFORCE(map_node_to_node3.find(pair.first) ==
    // map_node_to_node3.end(),
    //          "This node index ", pair.first,
    //          " was already added as a TreeNodeElement3.");
    node = pair.second;
    if (!node->is_not_leaf()) {
      continue;
    }
    true_node = node + node->truenode_inc_or_first_weight;
    false_node = node + node->falsenode_inc_or_n_weights;
    if (!true_node->is_not_leaf() || !false_node->is_not_leaf()) {
      continue;
    }
    TreeNodeElement3<ThresholdType> node3;
    node3.node_id[0] = static_cast<int32_t>(pair.first) + node->falsenode_inc_or_n_weights +
                       false_node->falsenode_inc_or_n_weights;
    node3.node_id[1] = static_cast<int32_t>(pair.first) + node->falsenode_inc_or_n_weights +
                       false_node->truenode_inc_or_first_weight;
    node3.node_id[2] = static_cast<int32_t>(pair.first) + node->truenode_inc_or_first_weight +
                       true_node->falsenode_inc_or_n_weights;
    node3.node_id[3] = static_cast<int32_t>(pair.first) + node->truenode_inc_or_first_weight +
                       true_node->truenode_inc_or_first_weight;

    node3.feature_id[0] = false_node->feature_id;
    node3.feature_id[1] = true_node->feature_id;
    node3.feature_id[2] = node->feature_id;

    node3.thresholds[0] = false_node->value_or_unique_weight;
    node3.thresholds[2] = true_node->value_or_unique_weight;
    node3.thresholds[1] = node->value_or_unique_weight;
    node3.thresholds[3] = node->value_or_unique_weight; // repeated for AVX

    node3.flags = node->mode() | (false_node->is_missing_track_true() * MissingTrack3::kTrue0) |
                  (true_node->is_missing_track_true() * MissingTrack3::kTrue1) |
                  (node->is_missing_track_true() * MissingTrack3::kTrue2);

    auto node3_index = nodes3_.size();
    bool add = true;
    for (std::size_t i = 0; i < 4; ++i) {
      auto it = map_node_to_node3.find(node3.node_id[i]);
      if (it != map_node_to_node3.end()) {
        // A node already points to another node converted into node3.
        // This happens when a child node points to another node at a lower
        // level (closer to the root).
        add = false;
        break;
      }
    }
    if (!add) {
      // Unable to handle this node.
      continue;
    }
    for (std::size_t i = 0; i < 4; ++i) {
      stack.push_back(std::pair<std::size_t, TreeNodeElement<ThresholdType> *>(
          node3.node_id[i], &(nodes_[node3.node_id[i]])));
    }
    map_node_to_node3[pair.first] = node3_index;
    map_node_to_node3[static_cast<int32_t>(pair.first) + node->truenode_inc_or_first_weight] =
        node3_index;
    map_node_to_node3[static_cast<int32_t>(pair.first) + node->falsenode_inc_or_n_weights] =
        node3_index;
    nodes3_.emplace_back(node3);
    to_remove.push_back(pair.first);
    to_remove.push_back(pair.first + node->falsenode_inc_or_n_weights);
    to_remove.push_back(pair.first + node->truenode_inc_or_first_weight);
  }
  // Every node3 points to a node. It needs to be changed.
  int changed;
  for (std::size_t i = last_node3; i < nodes3_.size(); ++i) {
    TreeNodeElement3<ThresholdType> &n3 = nodes3_[i];
    changed = 0;
    for (std::size_t j = 0; j < 4; ++j) {
      auto it = map_node_to_node3.find(n3.node_id[j]);
      if (it == map_node_to_node3.end())
        break;
      ++changed;
    }
    if (changed == 4) {
      n3.flags |= MissingTrack3::kChildren3;
      for (std::size_t j = 0; j < 4; ++j) {
        auto it = map_node_to_node3.find(n3.node_id[j]);
        n3.node_id[j] = static_cast<int32_t>(it->second);
      }
    }
  }
  return nodes3_.size() > last_node3 ? static_cast<int>(last_node3) : -1;
}

template <typename FeatureType, typename ThresholdType, typename OutputType>
Status TreeEnsembleCommon<FeatureType, ThresholdType, OutputType>::Compute(
    int64_t n_rows, int64_t n_features, const InputType *X, OutputType *Y,
    int64_t *label) const {
  FeatureType features(X, n_rows, n_features);

  switch (aggregate_function_) {
  case AGGREGATE_FUNCTION::AVERAGE:
    DEBUG_PRINT("Compute AVERAGE")
    ComputeAgg(features, Y, label,
               TreeAggregatorAverage<FeatureType, ThresholdType, OutputType>(
                   roots_.size(), n_targets_or_classes_, post_transform_, base_values_, bias_));
    return Status::OK();
  case AGGREGATE_FUNCTION::SUM:
    DEBUG_PRINT("Compute SUM")
    ComputeAgg(features, Y, label,
               TreeAggregatorSum<FeatureType, ThresholdType, OutputType>(
                   roots_.size(), n_targets_or_classes_, post_transform_, base_values_, bias_));
    return Status::OK();
  case AGGREGATE_FUNCTION::MIN:
    DEBUG_PRINT("Compute MIN")
    ComputeAgg(features, Y, label,
               TreeAggregatorMin<FeatureType, ThresholdType, OutputType>(
                   roots_.size(), n_targets_or_classes_, post_transform_, base_values_, bias_));
    return Status::OK();
  case AGGREGATE_FUNCTION::MAX:
    DEBUG_PRINT("Compute MAX")
    ComputeAgg(features, Y, label,
               TreeAggregatorMax<FeatureType, ThresholdType, OutputType>(
                   roots_.size(), n_targets_or_classes_, post_transform_, base_values_, bias_));
    return Status::OK();
  default:
    EXT_THROW("Unknown aggregation function in TreeEnsemble.");
  }
}

template <typename FeatureType, typename ThresholdType, typename OutputType>
template <typename AGG>
void TreeEnsembleCommon<FeatureType, ThresholdType, OutputType>::ComputeAgg(
    const FeatureType &features, OutputType *Y, int64_t *labels, const AGG &agg) const {
  int64_t N = features.n_rows;
  int64_t C = features.n_features;
  if (max_feature_id_ >= C) {
    throw std::runtime_error(MakeString("One path in the graph requests feature ",
                                        max_feature_id_, " but input tensor has ", C,
                                        " features."));
  }
  OutputType *z_data = Y;

  int64_t *label_data = labels;
  int64_t max_n_threads = omp_get_max_threads();
  int64_t parallel_tree_n = parallel_tree_N_;

  DEBUG_PRINT("max_n_threads=", max_n_threads)
  DEBUG_PRINT("parallel_tree_N_=", parallel_tree_N_)
  DEBUG_PRINT("parallel_tree_n=", parallel_tree_n)
  DEBUG_PRINT("n_targets_or_classes_=", n_targets_or_classes_, " N=", N,
              " agg.kind()=", agg.kind())

  // printf("n_targets_or_classes_=%d N=%d n_trees_=%d parallel_tree_=%d
  // parallel_N_=%d max_n_threads=%d\n",
  //   n_targets_or_classes_, N, n_trees_, parallel_tree_, parallel_N_,
  //   max_n_threads);
  if (n_targets_or_classes_ == 1) {
    DEBUG_PRINT()
    if (N == 1) {
      DEBUG_PRINT()
      ScoreValue<ThresholdType> score = {0, 0};
      if (n_trees_ <= parallel_tree_ ||
          max_n_threads == 1) { /* section A: 1 output, 1 row and not enough
                                     trees to parallelize */
        DEBUG_PRINT_STEP("S:N1:TN")
        DEBUG_PRINT()
        auto acc = features.get(0);
        for (int64_t j = 0; j < n_trees_; ++j) {
          agg.ProcessTreeNodePrediction1(
              score, *ProcessTreeNodeLeave(static_cast<std::size_t>(j), acc));
        }
        DEBUG_PRINT()
      } else { /* section B: 1 output, 1 row and enough trees to parallelize
                */
        DEBUG_PRINT()
        DEBUG_PRINT_STEP("S:N1:TN-P")
        std::vector<ScoreValue<ThresholdType>> scores(static_cast<std::size_t>(n_trees_),
                                                      {0, 0});
        auto acc = features.get(0);
        TryBatchParallelFor(max_n_threads, this->batch_size_tree_, n_trees_,
                            [this, &scores, &agg, &acc](int64_t j) {
                              agg.ProcessTreeNodePrediction1(scores[j],
                                                             *ProcessTreeNodeLeave(j, acc));
                            });

        for (auto it = scores.cbegin(); it != scores.cend(); ++it) {
          agg.MergePrediction1(score, *it);
        }
        DEBUG_PRINT()
      }
      agg.FinalizeScores1(z_data, score, label_data);
      DEBUG_PRINT()
    } else if ((N <= parallel_N_ && n_trees_ <= parallel_tree_) ||
               max_n_threads == 1) { /* section C: 1 output, 2+ rows but not
                                        enough rows to parallelize */
      // Not enough data to parallelize but the computation is split into
      // batches of 128 rows, and then loop on trees to evaluate every tree on
      // this batch. This change was introduced by PR:
      // https://github.com/microsoft/onnxruntime/pull/13835. The input tensor
      // (2D) is stored in a contiguous array. Therefore, it is faster to loop
      // on tree first and inside that loop evaluate a tree on the input
      // tensor (inner loop). The processor is faster when it has to move
      // chunks of a contiguous array (branching). However, if the input
      // tensor is too big, the data does not hold on caches (L1, L2, L3). In
      // that case, looping first on tree or on data is almost the same.
      // That's why the first loop split into batch so that every batch holds
      // on caches, then loop on trees and finally loop on the batch rows.
      DEBUG_PRINT()
      DEBUG_PRINT_STEP("S:NN:TN")
      std::vector<ScoreValue<ThresholdType>> scores(std::min(parallel_tree_n, N));
      std::vector<typename FeatureType::RowAccessor> acc(scores.size());
      std::size_t j;
      int64_t i, batch, batch_end;

      for (batch = 0; batch < N; batch += parallel_tree_n) {
        batch_end = std::min(N, batch + parallel_tree_n);
        for (i = batch; i < batch_end; ++i) {
          scores[static_cast<int64_t>(i - batch)] = {0, 0};
          acc[i - batch] = features.get(i);
        }
        for (j = 0; j < static_cast<std::size_t>(n_trees_); ++j) {
          for (i = batch; i < batch_end; ++i) {
            agg.ProcessTreeNodePrediction1(scores[static_cast<int64_t>(i - batch)],
                                           *ProcessTreeNodeLeave(j, acc[i - batch]));
          }
        }
        for (i = batch; i < batch_end; ++i) {
          agg.FinalizeScores1(z_data + i, scores[static_cast<int64_t>(i - batch)],
                              label_data == nullptr ? nullptr : (label_data + i));
        }
      }
      DEBUG_PRINT()
    } else if (n_trees_ > max_n_threads &&
               n_trees_ >= parallel_tree_) { /* section D: 1 output, 2+ rows and
                                                enough trees to parallelize */
      DEBUG_PRINT()
      DEBUG_PRINT_STEP("S:NNB:TN-PG")
      auto n_threads = std::min<int32_t>(static_cast < int32_t>(max_n_threads), static_cast<int32_t>(n_trees_));
      int n_batches =
          static_cast<int>(this->batch_size_tree_ <= 1 ? n_trees_ : n_trees_ / this->batch_size_tree_ + 1);
      int max_n = static_cast<int> (std::min(N, parallel_tree_n));
      std::vector<ScoreValue<ThresholdType>> scores(
          static_cast<std::size_t>(n_batches * max_n));
      std::vector<typename FeatureType::RowAccessor> acc(max_n);
      int64_t end_n, begin_n = 0;
      while (begin_n < N) {
        end_n = std::min(N, begin_n + parallel_tree_n);

        // initialization of scores to 0
        TrySimpleParallelFor(n_threads, n_threads * 2, [n_threads, &scores](int64_t batch_num) {
          auto work = PartitionWork(batch_num, n_threads * 2, scores.size());
          for (int64_t i = work.start; i < work.end; ++i) {
            DEBUG_INDEX(i, scores.size(), "ERROR i=", i, " scores.size()=", scores.size(),
                        " batch_num=", batch_num, " n_threads=", n_threads);
            scores[i] = {0, 0};
          }
        });

        TrySimpleParallelFor(n_threads, n_threads * 2,
                             [n_threads, begin_n, end_n, &acc, &features](int64_t batch_num) {
                               auto work =
                                   PartitionWork(batch_num, n_threads * 2, end_n - begin_n);
                               for (int64_t i = work.start; i < work.end; ++i) {
                                 acc[i] = features.get(i + begin_n);
                               }
                             });

        // computing tree predictions
        TrySimpleParallelFor(
            n_threads, n_batches,
            [this, &agg, &scores, n_batches, &acc, begin_n, end_n, max_n](int64_t batch_num) {
              auto work = PartitionWork(batch_num, n_batches, this->n_trees_);
              int score_index;
              for (auto j = work.start; j < work.end; ++j) {
                score_index = static_cast<int>(batch_num) * max_n;
                for (int64_t i = begin_n; i < end_n; ++i, ++score_index) {
                  DEBUG_INDEX(score_index, scores.size(), "ERROR score_index=", score_index,
                              " max_n=", max_n, " i-begin_n=", i - begin_n,
                              " scores.size()=", scores.size());
                  agg.ProcessTreeNodePrediction1(scores[score_index],
                                                 *ProcessTreeNodeLeave(j, acc[i - begin_n]));
                }
              }
            });

        // reducing the predictions
        TrySimpleParallelFor(
            n_threads, n_threads * 2,
            [&agg, &scores, n_threads, begin_n, end_n, n_batches, max_n, z_data,
             label_data](int64_t batch_num) {
              auto work = PartitionWork(batch_num, n_threads * 2, end_n - begin_n);
              for (auto i = work.start; i < work.end; ++i) {
                for (int64_t j = 1; j < n_batches; ++j) {
                  DEBUG_INDEX(i, scores.size(), "ERROR i=", i,
                              " scores.size()=", scores.size());
                  DEBUG_INDEX(j * static_cast<int64_t>(max_n) + i, scores.size(), "ERROR i=", i,
                              "j=", j, " max_n=", max_n, " scores.size()=", scores.size());
                  agg.MergePrediction1(scores[i], scores[j * static_cast<int64_t>(max_n) + i]);
                }
                agg.FinalizeScores1(z_data + (begin_n + i), scores[i],
                                    label_data == nullptr ? nullptr
                                                          : (label_data + begin_n + i));
              }
            });

        begin_n = end_n;
      }
      DEBUG_PRINT()
    } else { /* section E: 1 output, 2+ rows, parallelization by rows */
      DEBUG_PRINT()
      DEBUG_PRINT_STEP("S:NN-P:TN")
      TryBatchParallelFor(
          max_n_threads, batch_size_rows_, N,
          [this, &agg, z_data, label_data, &features](int64_t i) {
            auto acc = features.get(i);
            ScoreValue<ThresholdType> score = {0, 0};
            for (std::size_t j = 0; j < static_cast<std::size_t>(n_trees_); ++j) {
              agg.ProcessTreeNodePrediction1(score, *ProcessTreeNodeLeave(j, acc));
            }
            agg.FinalizeScores1(z_data + i, score,
                                label_data == nullptr ? nullptr : (label_data + i));
          });
      DEBUG_PRINT()
    }
  } else {
    DEBUG_PRINT("C>1")
    if (N == 1) { /* section A2: 2+ outputs, 1 row, not enough trees to
                     parallelize */
      DEBUG_PRINT()
      if (n_trees_ <= parallel_tree_ || max_n_threads == 1) { /* section A2 */
        DEBUG_PRINT()
        DEBUG_PRINT_STEP("M:N1:TN")
        InlinedVector<ScoreValue<ThresholdType>> scores(
            static_cast<std::size_t>(n_targets_or_classes_), {0, 0});
        auto acc = features.get(0);
        for (int64_t j = 0; j < n_trees_; ++j) {
          agg.ProcessTreeNodePrediction(
              scores, *ProcessTreeNodeLeave(static_cast<std::size_t>(j), acc), weights_);
        }
        agg.FinalizeScores(scores, z_data, -1, label_data);
        DEBUG_PRINT()
      } else { /* section B2: 2+ outputs, 1 row, enough trees to parallelize
                */
        DEBUG_PRINT()
        DEBUG_PRINT_STEP("M:N1:TN-P")
        auto n_threads = std::min<int32_t>(static_cast<int32_t>(max_n_threads), static_cast<int32_t>(n_trees_));
        std::vector<InlinedVector<ScoreValue<ThresholdType>>> scores(n_threads * 2);
        auto acc = features.get(0);
        TrySimpleParallelFor(
            n_threads, n_threads * 2,
            [this, &agg, &scores, n_threads, &acc](int64_t batch_num) {
              DEBUG_INDEX(batch_num, scores.size(), "ERROR batch_num=", batch_num,
                          " scores.size()=", scores.size());
              scores[batch_num].resize(static_cast<std::size_t>(this->n_targets_or_classes_),
                                       {0, 0});
              auto work = PartitionWork(batch_num, n_threads * 2, this->n_trees_);
              for (auto j = work.start; j < work.end; ++j) {
                agg.ProcessTreeNodePrediction(scores[batch_num], *ProcessTreeNodeLeave(j, acc),
                                              weights_);
              }
            });
        for (std::size_t i = 1, limit = scores.size(); i < limit; ++i) {
          agg.MergePrediction(scores[0], scores[i]);
        }
        agg.FinalizeScores(scores[0], z_data, -1, label_data);
        DEBUG_PRINT()
      }
    } else if ((N <= parallel_N_ && n_trees_ <= parallel_tree_) ||
               max_n_threads == 1) { /* section C2: 2+ outputs, 2+ rows, not
                                        enough rows to parallelize */
      DEBUG_PRINT_STEP("M:NN:TN-P")
      DEBUG_PRINT("n_targets_or_classes_=", n_targets_or_classes_, " N=", N)
      std::size_t j, limit;
      int64_t i, batch, batch_end;
      batch_end = std::min(N, static_cast<int64_t>(parallel_tree_n));
      std::vector<InlinedVector<ScoreValue<ThresholdType>>> scores(batch_end);
      std::vector<typename FeatureType::RowAccessor> acc(scores.size());
      for (i = 0; i < batch_end; ++i) {
        scores[i].resize(static_cast<std::size_t>(n_targets_or_classes_));
        acc[i] = features.get(i);
      }
      for (batch = 0; batch < N; batch += parallel_tree_n) {
        batch_end = std::min(N, batch + parallel_tree_n);
        for (i = batch; i < batch_end; ++i) {
          std::fill(scores[i - batch].begin(), scores[i - batch].end(),
                    ScoreValue<ThresholdType>({0, 0}));
        }
        for (j = 0, limit = roots_.size(); j < limit; ++j) {
          for (i = batch; i < batch_end; ++i) {
            agg.ProcessTreeNodePrediction(scores[i - batch], *ProcessTreeNodeLeave(j, acc[i]),
                                          weights_);
          }
        }
        for (i = batch; i < batch_end; ++i) {
          agg.FinalizeScores(scores[i - batch], z_data + i * n_targets_or_classes_, -1,
                             label_data == nullptr ? nullptr : (label_data + i));
        }
      }
      DEBUG_PRINT()
    } else if (n_trees_ >= max_n_threads &&
               n_trees_ >= parallel_tree_) { /* section: D2: 2+ outputs, 2+ rows,
                                                enough trees to parallelize*/
      DEBUG_PRINT_STEP("M:NNB:TN-PG")
      DEBUG_PRINT()
      auto n_threads = std::min<int32_t>(static_cast<int32_t>(max_n_threads), static_cast<int32_t>(n_trees_));
      int n_batches =
          static_cast<int>(this->batch_size_tree_ <= 1 ? n_trees_ : n_trees_ / this->batch_size_tree_ + 1);
      int max_n = static_cast<int>(std::min(N, parallel_tree_n));
      std::vector<InlinedVector<ScoreValue<ThresholdType>>> scores(
          static_cast<std::size_t>(n_batches * max_n));
      for (std::size_t ind = 0; ind < scores.size(); ++ind) {
        scores[ind].resize(static_cast<std::size_t>(n_targets_or_classes_));
      }

      int64_t end_n, begin_n = 0;
      while (begin_n < N) {
        end_n = std::min(N, begin_n + parallel_tree_n);

        // initialization of scores to 0
        TrySimpleParallelFor(n_threads, n_threads * 2, [n_threads, &scores](int64_t batch_num) {
          auto work = PartitionWork(batch_num, n_threads * 2, scores.size());
          for (int64_t i = work.start; i < work.end; ++i) {
            DEBUG_INDEX(i, scores.size(), "ERROR i=", i, " scores.size()=", scores.size(),
                        " batch_num=", batch_num, " n_threads=", n_threads);
            std::fill(scores[i].begin(), scores[i].end(), ScoreValue<ThresholdType>({0, 0}));
          }
        });

        // computing tree predictions
        TrySimpleParallelFor(
            n_threads, n_batches,
            [this, &agg, &scores, n_batches, begin_n, end_n, max_n,
             &features](int64_t batch_num) {
              auto work = PartitionWork(batch_num, n_batches, this->n_trees_);
              int score_index;
              for (auto j = work.start; j < work.end; ++j) {
                score_index = static_cast<int>(batch_num )* max_n;
                for (int64_t i = begin_n; i < end_n; ++i, ++score_index) {
                  DEBUG_INDEX(score_index, scores.size(), "ERROR score_index=", score_index,
                              " max_n=", max_n, " i-begin_n=", i - begin_n,
                              " scores.size()=", scores.size());
                  agg.ProcessTreeNodePrediction(
                      scores[score_index], *ProcessTreeNodeLeave(j, features.get(i)), weights_);
                }
              }
            });

        // reducing the predictions
        TrySimpleParallelFor(
            n_threads, n_threads * 2,
            [&agg, &scores, this, n_threads, begin_n, end_n, n_batches, max_n, z_data,
             label_data](int64_t batch_num) {
              auto work = PartitionWork(batch_num, n_threads * 2, end_n - begin_n);
              for (auto i = work.start; i < work.end; ++i) {
                for (int64_t j = 1; j < n_batches; ++j) {
                  DEBUG_INDEX(i, scores.size(), "ERROR i=", i,
                              " scores.size()=", scores.size());
                  DEBUG_INDEX(j * static_cast<int64_t>(max_n) + i, scores.size(), "ERROR i=", i,
                              "j=", j, " max_n=", max_n, " scores.size()=", scores.size());
                  agg.MergePrediction(scores[i], scores[j * static_cast<int64_t>(max_n) + i]);
                }
                agg.FinalizeScores(
                    scores[i], z_data + ((begin_n + i) * n_targets_or_classes_), -1,
                    label_data == nullptr ? nullptr : (label_data + begin_n + i));
              }
            });

        begin_n = end_n;
      }
      DEBUG_PRINT()
    } else { /* section E2: 2+ outputs, 2+ rows, parallelization by rows */
      DEBUG_PRINT_STEP("M:NNB-P:TN")
      DEBUG_PRINT()
      auto n_threads = std::min<int32_t>(static_cast<int32_t>(max_n_threads), static_cast<int32_t>(N));
      auto n_batches = N / n_threads + 1;
      TrySimpleParallelFor(
          n_threads, n_batches,
          [this, &agg, z_data, label_data, N, &features, n_batches](int64_t batch_num) {
            auto work = PartitionWork(batch_num, n_batches, N);
            for (int64_t i = work.start; i < work.end; ++i) {
              std::size_t j, limit;
              InlinedVector<ScoreValue<ThresholdType>> scores(
                  static_cast<std::size_t>(n_targets_or_classes_));

              std::fill(scores.begin(), scores.end(), ScoreValue<ThresholdType>({0, 0}));
              for (j = 0, limit = roots_.size(); j < limit; ++j) {
                agg.ProcessTreeNodePrediction(scores, *ProcessTreeNodeLeave(j, features.get(i)),
                                              weights_);
              }

              agg.FinalizeScores(scores, z_data + i * n_targets_or_classes_, -1,
                                 label_data == nullptr ? nullptr : (label_data + i));
            }
          });
      DEBUG_PRINT()
    }
  }
  DEBUG_PRINT()
} // namespace detail

#define TREE_FIND_VALUE(CMP)                                                                   \
  if (has_missing_tracks_) {                                                                   \
    while (root->is_not_leaf()) {                                                              \
      val = row.get(root->feature_id);                                                         \
      root += (val CMP root->value_or_unique_weight ||                                         \
               (root->is_missing_track_true() && _isnan_(val)))                                \
                  ? root->truenode_inc_or_first_weight                                         \
                  : root->falsenode_inc_or_n_weights;                                          \
    }                                                                                          \
  } else {                                                                                     \
    while (root->is_not_leaf()) {                                                              \
      val = row.get(root->feature_id);                                                         \
      root += val CMP root->value_or_unique_weight ? root->truenode_inc_or_first_weight        \
                                                   : root->falsenode_inc_or_n_weights;         \
    }                                                                                          \
  }

template <typename FeatureType, typename ThresholdType>
inline int GetLeave3IndexLEQ(typename FeatureType::ValueType *features,
                             const TreeNodeElement3<ThresholdType> *node3,
                             typename FeatureType::RowAccessor row) {
  features[0] = row.get(node3->feature_id[0]); // x_data[node3->feature_id[0]];
  features[1] = row.get(node3->feature_id[2]); // x_data[node3->feature_id[2]];
  features[2] = row.get(node3->feature_id[1]); // x_data[node3->feature_id[1]];
  // features[3] = x_data[node3->feature_id[2]];
  return node3->node_id[features[1] <= node3->thresholds[1]
                            ? (features[2] <= node3->thresholds[2] ? 3 : 2)
                            : (features[0] <= node3->thresholds[0] ? 1 : 0)];
}

#if 0
template <>
inline int GetLeave3IndexLEQ(typename FeatureType::ValueType* features, const TreeNodeElement3<float>* node3, typename FeatureType::RowAccessor& row) {
  features[0] = x_data[node3->feature_id[0]];
  features[1] = x_data[node3->feature_id[2]];
  features[2] = x_data[node3->feature_id[1]];
  features[3] = x_data[node3->feature_id[2]];

  // maybe we could align these pointers
  /*
  __m128 c1 = _mm_load_ps1(features);
  __m128 c2 = _mm_load_ps1(node3->thresholds);
  __m128 cmp = _mm_cmpgt_ps(c2, c1);  // does not work
  float out[4];
  _mm_store_ps1(out, cmp);
  __m128i res = _mm_castps_si128(cmp);
  int32_t iout[4];
  _mm_storeu_si128((__m128i*)iout, res);
  __m128i res2 = _mm_cmpeq_epi32(res, _mm_set1_epi32(-1));
  int32_t iout2[4];
  _mm_storeu_si128((__m128i*)iout2, res2);
  cmp = _mm_castsi128_ps(res2);
  */
  __m128i c1 = _mm_loadu_si128((const __m128i*)features);           // _mm_load_si128 aligned
  __m128i c2 = _mm_loadu_si128((const __m128i*)node3->thresholds);  // _mm_load_si128 aligned
  __m128i zero = _mm_set1_epi32(0);
  __m128i signc1 = _mm_cmplt_epi32(c1, zero);
  __m128i signc2 = _mm_cmplt_epi32(c2, zero);
  __m128i cmp_sign = _mm_cmplt_epi32(signc1, signc2);
  __m128i cmp = _mm_cmpgt_epi32(c2, c1);
  __m128i final = _mm_xor_si128(cmp, cmp_sign);

  uint32_t s1[4]={5, 6, 7, 8};
  _mm_storeu_si128((__m128i*)s1, signc1);
  uint32_t s2[4] = {51, 61, 71, 81};
  _mm_storeu_si128((__m128i*)s2, signc2);

  uint32_t iout[4] = {52, 62, 72, 82};
  _mm_storeu_si128((__m128i*)iout, cmp);
  uint32_t iout2[4] = {53, 63, 73, 83};
  _mm_storeu_si128((__m128i*)iout2, cmp_sign);
  uint32_t iout3[4] = {54, 64, 74, 84};
  _mm_storeu_si128((__m128i*)iout3, final);

  int ind = _mm_movemask_ps(_mm_castsi128_ps(final));
  int ind2 = (ind >> (ind & 2)) & 3;
  auto exp = features[1] <= node3->thresholds[1]
                 ? (features[2] <= node3->thresholds[2] ? 3 : 2)
                 : (features[0] <= node3->thresholds[0] ? 1 : 0);
  EXT_ENFORCE(exp == ind2, "\n--exp=", exp, " ind=", ind, " ind2=", ind2, " FF\n",
              features[0], "<=", node3->thresholds[0], " -- ",
              features[1], "<=", node3->thresholds[1], " -- ",
              features[2], "<=", node3->thresholds[2], " -- ",
              features[3], "<=", node3->thresholds[3], " --\n",
              s1[0], " # ", s1[1], " # ", s1[2], " # ", s1[3], "\n",
              s2[0], " # ", s2[1], " # ", s2[2], " # ", s2[3],
              "\n???? ", iout[0], ",", iout[1], ",", iout[2], ",", iout[3],
              "\n???? ", iout2[0], ",", iout2[1], ",", iout2[2], ",", iout2[3],
              "\n???? ", iout3[0], ",", iout3[1], ",", iout3[2], ",", iout3[3]);
  return node3->node_id[ind2];
}
#endif

template <typename FeatureType, typename ThresholdType, typename OutputType>
const TreeNodeElement<ThresholdType> *
TreeEnsembleCommon<FeatureType, ThresholdType, OutputType>::ProcessTreeNodeLeave3(
    std::size_t root_id, const typename FeatureType::RowAccessor &row) const {
  EXT_ENFORCE(same_mode_, "This optimization is only available when all node "
                          "follow the same mode.");
  const TreeNodeElement3<ThresholdType> *root3 = roots3_[root_id];
  const TreeNodeElement<ThresholdType> *root;
  EXT_ENFORCE(root3 != nullptr, "No optimization for tree ", (int64_t)root_id, ".");
  InputType features[4];
  int node_id;
  switch (root3->mode()) {
  case NODE_MODE::BRANCH_LEQ:
    if (has_missing_tracks_) {
      EXT_THROW("TreeNodeElement3 not yet implement with has_missing_tracks_.");
    } else {
      while (root3->children_are_tree_element3()) {
        node_id = GetLeave3IndexLEQ<FeatureType, ThresholdType>(features, root3, row);
        root3 = &(nodes3_[node_id]);
      }
      node_id = GetLeave3IndexLEQ<FeatureType, ThresholdType>(features, root3, row);
      root = &(nodes_[node_id]);
      while (root->is_not_leaf()) {
        root += row.get(root->feature_id) <= root->value_or_unique_weight
                    ? root->truenode_inc_or_first_weight
                    : root->falsenode_inc_or_n_weights;
      }
      return root;
    }
    break;
  default:
    EXT_THROW("TreeNodeElement3 not yet implement with mode ", (int64_t)root3->mode(), ".");
  }
}

template <typename FeatureType, typename ThresholdType, typename OutputType>
const TreeNodeElement<ThresholdType> *
TreeEnsembleCommon<FeatureType, ThresholdType, OutputType>::ProcessTreeNodeLeave(
    std::size_t root_id, const typename FeatureType::RowAccessor &row) const {
  if (!nodes3_.empty() && (roots3_[root_id] != nullptr)) {
    return ProcessTreeNodeLeave3(root_id, row);
  }
  DEBUG_INDEX(root_id, roots_.size(), "ERROR ProcessTreeNodeLeave root_id=", (int64_t)root_id,
              " roots_.size()=", (int64_t)roots_.size(), ".");
  const TreeNodeElement<ThresholdType> *root = roots_[root_id];
  InputType val;
  if (same_mode_) {
    switch (root->mode()) {
    case NODE_MODE::BRANCH_LEQ:
      if (has_missing_tracks_) {
        DEBUG_PRINT("LEQ1")
        while (root->is_not_leaf()) {
          val = row.get(root->feature_id);
          root += (val <= root->value_or_unique_weight ||
                   (root->is_missing_track_true() && _isnan_(val)))
                      ? root->truenode_inc_or_first_weight
                      : root->falsenode_inc_or_n_weights;
        }
      } else {
        DEBUG_PRINT("LEQ2")
        while (root->is_not_leaf()) {
          val = row.get(root->feature_id);
          root += val <= root->value_or_unique_weight ? root->truenode_inc_or_first_weight
                                                      : root->falsenode_inc_or_n_weights;
        }
      }
      DEBUG_PRINT("LEQ3")
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
  } else { // Different rules to compare to node thresholds.
    ThresholdType threshold;
    while (root->is_not_leaf()) {
      val = row.get(root->feature_id);
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

} // namespace onnx_c_ops
