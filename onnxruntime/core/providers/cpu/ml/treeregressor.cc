// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/ml/treeregressor.h"

namespace onnxruntime {
namespace ml {

ONNX_CPU_OPERATOR_ML_KERNEL(
    TreeEnsembleRegressor,
    1,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()).MayInplace(0, 0),
    TreeEnsembleRegressor<float>);

template <typename T>
TreeEnsembleRegressor<T>::TreeEnsembleRegressor(const OpKernelInfo& info)
    : OpKernel(info),
      nodes_treeids_(info.GetAttrsOrDefault<int64_t>("nodes_treeids")),
      nodes_nodeids_(info.GetAttrsOrDefault<int64_t>("nodes_nodeids")),
      nodes_featureids_(info.GetAttrsOrDefault<int64_t>("nodes_featureids")),
      nodes_values_(info.GetAttrsOrDefault<float>("nodes_values")),
      nodes_hitrates_(info.GetAttrsOrDefault<float>("nodes_hitrates")),
      nodes_truenodeids_(info.GetAttrsOrDefault<int64_t>("nodes_truenodeids")),
      nodes_falsenodeids_(info.GetAttrsOrDefault<int64_t>("nodes_falsenodeids")),
      missing_tracks_true_(info.GetAttrsOrDefault<int64_t>("nodes_missing_value_tracks_true")),
      target_nodeids_(info.GetAttrsOrDefault<int64_t>("target_nodeids")),
      target_treeids_(info.GetAttrsOrDefault<int64_t>("target_treeids")),
      target_ids_(info.GetAttrsOrDefault<int64_t>("target_ids")),
      target_weights_(info.GetAttrsOrDefault<float>("target_weights")),
      base_values_(info.GetAttrsOrDefault<float>("base_values")),
      transform_(::onnxruntime::ml::MakeTransform(info.GetAttrOrDefault<std::string>("post_transform", "NONE"))),
      aggregate_function_(::onnxruntime::ml::MakeAggregateFunction(info.GetAttrOrDefault<std::string>("aggregate_function", "SUM"))) {
  ORT_ENFORCE(info.GetAttr<int64_t>("n_targets", &n_targets_).IsOK());

  //update nodeids to start at 0
  ORT_ENFORCE(!nodes_treeids_.empty());
  int64_t current_tree_id = 1234567891L;
  std::vector<int64_t> tree_offsets;

  for (size_t i = 0; i < nodes_treeids_.size(); i++) {
    if (nodes_treeids_[i] != current_tree_id) {
      tree_offsets.push_back(nodes_nodeids_[i]);
      current_tree_id = nodes_treeids_[i];
    }
    int64_t offset = tree_offsets[tree_offsets.size() - 1];
    nodes_nodeids_[i] = nodes_nodeids_[i] - offset;
    if (nodes_falsenodeids_[i] >= 0) {
      nodes_falsenodeids_[i] = nodes_falsenodeids_[i] - offset;
    }
    if (nodes_truenodeids_[i] >= 0) {
      nodes_truenodeids_[i] = nodes_truenodeids_[i] - offset;
    }
  }
  for (size_t i = 0; i < target_nodeids_.size(); i++) {
    int64_t offset = tree_offsets[target_treeids_[i]];
    target_nodeids_[i] = target_nodeids_[i] - offset;
  }

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

  max_tree_depth_ = 1000;
  offset_ = four_billion_;
  //leafnode data, these are the votes that leaves do
  for (size_t i = 0; i < target_nodeids_.size(); i++) {
    leafnode_data_.push_back(std::make_tuple(target_treeids_[i], target_nodeids_[i], target_ids_[i], target_weights_[i]));
  }
  std::sort(begin(leafnode_data_), end(leafnode_data_), [](auto const& t1, auto const& t2) {
    if (std::get<0>(t1) != std::get<0>(t2))
      return std::get<0>(t1) < std::get<0>(t2);

    return std::get<1>(t1) < std::get<1>(t2);
  });
  //make an index so we can find the leafnode data quickly when evaluating
  int64_t field0 = -1;
  int64_t field1 = -1;
  for (size_t i = 0; i < leafnode_data_.size(); i++) {
    int64_t id0 = std::get<0>(leafnode_data_[i]);
    int64_t id1 = std::get<1>(leafnode_data_[i]);
    if (id0 != field0 || id1 != field1) {
      int64_t id = id0 * four_billion_ + id1;
      auto p3 = std::make_pair(id, i);  // position is i
      leafdata_map_.insert(p3);
      field0 = id;
      field1 = static_cast<int64_t>(i);
    }
  }
  //treenode ids, some are roots, and roots have no parents
  std::unordered_map<int64_t, size_t> parents;  //holds count of all who point to you
  std::unordered_map<int64_t, size_t> indices;
  //add all the nodes to a map, and the ones that have parents are not roots
  std::unordered_map<int64_t, size_t>::iterator it;
  size_t start_counter = 0L;
  for (size_t i = 0; i < nodes_treeids_.size(); i++) {
    //make an index to look up later
    int64_t id = nodes_treeids_[i] * four_billion_ + nodes_nodeids_[i];
    auto p3 = std::make_pair(id, i);  // i is the position
    indices.insert(p3);
    it = parents.find(id);
    if (it == parents.end()) {
      //start counter at 0
      auto p1 = std::make_pair(id, start_counter);
      parents.insert(p1);
    }
  }
  //all true nodes aren't roots
  for (size_t i = 0; i < nodes_truenodeids_.size(); i++) {
    if (nodes_modes_[i] == ::onnxruntime::ml::NODE_MODE::LEAF) continue;
    //they must be in the same tree
    int64_t id = nodes_treeids_[i] * offset_ + nodes_truenodeids_[i];
    it = parents.find(id);
    ORT_ENFORCE(it != parents.end());
    it->second++;
  }
  //all false nodes aren't roots
  for (size_t i = 0; i < nodes_falsenodeids_.size(); i++) {
    if (nodes_modes_[i] == ::onnxruntime::ml::NODE_MODE::LEAF) continue;
    //they must be in the same tree
    int64_t id = nodes_treeids_[i] * offset_ + nodes_falsenodeids_[i];
    it = parents.find(id);
    ORT_ENFORCE(it != parents.end());
    it->second++;
  }
  //find all the nodes that dont have other nodes pointing at them
  for (auto& parent : parents) {
    if (parent.second == 0) {
      int64_t id = parent.first;
      it = indices.find(id);
      ORT_ENFORCE(it != indices.end());
      roots_.push_back(it->second);
    }
  }
  ORT_ENFORCE(base_values_.empty() || base_values_.size() == static_cast<size_t>(n_targets_));
}

template <typename T>
common::Status TreeEnsembleRegressor<T>::ProcessTreeNode(std::unordered_map < int64_t, std::tuple<float, float, float>>& classes, int64_t treeindex, const T* Xdata, int64_t feature_base) const {
  //walk down tree to the leaf
  auto mode = static_cast<::onnxruntime::ml::NODE_MODE>(nodes_modes_[treeindex]);
  int64_t loopcount = 0;
  int64_t root = treeindex;
  while (mode != ::onnxruntime::ml::NODE_MODE::LEAF) {
    T val = Xdata[feature_base + nodes_featureids_[treeindex]];
    bool tracktrue = true;
    if (missing_tracks_true_.size() != nodes_truenodeids_.size()) {
      tracktrue = false;
    } else {
      tracktrue = (missing_tracks_true_[treeindex] != 0) && std::isnan(val);
    }
    float threshold = nodes_values_[treeindex];
    if (mode == ::onnxruntime::ml::NODE_MODE::BRANCH_LEQ) {
      treeindex = val <= threshold || tracktrue ? nodes_truenodeids_[treeindex] : nodes_falsenodeids_[treeindex];
    } else if (mode == ::onnxruntime::ml::NODE_MODE::BRANCH_LT) {
      treeindex = val < threshold || tracktrue ? nodes_truenodeids_[treeindex] : nodes_falsenodeids_[treeindex];
    } else if (mode == ::onnxruntime::ml::NODE_MODE::BRANCH_GTE) {
      treeindex = val >= threshold || tracktrue ? nodes_truenodeids_[treeindex] : nodes_falsenodeids_[treeindex];
    } else if (mode == ::onnxruntime::ml::NODE_MODE::BRANCH_GT) {
      treeindex = val > threshold || tracktrue ? nodes_truenodeids_[treeindex] : nodes_falsenodeids_[treeindex];
    } else if (mode == ::onnxruntime::ml::NODE_MODE::BRANCH_EQ) {
      treeindex = val == threshold || tracktrue ? nodes_truenodeids_[treeindex] : nodes_falsenodeids_[treeindex];
    } else if (mode == ::onnxruntime::ml::NODE_MODE::BRANCH_NEQ) {
      treeindex = val != threshold || tracktrue ? nodes_truenodeids_[treeindex] : nodes_falsenodeids_[treeindex];
    }

    if (treeindex < 0) {
      return common::Status(common::ONNXRUNTIME, common::RUNTIME_EXCEPTION,
                            "treeindex evaluated to a negative value, which should not happen.");
    }
    treeindex = treeindex + root;
    mode = (::onnxruntime::ml::NODE_MODE)nodes_modes_[treeindex];
    loopcount++;
    if (loopcount > max_tree_depth_) break;
  }
  //should be at leaf
  int64_t id = nodes_treeids_[treeindex] * four_billion_ + nodes_nodeids_[treeindex];
  //auto it_lp = leafdata_map.find(id);
  auto it_lp = leafdata_map_.find(id);
  if (it_lp != leafdata_map_.end()) {
    size_t index = it_lp->second;
    int64_t treeid = std::get<0>(leafnode_data_[index]);
    int64_t nodeid = std::get<1>(leafnode_data_[index]);
    while (treeid == nodes_treeids_[treeindex] && nodeid == nodes_nodeids_[treeindex]) {
      int64_t classid = std::get<2>(leafnode_data_[index]);
      float weight = std::get<3>(leafnode_data_[index]);
      auto it_classes = classes.find(classid);
      if (it_classes != classes.end()) {
        auto& tuple = it_classes->second;
        std::get<0>(tuple) += weight;
        if (weight < std::get<1>(tuple)) std::get<1>(tuple) = weight;
        if (weight > std::get<2>(tuple)) std::get<2>(tuple) = weight;
      } else {
        std::tuple<float, float, float> tuple = std::make_tuple(weight, weight, weight);
        auto p1 = std::make_pair(classid, tuple);
        classes.insert(p1);
      }
      index++;
      if (index >= leafnode_data_.size()) {
        break;
      }
      treeid = std::get<0>(leafnode_data_[index]);
      nodeid = std::get<1>(leafnode_data_[index]);
    }
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

  int64_t write_index = 0;
  const auto* x_data = X->template Data<T>();

  for (int64_t i = 0; i < N; i++)  //for each class
  {
    int64_t current_weight_0 = i * stride;
    std::unordered_map<int64_t, std::tuple<float, float, float>> scores; // sum, min, max
    //for each tree
    for (size_t j = 0; j < roots_.size(); j++) {
      //walk each tree from its root
      ORT_RETURN_IF_ERROR(ProcessTreeNode(scores, roots_[j], x_data, current_weight_0));
    }
    //find aggregate, could use a heap here if there are many classes
    std::vector<float> outputs;
    for (int64_t j = 0; j < n_targets_; j++) {
      //reweight scores based on number of voters
      auto it_scores = scores.find(j);
      float val = base_values_.size() == (size_t)n_targets_ ? base_values_[j] : 0.f;
      if (it_scores != scores.end()) {
        if (aggregate_function_ == ::onnxruntime::ml::AGGREGATE_FUNCTION::AVERAGE) {
          val += std::get<0>(scores[j]) / roots_.size();   //first element of tuple is already a sum
        } else if (aggregate_function_ == ::onnxruntime::ml::AGGREGATE_FUNCTION::SUM) {
          val += std::get<0>(scores[j]);
        } else if (aggregate_function_ == ::onnxruntime::ml::AGGREGATE_FUNCTION::MIN) {
          val += std::get<1>(scores[j]);  // second element of tuple is min
        } else if (aggregate_function_ == ::onnxruntime::ml::AGGREGATE_FUNCTION::MAX) {
          val += std::get<2>(scores[j]);  // third element of tuple is max
        }
      }
      outputs.push_back(val);
    }
    write_scores(outputs, transform_, write_index, Y, -1);
    write_index += scores.size();
  }
  return Status::OK();
}

}  // namespace ml
}  // namespace onnxruntime
