// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef ENABLE_TRITON

#include <nlohmann/json.hpp>
#include <string_view>

#include "orttraining/core/optimizer/triton_fusion.h"

#include "core/framework/compute_capability.h"
#include "core/graph/model.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"
#include "core/providers/partitioning_utils.h"

using namespace ONNX_NAMESPACE;
using json = nlohmann::json;

namespace onnxruntime {

namespace {

using SizeTypeVec = InlinedVector<size_t>;
using NodeVec = InlinedVector<Node*>;
using NodeArgVec = InlinedVector<NodeArg*>;
using ConstNodeArgVec = InlinedVector<const NodeArg*>;
using NodeArgSet = InlinedHashSet<NodeArg*>;
using IsSupportedFunc = std::function<bool(const Graph&, const Node&)>;

int64_t Hash(const std::string& str) {
  uint32_t hash = 0;
  for (char const& c : str) {
    hash = hash * 101 + c;
  }

  return static_cast<int64_t>(hash);
}

bool CheckAxis(const Node& node, int64_t expected_axis) {
  const auto& attributes = node.GetAttributes();
  if (attributes.find("axis") == attributes.end() || !attributes.at("axis").has_i()) return false;
  int64_t axis = attributes.at("axis").i();
  if (axis == expected_axis) return true;
  if (axis < 0 || expected_axis < 0) {
    const auto& input_shape = node.InputDefs()[0]->Shape();
    if (input_shape == nullptr) return false;
    int64_t rank = static_cast<int64_t>(input_shape->dim_size());
    if (axis < 0) axis += rank;
    int64_t non_neg_expected_axis = expected_axis < 0 ? expected_axis + rank : expected_axis;
    return axis == non_neg_expected_axis;
  }
  return false;
}

bool CheckAxes(const Graph& graph, const Node& node, bool single_axis, const std::vector<int>& expected_axes) {
  std::vector<int64_t> axes_values;
  const auto& attributes = node.GetAttributes();
  if (attributes.find("axes") != attributes.end()) {
    axes_values = RetrieveValues<int64_t>(attributes.at("axes"));
  } else if (node.InputDefs().size() == 2) {
    auto axes_const = graph.GetConstantInitializer(node.InputDefs()[1]->Name(), true);
    if (!axes_const) {
      return false;
    }
    Initializer initializer{*axes_const, graph.ModelPath()};
    axes_values.insert(axes_values.end(), initializer.DataAsSpan<int64_t>().begin(),
                       initializer.DataAsSpan<int64_t>().end());
  } else {
    return false;
  }

  if (expected_axes.empty()) {
    return !single_axis || axes_values.size() == 1;
  }

  std::vector<int64_t> expected_axes_values(expected_axes.begin(), expected_axes.end());
  bool has_negative_axis = false;
  for (auto axis : axes_values) {
    if (axis < 0) {
      has_negative_axis = true;
      break;
    }
  }
  if (!has_negative_axis) {
    for (auto axis : expected_axes_values) {
      if (axis < 0) {
        has_negative_axis = true;
        break;
      }
    }
  }
  if (has_negative_axis) {
    const auto& input_shape = node.InputDefs()[0]->Shape();
    if (input_shape == nullptr) return false;
    int64_t rank = static_cast<int64_t>(input_shape->dim_size());
    for (auto& axis : axes_values) {
      if (axis < 0) axis += rank;
    }
    for (auto& axis : expected_axes_values) {
      if (axis < 0) axis += rank;
    }
  }
  std::sort(axes_values.begin(), axes_values.end());
  std::sort(expected_axes_values.begin(), expected_axes_values.end());
  return axes_values == expected_axes_values;
}

// A TritonOpPartition contains all connected nodes in ONNX graph which are supported by ORTModule's Triton module.
// When building the TritonOpPartition, we keep all the dependencies and output ref counts so that to make sure
// there is no dependency between two nodes in the partition through any node outside the partition.
struct TritonOpPartition {
  NodeVec nodes;
  NodeArgSet outputs;
  NodeArgSet dependencies;
  size_t output_ref_count;

  void MergeFrom(const TritonOpPartition& other) {
    nodes.insert(nodes.end(), other.nodes.begin(), other.nodes.end());
    outputs.insert(other.outputs.begin(), other.outputs.end());
    dependencies.insert(other.dependencies.begin(), other.dependencies.end());
    output_ref_count += other.output_ref_count;
  }

  bool IsValid(const TritonFusionConfig& config) const {
    size_t count = 0;
    bool all_ignore_min_nodes = true;
    for (const auto& node : nodes) {
      if (!config.IsNoOp(*node)) {
        ++count;
        if (count >= config.min_nodes) return true;
      }
      if (!config.IgnoreMinNodes(*node)) all_ignore_min_nodes = false;
    }
    return all_ignore_min_nodes;
  }
};

}  // namespace

void from_json(const json& j, TritonFusionConfig::OpInfo& op_info) {
  if (j.contains("domain")) j.at("domain").get_to(op_info.domain);
  j.at("versions").get_to(op_info.versions);
  if (j.contains("is_no_op")) j.at("is_no_op").get_to(op_info.is_no_op);
  if (j.contains("conditions")) j.at("conditions").get_to(op_info.conditions);
  if (j.contains("ignore_min_nodes")) j.at("ignore_min_nodes").get_to(op_info.ignore_min_nodes);
}

TritonFusionConfig::TritonFusionConfig(std::string_view config_json) {
  const auto& config = json::parse(config_json);
  if (config.contains("ops")) {
    ops = config.at("ops").get<std::unordered_map<std::string, OpInfo>>();
  }
  if (config.contains("initializer")) {
    initializer = config.at("initializer").get<std::string>();
  }
  if (config.contains("min_nodes")) {
    min_nodes = static_cast<size_t>(config.at("min_nodes").get<int>());
  }
}

bool TritonFusionConfig::IsSupported(const Graph& graph, const Node& node) const {
  const auto& op_type = node.OpType();
  auto it = ops.find(op_type);
  if (it == ops.end()) {
    return false;
  }

  const auto& op_info = it->second;
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, op_type, op_info.versions, op_info.domain)) {
    return false;
  }

  for (const auto& pair : op_info.conditions) {
    if (pair.first == "axis") {
      if (!CheckAxis(node, static_cast<int64_t>(std::stoi(pair.second)))) {
        return false;
      }
    } else if (pair.first == "axes") {
      if (pair.second == "constant") {
        if (!CheckAxes(graph, node, false, {})) {
          return false;
        }
      } else if (pair.second == "single") {
        if (!CheckAxes(graph, node, true, {})) {
          return false;
        }
      } else {
        const auto& axes = json::parse(pair.second);
        std::vector<int> axes_values = axes.get<std::vector<int>>();
        if (!CheckAxes(graph, node, false, axes_values)) {
          return false;
        }
      }
    } else {
      return false;
    }
  }

  return true;
}

bool TritonFusionConfig::IsNoOp(const Node& node) const {
  const auto& op_type = node.OpType();
  return ops.find(op_type) != ops.end() && ops.at(op_type).is_no_op;
}

bool TritonFusionConfig::IgnoreMinNodes(const Node& node) const {
  const auto& op_type = node.OpType();
  return ops.find(op_type) != ops.end() && ops.at(op_type).ignore_min_nodes;
}

const ONNX_NAMESPACE::TensorProto* TritonFusionConfig::TryGetInitializer(const Graph& graph, const Node& node,
                                                                         NodeArg* node_arg) const {
  if (initializer == "none" || !graph_utils::IsInitializer(graph, node_arg->Name(), true)) {
    return nullptr;
  }

  const ONNX_NAMESPACE::TensorProto* tensor = nullptr;
  if (!graph.GetInitializedTensor(node_arg->Name(), tensor) || !tensor) {
    return nullptr;
  }

  if (initializer == "all") {
    return tensor;
  }

  if ((node.OpType() == "ReduceSum" || node.OpType() == "ReduceMean" || node.OpType() == "ReduceMax" ||
       node.OpType() == "ReduceMin") &&
      node.InputDefs().size() >= 2 and node.InputDefs()[1]->Name() == node_arg->Name()) {
    return tensor;
  }

  return optimizer_utils::IsScalar(*node_arg) ? tensor : nullptr;
}

Status TritonFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  size_t global_id = 0;
  InlinedHashMap<size_t, TritonOpPartition> partitions;
  InlinedHashMap<size_t, TritonOpPartition> partitions_to_fuse;
  InlinedHashMap<NodeArg*, size_t> active_outputs;
  for (auto node_index : node_topology_list) {
    auto* p_node = graph.GetNode(node_index);
    if (!p_node) continue;
    auto& node = *p_node;
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));
    bool is_supported =
        graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders()) && config_.IsSupported(graph, node);
    SizeTypeVec partitions_to_merge;
    for (auto& pair : partitions) {
      auto& partition = pair.second;
      bool connect_to_output = false;
      bool connect_to_dependency = false;
      for (auto& input : node.MutableInputDefs()) {
        if (partition.outputs.find(input) != partition.outputs.end()) {
          partition.output_ref_count--;
          connect_to_output = true;
        }
        if (partition.dependencies.find(input) != partition.dependencies.end()) {
          connect_to_dependency = true;
        }
      }
      if (is_supported && connect_to_output && !connect_to_dependency) {
        partitions_to_merge.emplace_back(pair.first);
      } else if (connect_to_output || connect_to_dependency) {
        for (auto& output : node.MutableOutputDefs()) {
          partition.dependencies.emplace(output);
        }
      }
    }

    if (!partitions_to_merge.empty()) {
      std::sort(partitions_to_merge.begin(), partitions_to_merge.end());
      TritonOpPartition& dst = partitions.at(partitions_to_merge[0]);
      for (size_t i = partitions_to_merge.size() - 1; i > 0; --i) {
        dst.MergeFrom(partitions.at(partitions_to_merge[i]));
        partitions.erase(partitions_to_merge[i]);
      }

      dst.nodes.emplace_back(&node);
      for (auto& output : node.MutableOutputDefs()) {
        dst.outputs.emplace(output);
      }
      dst.output_ref_count += node.GetOutputEdgesCount();
    } else if (is_supported) {
      TritonOpPartition partition;
      partition.nodes.emplace_back(&node);
      for (auto& node_def : node.MutableOutputDefs()) {
        partition.outputs.emplace(node_def);
      }
      partition.output_ref_count = node.GetOutputEdgesCount();
      partitions.emplace(global_id++, partition);
    }

    SizeTypeVec partitions_to_erase;
    for (auto& pair : partitions) {
      if (pair.second.output_ref_count == 0) {
        if (pair.second.IsValid(config_)) {
          pair.second.outputs.clear();
          pair.second.dependencies.clear();
          partitions_to_fuse.emplace(pair);
        }
        partitions_to_erase.emplace_back(pair.first);
      }
    }

    for (auto& id : partitions_to_erase) {
      partitions.erase(id);
    }

    for (auto& input : node.MutableInputDefs()) {
      if (active_outputs.find(input) != active_outputs.end()) {
        active_outputs.at(input)--;
        if (active_outputs.at(input) == 0) {
          active_outputs.erase(input);
          for (auto& pair : partitions) {
            pair.second.dependencies.erase(input);
          }
        }
      }
    }

    for (auto it = node.OutputEdgesBegin(), end = node.OutputEdgesEnd(); it != end; ++it) {
      NodeArg* output = node.MutableOutputDefs()[it->GetSrcArgIndex()];
      if (active_outputs.find(output) == active_outputs.end()) {
        active_outputs.emplace(output, 1);
      } else {
        active_outputs.at(output)++;
      }
    }
  }

  SizeTypeVec partition_ids;
  for (auto& pair : partitions_to_fuse) {
    partition_ids.emplace_back(pair.first);
  }
  std::sort(partition_ids.begin(), partition_ids.end());

  for (auto& id : partition_ids) {
    auto& partition = partitions_to_fuse.at(id);

    Model sub_model("test", false, logger);
    Graph& sub_graph = sub_model.MainGraph();

    NodeArgVec graph_inputs;
    NodeArgVec node_outputs;
    NodeArgSet graph_input_set;
    NodeArgSet initializers;
    ConstNodeArgVec graph_const_inputs;
    InlinedHashMap<NodeArg*, size_t> output_consumer_counts;
    NodeArgSet no_consumer_outputs;
    for (auto& p_node : partition.nodes) {
      auto& node = *p_node;
      sub_graph.AddNode(node);
      for (auto& input : node.MutableInputDefs()) {
        if (initializers.find(input) != initializers.end()) {
          continue;
        }
        const ONNX_NAMESPACE::TensorProto* tensor = config_.TryGetInitializer(graph, node, input);
        if (tensor) {
          initializers.emplace(input);
          sub_graph.AddInitializedTensor(*tensor);
          continue;
        }

        if (output_consumer_counts.find(input) != output_consumer_counts.end()) {
          output_consumer_counts.at(input)--;
          if (output_consumer_counts.at(input) == 0) {
            output_consumer_counts.erase(input);
          }
        } else if (graph_input_set.find(input) == graph_input_set.end()) {
          graph_inputs.emplace_back(input);
          graph_input_set.insert(input);
          graph_const_inputs.emplace_back(input);
        }
      }

      for (auto it = p_node->OutputEdgesBegin(), end = p_node->OutputEdgesEnd(); it != end; ++it) {
        NodeArg* output = p_node->MutableOutputDefs()[it->GetSrcArgIndex()];
        if (output_consumer_counts.find(output) == output_consumer_counts.end()) {
          output_consumer_counts.emplace(output, 1);
        } else {
          output_consumer_counts.at(output)++;
        }
      }

      for (auto& output : node.MutableOutputDefs()) {
        if (output->Name() != "") {
          node_outputs.emplace_back(output);
          if (output_consumer_counts.find(output) == output_consumer_counts.end()) {
            no_consumer_outputs.emplace(output);
          }
        }
      }
    }

    NodeArgVec graph_outputs;
    ConstNodeArgVec graph_const_outputs;
    for (auto& output : node_outputs) {
      if (no_consumer_outputs.find(output) != no_consumer_outputs.end() ||
          output_consumer_counts.find(output) != output_consumer_counts.end()) {
        graph_outputs.emplace_back(output);
        graph_const_outputs.emplace_back(output);
      }
    }

    sub_graph.SetInputs(graph_const_inputs);
    sub_graph.SetOutputs(graph_const_outputs);

    auto model_proto = sub_model.ToProto();
    std::string model_str;
    model_proto.SerializeToString(&model_str);

    Node& fused_node = graph.AddNode(graph.GenerateNodeName("TritonOp"), "TritonOp", "Fused nodes for TritonOp",
                                     graph_inputs, graph_outputs, {}, kMSDomain);
    fused_node.AddAttribute("onnx_key", Hash(model_str));
    fused_node.AddAttribute("onnx_string", model_str);
    fused_node.SetExecutionProviderType(partition.nodes[0]->GetExecutionProviderType());

    for (auto& p_node : partition.nodes) {
      graph_utils::RemoveNodeOutputEdges(graph, *p_node);
      graph.RemoveNode(p_node->Index());
    }

    modified = true;
  }

  return Status::OK();
}

}  // namespace onnxruntime

#endif  // ENABLE_TRITON
