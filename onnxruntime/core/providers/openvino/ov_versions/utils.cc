// Copyright (C) 2019-2022 Intel Corporation
// Licensed under the MIT License

#include "core/providers/shared_library/provider_api.h"

#if defined(_MSC_VER)
#pragma warning(disable : 4244 4245 5208)
#elif __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#include "openvino/core/deprecated.hpp"
#define IN_OV_COMPONENT
#define NGRAPH_LEGACY_HEADER_INCLUDED
#include <ngraph/frontend/onnx_import/onnx.hpp>

#undef NGRAPH_LEGACY_HEADER_INCLUDED
#undef IN_OV_COMPONENT

#if defined(_MSC_VER)
#pragma warning(default : 4244 4245)
#elif __GNUC__
#pragma GCC diagnostic pop
#endif

namespace onnxruntime {
namespace openvino_ep {

// Gets the input count of given node
int GetInputCount(const Node* node, const InitializedTensorSet& initializer_set) {
  int count = 0;
  for (const auto& input : node->InputDefs()) {
    auto name = input->Name();
    auto it = initializer_set.find(name);
    if (it == initializer_set.end()) {
      count++;
    }
  }
  return count;
}

// Ops which are supported only in models(as intermediate nodes) and not in unit tests
bool IsOpSupportedOnlyInModel(std::string name) {
  std::set<std::string> ops_supported_only_in_model = {
      "Cast",
      "Concat",
      "ConstantOfShape",
      "Dropout",
      "Einsum",
      "Expand",
      "EyeLike",
      "Exp",
      "GatherND",
      "Identity",
      "LayerNormalization",
      "NonMaxSuppression",
      "NonZero",
      "Not",
      "OneHot",
      "Pad",
      "Range",
      "ReduceMin",
      "Resize",
      "Round",
      "Shape",
      "Split",
      "TopK"};
  return ops_supported_only_in_model.find(name) != ops_supported_only_in_model.end();
}

void AppendClusterToSubGraph(const std::vector<NodeIndex>& nodes,
                             const std::vector<std::string>& inputs,
                             const std::vector<std::string>& outputs,
                             std::vector<std::unique_ptr<ComputeCapability>>& result) {
  static size_t op_counter = 0;

  auto meta_def = IndexedSubGraph_MetaDef::Create();
  meta_def->name() = "OpenVINO-EP-subgraph_" + std::to_string(++op_counter);
  meta_def->domain() = kNGraphDomain;
  meta_def->since_version() = 1;
  meta_def->status() = ONNX_NAMESPACE::EXPERIMENTAL;
  meta_def->inputs() = inputs;
  meta_def->outputs() = outputs;

  auto sub_graph = IndexedSubGraph::Create();
  sub_graph->Nodes() = nodes;
  sub_graph->SetMetaDef(std::move(meta_def));
  result.push_back(ComputeCapability::Create(std::move(sub_graph)));
}

int GetOnnxOpSet(const GraphViewer& graph_viewer) {
  const auto& dm_to_ver = graph_viewer.DomainToVersionMap();
  return dm_to_ver.at(kOnnxDomain);
}

std::map<std::string, std::set<std::string>> GetNgSupportedOps(const int onnx_opset) {
  std::map<std::string, std::set<std::string>> ng_supported_ops;
  OPENVINO_SUPPRESS_DEPRECATED_START
  ng_supported_ops.emplace(kOnnxDomain, ngraph::onnx_import::get_supported_operators(onnx_opset, kOnnxDomain));

  const std::set<std::string> ng_disabled_ops = {"LSTM"};  // Place-holder for ops not supported.

  for (const auto& disabled_op : ng_disabled_ops) {
    ng_supported_ops.at(kOnnxDomain).erase(disabled_op);
  }
  OPENVINO_SUPPRESS_DEPRECATED_END
  return ng_supported_ops;
}

/**
 * Returns a vector clusters(or node_idx). For each unsupported node, the graph is split into 3 parts.
 * supported_cluster + (UNsupported_node + rest_of_the_graph). This functions returns vector of all supported_clusters by nGraph
 */
std::vector<std::vector<NodeIndex>>
GetPartitionedClusters(const std::vector<NodeIndex>& topological_order, const std::vector<NodeIndex>& unsupported_nodes) {
  std::vector<std::vector<NodeIndex>> ng_clusters;

  auto prev = topological_order.begin();

  for (const auto& unsup_node : unsupported_nodes) {
    auto it = std::find(prev, topological_order.end(), unsup_node);
    // Create a cluster vector[supported_node_idx, unsupported_node_idx) and append it to return list.
    std::vector<NodeIndex> this_cluster{prev, it};
    if (!this_cluster.empty()) {
      ng_clusters.push_back(std::move(this_cluster));
    }
    if (it != topological_order.end()) {
      // Point prev to node idx past this unsuported node.
      prev = ++it;
    }
  }

  // Tail
  std::vector<NodeIndex> this_cluster{prev, topological_order.end()};
  if (!this_cluster.empty()) {
    ng_clusters.push_back(std::move(this_cluster));
  }

  return ng_clusters;
}

void IdentifyConnectedNodes(const GraphViewer& graph_viewer, NodeIndex curr_node_index, std::vector<NodeIndex>& cluster, std::vector<NodeIndex>& sub_cluster) {
  if (std::find(cluster.begin(), cluster.end(), curr_node_index) == cluster.end())
    return;

  sub_cluster.emplace_back(curr_node_index);
  cluster.erase(std::remove(cluster.begin(), cluster.end(), curr_node_index), cluster.end());
  auto curr_node = graph_viewer.GetNode(curr_node_index);

  for (auto node = curr_node->InputNodesBegin(); node != curr_node->InputNodesEnd(); ++node) {
    IdentifyConnectedNodes(graph_viewer, (*node).Index(), cluster, sub_cluster);
  }
  for (auto node = curr_node->OutputNodesBegin(); node != curr_node->OutputNodesEnd(); ++node) {
    IdentifyConnectedNodes(graph_viewer, (*node).Index(), cluster, sub_cluster);
  }
}

std::vector<std::vector<NodeIndex>>
GetConnectedClusters(const GraphViewer& graph_viewer, const std::vector<std::vector<NodeIndex>>& clusters) {
  std::vector<std::vector<NodeIndex>> connected_clusters;

  for (auto this_cluster : clusters) {
    while (this_cluster.size() > 0) {
      std::vector<NodeIndex> sub_cluster;
      IdentifyConnectedNodes(graph_viewer, this_cluster[0], this_cluster, sub_cluster);
      connected_clusters.emplace_back(sub_cluster);
    }
  }
  return connected_clusters;
}

void GetInputsOutputsOfCluster(const GraphViewer& graph_viewer,
                               const std::vector<NodeIndex>& cluster,
                               const std::unordered_set<std::string>& ng_required_initializers,
                               /*out*/ std::vector<std::string>& cluster_graph_inputs,
                               /*out*/ std::vector<std::string>& cluster_inputs,
                               /*out*/ std::vector<std::string>& constant_inputs,
                               /*out*/ std::vector<std::string>& cluster_outputs) {
  std::unordered_set<std::string> input_args;
  std::vector<std::string> ordered_input_args;
  std::unordered_set<std::string> output_args;
  std::unordered_set<std::string> external_output_args;

  for (const auto& node_idx : cluster) {
    const auto& node = graph_viewer.GetNode(node_idx);
    // Collect all inputs and outputs
    node->ForEachDef(
        [&input_args, &ordered_input_args, &output_args](const NodeArg& node_arg, bool is_input) {
          if (node_arg.Name() != "") {
            if (is_input) {
              if (!input_args.count(node_arg.Name())) {
                ordered_input_args.push_back(node_arg.Name());
              }
              input_args.insert(node_arg.Name());
            } else {
              output_args.insert(node_arg.Name());
            }
          }
        },
        true);

    // Check if output of this node is used by nodes outside this_cluster. If yes add this to cluster outputs
    for (auto it = node->OutputNodesBegin(); it != node->OutputNodesEnd(); ++it) {
      const auto& ext_node = graph_viewer.GetNode((*it).Index());

      if (std::find(cluster.begin(), cluster.end(), ext_node->Index()) == cluster.end()) {
        // Node is external to this_cluster. Search through its inputs to find the output that is generated by this_cluster.
        std::set<std::string> ext_node_inputs;
        ext_node->ForEachDef(
            [&ext_node_inputs](const NodeArg& arg, bool is_input) {
              if (is_input) {
                ext_node_inputs.insert(arg.Name());
              }
            },
            true);

        for (const auto& out_def : node->OutputDefs()) {
          if (ext_node_inputs.find(out_def->Name()) != ext_node_inputs.end()) {
            external_output_args.insert(out_def->Name());
          }
        }
      }
    }
  }

  // Extract initializers used by this_cluster.
  std::unordered_set<std::string> original_graph_inputs;
  for (const auto& node_arg : graph_viewer.GetInputsIncludingInitializers()) {
    original_graph_inputs.insert(node_arg->Name());
  }

  const auto& initializers = graph_viewer.GetAllInitializedTensors();
  for (const auto& in_arg : ordered_input_args) {
    if ((initializers.count(in_arg) && !original_graph_inputs.count(in_arg)) ||
        ng_required_initializers.count(in_arg)) {
      constant_inputs.push_back(in_arg);
    }
  }

  for (const auto& in_arg : ordered_input_args) {
    if (!output_args.count(in_arg) &&
        !((initializers.count(in_arg) && !original_graph_inputs.count(in_arg)) ||
          ng_required_initializers.count(in_arg))) {
      cluster_inputs.push_back(in_arg);
    }
  }
  for (const auto& input : cluster_inputs) {
    cluster_graph_inputs.push_back(input);
  }

  for (const auto& in_arg : constant_inputs) {
    cluster_inputs.push_back(in_arg);
  }

  std::copy(external_output_args.begin(), external_output_args.end(), std::back_inserter(cluster_outputs));
  for (const auto& node_arg : graph_viewer.GetOutputs()) {
    const auto& name = node_arg->Name();
    if (output_args.count(name) && !external_output_args.count(name)) {
      cluster_outputs.push_back(name);
    }
  }
}

}  // namespace openvino_ep
}  // namespace onnxruntime
