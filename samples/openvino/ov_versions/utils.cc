// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include <iterator>
#include "utils.h"

#if defined(_MSC_VER)
#pragma warning(disable : 4244 4245 5208)
#elif __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#if defined(_MSC_VER)
#pragma warning(default : 4244 4245)
#elif __GNUC__
#pragma GCC diagnostic pop
#endif

namespace onnxruntime {
namespace openvino_ep {

// Gets the input count of given node
//int GetInputCount(const Node* node, const InitializedTensorSet& initializer_set) {
//  int count = 0;
//  for (const auto& input : node->InputDefs()) {
//    const auto& name = input->Name();
//    auto it = initializer_set.find(name);
//    if (it == initializer_set.end()) {
//      count++;
//    }
//  }
//  return count;
//}

// Ops which are supported only in models(as intermediate nodes) and not in unit tests
//bool IsOpSupportedOnlyInModel(std::string name) {
//  std::set<std::string> ops_supported_only_in_model = {
//      "Cast",
//      "Concat",
//      "ConstantOfShape",
//      "Dropout",
//      "Einsum",
//      "Expand",
//      "EyeLike",
//      "Exp",
//      "GatherND",
//      "Identity",
//      "LayerNormalization",
//      "NonMaxSuppression",
//      "NonZero",
//      "Not",
//      "OneHot",
//      "Pad",
//      "Range",
//      "ReduceMin",
//      "Resize",
//      "Round",
//      "Shape",
//      "Split",
//      "TopK"};
//  return ops_supported_only_in_model.find(name) != ops_supported_only_in_model.end();
//}

void AppendClusterToSubGraph(const size_t* node_index, size_t node_count,
                             const std::vector<std::string>& inputs,
                             const std::vector<std::string>& outputs,
                             std::vector<OrtIndexedSubGraph*>& cache) {
  static size_t op_counter = 0;

  OrtMetaDef* meta_def = new OrtMetaDef();
  std::string name = "OpenVINO-EP-subgraph_" + std::to_string(++op_counter);
  meta_def->name = new char [name.length() + 1];
  strcpy(meta_def->name, name.c_str());
  meta_def->domain = "com.intel.ai";
  meta_def->since_version = 1;
  // TODO(leca): meta_def->status() = ONNX_NAMESPACE::EXPERIMENTAL;
  meta_def->input_len = inputs.size();
  meta_def->inputs = new char* [inputs.size()];
  for (int i = 0; i < inputs.size(); i++) {
    meta_def->inputs[i] = new char [inputs[i].length() + 1];
    strcpy(meta_def->inputs[i], inputs[i].c_str());
  }
  meta_def->output_len = outputs.size();
  meta_def->outputs = new char* [outputs.size()];
  for (int i = 0; i < outputs.size(); i++) {
    meta_def->outputs[i] = new char [outputs[i].length() + 1];
    strcpy(meta_def->outputs[i], outputs[i].c_str());
  }

  OrtIndexedSubGraph* indexed_sub_graph = new OrtIndexedSubGraph();
  indexed_sub_graph->meta_def = meta_def;
  indexed_sub_graph->node_index_len = node_count;
  indexed_sub_graph->node_index = new size_t [node_count];
  for (int i = 0; i < node_count; i++) {
    indexed_sub_graph->node_index[i] = node_index[i];
  }

  cache.push_back(indexed_sub_graph);
}

//int GetOnnxOpSet(const GraphViewer& graph_viewer) {
//  const auto& dm_to_ver = graph_viewer.DomainToVersionMap();
//  return dm_to_ver.at(kOnnxDomain);
//}

/**
 * Returns a vector clusters(or node_idx). For each unsupported node, the graph is split into 3 parts.
 * supported_cluster + (UNsupported_node + rest_of_the_graph). This functions returns vector of all supported_clusters by nGraph
 */
std::vector<std::vector<NodeIndex>>
GetPartitionedClusters(const std::vector<NodeIndex>& topological_order,
                       const std::vector<NodeIndex>& unsupported_nodes) {
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

void IdentifyConnectedNodes(const OrtGraphApi* graph_api,
                            const OrtGraphViewer* graph_viewer,
                            NodeIndex curr_node_index,
                            std::vector<NodeIndex>& cluster,
                            std::vector<NodeIndex>& sub_cluster) {
  if (std::find(cluster.begin(), cluster.end(), curr_node_index) == cluster.end())
    return;

  sub_cluster.emplace_back(curr_node_index);
  cluster.erase(std::remove(cluster.begin(), cluster.end(), curr_node_index), cluster.end());
  const OrtNode* curr_node = nullptr;
  graph_api->OrtGraph_GetOrtNode(graph_viewer, curr_node_index, &curr_node);

  // TODO(leca): equivalent to for (auto node = curr_node->InputNodesBegin(); node != curr_node->InputNodesEnd(); ++node)?
  // TODO(leca): consider implicit input?
  size_t num_inputs = 0;
  graph_api->OrtNode_GetNumInputs(curr_node, &num_inputs);
  for (int i = 0; i < num_inputs; i++) {
    const char* input_name = nullptr;
    graph_api->OrtNode_GetIthInputName(curr_node, i, &input_name);
    const OrtNode* producer_node = nullptr;
    graph_api->OrtGraph_GetNodeProducingOutput(graph_viewer, input_name, &producer_node);
    size_t producer_index = 0;
    graph_api->OrtNode_GetIndex(producer_node, &producer_index);
    IdentifyConnectedNodes(graph_api, graph_viewer, producer_index, cluster, sub_cluster);
  }

  // TODO(leca): equal to for (auto node = curr_node->OutputNodesBegin(); node != curr_node->OutputNodesEnd(); ++node) ?
  size_t num_outputs = 0;
  graph_api->OrtNode_GetNumOutputs(curr_node, &num_outputs);
  for (int i = 0; i < num_outputs; i++) {
    const char* output_name = nullptr;
    graph_api->OrtNode_GetIthOutputName(curr_node, i, &output_name);
    const OrtNode** consumer_nodes = nullptr;
    size_t num_consumers = 0;
    // TODO(leca): if there is one consumer consuming more than 1 output of curr_node, would it be visited twice?
    graph_api->OrtGraph_GetNodesConsumingInput(graph_viewer, output_name, &consumer_nodes, &num_consumers);
    for (int j = 0; j < num_consumers; j++) {
      size_t consumer_index = 0;
      graph_api->OrtNode_GetIndex(consumer_nodes[j], &consumer_index);
      IdentifyConnectedNodes(graph_api, graph_viewer, consumer_index, cluster, sub_cluster);
    }
    graph_api->ReleaseOrtNodeArray(consumer_nodes);
  }
}

std::vector<std::vector<NodeIndex>>
GetConnectedClusters(const OrtGraphApi* graph_api, const OrtGraphViewer* graph_viewer, const std::vector<std::vector<NodeIndex>>& clusters) {
  std::vector<std::vector<NodeIndex>> connected_clusters;

  for (auto this_cluster : clusters) {
    while (this_cluster.size() > 0) {
      std::vector<NodeIndex> sub_cluster;
      IdentifyConnectedNodes(graph_api, graph_viewer, this_cluster[0], this_cluster, sub_cluster);
      connected_clusters.emplace_back(sub_cluster);
    }
  }
  return connected_clusters;
}

void GetInputsOutputsOfCluster(const OrtGraphApi* graph_api,
                               const OrtGraphViewer* graph_viewer,
                               const std::vector<NodeIndex>& cluster,
                               const std::unordered_set<std::string>& ng_required_initializers,
                               /*out*/ std::vector<std::string>& cluster_graph_inputs,
                               /*out*/ std::vector<std::string>& cluster_inputs,
                               /*out*/ std::vector<std::string>& cluster_outputs) {
  std::unordered_set<std::string> input_args;
  std::vector<std::string> ordered_input_args;
  std::unordered_set<std::string> output_args;
  std::unordered_set<std::string> external_output_args;
  std::vector<std::string> constant_inputs;

  for (const auto& node_idx : cluster) {
    const OrtNode* node = nullptr;
    graph_api->OrtGraph_GetOrtNode(graph_viewer, node_idx, &node);
    // Collect all inputs and outputs
    ForEachNodeDef(graph_api, graph_viewer, node,
                   [&input_args, &ordered_input_args, &output_args](const char* arg_name, const OrtValueInfoRef*, bool is_input) {
          if (strcmp(arg_name, "") != 0) {
            if (is_input) {
              if (!input_args.count(std::string(arg_name))) {
                ordered_input_args.push_back(std::string(arg_name));
              }
              input_args.insert(std::string(arg_name));
            } else {
              output_args.insert(std::string(arg_name));
            }
          }
    });

    // Check if output of this node is used by nodes outside this_cluster. If yes add this to cluster outputs
    // TODO(leca): equal to for (auto node = curr_node->OutputNodesBegin(); node != curr_node->OutputNodesEnd(); ++node) ?
    size_t num_outputs = 0;
    graph_api->OrtNode_GetNumOutputs(node, &num_outputs);
    for (int i = 0; i < num_outputs; i++) {
      const char* output_name = nullptr;
      graph_api->OrtNode_GetIthOutputName(node, i, &output_name);
      const OrtNode** consumer_nodes = nullptr;
      size_t num_consumers = 0;
      // TODO(leca): if there is one consumer consuming more than 1 output of curr_node, would it be visited twice?
      graph_api->OrtGraph_GetNodesConsumingInput(graph_viewer, output_name, &consumer_nodes, &num_consumers);
      for (int j = 0; j < num_consumers; j++) {
        size_t consumer_index = 0;
        graph_api->OrtNode_GetIndex(consumer_nodes[j], &consumer_index);

        if (std::find(cluster.begin(), cluster.end(), consumer_index) == cluster.end()) {
          // Node is external to this_cluster. Search through its inputs to
          // find the output that is generated by this_cluster.
          std::set<std::string> ext_node_inputs;
          ForEachNodeDef(graph_api, graph_viewer, consumer_nodes[j],
                         [&ext_node_inputs](const char* arg_name, const OrtValueInfoRef*, bool is_input) {
              if (is_input) {
                ext_node_inputs.insert(std::string(arg_name));
              }
          });

          for (int j = 0; j < num_outputs; j++) {
            const char* out_def = nullptr;
            graph_api->OrtNode_GetIthOutputName(node, j, &out_def);
            if (ext_node_inputs.find(std::string(out_def)) != ext_node_inputs.end()) {
              external_output_args.insert(std::string(out_def));
            }
          }
        }
      }
      graph_api->ReleaseOrtNodeArray(consumer_nodes);
    }
  }

  // Extract initializers used by this_cluster.
  std::unordered_set<std::string> original_graph_inputs;
  const char** input_names = nullptr;
  size_t input_len = 0;
  graph_api->OrtGraph_GetAllInputs(graph_viewer, &input_names, &input_len);
  for (int i = 0; i < input_len; i++) {
    original_graph_inputs.insert(std::string(input_names[i]));
  }
  graph_api->ReleaseCharArray(input_names);

  const char** initializers = nullptr;
  size_t initializer_len = 0;
  graph_api->OrtGraph_GetAllInitializers(graph_viewer, &initializers, &initializer_len);
  for (const auto& in_arg : ordered_input_args) {
    bool initializers_contain_in_arg = false;
    for (int i = 0; i < initializer_len; i++) {
      if (!strcmp(initializers[i], in_arg.c_str())) {
        initializers_contain_in_arg = true;
        break;
      }
    }

    if ((initializers_contain_in_arg && !original_graph_inputs.count(in_arg)) ||
        ng_required_initializers.count(in_arg)) constant_inputs.push_back(in_arg);
    if (!output_args.count(in_arg) &&
        !((initializers_contain_in_arg && !original_graph_inputs.count(in_arg)) ||
          ng_required_initializers.count(in_arg))) cluster_inputs.push_back(in_arg);
  }

  for (const auto& input : cluster_inputs) {
    cluster_graph_inputs.push_back(input);
  }

  for (const auto& in_arg : constant_inputs) {
    cluster_inputs.push_back(in_arg);
  }

  std::copy(external_output_args.begin(), external_output_args.end(), std::back_inserter(cluster_outputs));
  size_t output_count = 0;
  graph_api->OrtGraph_GetOutputSize(graph_viewer, &output_count);
  for (int i = 0; i < output_count; i++) {
    const char* name = nullptr;
    graph_api->OrtGraph_GetIthOutputName(graph_viewer, i, &name);
    if (output_args.count(name) && !external_output_args.count(name)) {
      cluster_outputs.push_back(name);
    }
  }
}

}  // namespace openvino_ep
}  // namespace onnxruntime
