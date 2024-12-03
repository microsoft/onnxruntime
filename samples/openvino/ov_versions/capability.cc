// Copyright (C) 2019- Intel Corporation
// Licensed under the MIT License
#include <map>
#include <unordered_set>

#include "../backend_utils.h"
#include "../backend_manager.h"
#include "capability.h"
#include "utils.h"
#include "openvino/core/version.hpp"

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

const OrtGraphApi* GetCapability::graph_api_ = OrtGetApiBase()->GetApi(ORT_API_VERSION)->GetGraphApi(ORT_API_VERSION);

// Constructor
GetCapability::GetCapability(const OrtGraphViewer* graph_viewer_param,
                             const std::string device_type_param,
                             const bool enable_qdq_optimizer)
    : graph_viewer_(graph_viewer_param), device_type_(device_type_param) {
  bool npu_qdq_optimizer_enabled = false;
  if (device_type_.find("NPU") != std::string::npos) {
    device_type_ = "CPU";
    if (enable_qdq_optimizer) npu_qdq_optimizer_enabled = true;
  }
#if OPENVINO_VERSION_MAJOR == 2023 && OPENVINO_VERSION_MINOR == 1
  data_ops_ = new DataOps(graph_viewer_, V_2023_1, device_type_, npu_qdq_optimizer_enabled);
#elif OPENVINO_VERSION_MAJOR == 2023 && OPENVINO_VERSION_MINOR == 2
  data_ops_ = new DataOps(graph_viewer_, V_2023_2, device_type_, npu_qdq_optimizer_enabled);
#elif OPENVINO_VERSION_MAJOR == 2023 && OPENVINO_VERSION_MINOR == 3
  data_ops_ = new DataOps(graph_viewer_, V_2023_3, device_type_, npu_qdq_optimizer_enabled);
#elif OPENVINO_VERSION_MAJOR == 2024 && OPENVINO_VERSION_MINOR == 0
  data_ops_ = new DataOps(graph_viewer_, V_2024_0, device_type_, npu_qdq_optimizer_enabled);
#elif OPENVINO_VERSION_MAJOR == 2024 && OPENVINO_VERSION_MINOR == 1
  data_ops_ = new DataOps(graph_viewer_, V_2024_1, device_type_, npu_qdq_optimizer_enabled);
#else
  data_ops_ = new DataOps(graph_viewer_, V_2024_1, device_type_, npu_qdq_optimizer_enabled);
#endif
}

size_t GetCapability::Execute(OrtIndexedSubGraph*** indexed_sub_graph) {
  // Check if it is a subgraph
  bool is_subgraph = false;
  graph_api_->OrtGraph_IsSubgraph(graph_viewer_, &is_subgraph);
  const char* graph_name = nullptr;
  graph_api_->OrtGraph_GetName(graph_viewer_, &graph_name);
  if (is_subgraph && !strcmp(graph_name, "tf2onnx")) return 0;

  // This is a list of initializers that nGraph considers as constants. Example weights, reshape shape etc.
  std::unordered_set<std::string> ng_required_initializers;

  const auto unsupported_nodes = data_ops_->GetUnsupportedNodeIndices(ng_required_initializers);
#ifndef NDEBUG
  if (openvino_ep::backend_utils::IsDebugEnabled()) {
    std::cout << "No of unsupported nodes " << unsupported_nodes.size() << std::endl;
    for (size_t i = 0; i < unsupported_nodes.size(); i++) {
      const OrtNode* node = nullptr;
      graph_api_->OrtGraph_GetOrtNode(graph_viewer_, unsupported_nodes[i], &node);
      const char* optype = nullptr;
      graph_api_->OrtNode_GetOpType(node, &optype);
      std::cout << "Unsupported node op " << optype << std::endl;
    }
  }
#endif

  // If all ops are supported, no partitioning is required. Short-circuit and avoid splitting.
  std::vector<OrtIndexedSubGraph*> cache;
  if (unsupported_nodes.empty()) {
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    // Fill inputs with names
    const char** input_names = nullptr;
    size_t input_count = 0;
    graph_api_->OrtGraph_GetRequiredInputs(graph_viewer_, &input_names, &input_count);
    for (int i = 0; i < input_count; i++) inputs.push_back(std::string(input_names[i]));
    graph_api_->ReleaseCharArray(input_names);

    /* In scenarios, when there are no inputs or all inputs being initializers,
         ConstantFolding optimization in onnxruntime pre-computes the value.*/
    if (inputs.empty()) {
      return 0;
    }

    const size_t* nodes = nullptr;
    size_t num_nodes;
    graph_api_->OrtGraph_GetNodesIndexInTopologicalOrder(graph_viewer_, 0, &nodes, &num_nodes);

    const OrtNode* node = nullptr;
    graph_api_->OrtGraph_GetOrtNode(graph_viewer_, nodes[0], &node);

    // Handle cases where lone, reoccuring Ops in smaller models cannot be supported in OpenVINO
    // If only a node of the same lone,unsupported type is present, then do not proceed with the subgraph
    if (num_nodes <= 3) {
      const char* optype = nullptr;
      graph_api_->OrtNode_GetOpType(node, &optype);
      if (data_ops_->IsOpSupportedOnlyInModel(optype)) {
        return 0;
      }
    }

    // Nodes that work well in models but not as a single node
    if (num_nodes == 1) {
      // If reshape is not an intermediate node, shape needs to be an initializer
      if (data_ops_->SpecialConditionForClusterSizeOne(ng_required_initializers, node)) {
        return 0;
      }
    }

    // Initializers need to be part of meta_def->inputs
    std::for_each(ng_required_initializers.begin(), ng_required_initializers.end(),
                  [&inputs](const std::string& initializer) { inputs.push_back(initializer); });

    // Fill outputs with names
    size_t output_count = 0;
    graph_api_->OrtGraph_GetOutputSize(graph_viewer_, &output_count);
    for (int i = 0; i < output_count; i++) {
      const char* output_name = nullptr;
      graph_api_->OrtGraph_GetIthOutputName(graph_viewer_, i, &output_name);
      outputs.push_back(std::string(output_name));
    }

    // Create and add this graph to result.
    AppendClusterToSubGraph(nodes, num_nodes, inputs, outputs, cache);

//    LOGS_DEFAULT(INFO) << "[OpenVINO-EP] Model is fully supported by OpenVINO";
    // Enable CI Logs
    if (backend_utils::IsCILogEnabled()) {
      std::cout << "Model is fully supported on OpenVINO" << std::endl;
    }
    is_wholly_supported_graph_ = true;

  } else {                                     // unsupported_nodes_idx.empty()
#if defined(OPENVINO_DISABLE_GRAPH_PARTITION)  // disables graph partition at build time
//    LOGS_DEFAULT(INFO) << "[OpenVINO-EP] DISABLE_GRAPH_PARTITION option is set";
//    LOGS_DEFAULT(INFO) << "[OpenVINO-EP] Model is not fully supported by OpenVINO, "
//                       << "so making the full model fall back to default CPU Execution Provider";
    return result;
#endif

    std::vector<NodeIndex> modified_unsupported_nodes;
    const size_t* topo_order = nullptr;
    size_t num_nodes = 0;
    graph_api_->OrtGraph_GetNodesIndexInTopologicalOrder(graph_viewer_, 0, &topo_order, &num_nodes);
    for (int i = 0; i < num_nodes; i++) {
      const NodeIndex node_idx = topo_order[i];
      if (find(unsupported_nodes.begin(), unsupported_nodes.end(), node_idx) != unsupported_nodes.end()) {
        modified_unsupported_nodes.push_back(node_idx);
      } else {
        const OrtNode* node = nullptr;
        graph_api_->OrtGraph_GetOrtNode(graph_viewer_, node_idx, &node);
        const char* optype = nullptr;
        graph_api_->OrtNode_GetOpType(node, &optype);
        if (data_ops_->InsertNode(optype)) {
          modified_unsupported_nodes.push_back(node_idx);
        }
      }
    }

    std::vector<size_t> topo_vec(topo_order, topo_order + num_nodes);
    auto ng_clusters = GetPartitionedClusters(topo_vec, modified_unsupported_nodes);

    auto connected_clusters = GetConnectedClusters(graph_api_, graph_viewer_, ng_clusters);

    int no_of_clusters = 0;

    for (auto this_cluster : connected_clusters) {
      // If subgraph has less then three, graph is considered trivial
      if (this_cluster.size() < 3) {
        continue;
      }

      std::vector<std::string> cluster_graph_inputs, cluster_inputs, cluster_outputs;

      GetInputsOutputsOfCluster(graph_api_, graph_viewer_,
                                this_cluster,
                                ng_required_initializers,
                                cluster_graph_inputs,
                                cluster_inputs,
                                cluster_outputs);

      bool omit_subgraph = false;
      // Omitting zero dim subgraphs
      for (auto index : this_cluster) {
        const OrtNode* node = nullptr;
        graph_api_->OrtGraph_GetOrtNode(graph_viewer_, index, &node);
        const char* optype = nullptr;
        graph_api_->OrtNode_GetOpType(node, &optype);
        if (data_ops_->DoNotOmitSubGraph(optype)) {
          size_t num_inputs = 0;
          graph_api_->OrtNode_GetNumInputs(node, &num_inputs);
          for (int i = 0; i < num_inputs; i++) {
            const char* input_name = nullptr;
            graph_api_->OrtNode_GetIthInputName(node, i, &input_name);
            auto it = find(cluster_graph_inputs.begin(), cluster_graph_inputs.end(), std::string(input_name));
            if (it != cluster_graph_inputs.end()) {
              omit_subgraph = true;
              break;
            }
          }
        }

        if (strcmp(optype, "Conv") == 0 || strcmp(optype, "Identity") == 0) {
          const char* output_name = nullptr;
          graph_api_->OrtNode_GetIthOutputName(node, 0, &output_name);
          auto it = find(cluster_outputs.begin(), cluster_outputs.end(), std::string(output_name));
          size_t outputs_count = 0;
          graph_api_->OrtNode_GetNumOutputs(node, &outputs_count);  // TODO(leca): equivelant to node->GetOutputEdgesCount()?
          if (it != cluster_outputs.end() && outputs_count != 0) {
            omit_subgraph = true;
            break;
          }
        }

        std::map<std::string, int> slice_map;
        if (!strcmp(optype, "Slice")) {
          const char* input_name = nullptr;
          graph_api_->OrtNode_GetIthInputName(node, 0, &input_name);
          auto it = find(cluster_graph_inputs.begin(), cluster_graph_inputs.end(), std::string(input_name));
          if (it != cluster_graph_inputs.end()) {
            if (slice_map.count(input_name) == 0) {
              slice_map[input_name] = 1;
            } else {
              omit_subgraph = true;
              break;
            }
          }
        }
      }
      if (omit_subgraph)
        continue;

      /* In scenarios, when there are no inputs or all inputs being initializers,
         ConstantFolding optimization in onnxruntime pre-computes the value.*/
      if (!cluster_inputs.empty()) {
        AppendClusterToSubGraph(this_cluster.data(), this_cluster.size(), cluster_inputs, cluster_outputs, cache);
        no_of_clusters++;
      }
    }
//    LOGS_DEFAULT(INFO) << "[OpenVINO-EP] Supported subgraphs on OpenVINO: " << no_of_clusters;
  }
  *indexed_sub_graph = new OrtIndexedSubGraph* [cache.size()];
  for (int i = 0; i < cache.size(); i++) (*indexed_sub_graph)[i] = cache[i];
  return cache.size();
}

}  // namespace openvino_ep
}  // namespace onnxruntime
