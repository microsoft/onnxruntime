// Copyright (C) 2019-2022 Intel Corporation
// Licensed under the MIT License

#include "../backend_utils.h"
#include "../backend_manager.h"
#include "capabilities.h"
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

// Constructor
GetCapability::GetCapability(const onnxruntime::interface::GraphViewRef& graph_viewer_param, std::string device_type_param,
                             const std::string version_param) : graph_viewer_(graph_viewer_param), device_type_(device_type_param) {
  if (version_param == "V_2022_1") {
    data_ops_ = new DataOps(graph_viewer_, V_2022_1, device_type_);
  } else if (version_param == "V_2022_2") {
    data_ops_ = new DataOps(graph_viewer_, V_2022_2, device_type_);
  } else if (version_param == "V_2022_3") {
    data_ops_ = new DataOps(graph_viewer_, V_2022_3, device_type_);
  } else if (version_param == "V_2023_0") {
    data_ops_ = new DataOps(graph_viewer_, V_2023_0, device_type_);
  } else if (version_param == "V_2023_1") {
    data_ops_ = new DataOps(graph_viewer_, V_2023_1, device_type_);
  } else {
    data_ops_ = new DataOps(graph_viewer_, V_2023_1, device_type_);
  }
}

std::vector<std::unique_ptr<SubGraphDef>> GetCapability::Execute() {
  std::vector<std::unique_ptr<SubGraphDef>> result;

  // Check if it is a subgraph
  if (graph_viewer_.IsSubGraph() && graph_viewer_.Name() == "tf2onnx") {
    return result;
  }

  // This is a list of initializers that nGraph considers as constants. Example weights, reshape shape etc.
  std::unordered_set<std::string> ng_required_initializers;

  const auto unsupported_nodes = data_ops_->GetUnsupportedNodeIndices(ng_required_initializers);
#ifndef NDEBUG
  if (openvino_ep::backend_utils::IsDebugEnabled()) {
    std::cout << "No of unsupported nodes " << unsupported_nodes.size() << std::endl;
    for (size_t i = 0; i < unsupported_nodes.size(); i++) {
      std::cout << "Unsupported node op " << unsupported_nodes[i]->OpType() << std::endl;
    }
  }
#endif

  // If all ops are supported, no partitioning is required. Short-circuit and avoid splitting.
  if (unsupported_nodes.empty()) {
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    // Fill inputs with names
    std::for_each(graph_viewer_.GetInputs().begin(), graph_viewer_.GetInputs().end(),
                  [&inputs](std::string_view node_arg) { inputs.push_back(std::string(node_arg)); });

    /* In scenarios, when there are no inputs or all inputs being initializers,
         ConstantFolding optimization in onnxruntime pre-computes the value.*/
    if (inputs.empty()) {
      return result;
    }

    const auto& nodes = graph_viewer_.NodeViews();

    const auto& node = nodes[0];

    // Handle cases where lone, reoccuring Ops in smaller models cannot be supported in OpenVINO
    // If only a node of the same lone,unsupported type is present, then do not proceed with the subgraph
    if (nodes.size() <= 3) {
      if (data_ops_->IsOpSupportedOnlyInModel(std::string(node->OpType()))) {
        return result;
      }
    }

    // Nodes that work well in models but not as a single node
    if (nodes.size() == 1) {
      // If reshape is not an intermediate node, shape needs to be an initializer
      if (data_ops_->SpecialConditionForClusterSizeOne(ng_required_initializers, node.get())) {
        return result;
      }
    }

    // Initializers need to be part of meta_def->inputs
    std::for_each(ng_required_initializers.begin(), ng_required_initializers.end(),
                  [&inputs](const std::string& initializer) { inputs.push_back(initializer); });

    // Fill outputs with names
    std::for_each(graph_viewer_.GetOutputs().begin(), graph_viewer_.GetOutputs().end(),
                  [&outputs](std::string_view node_arg) { outputs.push_back(std::string(node_arg)); });

    // Create and add this graph to result.
    AppendClusterToSubGraph(graph_viewer_.GetNodesInTopologicalOrder(), inputs, outputs, result);

    //LOGS_DEFAULT(INFO) << "[OpenVINO-EP] Model is fully supported by OpenVINO";
    // Enable CI Logs
    if (backend_utils::IsCILogEnabled()) {
      std::cout << "Model is fully supported on OpenVINO" << std::endl;
    }
    openvino_ep::BackendManager::GetGlobalContext().is_wholly_supported_graph = true;

  } else {  // unsupported_nodes_idx.empty()

#if defined(OPENVINO_DISABLE_GRAPH_PARTITION)  // disables graph partition at build time
    //LOGS_DEFAULT(INFO) << "[OpenVINO-EP] DISABLE_GRAPH_PARTITION option is set";
    //LOGS_DEFAULT(INFO) << "[OpenVINO-EP] Model is not fully supported by OpenVINO, so making the full model fall back to default CPU Execution Provider";
    return result;
#endif

    std::vector<size_t> modified_unsupported_nodes;
    for (std::unique_ptr<interface::NodeViewRef>& node : graph_viewer_.NodeViews()) {
      if (std::find_if(unsupported_nodes.begin(), unsupported_nodes.end(), [&node](interface::NodeViewRef* unsupported) { return unsupported->Index() == node->Index(); }) != unsupported_nodes.end()) {
        modified_unsupported_nodes.push_back(node->Index());
      } else {
        std::string_view optype = node->OpType();
        if (data_ops_->InsertNode(std::string(optype))) {
          modified_unsupported_nodes.push_back(node->Index());
        }
      }
    }

    auto ng_clusters = GetPartitionedClusters(graph_viewer_.GetNodesInTopologicalOrder(), modified_unsupported_nodes);

    auto connected_clusters = GetConnectedClusters(graph_viewer_, ng_clusters);

    int no_of_clusters = 0;

    for (auto this_cluster : connected_clusters) {
      // If subgraph has less then three, graph is considered trivial
      if (this_cluster.size() < 3) {
        continue;
      } else {
        // If subgraph only has Identity node, EyeLike or Dropout, OpenVINO EP doesn't support it.
        if (this_cluster.size() == 1) {
          const auto& node = graph_viewer_.GetNode(this_cluster[0]);
          if (IsOpSupportedOnlyInModel(std::string(node->OpType())))
            continue;
          // If reshape is not an intermediate node, shape needs to be an initializer
          if (data_ops_->SpecialConditionForClusterSizeOne(ng_required_initializers, node.get()))
            continue;
        }
      }

      std::vector<std::string> cluster_graph_inputs, cluster_inputs, const_inputs, cluster_outputs;

      GetInputsOutputsOfCluster(graph_viewer_, this_cluster, ng_required_initializers, cluster_graph_inputs, cluster_inputs, const_inputs, cluster_outputs);

      bool omit_subgraph = false;
      // Omitting zero dim subgraphs
      for (auto index : this_cluster) {
        std::unique_ptr<interface::NodeViewRef> node = graph_viewer_.GetNode(index);
        if (data_ops_->DoNotOmitSubGraph(std::string(node->OpType()))) {
          for (std::string_view input_name : node->Inputs()) {
            auto it = find(cluster_graph_inputs.begin(), cluster_graph_inputs.end(), std::string(input_name));
            if (it != cluster_graph_inputs.end()) {
              omit_subgraph = true;
              break;
            }
          }
        }

        if (node->OpType() == "Conv" || node->OpType() == "Identity") {
          auto output_name = node->Outputs()[0];
          auto it = find(cluster_outputs.begin(), cluster_outputs.end(), std::string(output_name));
          if (it != cluster_outputs.end() && node->Outputs().size() != 0) {
            omit_subgraph = true;
            break;
          }
        }

        std::map<std::string, int> slice_map;
        if (node->OpType() == "Slice") {
          auto input_name = node->Inputs()[0];
          auto it = find(cluster_graph_inputs.begin(), cluster_graph_inputs.end(), std::string(input_name));
          if (it != cluster_graph_inputs.end()) {
            if (slice_map.count(std::string(input_name)) == 0) {
              slice_map[std::string(input_name)] = 1;
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
        AppendClusterToSubGraph(this_cluster, cluster_inputs, cluster_outputs, result);
        no_of_clusters++;
      }
    }
    //LOGS_DEFAULT(INFO) << "[OpenVINO-EP] Supported subgraphs on OpenVINO: " << no_of_clusters;
  }

  return result;
}

}  // namespace openvino_ep
}  // namespace onnxruntime
