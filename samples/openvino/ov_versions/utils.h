// Copyright (C) Intel Corporation
// Licensed under the MIT License
#pragma once

#include <memory>
#include <map>
#include <utility>
#include <vector>
#include <set>
#include <algorithm>
#include <string>
#include <unordered_set>
#include "../openvino_utils.h"

namespace onnxruntime {
namespace openvino_ep {

//int GetInputCount(const Node* node, const InitializedTensorSet& initializer_set);

//bool IsOpSupportedOnlyInModel(std::string name);

void AppendClusterToSubGraph(const size_t* node_index, size_t node_count,
                             const std::vector<std::string>& inputs,
                             const std::vector<std::string>& outputs,
                             std::vector<OrtIndexedSubGraph*>& cache);

//int GetOnnxOpSet(const GraphViewer& graph_viewer);

//std::map<std::string, std::set<std::string>> GetNgSupportedOps(const int onnx_opset);

std::vector<std::vector<NodeIndex>>
GetPartitionedClusters(
    const std::vector<NodeIndex>& topological_order, const std::vector<NodeIndex>& unsupported_nodes);

void IdentifyConnectedNodes(
    const OrtGraphApi* graph_api,
    const OrtGraphViewer* graph_viewer,
    NodeIndex curr_node_index,
    std::vector<NodeIndex>& cluster,
    std::vector<NodeIndex>& sub_cluster);

std::vector<std::vector<NodeIndex>>
GetConnectedClusters(const OrtGraphApi* graph_api, const OrtGraphViewer* graph_viewer, const std::vector<std::vector<NodeIndex>>& clusters);

//void GetInputsOutputsOfCluster(const OrtGraphApi* graph_api,
//                               const GraphViewer& graph_viewer,
//                               const std::vector<NodeIndex>& cluster,
//                               const std::unordered_set<std::string>& ng_required_initializers,
//                               /*out*/ std::vector<std::string>& cluster_graph_inputs,
//                               /*out*/ std::vector<std::string>& cluster_inputs,
//                               /*out*/ std::vector<std::string>& cluster_outputs);

}  // namespace openvino_ep
}  // namespace onnxruntime
