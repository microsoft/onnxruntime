// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

namespace onnxruntime {
namespace openvino_ep {

int GetInputCount(const Provider_Node* node, const Provider_InitializedTensorSet& initializer_set);

bool IsOpSupportedOnlyInModel(std::string name);

void AppendClusterToSubGraph(const std::vector<NodeIndex>& nodes,
                             const std::vector<std::string>& inputs,
                             const std::vector<std::string>& outputs,
                             std::vector<std::unique_ptr<Provider_ComputeCapability>>& result);

int GetOnnxOpSet(const Provider_GraphViewer& graph_viewer);

std::map<std::string, std::set<std::string>> GetNgSupportedOps(const int onnx_opset);

std::vector<std::vector<NodeIndex>>
GetPartitionedClusters(const std::vector<NodeIndex>& topological_order, const std::vector<NodeIndex>& unsupported_nodes);

void IdentifyConnectedNodes(const Provider_GraphViewer& graph_viewer, NodeIndex curr_node_index, std::vector<NodeIndex>& cluster, std::vector<NodeIndex>& sub_cluster);

std::vector<std::vector<NodeIndex>>
GetConnectedClusters(const Provider_GraphViewer& graph_viewer, const std::vector<std::vector<NodeIndex>>& clusters);

void GetInputsOutputsOfCluster(const Provider_GraphViewer& graph_viewer,
                               const std::vector<NodeIndex>& cluster,
                               const std::unordered_set<std::string>& ng_required_initializers,
                               /*out*/ std::vector<std::string>& cluster_graph_inputs,
                               /*out*/ std::vector<std::string>& cluster_inputs,
                               /*out*/ std::vector<std::string>& constant_inputs,
                               /*out*/ std::vector<std::string>& cluster_outputs);

}  // namespace openvino_ep
}  // namespace onnxruntime
