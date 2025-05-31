// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/ep_api.h"

#include <algorithm>
#include <vector>
#include "core/framework/error_code_helper.h"
#include "core/framework/func_api.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/ep_api_types.h"
#include "core/session/abi_devices.h"
#include "core/session/abi_ep_types.h"
#include "core/session/ort_apis.h"

using namespace onnxruntime;
namespace OrtExecutionProviderApi {
ORT_API_STATUS_IMPL(CreateEpDevice, _In_ OrtEpFactory* ep_factory,
                    _In_ const OrtHardwareDevice* hardware_device,
                    _In_opt_ const OrtKeyValuePairs* ep_metadata,
                    _In_opt_ const OrtKeyValuePairs* ep_options,
                    _Out_ OrtEpDevice** ort_ep_device) {
  API_IMPL_BEGIN
  auto ep_device = std::make_unique<OrtEpDevice>();
  ep_device->device = hardware_device;
  ep_device->ep_factory = ep_factory;
  ep_device->ep_name = ep_factory->GetName(ep_factory);
  ep_device->ep_vendor = ep_factory->GetVendor(ep_factory);

  if (ep_metadata) {
    ep_device->ep_metadata = *ep_metadata;
  }

  if (ep_options) {
    ep_device->ep_options = *ep_options;
  }

  *ort_ep_device = ep_device.release();
  return nullptr;
  API_IMPL_END
}

ORT_API(void, ReleaseEpDevice, _Frees_ptr_opt_ OrtEpDevice* device) {
  delete device;
}

ORT_API_STATUS_IMPL(EpGraphSupportInfo_AddSupportedNodes, _In_ OrtEpGraphSupportInfo* ort_graph_support_info,
                    _In_reads_(num_supported_nodes) const OrtNode* const* supported_nodes,
                    size_t num_supported_nodes,
                    _In_ const OrtHardwareDevice* hardware_device) {
  API_IMPL_BEGIN
  if (ort_graph_support_info == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Must specify a valid OrtGraph instance");
  }

  if (num_supported_nodes == 0 || supported_nodes == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Must specify a valid array of 1 or more supported nodes");
  }

  // TODO: Check that the OrtNodes are all contained by OrtEpGraphSupportInfo.
  gsl::span<const OrtNode* const> nodes_span(supported_nodes,
                                             supported_nodes + num_supported_nodes);
  ORT_API_RETURN_IF_STATUS_NOT_OK(ort_graph_support_info->AddSupportedNodes(hardware_device, nodes_span));
  return nullptr;
  API_IMPL_END
}

//
// OrtGraph
//

ORT_API_STATUS_IMPL(Graph_GetName, _In_ const OrtGraph* graph, _Out_ const char** name) {
  API_IMPL_BEGIN
  const onnxruntime::EpGraph* ep_graph = onnxruntime::EpGraph::ToInternal(graph);

  if (ep_graph == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Invalid OrtGraph variant for use in OrtEpApi");
  }

  *name = ep_graph->graph_viewer.Name().c_str();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(Graph_GetNumNodes, _In_ const OrtGraph* graph, _Out_ size_t* num_nodes) {
  API_IMPL_BEGIN
  const onnxruntime::EpGraph* ep_graph = onnxruntime::EpGraph::ToInternal(graph);

  if (ep_graph == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Invalid OrtGraph variant for use in OrtEpApi");
  }

  *num_nodes = ep_graph->graph_viewer.NumberOfNodes();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(Graph_GetNodes, const OrtGraph* graph, int order,
                    _Out_writes_all_(max_num_nodes) const OrtNode** nodes, _In_ size_t max_num_nodes) {
  API_IMPL_BEGIN
  const onnxruntime::EpGraph* ep_graph = onnxruntime::EpGraph::ToInternal(graph);

  if (ep_graph == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Invalid OrtGraph variant for use in OrtEpApi");
  }

  // TODO: make order an enum value.
  if (order < 0 || order > 2) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 "Invalid `order` value passed to OrtGraph_GetNodes(); only accepts values "
                                 "0, 1, or 2.");
  }

  ExecutionOrder execution_order = static_cast<ExecutionOrder>(order);
  const std::vector<NodeIndex>& node_indices = ep_graph->graph_viewer.GetNodesInTopologicalOrder(execution_order);
  size_t num_nodes = std::min(max_num_nodes, node_indices.size());

  for (size_t i = 0; i < num_nodes; i++) {
    NodeIndex node_idx = node_indices[i];
    auto node_it = ep_graph->index_to_node.find(node_idx);
    ORT_ENFORCE(node_it != ep_graph->index_to_node.end());
    nodes[i] = node_it->second->ToExternal();
  }

  return nullptr;
  API_IMPL_END
}

//
// OrtNode
//

ORT_API_STATUS_IMPL(Node_GetName, const OrtNode* node, _Outptr_ const char** name) {
  API_IMPL_BEGIN
  const onnxruntime::EpNode* ep_node = onnxruntime::EpNode::ToInternal(node);

  if (ep_node == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Invalid OrtNode variant for use in OrtEpApi");
  }

  *name = ep_node->node.Name().c_str();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(Node_GetOperatorType, const OrtNode* node, _Outptr_ const char** op_type) {
  API_IMPL_BEGIN
  const onnxruntime::EpNode* ep_node = onnxruntime::EpNode::ToInternal(node);

  if (ep_node == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Invalid OrtNode variant for use in OrtEpApi");
  }

  *op_type = ep_node->node.OpType().c_str();
  return nullptr;
  API_IMPL_END
}

//
// OrtCompiledNodeComputeContext
//

ORT_API(const char*, NodeComputeContext_NodeName, _In_ const OrtNodeComputeContext* context) {
  const auto* compute_context = reinterpret_cast<const onnxruntime::ComputeContext*>(context);
  return compute_context->node_name;
}

static constexpr OrtEpApi ort_ep_api = {
    // NOTE: ABI compatibility depends on the order within this struct so all additions must be at the end,
    // and no functions can be removed (the implementation needs to change to return an error).

    &OrtExecutionProviderApi::CreateEpDevice,
    &OrtExecutionProviderApi::ReleaseEpDevice,
    // End of Version 22 - DO NOT MODIFY ABOVE

    &OrtExecutionProviderApi::EpGraphSupportInfo_AddSupportedNodes,
    &OrtExecutionProviderApi::Graph_GetName,
    &OrtExecutionProviderApi::Graph_GetNumNodes,
    &OrtExecutionProviderApi::Graph_GetNodes,
    &OrtExecutionProviderApi::Node_GetName,
    &OrtExecutionProviderApi::Node_GetOperatorType,

    &OrtExecutionProviderApi::NodeComputeContext_NodeName,
};

// checks that we don't violate the rule that the functions must remain in the slots they were originally assigned
static_assert(offsetof(OrtEpApi, ReleaseEpDevice) / sizeof(void*) == 1,
              "Size of version 22 API cannot change");  // initial version in ORT 1.22

}  // namespace OrtExecutionProviderApi

ORT_API(const OrtEpApi*, OrtExecutionProviderApi::GetEpApi) {
  return &ort_ep_api;
}
