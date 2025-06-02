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

ORT_API(const char*, Graph_GetName, _In_ const OrtGraph* graph) {
  return graph->Name().c_str();
}

ORT_API(size_t, Graph_GetNumNodes, _In_ const OrtGraph* graph) {
  return graph->NumberOfNodes();
}

ORT_API_STATUS_IMPL(Graph_GetNodes, const OrtGraph* graph, int order,
                    _Out_writes_all_(max_num_nodes) const OrtNode** nodes, _In_ size_t max_num_nodes) {
  API_IMPL_BEGIN
  // TODO: make order an enum value.
  if (order < 0 || order > 2) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 "Invalid `order` value passed to OrtGraph_GetNodes(); only accepts values "
                                 "0, 1, or 2.");
  }

  std::vector<const OrtNode*> sorted_nodes = graph->GetNodes(order);
  size_t num_nodes = std::min(max_num_nodes, sorted_nodes.size());

  for (size_t i = 0; i < num_nodes; i++) {
    nodes[i] = sorted_nodes[i];
  }

  return nullptr;
  API_IMPL_END
}

//
// OrtNode
//

ORT_API(const char*, Node_GetName, const OrtNode* node) {
  return node->Name().c_str();
}

ORT_API(const char*, Node_GetOperatorType, const OrtNode* node) {
  return node->OpType().c_str();
}

ORT_API(const char*, Node_GetDomain, const OrtNode* node) {
  return node->Domain().c_str();
}

ORT_API(size_t, Node_GetNumInputs, const OrtNode* node) {
  return node->GetNumInputs();
}

ORT_API(size_t, Node_GetNumOutputs, const OrtNode* node) {
  return node->GetNumOutputs();
}

ORT_API_STATUS_IMPL(Node_GetInputs, _In_ const OrtNode* node,
                    _Out_writes_all_(max_num_inputs) const OrtValueInfo** inputs, _In_ size_t max_num_inputs) {
  API_IMPL_BEGIN
  onnxruntime::InlinedVector<const OrtValueInfo*> node_inputs;
  ORT_API_RETURN_IF_STATUS_NOT_OK(node->GetInputs(node_inputs));

  size_t num_inputs = std::min(max_num_inputs, node_inputs.size());
  for (size_t i = 0; i < num_inputs; i++) {
    inputs[i] = node_inputs[i];
  }
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(Node_GetOutputs, _In_ const OrtNode* node,
                    _Out_writes_all_(max_num_outputs) const OrtValueInfo** outputs, _In_ size_t max_num_outputs) {
  API_IMPL_BEGIN
  onnxruntime::InlinedVector<const OrtValueInfo*> node_outputs;
  ORT_API_RETURN_IF_STATUS_NOT_OK(node->GetOutputs(node_outputs));

  size_t num_outputs = std::min(max_num_outputs, node_outputs.size());
  for (size_t i = 0; i < num_outputs; i++) {
    outputs[i] = node_outputs[i];
  }
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
    &OrtExecutionProviderApi::Node_GetDomain,
    &OrtExecutionProviderApi::Node_GetNumInputs,
    &OrtExecutionProviderApi::Node_GetNumOutputs,
    &OrtExecutionProviderApi::Node_GetInputs,
    &OrtExecutionProviderApi::Node_GetOutputs,

    &OrtExecutionProviderApi::NodeComputeContext_NodeName,
};

// checks that we don't violate the rule that the functions must remain in the slots they were originally assigned
static_assert(offsetof(OrtEpApi, ReleaseEpDevice) / sizeof(void*) == 1,
              "Size of version 22 API cannot change");  // initial version in ORT 1.22

}  // namespace OrtExecutionProviderApi

ORT_API(const OrtEpApi*, OrtExecutionProviderApi::GetEpApi) {
  return &ort_ep_api;
}
