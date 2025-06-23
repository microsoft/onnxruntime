// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/ep_api.h"

#include <algorithm>
#include <vector>
#include "core/framework/error_code_helper.h"
#include "core/framework/func_api.h"
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

ORT_API_STATUS_IMPL(CreateNodeFusionOptions, _In_reads_(num_nodes) const OrtNode* const* nodes, _In_ size_t num_nodes,
                    _Outptr_ OrtNodeFusionOptions** out) {
  API_IMPL_BEGIN
  if (nodes == nullptr || num_nodes == 0) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Must specify one or more valid nodes.");
  }

  auto node_fusion_options = std::make_unique<OrtNodeFusionOptions>();
  node_fusion_options->nodes.reserve(num_nodes);

  for (size_t i = 0; i < num_nodes; ++i) {
    const OrtNode* ort_node = nodes[i];

    if (ort_node == nullptr) {
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "OrtNode instance is NULL.");
    }

    const auto* ep_node = onnxruntime::EpNode::ToInternal(ort_node);

    if (ep_node == nullptr) {
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                   "Unexpected variant of OrtNode that is not compatible with OrtEpApi.");
    }

    node_fusion_options->nodes.push_back(ep_node);
  }

  *out = node_fusion_options.release();
  return nullptr;
  API_IMPL_END
}

ORT_API(void, ReleaseNodeFusionOptions, _Frees_ptr_opt_ OrtNodeFusionOptions* options) {
  delete options;
}

ORT_API_STATUS_IMPL(NodeFusionOptions_DropConstantInitializers, _In_ OrtNodeFusionOptions* options,
                    _In_ bool drop) {
  API_IMPL_BEGIN
  options->drop_constant_initializers = drop;
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(EpGraphSupportInfo_AddNodesToFuse, _In_ OrtEpGraphSupportInfo* ort_graph_support_info,
                    _Inout_ OrtNodeFusionOptions* node_fusion_options) {
  API_IMPL_BEGIN
  std::unique_ptr<OrtNodeFusionOptions> owned_options(node_fusion_options);  // Take ownership

  if (node_fusion_options == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "'node_fusion_options' argument is NULL.");
  }

  ort_graph_support_info->AddNodesToFuse(*owned_options);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(EpGraphSupportInfo_AddSingleNode, _In_ OrtEpGraphSupportInfo* ort_graph_support_info,
                    _In_ const OrtNode* node) {
  API_IMPL_BEGIN
  if (ort_graph_support_info == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Must specify a valid OrtGraph instance");
  }

  if (node == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Must specify a non-null OrtNode");
  }

  const auto* ep_node = onnxruntime::EpNode::ToInternal(node);

  if (ep_node == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Unexpected variant of OrtNode is not compatible with OrtEpApi.");
  }

  ort_graph_support_info->AddSingleNode(*ep_node);
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

    &OrtExecutionProviderApi::CreateNodeFusionOptions,
    &OrtExecutionProviderApi::ReleaseNodeFusionOptions,
    &OrtExecutionProviderApi::NodeFusionOptions_DropConstantInitializers,
    &OrtExecutionProviderApi::EpGraphSupportInfo_AddNodesToFuse,
    &OrtExecutionProviderApi::EpGraphSupportInfo_AddSingleNode,
    &OrtExecutionProviderApi::NodeComputeContext_NodeName,
};

// checks that we don't violate the rule that the functions must remain in the slots they were originally assigned
static_assert(offsetof(OrtEpApi, ReleaseEpDevice) / sizeof(void*) == 1,
              "Size of version 22 API cannot change");  // initial version in ORT 1.22

}  // namespace OrtExecutionProviderApi

ORT_API(const OrtEpApi*, OrtExecutionProviderApi::GetEpApi) {
  return &ort_ep_api;
}
