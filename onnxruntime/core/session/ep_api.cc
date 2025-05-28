// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/ep_api.h"

#include "core/framework/error_code_helper.h"
#include "core/session/abi_devices.h"
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

ORT_API_STATUS_IMPL(CreateEpSupportedSubgraph, _In_ const OrtGraph* graph,
                    _In_ const char* subgraph_name,
                    _In_ const OrtHardwareDevice* hardware_device,
                    _In_reads_(num_supported_nodes) const OrtNode* const* supported_nodes,
                    size_t num_supported_nodes,
                    _Outptr_ OrtEpSupportedSubgraph** out) {
  API_IMPL_BEGIN
  if (graph == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Must specify a valid OrtGraph instance");
  }

  if (num_supported_nodes == 0 || supported_nodes == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Must specify a valid array of 1 or more supported nodes");
  }

  // TODO: Check that the OrtNodes are all contained by OrtGraph.

  auto ep_subgraph = std::make_unique<OrtEpSupportedSubgraph>();
  ep_subgraph->name = subgraph_name;
  ep_subgraph->hardware_device = hardware_device;
  ep_subgraph->nodes = std::vector<const OrtNode*>(supported_nodes, supported_nodes + num_supported_nodes);
  *out = ep_subgraph.release();
  return nullptr;
  API_IMPL_END
}

ORT_API(void, ReleaseEpSupportedSubgraph, _Frees_ptr_opt_ OrtEpSupportedSubgraph* subgraph) {
  delete subgraph;
}

static constexpr OrtEpApi ort_ep_api = {
    // NOTE: ABI compatibility depends on the order within this struct so all additions must be at the end,
    // and no functions can be removed (the implementation needs to change to return an error).

    &OrtExecutionProviderApi::CreateEpDevice,
    &OrtExecutionProviderApi::ReleaseEpDevice,
    // End of Version 22 - DO NOT MODIFY ABOVE

    &OrtExecutionProviderApi::CreateEpSupportedSubgraph,
    &OrtExecutionProviderApi::ReleaseEpSupportedSubgraph,
};

// checks that we don't violate the rule that the functions must remain in the slots they were originally assigned
static_assert(offsetof(OrtEpApi, ReleaseEpDevice) / sizeof(void*) == 1,
              "Size of version 22 API cannot change");  // initial version in ORT 1.22

}  // namespace OrtExecutionProviderApi

ORT_API(const OrtEpApi*, OrtExecutionProviderApi::GetEpApi) {
  return &ort_ep_api;
}
