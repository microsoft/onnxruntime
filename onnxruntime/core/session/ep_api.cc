// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/ep_api.h"

#include <algorithm>
#include <vector>

#include "core/common/semver.h"
#include "core/framework/error_code_helper.h"
#include "core/framework/func_api.h"
#include "core/framework/ort_value.h"
#include "core/framework/ortdevice.h"
#include "core/framework/ortmemoryinfo.h"
#include "core/framework/tensor.h"
#include "core/graph/ep_api_types.h"
#include "core/session/abi_devices.h"
#include "core/session/abi_ep_types.h"
#include "core/session/onnxruntime_ep_device_ep_metadata_keys.h"
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

  // Add EP version from OrtEpFactory to metadata. OrtEpFactory::GetVersion is supported since 1.23.
  if (ep_factory->ort_version_supported >= uint32_t{23}) {
    if (ep_device->ep_metadata.Entries().find(kOrtEpDevice_EpMetadataKey_Version) !=
        ep_device->ep_metadata.Entries().end()) {
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                   "The provided EP metadata should not explicitly specify the EP version.");
    }

    {
      std::string ep_version = ep_factory->GetVersion(ep_factory);
      ORT_API_RETURN_IF_STATUS_NOT_OK(ParseSemVerVersion(ep_version, nullptr));
      ep_device->ep_metadata.Add(kOrtEpDevice_EpMetadataKey_Version, std::move(ep_version));
    }
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

ORT_API_STATUS_IMPL(EpGraphSupportInfo_AddNodesToFuse, _In_ OrtEpGraphSupportInfo* ort_graph_support_info,
                    _In_reads_(num_nodes) const OrtNode* const* nodes, size_t num_nodes,
                    _In_opt_ const OrtNodeFusionOptions* node_fusion_options) {
  API_IMPL_BEGIN
  if (ort_graph_support_info == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Must specify a valid OrtGraph instance");
  }

  if (num_nodes == 0 || nodes == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Must specify a valid array of 1 or more supported nodes");
  }

  gsl::span<const OrtNode* const> nodes_span(nodes, nodes + num_nodes);
  ORT_API_RETURN_IF_STATUS_NOT_OK(ort_graph_support_info->AddNodesToFuse(nodes_span, node_fusion_options));
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

  ORT_API_RETURN_IF_STATUS_NOT_OK(ort_graph_support_info->AddSingleNode(node));
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

ORT_API_STATUS_IMPL(EpDevice_AddAllocatorInfo, _In_ OrtEpDevice* ep_device,
                    _In_ const OrtMemoryInfo* allocator_memory_info) {
  const OrtDevice& info = allocator_memory_info->device;
  switch (info.MemType()) {
    case OrtDevice::MemType::DEFAULT:
      ep_device->device_memory_info = allocator_memory_info;
      break;
    case OrtDevice::MemType::HOST_ACCESSIBLE:
      ep_device->host_accessible_memory_info = allocator_memory_info;
      break;
    default:
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Memory type must be DEFAULT or HOST_ACCESSIBLE.");
  }

  return nullptr;
}

ORT_API(const OrtMemoryDevice*, MemoryInfo_GetMemoryDevice, _In_ const OrtMemoryInfo* memory_info) {
  return static_cast<const OrtMemoryDevice*>(&memory_info->device);
}

ORT_API_STATUS_IMPL(Value_GetMemoryDevice, _In_ const OrtValue* value, _Out_ const OrtMemoryDevice** device) {
  *device = nullptr;
  if (value == nullptr || value->IsTensor() == false) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "OrtValue does not contain an allocated tensor.");
  }

  auto& tensor = value->Get<Tensor>();
  *device = static_cast<const OrtMemoryDevice*>(&tensor.Location().device);

  return nullptr;
}

ORT_API(bool, MemoryDevice_AreEqual, _In_ const OrtMemoryDevice* a, _In_ const OrtMemoryDevice* b) {
  // don't care if they're both null as you don't need to call this function if they are
  if (a == nullptr || b == nullptr) {
    return false;
  }

  // TODO: Validate this calls OrtDevice::operator== as expected
  return *a == *b;
}

ORT_API(OrtMemoryInfoDeviceType, MemoryDevice_GetDeviceType, _In_ const OrtMemoryDevice* memory_device) {
  switch (memory_device->Type()) {
    case OrtDevice::GPU:
      return OrtMemoryInfoDeviceType_GPU;
    case OrtDevice::NPU:
      return OrtMemoryInfoDeviceType_NPU;
    case OrtDevice::FPGA:
      return OrtMemoryInfoDeviceType_FPGA;
    case OrtDevice::CPU:
    default:  // should never happen. means we're out of sync with CreateMemoryInfo_V2
      return OrtMemoryInfoDeviceType_CPU;
  }
}

ORT_API(OrtDeviceMemoryType, MemoryDevice_GetMemoryType, _In_ const OrtMemoryDevice* memory_device) {
  return memory_device->MemType() == OrtDevice::MemType::DEFAULT ? OrtDeviceMemoryType_DEFAULT
                                                                 : OrtDeviceMemoryType_HOST_ACCESSIBLE;
}

ORT_API(uint32_t, MemoryDevice_GetVendorId, _In_ const OrtMemoryDevice* memory_device) {
  return memory_device->Vendor();
}

ORT_API(uint32_t, MemoryDevice_GetDeviceId, _In_ const OrtMemoryDevice* memory_device) {
  return memory_device->Id();
}

static constexpr OrtEpApi ort_ep_api = {
    // NOTE: ABI compatibility depends on the order within this struct so all additions must be at the end,
    // and no functions can be removed (the implementation needs to change to return an error).

    &OrtExecutionProviderApi::CreateEpDevice,
    &OrtExecutionProviderApi::ReleaseEpDevice,
    // End of Version 22 - DO NOT MODIFY ABOVE

    &OrtExecutionProviderApi::EpGraphSupportInfo_AddNodesToFuse,
    &OrtExecutionProviderApi::EpGraphSupportInfo_AddSingleNode,
    &OrtExecutionProviderApi::NodeComputeContext_NodeName,
    &OrtExecutionProviderApi::EpDevice_AddAllocatorInfo,

    &OrtExecutionProviderApi::MemoryInfo_GetMemoryDevice,
    &OrtExecutionProviderApi::Value_GetMemoryDevice,

    &OrtExecutionProviderApi::MemoryDevice_AreEqual,
    &OrtExecutionProviderApi::MemoryDevice_GetDeviceType,
    &OrtExecutionProviderApi::MemoryDevice_GetMemoryType,
    &OrtExecutionProviderApi::MemoryDevice_GetVendorId,
    &OrtExecutionProviderApi::MemoryDevice_GetDeviceId,
};

// checks that we don't violate the rule that the functions must remain in the slots they were originally assigned
static_assert(offsetof(OrtEpApi, ReleaseEpDevice) / sizeof(void*) == 1,
              "Size of version 22 API cannot change");  // initial version in ORT 1.22

}  // namespace OrtExecutionProviderApi

ORT_API(const OrtEpApi*, OrtExecutionProviderApi::GetEpApi) {
  return &ort_ep_api;
}
