// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/plugin_ep/ep_api.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/common/semver.h"
#include "core/framework/error_code_helper.h"
#include "core/framework/func_api.h"
#include "core/framework/op_kernel_info.h"
#include "core/framework/ort_value.h"
#include "core/framework/ortdevice.h"
#include "core/framework/ortmemoryinfo.h"
#include "core/framework/plugin_ep_stream.h"
#include "core/framework/tensor.h"
#include "core/graph/ep_api_types.h"
#include "core/session/abi_devices.h"
#include "core/session/abi_ep_types.h"
#include "core/session/onnxruntime_ep_device_ep_metadata_keys.h"
#include "core/session/ort_apis.h"
#include "core/session/plugin_ep/ep_kernel_registration.h"
#include "core/session/utils.h"

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
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Must specify a valid OrtEpGraphSupportInfo instance");
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
      if (allocator_memory_info->alloc_type == OrtReadOnlyAllocator) {
        ep_device->read_only_device_memory_info = allocator_memory_info;
      } else {
        ep_device->device_memory_info = allocator_memory_info;
      }
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

ORT_API(const OrtMemoryDevice*, Value_GetMemoryDevice, _In_ const OrtValue* value) {
  if (value == nullptr || value->IsTensor() == false) {
    return nullptr;  // Tensor always has a device, so we don't need a more specific error here.
  }

  auto& tensor = value->Get<Tensor>();
  return static_cast<const OrtMemoryDevice*>(&tensor.Location().device);
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

ORT_API(const OrtSyncStreamImpl*, SyncStream_GetImpl, _In_ const OrtSyncStream* ort_stream) {
  // the EP API should only ever see plugin_ep::Stream instances
  const auto& stream = *reinterpret_cast<const plugin_ep::Stream*>(ort_stream);
  return &stream.GetImpl();
}

ORT_API(uint64_t, SyncStream_GetSyncId, _In_ const OrtSyncStream* stream) {
  return static_cast<const Stream*>(stream)->GetSyncId();
}

ORT_API(uint64_t, GetSyncIdForLastWaitOnSyncStream, _In_ const OrtSyncStream* producer_stream,
        _In_ const OrtSyncStream* consumer_stream) {
  uint64_t id{0};
  if (producer_stream && consumer_stream) {
    const auto& producer = *static_cast<const Stream*>(producer_stream);
    const auto& consumer = *static_cast<const Stream*>(consumer_stream);

    // If both streams are valid, we can return the sync id for the last wait on the producer stream.
    // This is useful for synchronizing operations between different streams.
    id = consumer.GetSyncIdForLastWaitOnStream(producer);
  }

  return id;
}

ORT_API_STATUS_IMPL(CreateHardwareDevice, _In_ OrtHardwareDeviceType type,
                    _In_ uint32_t vendor_id,
                    _In_ uint32_t device_id,
                    _In_ const char* vendor_name,
                    _In_opt_ const OrtKeyValuePairs* metadata,
                    _Out_ OrtHardwareDevice** hardware_device) {
  API_IMPL_BEGIN
  auto device = std::make_unique<OrtHardwareDevice>();
  device->type = type;
  device->vendor_id = vendor_id;
  device->device_id = device_id;
  device->vendor = std::string(vendor_name);

  if (metadata) {
    device->metadata = *metadata;
  }

  *hardware_device = device.release();
  return nullptr;
  API_IMPL_END
}

ORT_API(void, ReleaseHardwareDevice, _Frees_ptr_opt_ OrtHardwareDevice* device) {
  delete device;
}

ORT_API_STATUS_IMPL(CreateKernelRegistry, _Outptr_ OrtKernelRegistry** kernel_registry) {
  API_IMPL_BEGIN
  auto unique_kernel_registry = std::make_unique<onnxruntime::KernelRegistry>();

  *kernel_registry = reinterpret_cast<OrtKernelRegistry*>(unique_kernel_registry.release());
  return nullptr;
  API_IMPL_END
}

ORT_API(void, ReleaseKernelRegistry, _Frees_ptr_opt_ OrtKernelRegistry* kernel_registry) {
  delete reinterpret_cast<onnxruntime::KernelRegistry*>(kernel_registry);
}

ORT_API_STATUS_IMPL(KernelRegistry_AddKernel, _In_ OrtKernelRegistry* kernel_registry,
                    _In_ const OrtKernelDef* kernel_def, _In_ OrtKernelCreateFunc kernel_create_func,
                    _In_ void* kernel_create_func_state) {
  API_IMPL_BEGIN
  if (kernel_registry == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Must specify a non-null OrtKernelRegistry");
  }

  if (kernel_def == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Must specify a non-null OrtKernelDef");
  }

  if (kernel_create_func == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Must specify a non-null OrtKernelCreateFunc");
  }

  auto* internal_kernel_def = reinterpret_cast<const onnxruntime::KernelDef*>(kernel_def);
  onnxruntime::KernelCreateInfo kernel_create_info = MakePluginEpKernelCreateInfo(internal_kernel_def,
                                                                                  kernel_create_func,
                                                                                  kernel_create_func_state);

  auto* actual_registry = reinterpret_cast<onnxruntime::KernelRegistry*>(kernel_registry);
  ORT_API_RETURN_IF_STATUS_NOT_OK(actual_registry->Register(std::move(kernel_create_info)));
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(CreateKernelDefBuilder, _Outptr_ OrtKernelDefBuilder** kernel_def_builder_out) {
  API_IMPL_BEGIN
  auto builder = onnxruntime::KernelDefBuilder::Create();
  *kernel_def_builder_out = reinterpret_cast<OrtKernelDefBuilder*>(builder.release());
  return nullptr;
  API_IMPL_END
}

ORT_API(void, ReleaseKernelDefBuilder, _Frees_ptr_opt_ OrtKernelDefBuilder* kernel_def_builder) {
  delete reinterpret_cast<onnxruntime::KernelDefBuilder*>(kernel_def_builder);
}

ORT_API_STATUS_IMPL(KernelDefBuilder_SetOperatorType, _In_ OrtKernelDefBuilder* kernel_def_builder,
                    _In_ const char* op_type) {
  API_IMPL_BEGIN
  auto* builder = reinterpret_cast<onnxruntime::KernelDefBuilder*>(kernel_def_builder);
  builder->SetName(op_type);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(KernelDefBuilder_SetDomain, _In_ OrtKernelDefBuilder* kernel_def_builder,
                    _In_ const char* domain) {
  API_IMPL_BEGIN
  auto* builder = reinterpret_cast<onnxruntime::KernelDefBuilder*>(kernel_def_builder);
  builder->SetDomain(domain);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(KernelDefBuilder_SetSinceVersion, _In_ OrtKernelDefBuilder* kernel_def_builder,
                    _In_ int since_version_start, _In_ int since_version_end) {
  API_IMPL_BEGIN
  auto* builder = reinterpret_cast<onnxruntime::KernelDefBuilder*>(kernel_def_builder);

  // start version must be >= 1
  if (since_version_start < 1) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Start version must be >= 1");
  }

  // end version must >= start version
  if (since_version_end < since_version_start) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "End version must be >= to the start version");
  }

  builder->SinceVersion(since_version_start, since_version_end);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(KernelDefBuilder_SetExecutionProvider, _In_ OrtKernelDefBuilder* kernel_def_builder,
                    _In_ const char* ep_name) {
  API_IMPL_BEGIN
  auto* builder = reinterpret_cast<onnxruntime::KernelDefBuilder*>(kernel_def_builder);
  builder->Provider(ep_name);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(KernelDefBuilder_SetInputMemType, _In_ OrtKernelDefBuilder* kernel_def_builder,
                    _In_ size_t input_index, _In_ OrtMemType mem_type) {
  API_IMPL_BEGIN
  auto* builder = reinterpret_cast<onnxruntime::KernelDefBuilder*>(kernel_def_builder);
  builder->InputMemoryType(mem_type, static_cast<int>(input_index));
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(KernelDefBuilder_SetOutputMemType, _In_ OrtKernelDefBuilder* kernel_def_builder,
                    _In_ size_t output_index, _In_ OrtMemType mem_type) {
  API_IMPL_BEGIN
  auto* builder = reinterpret_cast<onnxruntime::KernelDefBuilder*>(kernel_def_builder);
  builder->OutputMemoryType(mem_type, static_cast<int>(output_index));
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(KernelDefBuilder_AddTypeConstraint, _In_ OrtKernelDefBuilder* kernel_def_builder,
                    _In_ const char* arg_name, _In_reads_(num_types) const OrtDataType* const* types,
                    _In_ size_t num_types) {
  API_IMPL_BEGIN
  if (num_types == 0) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Must specify at least one OrtDataType instance");
  }

  if (types == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Must specify a valid array of OrtDataType instances");
  }

  if (arg_name == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 "Must specify a valid name for a kernel definition's type constraint");
  }

  std::vector<onnxruntime::MLDataType> ml_types;
  ml_types.reserve(num_types);

  for (size_t i = 0; i < num_types; i++) {
    ml_types.push_back(reinterpret_cast<const onnxruntime::DataTypeImpl*>(types[i]));
  }

  auto* builder = reinterpret_cast<onnxruntime::KernelDefBuilder*>(kernel_def_builder);
  builder->TypeConstraint(arg_name, std::move(ml_types));
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(KernelDefBuilder_AddInputOutputAliases, _In_ OrtKernelDefBuilder* kernel_def_builder,
                    _In_reads_(num_io_indices) int const* input_indices,
                    _In_reads_(num_io_indices) int const* output_indices,
                    _In_ size_t num_io_indices) {
  API_IMPL_BEGIN
  if (num_io_indices == 0) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Must specify at least one input/output alias");
  }

  if (input_indices == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Must specify a valid array of input indices to alias");
  }

  if (output_indices == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Must specify a valid array of output indices to alias");
  }

  auto* builder = reinterpret_cast<onnxruntime::KernelDefBuilder*>(kernel_def_builder);

  if (num_io_indices == 1) {
    builder->Alias(input_indices[0], output_indices[0]);
  } else {
    std::vector<std::pair<int, int>> pairs;
    pairs.reserve(num_io_indices);

    for (size_t i = 0; i < num_io_indices; ++i) {
      pairs.push_back({input_indices[i], output_indices[i]});
    }

    builder->Alias(pairs);
  }

  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(KernelDefBuilder_AddInputOutputMutableAliases, _In_ OrtKernelDefBuilder* kernel_def_builder,
                    _In_reads_(num_io_indices) int const* input_indices,
                    _In_reads_(num_io_indices) int const* output_indices,
                    _In_ size_t num_io_indices) {
  API_IMPL_BEGIN
  if (num_io_indices == 0) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Must specify at least one input/output alias (mutable)");
  }

  if (input_indices == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 "Must specify a valid array of input indices to alias (mutable)");
  }

  if (output_indices == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 "Must specify a valid array of output indices to alias (mutable)");
  }

  auto* builder = reinterpret_cast<onnxruntime::KernelDefBuilder*>(kernel_def_builder);

  if (num_io_indices == 1) {
    builder->MayInplace(input_indices[0], output_indices[0]);
  } else {
    std::vector<std::pair<int, int>> pairs;
    pairs.reserve(num_io_indices);

    for (size_t i = 0; i < num_io_indices; ++i) {
      pairs.push_back({input_indices[i], output_indices[i]});
    }

    builder->MayInplace(pairs);
  }

  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(KernelDefBuilder_Build, _In_ OrtKernelDefBuilder* kernel_def_builder,
                    _Outptr_ OrtKernelDef** kernel_def_out) {
  API_IMPL_BEGIN
  auto* builder = reinterpret_cast<onnxruntime::KernelDefBuilder*>(kernel_def_builder);
  *kernel_def_out = reinterpret_cast<OrtKernelDef*>(builder->Build().release());
  return nullptr;
  API_IMPL_END
}

ORT_API(void, ReleaseKernelDef, _Frees_ptr_opt_ OrtKernelDef* kernel_def) {
  delete reinterpret_cast<onnxruntime::KernelDef*>(kernel_def);
}

ORT_API(const char*, KernelDef_GetOperatorType, _In_ const OrtKernelDef* kernel_def) {
  return reinterpret_cast<const onnxruntime::KernelDef*>(kernel_def)->OpName().c_str();
}

ORT_API(const char*, KernelDef_GetDomain, _In_ const OrtKernelDef* kernel_def) {
  return reinterpret_cast<const onnxruntime::KernelDef*>(kernel_def)->Domain().c_str();
}

ORT_API_STATUS_IMPL(KernelDef_GetSinceVersion, _In_ const OrtKernelDef* kernel_def,
                    _Out_ int* start_version, _Out_ int* end_version) {
  API_IMPL_BEGIN
  if (kernel_def == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Must specify a valid non-null OrtKernelDef");
  }

  if (start_version == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Must specify a non-null `start_version` output parameter");
  }

  if (end_version == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Must specify a non-null `end_version` output parameter");
  }

  auto* internal_kernel_def = reinterpret_cast<const onnxruntime::KernelDef*>(kernel_def);
  internal_kernel_def->SinceVersion(start_version, end_version);

  return nullptr;
  API_IMPL_END
}

ORT_API(const char*, KernelDef_GetExecutionProvider, _In_ const OrtKernelDef* kernel_def) {
  return reinterpret_cast<const onnxruntime::KernelDef*>(kernel_def)->Provider().c_str();
}

ORT_API_STATUS_IMPL(KernelDef_GetInputMemType, _In_ const OrtKernelDef* kernel_def,
                    _In_ size_t input_index, _Out_ OrtMemType* mem_type) {
  API_IMPL_BEGIN
  if (kernel_def == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Must specify a valid non-null OrtKernelDef");
  }

  if (mem_type == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Must specify a non-null `mem_type` output parameter");
  }

  auto* internal_kernel_def = reinterpret_cast<const onnxruntime::KernelDef*>(kernel_def);
  *mem_type = internal_kernel_def->InputMemoryType(input_index);

  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(KernelDef_GetOutputMemType, _In_ const OrtKernelDef* kernel_def,
                    _In_ size_t output_index, _Out_ OrtMemType* mem_type) {
  API_IMPL_BEGIN
  if (kernel_def == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Must specify a valid non-null OrtKernelDef");
  }

  if (mem_type == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Must specify a non-null `mem_type` output parameter");
  }

  auto* internal_kernel_def = reinterpret_cast<const onnxruntime::KernelDef*>(kernel_def);
  *mem_type = internal_kernel_def->OutputMemoryType(output_index);

  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(GetTensorDataType, _In_ ONNXTensorElementDataType elem_type,
                    _Outptr_ const OrtDataType** out) {
  API_IMPL_BEGIN
  const DataTypeImpl* ml_type = DataTypeImpl::TensorTypeFromONNXEnum(elem_type);
  *out = reinterpret_cast<const OrtDataType*>(ml_type);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(EpGraphSupportInfo_LookUpKernel, _In_ OrtEpGraphSupportInfo* graph_support_info,
                    _In_ const OrtNode* node, _Outptr_result_maybenull_ const OrtKernelDef** out_kernel_def) {
  API_IMPL_BEGIN
  if (out_kernel_def == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Must specify a non-null OrtKernelDef output parameter");
  }

  *out_kernel_def = nullptr;

  if (graph_support_info == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Must specify a valid non-null OrtEpGraphSupportInfo instance");
  }

  if (node == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Must specify a valid non-null OrtNode instance");
  }

  const onnxruntime::EpNode* ep_node = onnxruntime::EpNode::ToInternal(node);
  if (ep_node == nullptr) {
    return OrtApis::CreateStatus(OrtErrorCode::ORT_INVALID_ARGUMENT,
                                 "OrtNode created via the ModelEditor API is not supported");
  }

  const onnxruntime::KernelCreateInfo* create_info =
      graph_support_info->kernel_lookup.LookUpKernel(ep_node->GetInternalNode());

  *out_kernel_def = create_info != nullptr ? reinterpret_cast<const OrtKernelDef*>(create_info->kernel_def.get())
                                           : nullptr;
  return nullptr;
  API_IMPL_END
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

    &OrtExecutionProviderApi::SyncStream_GetImpl,
    &OrtExecutionProviderApi::SyncStream_GetSyncId,
    &OrtExecutionProviderApi::GetSyncIdForLastWaitOnSyncStream,
    // End of Version 23 - DO NOT MODIFY ABOVE

    &OrtExecutionProviderApi::CreateHardwareDevice,
    &OrtExecutionProviderApi::ReleaseHardwareDevice,

    &OrtExecutionProviderApi::CreateKernelRegistry,
    &OrtExecutionProviderApi::ReleaseKernelRegistry,
    &OrtExecutionProviderApi::KernelRegistry_AddKernel,
    &OrtExecutionProviderApi::CreateKernelDefBuilder,
    &OrtExecutionProviderApi::ReleaseKernelDefBuilder,
    &OrtExecutionProviderApi::KernelDefBuilder_SetOperatorType,
    &OrtExecutionProviderApi::KernelDefBuilder_SetDomain,
    &OrtExecutionProviderApi::KernelDefBuilder_SetSinceVersion,
    &OrtExecutionProviderApi::KernelDefBuilder_SetExecutionProvider,
    &OrtExecutionProviderApi::KernelDefBuilder_SetInputMemType,
    &OrtExecutionProviderApi::KernelDefBuilder_SetOutputMemType,
    &OrtExecutionProviderApi::KernelDefBuilder_AddTypeConstraint,
    &OrtExecutionProviderApi::KernelDefBuilder_AddInputOutputAliases,
    &OrtExecutionProviderApi::KernelDefBuilder_AddInputOutputMutableAliases,
    &OrtExecutionProviderApi::KernelDefBuilder_Build,
    &OrtExecutionProviderApi::ReleaseKernelDef,
    &OrtExecutionProviderApi::KernelDef_GetOperatorType,
    &OrtExecutionProviderApi::KernelDef_GetDomain,
    &OrtExecutionProviderApi::KernelDef_GetSinceVersion,
    &OrtExecutionProviderApi::KernelDef_GetExecutionProvider,
    &OrtExecutionProviderApi::KernelDef_GetInputMemType,
    &OrtExecutionProviderApi::KernelDef_GetOutputMemType,
    &OrtExecutionProviderApi::GetTensorDataType,
    &OrtExecutionProviderApi::EpGraphSupportInfo_LookUpKernel,
};

// checks that we don't violate the rule that the functions must remain in the slots they were originally assigned
static_assert(offsetof(OrtEpApi, ReleaseEpDevice) / sizeof(void*) == 1,
              "Size of version 22 API cannot change");  // initial version in ORT 1.22
static_assert(offsetof(OrtEpApi, GetSyncIdForLastWaitOnSyncStream) / sizeof(void*) == 15,
              "Size of version 23 API cannot change");

}  // namespace OrtExecutionProviderApi

ORT_API(const OrtEpApi*, OrtExecutionProviderApi::GetEpApi) {
  return &ort_ep_api;
}
