// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/session/onnxruntime_c_api.h"

namespace OrtExecutionProviderApi {
// implementation that returns the API struct
ORT_API(const OrtEpApi*, GetEpApi);

ORT_API_STATUS_IMPL(CreateEpDevice, _In_ OrtEpFactory* ep_factory,
                    _In_ const OrtHardwareDevice* hardware_device,
                    _In_opt_ const OrtKeyValuePairs* ep_metadata,
                    _In_opt_ const OrtKeyValuePairs* ep_options,
                    _Out_ OrtEpDevice** ep_device);

ORT_API(void, ReleaseEpDevice, _Frees_ptr_opt_ OrtEpDevice* device);

ORT_API_STATUS_IMPL(EpGraphSupportInfo_AddNodesToFuse, _In_ OrtEpGraphSupportInfo* graph_support_info,
                    _In_reads_(num_nodes) const OrtNode* const* nodes, _In_ size_t num_nodes,
                    _In_opt_ const OrtNodeFusionOptions* node_fusion_options);
ORT_API_STATUS_IMPL(EpGraphSupportInfo_AddSingleNode, _In_ OrtEpGraphSupportInfo* graph_support_info,
                    _In_ const OrtNode* node);
ORT_API(const char*, NodeComputeContext_NodeName, _In_ const OrtNodeComputeContext* context);

ORT_API_STATUS_IMPL(EpDevice_AddAllocatorInfo, _In_ OrtEpDevice* ep_device,
                    _In_ const OrtMemoryInfo* allocator_memory_info);

ORT_API(const OrtMemoryDevice*, MemoryInfo_GetMemoryDevice, _In_ const OrtMemoryInfo* memory_info);
ORT_API(const OrtMemoryDevice*, Value_GetMemoryDevice, _In_ const OrtValue* value);

ORT_API(bool, MemoryDevice_AreEqual, _In_ const OrtMemoryDevice* a, _In_ const OrtMemoryDevice* b);
ORT_API(OrtMemoryInfoDeviceType, MemoryDevice_GetDeviceType, _In_ const OrtMemoryDevice* memory_device);
ORT_API(OrtDeviceMemoryType, MemoryDevice_GetMemoryType, _In_ const OrtMemoryDevice* memory_device);
ORT_API(uint32_t, MemoryDevice_GetVendorId, _In_ const OrtMemoryDevice* memory_device);
ORT_API(uint32_t, MemoryDevice_GetDeviceId, _In_ const OrtMemoryDevice* memory_device);

ORT_API(const OrtSyncStreamImpl*, SyncStream_GetImpl, _In_ const OrtSyncStream* stream);
ORT_API(uint64_t, SyncStream_GetSyncId, _In_ const OrtSyncStream* stream);
ORT_API(uint64_t, GetSyncIdForLastWaitOnSyncStream, _In_ const OrtSyncStream* producer_stream,
        _In_ const OrtSyncStream* consumer_stream);

ORT_API_STATUS_IMPL(CreateKernelCreationInfo, _In_ const OrtKernelDef* kernel_def,
                    _In_ OrtKernelCreateFunc kernel_create_func,
                    _In_ void* ep_state,
                    _Outptr_ OrtKernelCreateInfo** kernel_create_info_out);
ORT_API(void, ReleaseKernelCreateInfo, _Frees_ptr_opt_ OrtKernelCreateInfo* kernel_create_info);

ORT_API_STATUS_IMPL(CreateKernelDefBuilder, _Outptr_ OrtKernelDefBuilder** kernel_def_builder_out);
ORT_API(void, ReleaseKernelDefBuilder, _Frees_ptr_opt_ OrtKernelDefBuilder* kernel_def_builder);
ORT_API_STATUS_IMPL(KernelDefBuilder_SetOperatorType, _In_ OrtKernelDefBuilder* kernel_def_builder,
                    _In_ const char* op_type);
ORT_API_STATUS_IMPL(KernelDefBuilder_SetDomain, _In_ OrtKernelDefBuilder* kernel_def_builder,
                    _In_ const char* domain);
ORT_API_STATUS_IMPL(KernelDefBuilder_SetSinceVersion, _In_ OrtKernelDefBuilder* kernel_def_builder,
                    _In_ int since_version_start, _In_ int since_version_end);
ORT_API_STATUS_IMPL(KernelDefBuilder_SetExecutionProvider, _In_ OrtKernelDefBuilder* kernel_def_builder,
                    _In_ const char* ep_name);
ORT_API_STATUS_IMPL(KernelDefBuilder_SetInputMemType, _In_ OrtKernelDefBuilder* kernel_def_builder,
                    _In_ size_t input_index, _In_ OrtMemType mem_type);
ORT_API_STATUS_IMPL(KernelDefBuilder_SetOutputMemType, _In_ OrtKernelDefBuilder* kernel_def_builder,
                    _In_ size_t output_index, _In_ OrtMemType mem_type);
ORT_API_STATUS_IMPL(KernelDefBuilder_AddTypeConstraint, _In_ OrtKernelDefBuilder* kernel_def_builder,
                    _In_ const char* arg_name, _In_reads_(num_types) const OrtMLDataType* const* types,
                    _In_ size_t num_types);
ORT_API_STATUS_IMPL(KernelDefBuilder_Build, _In_ OrtKernelDefBuilder* kernel_def_builder,
                    _Outptr_ OrtKernelDef** kernel_def_out);

ORT_API(void, ReleaseKernelDef, _Frees_ptr_opt_ OrtKernelDef* kernel_def);
ORT_API_STATUS_IMPL(GetTensorMLDataType, _In_ ONNXTensorElementDataType elem_type,
                    _Outptr_ const OrtMLDataType** out);

}  // namespace OrtExecutionProviderApi
