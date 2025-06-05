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

ORT_API_STATUS_IMPL(EpGraphSupportInfo_AddFusedNodes, _In_ OrtEpGraphSupportInfo* graph_support_info,
                    _In_reads_(num_nodes) const OrtNode* const* nodes, size_t num_nodes,
                    _In_ const OrtHardwareDevice* const* hardware_devices, size_t num_devices);
ORT_API_STATUS_IMPL(EpGraphSupportInfo_AddSingleNode, _In_ OrtEpGraphSupportInfo* graph_support_info,
                    _In_ const OrtNode* node, _In_ const OrtHardwareDevice* hardware_device);
ORT_API(const char*, NodeComputeContext_NodeName, _In_ const OrtNodeComputeContext* context);

}  // namespace OrtExecutionProviderApi
