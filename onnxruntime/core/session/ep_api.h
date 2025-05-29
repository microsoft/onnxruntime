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

ORT_API_STATUS_IMPL(EpGraphSupportInfo_AddSubgraph, _In_ OrtEpGraphSupportInfo* graph_support_info,
                    _In_ const char* subgraph_name,
                    _In_ const OrtHardwareDevice* hardware_device,
                    _In_reads_(num_supported_nodes) const OrtNode* const* supported_nodes,
                    size_t num_supported_nodes);

}  // namespace OrtExecutionProviderApi
