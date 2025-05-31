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

ORT_API_STATUS_IMPL(EpGraphSupportInfo_AddSupportedNodes, _In_ OrtEpGraphSupportInfo* graph_support_info,
                    _In_reads_(num_supported_nodes) const OrtNode* const* supported_nodes,
                    size_t num_supported_nodes,
                    _In_ const OrtHardwareDevice* hardware_device);
ORT_API_STATUS_IMPL(Graph_GetNumNodes, _In_ const OrtGraph* graph, _Out_ size_t* num_nodes);
ORT_API_STATUS_IMPL(Graph_GetNodes, const OrtGraph* graph, int order,
                    _Out_writes_all_(max_num_nodes) const OrtNode** nodes, _In_ size_t max_num_nodes);
ORT_API_STATUS_IMPL(Node_GetName, const OrtNode* node, _Outptr_ const char** name);
ORT_API_STATUS_IMPL(Node_GetOperatorType, const OrtNode* node, _Outptr_ const char** op_type);

ORT_API(const char*, NodeComputeContext_NodeName, _In_ const OrtNodeComputeContext* context);

}  // namespace OrtExecutionProviderApi
