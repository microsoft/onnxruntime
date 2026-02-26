// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Provider host bridge — legacy plugin path only.
// Initializes g_host, which provider_api.h's stub functions delegate to.
// In the adapter path (ORT_CUDA_PLUGIN_USE_ADAPTER), the bridge is
// bypassed entirely because kernels link against real framework symbols.

#ifndef ORT_CUDA_PLUGIN_USE_ADAPTER
#include "core/providers/shared/common.h"

namespace onnxruntime {

// Bridge provider-wrapped types to the host exported via Provider_GetHost.
ProviderHost* g_host = Provider_GetHost();

}  // namespace onnxruntime
#endif
