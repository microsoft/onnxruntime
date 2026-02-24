// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared/common.h"

namespace onnxruntime {

// Bridge provider-wrapped types to the host exported via Provider_GetHost.
ProviderHost* g_host = Provider_GetHost();

}  // namespace onnxruntime
