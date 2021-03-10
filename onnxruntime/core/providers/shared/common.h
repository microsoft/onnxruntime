// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

namespace onnxruntime {
struct ProviderHost;
}

extern "C" {
onnxruntime::ProviderHost* Provider_GetHost();
void Provider_SetHost(onnxruntime::ProviderHost*);
}
