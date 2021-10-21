// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common.h"

onnxruntime::ProviderHost* g_host{};

onnxruntime::ProviderHost* Provider_GetHost() {
  return g_host;
}

void Provider_SetHost(onnxruntime::ProviderHost* p) {
  g_host = p;
}
