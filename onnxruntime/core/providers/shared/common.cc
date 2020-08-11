// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common.h"

void* g_host{};

void* Provider_GetHost() {
  return g_host;
}

void Provider_SetHost(void* p) {
  g_host = p;
}
