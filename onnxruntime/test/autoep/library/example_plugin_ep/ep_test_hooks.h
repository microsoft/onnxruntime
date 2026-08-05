// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <cstdint>

// Export visibility
#if defined(_WIN32)
#ifdef EXAMPLE_PLUGIN_EP_BUILD
#define EXPORT_SYMBOL __declspec(dllexport)
#else
#define EXPORT_SYMBOL __declspec(dllimport)
#endif
#elif defined(__APPLE__)
#define EXPORT_SYMBOL __attribute__((visibility("default")))
#else
#define EXPORT_SYMBOL
#endif

extern "C" {
EXPORT_SYMBOL void ExampleEpTestHooks_ResetSyncCount();
EXPORT_SYMBOL uint64_t ExampleEpTestHooks_GetSyncCount();
}
