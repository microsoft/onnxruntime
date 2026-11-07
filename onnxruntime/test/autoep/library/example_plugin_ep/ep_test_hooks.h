// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <cstddef>
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

inline constexpr const char* kExampleEpTestEpContextDataSupport =
    "ep.example_ep.test_ep_context_data_support";
inline constexpr const char* kExampleEpTestOrtVersion = "ep.example_ep.test_ort_version";

extern "C" {
EXPORT_SYMBOL void ExampleEpTestHooks_ResetSyncCount();
EXPORT_SYMBOL uint64_t ExampleEpTestHooks_GetSyncCount();
EXPORT_SYMBOL void ExampleEpTestHooks_ResetPreallocatedOutputQuery();
EXPORT_SYMBOL int ExampleEpTestHooks_GetPreallocatedOutputQueryResult();
EXPORT_SYMBOL int ExampleEpTestHooks_GetPreallocatedOutputBadIndexRejected();
EXPORT_SYMBOL void ExampleEpTestHooks_SetCreateDataTransferFailure(int enabled);
EXPORT_SYMBOL int ExampleEpTestHooks_GetDlpackCurrentDevice();
EXPORT_SYMBOL void ExampleEpTestHooks_SetDlpackCurrentDevice(int device_id);
EXPORT_SYMBOL std::size_t ExampleEpTestHooks_GetDlpackDeviceEntryCount();
EXPORT_SYMBOL int ExampleEpTestHooks_GetDlpackSyncCount();
EXPORT_SYMBOL int ExampleEpTestHooks_GetDlpackReleaseCount();
EXPORT_SYMBOL void ExampleEpTestHooks_ResetDlpack();
EXPORT_SYMBOL void ExampleEpTestHooks_SetDlpackSyncBlocked(int blocked);
}

// Internal to the library; not exported.
void RecordPreallocatedOutputQueryResult(int has_preallocated_output);
void RecordPreallocatedOutputBadIndexRejected(int rejected);
bool ShouldFailCreateDataTransfer();
