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

inline constexpr const char* kExampleEpTestEpContextDataSupport =
    "ep.example_ep.test_ep_context_data_support";
inline constexpr const char* kExampleEpTestOrtVersion = "ep.example_ep.test_ort_version";
inline constexpr const char* kExampleEpTestStatefulMode = "ep.example_ep.test_stateful_mode";
using ExampleStatefulObserver = void (*)(void*, int);

extern "C" {
EXPORT_SYMBOL void ExampleEpTestHooks_ResetSyncCount();
EXPORT_SYMBOL uint64_t ExampleEpTestHooks_GetSyncCount();
EXPORT_SYMBOL void ExampleEpTestHooks_ResetPreallocatedOutputQuery();
EXPORT_SYMBOL int ExampleEpTestHooks_GetPreallocatedOutputQueryResult();
EXPORT_SYMBOL int ExampleEpTestHooks_GetPreallocatedOutputBadIndexRejected();
EXPORT_SYMBOL void ExampleEpTestHooks_SetCreateDataTransferFailure(int enabled);
EXPORT_SYMBOL void ExampleEpTestHooks_ResetStateful();
EXPORT_SYMBOL uint64_t ExampleEpTestHooks_GetStatefulCalls(int operation);
EXPORT_SYMBOL void ExampleEpTestHooks_SetStatefulFailure(int operation, int code);
EXPORT_SYMBOL void ExampleEpTestHooks_SetStatefulObserver(void* state, ExampleStatefulObserver observer);
EXPORT_SYMBOL uint64_t ExampleEpTestHooks_GetLegacySize();
}

// Internal to the library; not exported.
void RecordPreallocatedOutputQueryResult(int has_preallocated_output);
void RecordPreallocatedOutputBadIndexRejected(int rejected);
bool ShouldFailCreateDataTransfer();
void RecordStatefulCall(int operation);
int StatefulFailureOperation();
int StatefulFailureCode();
void RecordStatefulLifetime(int kind);
void RecordLegacySize(uint64_t size);
