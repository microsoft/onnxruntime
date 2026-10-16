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
// Overrides the OrtWeightlessSupport value returned by GetWeightlessSupport() (base-10 integer), to make it disagree
// with the EP metadata.
inline constexpr const char* kExampleEpTestWeightlessSupport = "ep.example_ep.test_weightless_support";

extern "C" {
EXPORT_SYMBOL void ExampleEpTestHooks_ResetSyncCount();
EXPORT_SYMBOL uint64_t ExampleEpTestHooks_GetSyncCount();
EXPORT_SYMBOL void ExampleEpTestHooks_ResetPreallocatedOutputQuery();
EXPORT_SYMBOL int ExampleEpTestHooks_GetPreallocatedOutputQueryResult();
EXPORT_SYMBOL int ExampleEpTestHooks_GetPreallocatedOutputBadIndexRejected();
EXPORT_SYMBOL void ExampleEpTestHooks_SetCreateDataTransferFailure(int enabled);
// Value of the deprecated "ep.enable_weightless" session option seen by the last CreateEp() call:
// -2: CreateEp() not called since the last reset, -1: not set, 0: "0", 1: "1", 2: any other value.
EXPORT_SYMBOL void ExampleEpTestHooks_ResetEnableWeightlessOption();
EXPORT_SYMBOL int ExampleEpTestHooks_GetEnableWeightlessOption();
// Number of constant initializers the EP copied since the last reset.
EXPORT_SYMBOL void ExampleEpTestHooks_ResetSavedInitializerCount();
EXPORT_SYMBOL uint64_t ExampleEpTestHooks_GetSavedInitializerCount();
// Weightless source model buffer seen by the last CreateEp() call (nullptr and 0 if not set or after a reset).
EXPORT_SYMBOL void ExampleEpTestHooks_ResetWeightlessSourceModelBuffer();
EXPORT_SYMBOL void ExampleEpTestHooks_GetWeightlessSourceModelBuffer(const void** data, size_t* length);
}

// Internal to the library; not exported.
void RecordPreallocatedOutputQueryResult(int has_preallocated_output);
void RecordPreallocatedOutputBadIndexRejected(int rejected);
bool ShouldFailCreateDataTransfer();
void RecordEnableWeightlessOption(int value);
void RecordSavedInitializer();
void RecordWeightlessSourceModelBuffer(const void* data, size_t length);
