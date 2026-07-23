// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <cstdint>

struct OrtHardwareDevice;

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

// Sets the OrtHardwareDevice that the example EP will attach to the fused Mul node via
// OrtNodeFusionOptions::fused_node_hardware_device in its next GetCapability() call. Pass nullptr to clear.
// Used to test EpAssignedSubgraph_GetHardwareDevices resolution of a plugin-declared per-subgraph device.
EXPORT_SYMBOL void ExampleEpTestHooks_SetFusedNodeHardwareDevice(const OrtHardwareDevice* device);
}
