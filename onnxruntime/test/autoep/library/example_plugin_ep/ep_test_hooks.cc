// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#define EXAMPLE_PLUGIN_EP_BUILD
#include "ep_test_hooks.h"
#include <atomic>

std::atomic<uint64_t> g_sync_count{0};
std::atomic<const OrtHardwareDevice*> g_fused_node_hardware_device{nullptr};

extern "C" void ExampleEpTestHooks_ResetSyncCount() { g_sync_count.store(0); }
extern "C" uint64_t ExampleEpTestHooks_GetSyncCount() { return g_sync_count.load(); }
extern "C" void ExampleEpTestHooks_SetFusedNodeHardwareDevice(const OrtHardwareDevice* device) {
  g_fused_node_hardware_device.store(device);
}