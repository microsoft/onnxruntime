// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#define EXAMPLE_PLUGIN_EP_BUILD
#include "ep_test_hooks.h"
#include <atomic>

std::atomic<uint64_t> g_sync_count{0};
std::atomic<int> g_preallocated_output_query_result{-1};
std::atomic<int> g_preallocated_output_bad_index_rejected{-1};

extern "C" void ExampleEpTestHooks_ResetSyncCount() { g_sync_count.store(0); }
extern "C" uint64_t ExampleEpTestHooks_GetSyncCount() { return g_sync_count.load(); }
extern "C" void ExampleEpTestHooks_ResetPreallocatedOutputQuery() {
  g_preallocated_output_query_result.store(-1);
  g_preallocated_output_bad_index_rejected.store(-1);
}
extern "C" int ExampleEpTestHooks_GetPreallocatedOutputQueryResult() {
  return g_preallocated_output_query_result.load();
}
extern "C" int ExampleEpTestHooks_GetPreallocatedOutputBadIndexRejected() {
  return g_preallocated_output_bad_index_rejected.load();
}
void RecordPreallocatedOutputQueryResult(int has_preallocated_output) {
  g_preallocated_output_query_result.store(has_preallocated_output);
}
void RecordPreallocatedOutputBadIndexRejected(int rejected) {
  g_preallocated_output_bad_index_rejected.store(rejected);
}
