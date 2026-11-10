// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#define EXAMPLE_PLUGIN_EP_BUILD
#include "ep_test_hooks.h"
#include <atomic>
#include <array>

std::atomic<uint64_t> g_sync_count{0};
std::atomic<int> g_preallocated_output_query_result{-1};
std::atomic<int> g_preallocated_output_bad_index_rejected{-1};
std::atomic<bool> g_create_data_transfer_should_fail{false};
namespace {
std::array<std::atomic<uint64_t>, 7> stateful_calls{};
std::atomic<int> stateful_failure_operation{-1};
std::atomic<int> stateful_failure_code{0};
std::atomic<uint64_t> legacy_size{0};
void* stateful_observer_state = nullptr;
ExampleStatefulObserver stateful_observer = nullptr;
}  // namespace

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
extern "C" void ExampleEpTestHooks_SetCreateDataTransferFailure(int enabled) {
  g_create_data_transfer_should_fail.store(enabled != 0);
}
void RecordPreallocatedOutputQueryResult(int has_preallocated_output) {
  g_preallocated_output_query_result.store(has_preallocated_output);
}
void RecordPreallocatedOutputBadIndexRejected(int rejected) {
  g_preallocated_output_bad_index_rejected.store(rejected);
}
bool ShouldFailCreateDataTransfer() {
  return g_create_data_transfer_should_fail.load();
}

extern "C" void ExampleEpTestHooks_ResetStateful() {
  for (auto& count : stateful_calls) count.store(0);
  stateful_failure_operation.store(-1);
  stateful_failure_code.store(0);
  legacy_size.store(0);
}
extern "C" uint64_t ExampleEpTestHooks_GetStatefulCalls(int operation) {
  return stateful_calls.at(static_cast<size_t>(operation)).load();
}
extern "C" void ExampleEpTestHooks_SetStatefulFailure(int operation, int code) {
  stateful_failure_operation.store(operation);
  stateful_failure_code.store(code);
}
extern "C" void ExampleEpTestHooks_SetStatefulObserver(void* state, ExampleStatefulObserver observer) {
  stateful_observer_state = state;
  stateful_observer = observer;
}
extern "C" uint64_t ExampleEpTestHooks_GetLegacySize() { return legacy_size.load(); }
void RecordStatefulCall(int operation) { ++stateful_calls.at(static_cast<size_t>(operation)); }
int StatefulFailureOperation() { return stateful_failure_operation.load(); }
int StatefulFailureCode() { return stateful_failure_code.load(); }
void RecordStatefulLifetime(int kind) {
  if (stateful_observer) stateful_observer(stateful_observer_state, kind);
}
void RecordLegacySize(uint64_t size) { legacy_size.store(size); }
