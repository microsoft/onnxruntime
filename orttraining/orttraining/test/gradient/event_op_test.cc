// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <bitset>
#include <cmath>
#include <random>
#include <thread>

#include "gtest/gtest.h"

#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/test_random_seed.h"
#include "test/util/include/default_providers.h"

#include "onnx/defs/attr_proto_util.h"

namespace onnxruntime {
namespace test {

// Run GPU op for GPU build. Otherwise, run GPU op.
void run_provider_specific_optest(OpTester& tester) {
  RunOptions run_option;
#ifdef USE_CUDA
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCudaExecutionProvider());
#else 
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCpuExecutionProvider());
#endif
  tester.Run(
      OpTester::ExpectResult::kExpectSuccess,
      "",
      std::unordered_set<std::string>(),
      &run_option,
      &providers);
}

void record_event(int64_t event_id) {
  OpTester test_record("RecordEvent", 1, onnxruntime::kMSDomain);
  test_record.AddInput<int64_t>("EventIdentifier", {}, {event_id});
  test_record.AddInput<bool>("InputSignal", {}, {true});
  test_record.AddOutput<bool>("OutputSignal", {}, {true});
  run_provider_specific_optest(test_record);
}

void record_event_multiple_inputs_and_outputs(int64_t event_id) {
  OpTester test_record("RecordEvent", 1, onnxruntime::kMSDomain);
  test_record.AddInput<int64_t>("EventIdentifier", {}, {event_id});
  test_record.AddInput<bool>("InputSignal", {}, {true});
  test_record.AddInput<float>("Input1", {3}, {9.4f, 1.7f, 3.6f});
  test_record.AddInput<float>("Input2", {1}, {1.6f});
  test_record.AddOutput<bool>("OutputSignal", {}, {true});
  test_record.AddOutput<float>("Output1", {3}, {9.4f, 1.7f, 3.6f});
  test_record.AddOutput<float>("Output2", {1}, {1.6f});
  run_provider_specific_optest(test_record);
}

void wait_event(int64_t event_id) {
  OpTester test_wait("WaitEvent", 1, onnxruntime::kMSDomain);
  test_wait.AddInput<int64_t>("EventIdentifier", {}, {event_id});
  test_wait.AddInput<bool>("InputSignal", {}, {true});
  test_wait.AddOutput<bool>("OutputSignal", {}, {true});
  run_provider_specific_optest(test_wait);
}

void wait_event_multiple_inputs_and_outputs(int64_t event_id) {
  OpTester test_wait("WaitEvent", 1, onnxruntime::kMSDomain);
  test_wait.AddInput<int64_t>("EventIdentifier", {}, {event_id});
  test_wait.AddInput<bool>("InputSignal", {}, {true});
  test_wait.AddInput<float>("Input1", {1}, {1.6f});
  test_wait.AddInput<float>("Input2", {3}, {9.4f, 1.7f, 3.6f});
  test_wait.AddOutput<bool>("OutputSignal", {}, {true});
  test_wait.AddOutput<float>("output1", {1}, {1.6f});
  test_wait.AddOutput<float>("output2", {3}, {9.4f, 1.7f, 3.6f});
  run_provider_specific_optest(test_wait);
}

TEST(Synchronization, RecordAndWaitEvent) {
  const int64_t event_id = static_cast<int64_t>(1736);
  record_event(event_id);
  wait_event(event_id);
}

TEST(Synchronization, WaitNullEvent) {
  wait_event(-1);
}

TEST(Synchronization, RecordAndWaitEventMultipleInputsAndOutputs) {
  const int64_t event_id = static_cast<int64_t>(995);
  record_event_multiple_inputs_and_outputs(event_id);
  wait_event_multiple_inputs_and_outputs(event_id);
}

TEST(Synchronization, WaitAndRecordEvent) {
  const int64_t event_id = static_cast<int64_t>(1228);
  std::thread waiting_thread(wait_event, event_id);
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  std::thread recording_thread(record_event, event_id);

  waiting_thread.join();
  recording_thread.join();
}

TEST(Synchronization, WaitAndRecordEventMany) {
  const size_t event_count = 16;
  for (int i = 0; i < 8; ++i) {
    std::thread thread_pool[2 * event_count];
    for (int j = 0; j < static_cast<int>(event_count); ++j) {
      thread_pool[j] = std::thread(wait_event, j);
      thread_pool[j + event_count] = std::thread(record_event, j);
    }
    for (size_t j = 0; j < event_count; ++j) {
      thread_pool[j].join();
      thread_pool[j + event_count].join();
    }
  }
}

}  // namespace test
}  // namespace onnxruntime