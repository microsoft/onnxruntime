// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/threadpool.h"

#include <core/common/make_unique.h>

#include "gtest/gtest.h"
#include <algorithm>
#include <memory>
#include <functional>
#include <mutex>

using namespace onnxruntime::concurrency;

namespace {

struct TestData {
  explicit TestData(int num) : data(num, 0) {}
  std::vector<int> data;
  std::mutex mutex;
};

// This unittest tests ThreadPool function by counting the number of calls to function with each index.
// the function should be called exactly once for each element.

std::unique_ptr<TestData> CreateTestData(int num) {
  return onnxruntime::make_unique<TestData>(num);
}

void IncrementElement(TestData& test_data, int i) {
  std::lock_guard<std::mutex> lock(test_data.mutex);
  test_data.data[i]++;
}

void ValidateTestData(TestData& test_data) {
  ASSERT_TRUE(std::count_if(test_data.data.cbegin(),
                            test_data.data.cend(),
                            [](int i) { return i != 1; }) == 0);
}

void CreateThreadPoolAndTest(const std::string& name, int num_threads, const std::function<void(ThreadPool*)>& test_body) {
  auto tp = onnxruntime::make_unique<ThreadPool>(name, num_threads);
  test_body(tp.get());
}

void TestParallelFor(const std::string& name, int num_threads, int num_tasks) {
  auto test_data = CreateTestData(num_tasks);
  CreateThreadPoolAndTest(name, num_threads, [&](ThreadPool* tp) {
    tp->ParallelFor(num_tasks, [&](int i) {
      IncrementElement(*test_data, i);
    });
  });
  ValidateTestData(*test_data);
}

void TestBatchParallelFor(const std::string& name, int num_threads, int num_tasks, int batch_size) {
  auto test_data = CreateTestData(num_tasks);
  CreateThreadPoolAndTest(name, num_threads, [&](ThreadPool* tp) {
    tp->BatchParallelFor(
        num_tasks, [&](int i) {
          IncrementElement(*test_data, i);
        },
        batch_size);
  });
  ValidateTestData(*test_data);
}

}  // namespace

TEST(ThreadPoolTest, TestParallelFor_2_Thread_NoTask) {
  TestParallelFor("TestParallelFor_2_Thread_NoTask", 2, 0);
}

TEST(ThreadPoolTest, TestParallelFor_2_Thread_50_Task) {
  TestParallelFor("TestParallelFor_2_Thread_50_Task", 2, 50);
}

TEST(ThreadPoolTest, TestParallelFor_1_Thread_50_Task) {
  TestParallelFor("TestParallelFor_1_Thread_50_Task", 1, 50);
}

TEST(ThreadPoolTest, TestBatchParallelFor_2_Thread_50_Task_10_Batch) {
  TestBatchParallelFor("TestBatchParallelFor_2_Thread_50_Task_10_Batch", 2, 50, 10);
}

TEST(ThreadPoolTest, TestBatchParallelFor_2_Thread_50_Task_0_Batch) {
  TestBatchParallelFor("TestBatchParallelFor_2_Thread_50_Task_0_Batch", 2, 50, 0);
}

TEST(ThreadPoolTest, TestBatchParallelFor_2_Thread_50_Task_1_Batch) {
  TestBatchParallelFor("TestBatchParallelFor_2_Thread_50_Task_1_Batch", 2, 50, 1);
}

TEST(ThreadPoolTest, TestBatchParallelFor_2_Thread_50_Task_100_Batch) {
  TestBatchParallelFor("TestBatchParallelFor_2_Thread_50_Task_100_Batch", 2, 50, 100);
}

TEST(ThreadPoolTest, TestBatchParallelFor_2_Thread_81_Task_20_Batch) {
  TestBatchParallelFor("TestBatchParallelFor_2_Thread_81_Task_20_Batch", 2, 81, 20);
}
