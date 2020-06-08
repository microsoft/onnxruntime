// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/threadpool.h"
#include "core/platform/EigenNonBlockingThreadPool.h"
#include "core/platform/ort_mutex.h"

#include <core/common/make_unique.h>

#include "gtest/gtest.h"
#include <algorithm>
#include <memory>
#include <functional>

#ifdef _WIN32
#include <Windows.h>
#endif

using namespace onnxruntime::concurrency;

namespace {

struct TestData {
  explicit TestData(int num) : data(num, 0) {
  }
  std::vector<int> data;
  onnxruntime::OrtMutex mutex;
};

// This unittest tests ThreadPool function by counting the number of calls to function with each index.
// the function should be called exactly once for each element.

std::unique_ptr<TestData> CreateTestData(int num) {
  return onnxruntime::make_unique<TestData>(num);
}

void IncrementElement(TestData& test_data, ptrdiff_t i) {
  std::lock_guard<onnxruntime::OrtMutex> lock(test_data.mutex);
  test_data.data[i]++;
}

void ValidateTestData(TestData& test_data) {
  ASSERT_TRUE(std::count_if(test_data.data.cbegin(), test_data.data.cend(), [](int i) { return i != 1; }) == 0);
}

void CreateThreadPoolAndTest(const std::string&, int num_threads, const std::function<void(ThreadPool*)>& test_body) {
  auto tp = onnxruntime::make_unique<ThreadPool>(&onnxruntime::Env::Default(), onnxruntime::ThreadOptions(), nullptr,
                                                 num_threads, true);
  test_body(tp.get());
}

void TestParallelFor(const std::string& name, int num_threads, int num_tasks) {
  auto test_data = CreateTestData(num_tasks);
  CreateThreadPoolAndTest(name, num_threads, [&](ThreadPool* tp) {
    tp->SimpleParallelFor(num_tasks, [&](std::ptrdiff_t i) { IncrementElement(*test_data, i); });
  });
  ValidateTestData(*test_data);
}

void TestBatchParallelFor(const std::string& name, int num_threads, int num_tasks, int batch_size) {
  auto test_data = CreateTestData(num_tasks);

  CreateThreadPoolAndTest(name, num_threads, [&](ThreadPool* tp) {
    onnxruntime::concurrency::ThreadPool::TryBatchParallelFor(
        tp, num_tasks, [&](ptrdiff_t i) { IncrementElement(*test_data, i); }, batch_size);
  });
  ValidateTestData(*test_data);
}

}  // namespace

namespace onnxruntime {
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

#ifdef _WIN32
TEST(ThreadPoolTest, TestStackSize) {
  ThreadOptions to;
  // For ARM, x86 and x64 machines, the default stack size is 1 MB
  // We change it to a different value to see if the setting works
  to.stack_size = 8 * 1024 * 1024;
  auto tp = onnxruntime::make_unique<ThreadPool>(&onnxruntime::Env::Default(), to, nullptr, 2, true);
  typedef void(WINAPI * FnGetCurrentThreadStackLimits)(_Out_ PULONG_PTR LowLimit, _Out_ PULONG_PTR HighLimit);

  Notification n;
  ULONG_PTR low_limit, high_limit;
  bool has_thread_limit_info = false;
  tp->Schedule([&]() {
    HMODULE kernel32_module = GetModuleHandle(TEXT("kernel32.dll"));
    assert(kernel32_module != nullptr);
    FnGetCurrentThreadStackLimits GetTS =
        (FnGetCurrentThreadStackLimits)GetProcAddress(kernel32_module, "GetCurrentThreadStackLimits");
    if (GetTS != nullptr) {
      GetTS(&low_limit, &high_limit);
      has_thread_limit_info = true;
    }
    n.Notify();
  });
  n.Wait();
  if (has_thread_limit_info)
    ASSERT_EQ(high_limit - low_limit, to.stack_size);
}
#endif

}  // namespace onnxruntime
