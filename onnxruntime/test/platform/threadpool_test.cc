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

void TestMultipleParallelFor(const std::string& name, int num_threads, int num_concurrent, int num_tasks) {
  // Test running multiple concurrent loops over the same thread pool.  This aims to provoke a
  // more diverse mix of interleavings than with a single loop running at a time.
  for (int rep = 0; rep < 5; rep++) {
    CreateThreadPoolAndTest(name, num_threads, [&](ThreadPool* tp) {
      std::vector<std::unique_ptr<TestData>> td;
      onnxruntime::Barrier b(num_concurrent - 1);

      // Each concurrent tests runs with its own set of counters
      for (int c = 0; c < num_concurrent; c++) {
        td.push_back(CreateTestData(num_tasks));
      }

      // For a range of scenarios, run some tests via the thread pool, and one directly
      for (int c = 0; c < num_concurrent - 1; c++) {
        tp->Schedule([&, c]() {
          tp->SimpleParallelFor(num_tasks, [&](std::ptrdiff_t i) {
            IncrementElement(*td[c], i);
          });
          b.Notify();
        });
      }

      tp->SimpleParallelFor(num_tasks, [&](std::ptrdiff_t i) {
        IncrementElement(*td[num_concurrent - 1], i);
      });

      // Validate all outputs
      b.Wait();
      for (int c = 0; c < num_concurrent; c++) {
        ValidateTestData(*td[c]);
      }
      td.clear();
    });
  }
}

void TestBurstScheduling(const std::string& name, int num_tasks) {
  // Test submitting a burst of functions for executing.  The aim is to provoke cases such
  // as the thread pool's work queues being full.
  for (int rep = 0; rep < 5; rep++) {
    std::atomic<int> ctr{0};
    // Schedule a burst of num_tasks back-to-back, and then cleanly shut down the thread
    // pool.  The synchronization barrier during shut down should ensure that all of the
    // tasks are complete.  Note that if the thread pool's work queues are full, then a
    // call to tp->Schedule() may run its argument synchronously.  In any case, we expect
    // ctr==num_tasks.
    CreateThreadPoolAndTest(name, 2, [&](ThreadPool* tp) {
      // First variant : schedule from outside the pool
      for (int tasks = 0; tasks < num_tasks; tasks++) {
        tp->Schedule([&]() {
          ctr++;
        });
      }
    });
    ASSERT_TRUE(ctr == num_tasks);
    CreateThreadPoolAndTest(name, 2, [&](ThreadPool* tp) {
      // Second variant : schedule from inside the pool
      tp->Schedule([&, tp]() {
        for (int tasks = 0; tasks < num_tasks; tasks++) {
          tp->Schedule([&]() {
            ctr++;
          });
        }
      });
    });
    ASSERT_TRUE(ctr == num_tasks*2);
  }
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

TEST(ThreadPoolTest, TestMultipleParallelFor_1Thread_1Conc_0Tasks) {
  TestMultipleParallelFor("TestMultipleParallelFor_1Thread_1Conc_0Tasks", 1, 1, 0);
}

TEST(ThreadPoolTest, TestMultipleParallelFor_1Thread_1Conc_1Tasks) {
  TestMultipleParallelFor("TestMultipleParallelFor_1Thread_1Conc_1Tasks", 1, 1, 1);
}

TEST(ThreadPoolTest, TestMultipleParallelFor_1Thread_1Conc_8Tasks) {
  TestMultipleParallelFor("TestMultipleParallelFor_1Thread_1Conc_8Tasks", 1, 1, 8);
}

TEST(ThreadPoolTest, TestMultipleParallelFor_1Thread_1Conc_1MTasks) {
  TestMultipleParallelFor("TestMultipleParallelFor_1Thread_1Conc_1MTasks", 1, 1, 1000000);
}

TEST(ThreadPoolTest, TestMultipleParallelFor_1Thread_4Conc_0Tasks) {
  TestMultipleParallelFor("TestMultipleParallelFor_1Thread_4Conc_0Tasks", 1, 4, 0);
}

TEST(ThreadPoolTest, TestMultipleParallelFor_1Thread_4Conc_1Tasks) {
  TestMultipleParallelFor("TestMultipleParallelFor_1Thread_4Conc_1Tasks", 1, 4, 1);
}

TEST(ThreadPoolTest, TestMultipleParallelFor_1Thread_4Conc_8Tasks) {
  TestMultipleParallelFor("TestMultipleParallelFor_1Thread_4Conc_8Tasks", 1, 4, 8);
}

TEST(ThreadPoolTest, TestMultipleParallelFor_1Thread_4Conc_1MTasks) {
  TestMultipleParallelFor("TestMultipleParallelFor_1Thread_4Conc_1MTasks", 1, 4, 1000000);
}

TEST(ThreadPoolTest, TestMultipleParallelFor_4Thread_1Conc_0Tasks) {
  TestMultipleParallelFor("TestMultipleParallelFor_4Thread_4Conc_0Tasks", 4, 1, 0);
}

TEST(ThreadPoolTest, TestMultipleParallelFor_4Thread_1Conc_1Tasks) {
  TestMultipleParallelFor("TestMultipleParallelFor_4Thread_4Conc_1Tasks", 4, 1, 1);
}

TEST(ThreadPoolTest, TestMultipleParallelFor_4Thread_1Conc_8Tasks) {
  TestMultipleParallelFor("TestMultipleParallelFor_4Thread_4Conc_8Tasks", 4, 1, 8);
}

TEST(ThreadPoolTest, TestMultipleParallelFor_4Thread_1Conc_1MTasks) {
  TestMultipleParallelFor("TestMultipleParallelFor_4Thread_4Conc_1MTasks", 4, 1, 1000000);
}

TEST(ThreadPoolTest, TestMultipleParallelFor_4Thread_4Conc_0Tasks) {
  TestMultipleParallelFor("TestMultipleParallelFor_4Thread_4Conc_0Tasks", 4, 4, 0);
}

TEST(ThreadPoolTest, TestMultipleParallelFor_4Thread_4Conc_1Tasks) {
  TestMultipleParallelFor("TestMultipleParallelFor_4Thread_4Conc_1Tasks", 4, 4, 1);
}

TEST(ThreadPoolTest, TestMultipleParallelFor_4Thread_4Conc_8Tasks) {
  TestMultipleParallelFor("TestMultipleParallelFor_4Thread_4Conc_8Tasks", 4, 4, 8);
}

TEST(ThreadPoolTest, TestMultipleParallelFor_4Thread_4Conc_1MTasks) {
  TestMultipleParallelFor("TestMultipleParallelFor_4Thread_4Conc_1MTasks", 4, 4, 1000000);
}

TEST(ThreadPoolTest, TestBurstScheduling_0Tasks) {
  TestBurstScheduling("TestBurstScheduling_0Tasks", 0);
}

TEST(ThreadPoolTest, TestBurstScheduling_1Task) {
  TestBurstScheduling("TestBurstScheduling_1Task", 1);
}

TEST(ThreadPoolTest, TestBurstScheduling_16Tasks) {
  TestBurstScheduling("TestBurstScheduling_16Tasks", 16);
}

TEST(ThreadPoolTest, TestBurstScheduling_65536Task) {
  // Attempt to exhaust the size of the queues used in the thread pool to
  // buffer tasks.
  TestBurstScheduling("TestBurstScheduling_65536Tasks", 65536);
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
