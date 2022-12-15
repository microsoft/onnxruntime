// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/threadpool.h"
#include "core/platform/EigenNonBlockingThreadPool.h"
#include "core/platform/ort_mutex.h"
#include "core/util/thread_utils.h"
#ifdef _WIN32
#include "test/platform/windows/env.h"
#include <Windows.h>
#endif

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
  return std::make_unique<TestData>(num);
}

void IncrementElement(TestData& test_data, ptrdiff_t i) {
  std::lock_guard<onnxruntime::OrtMutex> lock(test_data.mutex);
  test_data.data[i]++;
}

void ValidateTestData(TestData& test_data, int expected=1) {
  ASSERT_TRUE(std::count_if(test_data.data.cbegin(), test_data.data.cend(), [&](int i) { return i != expected; }) == 0);
}

// Run a test with a new thread pool created with num_threads threads
// in total (including the main thread).  If num_threads is 0 then we
// test the function with a null pointer, reflecting scenarios where we
// run with just the main thread.  Note that the thread pool API uses
// static methods and should operate across all of these cases.
void CreateThreadPoolAndTest(const std::string&, int num_threads, const std::function<void(ThreadPool*)>& test_body, int dynamic_block_base = 0, bool mock_hybrid = false) {
  if (num_threads > 0) {
    if (dynamic_block_base > 0) {
      onnxruntime::ThreadOptions thread_options;
      thread_options.dynamic_block_base_ = dynamic_block_base;
      auto tp_dynamic_block_size = std::make_unique<ThreadPool>(&onnxruntime::Env::Default(), thread_options, nullptr, num_threads, true, mock_hybrid);
      test_body(tp_dynamic_block_size.get());  // test thread pool with dynamic block size
    } else {
      auto tp_constant_block_size = std::make_unique<ThreadPool>(&onnxruntime::Env::Default(), onnxruntime::ThreadOptions{}, nullptr, num_threads, true, mock_hybrid);
      test_body(tp_constant_block_size.get()); // test thread pool with constant block size
    } 
  } else {
    test_body(nullptr);
  }
}

void TestParallelFor(const std::string& name, int num_threads, int num_tasks) {
  auto test_data = CreateTestData(num_tasks);
  CreateThreadPoolAndTest(name, num_threads, [&](ThreadPool* tp) {
      ThreadPool::TrySimpleParallelFor(tp, num_tasks, [&](std::ptrdiff_t i) { IncrementElement(*test_data, i); });
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

void TestConcurrentParallelFor(const std::string& name, int num_threads, int num_concurrent, int num_tasks, int dynamic_block_base = 0, bool mock_hybrid = false) {
  // Test running multiple concurrent loops over the same thread pool.  This aims to provoke a
  // more diverse mix of interleavings than with a single loop running at a time.
  for (int rep = 0; rep < 5; rep++) {
    CreateThreadPoolAndTest(
        name, num_threads, [&](ThreadPool* tp) {
          std::vector<std::unique_ptr<TestData>> td;
          onnxruntime::Barrier b(num_concurrent - 1);

          // Each concurrent tests runs with its own set of counters
          for (int c = 0; c < num_concurrent; c++) {
            td.push_back(CreateTestData(num_tasks));
          }

          // For a range of scenarios, run some tests via the thread pool, and one directly
          for (int c = 0; c < num_concurrent - 1; c++) {
            ThreadPool::Schedule(tp, [&, c]() {
              ThreadPool::TrySimpleParallelFor(tp, num_tasks, [&](std::ptrdiff_t i) {
                IncrementElement(*td[c], i);
              });
              b.Notify();
            });
          }

          ThreadPool::TrySimpleParallelFor(tp, num_tasks, [&](std::ptrdiff_t i) {
            IncrementElement(*td[num_concurrent - 1], i);
          });

          // Validate all outputs
          b.Wait();
          for (int c = 0; c < num_concurrent; c++) {
            ValidateTestData(*td[c]);
          }
          td.clear();
        },
        dynamic_block_base, mock_hybrid);
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
        ThreadPool::Schedule(tp, [&]() {
          ctr++;
        });
      }
    });
    ASSERT_TRUE(ctr == num_tasks);
    CreateThreadPoolAndTest(name, 2, [&](ThreadPool* tp) {
      // Second variant : schedule from inside the pool
      ThreadPool::Schedule(tp, [&, tp]() {
        for (int tasks = 0; tasks < num_tasks; tasks++) {
          ThreadPool::Schedule(tp, [&]() {
            ctr++;
          });
        }
      });
    });
    ASSERT_TRUE(ctr == num_tasks*2);
  }
}

void TestPoolCreation(const std::string&, int iter) {
  // Test creating and destroying thread pools.  This can be used with Valgrind to help
  // check for memory leaks related to the initialization and clean-up code.  For instance
  //
  //  valgrind --leak-check=full ./onnxruntime_test_all --gtest_filter=ThreadPoolTest.TestPoolCreation_10Iter
  //
  // We create #iter thread pools, and within each of them run a loop of #per_iter steps.
  std::atomic<std::ptrdiff_t> ctr{0};
  constexpr std::ptrdiff_t per_iter = 1024;
  constexpr int num_threads = 4;
  for (auto i = 0; i < iter; i++) {
    auto tp = std::make_unique<ThreadPool>(&onnxruntime::Env::Default(),
                                                   onnxruntime::ThreadOptions(),
                                                   nullptr,
                                                   num_threads,
                                                   true);
    ThreadPool::TryParallelFor(tp.get(), per_iter, 0.0,
                    [&](std::ptrdiff_t s, std::ptrdiff_t e) {
                      ctr += e - s;
                    });
  }
  ASSERT_EQ(ctr, iter * per_iter);
}

// Test multi-loop parallel sections, with a series of fixed-size loops
void TestMultiLoopSections(const std::string& name, int num_threads, int num_loops) {
  for (int rep = 0; rep < 5; rep++) {
    constexpr int num_tasks = 1024;
    auto test_data = CreateTestData(num_tasks);
    CreateThreadPoolAndTest(name, num_threads, [&](ThreadPool* tp) {
	ThreadPool::ParallelSection ps(tp);
	for (int l = 0; l < num_loops; l++) {
          ThreadPool::TrySimpleParallelFor(tp,
                                           num_tasks,
                                           [&](std::ptrdiff_t i) {
                                             IncrementElement(*test_data, i);
                                           });
	}
      });
    ValidateTestData(*test_data, num_loops);
  }
}

// Test multi-loop parallel sections, with alternating larger and
// smaller loops.  This helps test that we can dispatch work to
// differing numbers of threads over time.
void TestStagedMultiLoopSections(const std::string& name, int num_threads, int num_loops) {
  for (int rep = 0; rep < 5; rep++) {
    auto test_data1 = CreateTestData(num_threads/2);
    auto test_data2 = CreateTestData(num_threads);
    CreateThreadPoolAndTest(name, num_threads, [&](ThreadPool* tp) {
	ThreadPool::ParallelSection ps(tp);
	for (int l = 0; l < num_loops; l++) {
          // Loop needing few threads
          ThreadPool::TrySimpleParallelFor(tp,
                                           num_threads / 2,
                                           [&](std::ptrdiff_t i) {
                                             IncrementElement(*test_data1, i);
                                           });
          // Loop needing more threads, forcing growth of set of threads in use
          ThreadPool::TrySimpleParallelFor(tp,
                                           num_threads,
                                           [&](std::ptrdiff_t i) {
                                             IncrementElement(*test_data2, i);
                                           });
	}
      });
    ValidateTestData(*test_data1, num_loops);
    ValidateTestData(*test_data2, num_loops);
  }
}

}  // namespace

namespace onnxruntime {
TEST(ThreadPoolTest, TestParallelFor_0_Thread_NoTask) {
  TestParallelFor("TestParallelFor_0_Thread_NoTask", 0, 0);
}

TEST(ThreadPoolTest, TestParallelFor_0_Thread_50_Task) {
  TestParallelFor("TestParallelFor_0_Thread_50_Task", 0, 50);
}

TEST(ThreadPoolTest, TestParallelFor_2_Thread_NoTask) {
  TestParallelFor("TestParallelFor_2_Thread_NoTask", 2, 0);
}

TEST(ThreadPoolTest, TestParallelFor_2_Thread_50_Task) {
  TestParallelFor("TestParallelFor_2_Thread_50_Task", 2, 50);
}

TEST(ThreadPoolTest, TestParallelFor_1_Thread_50_Task) {
  TestParallelFor("TestParallelFor_1_Thread_50_Task", 1, 50);
}

TEST(ThreadPoolTest, TestBatchParallelFor_0_Thread_50_Task_10_Batch) {
  TestBatchParallelFor("TestBatchParallelFor_0_Thread_50_Task_10_Batch", 0, 50, 10);
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

TEST(ThreadPoolTest, TestConcurrentParallelFor_0Thread_1Conc_0Tasks) {
  TestConcurrentParallelFor("TestConcurrentParallelFor_0Thread_1Conc_0Tasks", 0, 1, 0);
}

TEST(ThreadPoolTest, TestConcurrentParallelFor_1Thread_1Conc_0Tasks) {
  TestConcurrentParallelFor("TestConcurrentParallelFor_1Thread_1Conc_0Tasks", 1, 1, 0);
}

TEST(ThreadPoolTest, TestConcurrentParallelFor_1Thread_1Conc_1Tasks) {
  TestConcurrentParallelFor("TestConcurrentParallelFor_1Thread_1Conc_1Tasks", 1, 1, 1);
}

TEST(ThreadPoolTest, TestConcurrentParallelFor_1Thread_1Conc_8Tasks) {
  TestConcurrentParallelFor("TestConcurrentParallelFor_1Thread_1Conc_8Tasks", 1, 1, 8);
}

TEST(ThreadPoolTest, TestConcurrentParallelFor_1Thread_1Conc_1MTasks) {
  TestConcurrentParallelFor("TestConcurrentParallelFor_1Thread_1Conc_1MTasks", 1, 1, 1000000);
}

TEST(ThreadPoolTest, TestConcurrentParallelFor_1Thread_4Conc_0Tasks) {
  TestConcurrentParallelFor("TestConcurrentParallelFor_1Thread_4Conc_0Tasks", 1, 4, 0);
}

TEST(ThreadPoolTest, TestConcurrentParallelFor_1Thread_4Conc_1Tasks) {
  TestConcurrentParallelFor("TestConcurrentParallelFor_1Thread_4Conc_1Tasks", 1, 4, 1);
}

TEST(ThreadPoolTest, TestConcurrentParallelFor_1Thread_4Conc_8Tasks) {
  TestConcurrentParallelFor("TestConcurrentParallelFor_1Thread_4Conc_8Tasks", 1, 4, 8);
}

TEST(ThreadPoolTest, TestConcurrentParallelFor_1Thread_4Conc_1MTasks) {
  TestConcurrentParallelFor("TestConcurrentParallelFor_1Thread_4Conc_1MTasks", 1, 4, 1000000);
}

TEST(ThreadPoolTest, TestConcurrentParallelFor_4Thread_1Conc_0Tasks) {
  TestConcurrentParallelFor("TestConcurrentParallelFor_4Thread_4Conc_0Tasks", 4, 1, 0);
}

TEST(ThreadPoolTest, TestConcurrentParallelFor_4Thread_1Conc_1Tasks) {
  TestConcurrentParallelFor("TestConcurrentParallelFor_4Thread_4Conc_1Tasks", 4, 1, 1);
}

TEST(ThreadPoolTest, TestConcurrentParallelFor_4Thread_1Conc_8Tasks) {
  TestConcurrentParallelFor("TestConcurrentParallelFor_4Thread_4Conc_8Tasks", 4, 1, 8);
}

TEST(ThreadPoolTest, TestConcurrentParallelFor_4Thread_1Conc_1MTasks) {
  TestConcurrentParallelFor("TestConcurrentParallelFor_4Thread_4Conc_1MTasks", 4, 1, 1000000);
}

TEST(ThreadPoolTest, TestConcurrentParallelFor_4Thread_4Conc_0Tasks) {
  TestConcurrentParallelFor("TestConcurrentParallelFor_4Thread_4Conc_0Tasks", 4, 4, 0);
}

TEST(ThreadPoolTest, TestConcurrentParallelFor_4Thread_4Conc_0Tasks_dynamic_block_base_1) {
  TestConcurrentParallelFor("TestConcurrentParallelFor_4Thread_4Conc_0Tasks_dynamic_block_base_1", 4, 4, 0, 1);
}

TEST(ThreadPoolTest, TestConcurrentParallelFor_4Thread_4Conc_0Tasks_dynamic_block_base_4) {
  TestConcurrentParallelFor("TestConcurrentParallelFor_4Thread_4Conc_0Tasks_dynamic_block_base_4", 4, 4, 0, 4);
}

TEST(ThreadPoolTest, TestConcurrentParallelFor_4Thread_4Conc_0Tasks_dynamic_block_base_16) {
  TestConcurrentParallelFor("TestConcurrentParallelFor_4Thread_4Conc_0Tasks_dynamic_block_base_16", 4, 4, 0, 16);
}

TEST(ThreadPoolTest, TestConcurrentParallelFor_4Thread_4Conc_0Tasks_dynamic_block_base_128) {
  TestConcurrentParallelFor("TestConcurrentParallelFor_4Thread_4Conc_0Tasks_dynamic_block_base_128", 4, 4, 0, 128);
}

TEST(ThreadPoolTest, TestConcurrentParallelFor_4Thread_4Conc_1Tasks) {
  TestConcurrentParallelFor("TestConcurrentParallelFor_4Thread_4Conc_1Tasks", 4, 4, 1);
}

TEST(ThreadPoolTest, TestConcurrentParallelFor_4Thread_4Conc_1Tasks_dynamic_block_base_1) {
  TestConcurrentParallelFor("TestConcurrentParallelFor_4Thread_4Conc_1Tasks_dynamic_block_base_1", 4, 4, 1, 1);
}

TEST(ThreadPoolTest, TestConcurrentParallelFor_4Thread_4Conc_1Tasks_dynamic_block_base_4) {
  TestConcurrentParallelFor("TestConcurrentParallelFor_4Thread_4Conc_1Tasks_dynamic_block_base_4", 4, 4, 1, 4);
}

TEST(ThreadPoolTest, TestConcurrentParallelFor_4Thread_4Conc_1Tasks_dynamic_block_base_16) {
  TestConcurrentParallelFor("TestConcurrentParallelFor_4Thread_4Conc_1Tasks_dynamic_block_base_16", 4, 4, 1, 16);
}

TEST(ThreadPoolTest, TestConcurrentParallelFor_4Thread_4Conc_1Tasks_dynamic_block_base_128) {
  TestConcurrentParallelFor("TestConcurrentParallelFor_4Thread_4Conc_1Tasks_dynamic_block_base_128", 4, 4, 1, 128);
}

TEST(ThreadPoolTest, TestConcurrentParallelFor_4Thread_4Conc_8Tasks) {
  TestConcurrentParallelFor("TestConcurrentParallelFor_4Thread_4Conc_8Tasks", 4, 4, 8);
}

TEST(ThreadPoolTest, TestConcurrentParallelFor_4Thread_4Conc_8Tasks_dynamic_block_base_1) {
  TestConcurrentParallelFor("TestConcurrentParallelFor_4Thread_4Conc_8Tasks_dynamic_block_base_1", 4, 4, 8, 1);
}

TEST(ThreadPoolTest, TestConcurrentParallelFor_4Thread_4Conc_8Tasks_dynamic_block_base_4) {
  TestConcurrentParallelFor("TestConcurrentParallelFor_4Thread_4Conc_8Tasks_dynamic_block_base_4", 4, 4, 8, 4);
}

TEST(ThreadPoolTest, TestConcurrentParallelFor_4Thread_4Conc_8Tasks_dynamic_block_base_16) {
  TestConcurrentParallelFor("TestConcurrentParallelFor_4Thread_4Conc_8Tasks_dynamic_block_base_16", 4, 4, 8, 16);
}

TEST(ThreadPoolTest, TestConcurrentParallelFor_4Thread_4Conc_8Tasks_dynamic_block_base_128) {
  TestConcurrentParallelFor("TestConcurrentParallelFor_4Thread_4Conc_8Tasks_dynamic_block_base_128", 4, 4, 8, 128);
}

TEST(ThreadPoolTest, TestConcurrentParallelFor_4Thread_4Conc_1MTasks) {
  TestConcurrentParallelFor("TestConcurrentParallelFor_4Thread_4Conc_1MTasks", 4, 4, 1000000);
}

TEST(ThreadPoolTest, TestConcurrentParallelFor_4Thread_4Conc_1MTasks_dynamic_block_base_1) {
  TestConcurrentParallelFor("TestConcurrentParallelFor_4Thread_4Conc_1MTasks_dynamic_block_base_1", 4, 4, 1000000, 1);
}

TEST(ThreadPoolTest, TestConcurrentParallelFor_4Thread_4Conc_1MTasks_dynamic_block_base_4) {
  TestConcurrentParallelFor("TestConcurrentParallelFor_4Thread_4Conc_1MTasks_dynamic_block_base_4", 4, 4, 1000000, 4);
}

TEST(ThreadPoolTest, TestConcurrentParallelFor_4Thread_4Conc_1MTasks_dynamic_block_base_16) {
  TestConcurrentParallelFor("TestConcurrentParallelFor_4Thread_4Conc_1MTasks_dynamic_block_base_16", 4, 4, 1000000, 16);
}

TEST(ThreadPoolTest, TestConcurrentParallelFor_4Thread_4Conc_1MTasks_dynamic_block_base_16_hybrid) {
  TestConcurrentParallelFor("TestConcurrentParallelFor_4Thread_4Conc_1MTasks_dynamic_block_base_16", 4, 4, 1000000, 16, true);
}

TEST(ThreadPoolTest, TestConcurrentParallelFor_4Thread_4Conc_1MTasks_dynamic_block_base_128) {
  TestConcurrentParallelFor("TestConcurrentParallelFor_4Thread_4Conc_1MTasks_dynamic_block_base_128", 4, 4, 1000000, 128);
}

TEST(ThreadPoolTest, TestConcurrentParallelFor_4Thread_4Conc_1MTasks_dynamic_block_base_128_hybrid) {
  TestConcurrentParallelFor("TestConcurrentParallelFor_4Thread_4Conc_1MTasks_dynamic_block_base_128", 4, 4, 1000000, 128, true);
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

TEST(ThreadPoolTest, TestPoolCreation_1Iter) {
  TestPoolCreation("TestPoolCreation_1Iter", 1);
}

TEST(ThreadPoolTest, TestPoolCreation_10Iter) {
  TestPoolCreation("TestPoolCreation_10Iter", 10);
}

TEST(ThreadPoolTest, TestPoolCreation_100Iter) {
  TestPoolCreation("TestPoolCreation_100Iter", 100);
}

TEST(ThreadPoolTest, TestMultiLoopSections_0Thread_0Loop) {
  TestMultiLoopSections("TestMultiLoopSections_0Thread_0Loop", 0, 0);
}

TEST(ThreadPoolTest, TestMultiLoopSections_0Thread_1Loop) {
  TestMultiLoopSections("TestMultiLoopSections_0Thread_1Loop", 0, 1);
}

TEST(ThreadPoolTest, TestMultiLoopSections_0Thread_100Loop) {
  TestMultiLoopSections("TestMultiLoopSections_0Thread_100Loop", 0, 100);
}

TEST(ThreadPoolTest, TestMultiLoopSections_1Thread_0Loop) {
  TestMultiLoopSections("TestMultiLoopSections_1Thread_0Loop", 1, 0);
}

TEST(ThreadPoolTest, TestMultiLoopSections_1Thread_1Loop) {
  TestMultiLoopSections("TestMultiLoopSections_1Thread_1Loop", 1, 1);
}

TEST(ThreadPoolTest, TestMultiLoopSections_1Thread_2Loop) {
  TestMultiLoopSections("TestMultiLoopSections_1Thread_2Loop", 1, 2);
}

TEST(ThreadPoolTest, TestMultiLoopSections_2Thread_0Loop) {
  TestMultiLoopSections("TestMultiLoopSections_2Thread_0Loop", 2, 0);
}

TEST(ThreadPoolTest, TestMultiLoopSections_2Thread_1Loop) {
  TestMultiLoopSections("TestMultiLoopSections_2Thread_1Loop", 2, 1);
}

TEST(ThreadPoolTest, TestMultiLoopSections_2Thread_2Loop) {
  TestMultiLoopSections("TestMultiLoopSections_2Thread_2Loop", 2, 2);
}

TEST(ThreadPoolTest, TestMultiLoopSections_2Thread_100Loop) {
  TestMultiLoopSections("TestMultiLoopSections_2Thread_100Loop", 2, 100);
}

TEST(ThreadPoolTest, TestMultiLoopSections_4Thread_1Loop) {
  TestMultiLoopSections("TestMultiLoopSections_4Thread_1Loop", 4, 1);
}

TEST(ThreadPoolTest, TestMultiLoopSections_4Thread_10Loop) {
  TestMultiLoopSections("TestMultiLoopSections_4Thread_10Loop", 4, 10);
}

TEST(ThreadPoolTest, TestMultiLoopSections_4Thread_100Loop) {
  TestMultiLoopSections("TestMultiLoopSections_4Thread_100Loop", 4, 100);
}

TEST(ThreadPoolTest, TestStagedMultiLoopSections_4Thread_1Loop) {
  TestStagedMultiLoopSections("TestStagedMultiLoopSections_4Thread_1Loop", 4, 1);
}

TEST(ThreadPoolTest, TestStagedMultiLoopSections_4Thread_10Loop) {
  TestStagedMultiLoopSections("TestStagedMultiLoopSections_4Thread_10Loop", 4, 10);
}

TEST(ThreadPoolTest, TestStagedMultiLoopSections_4Thread_100Loop) {
  TestStagedMultiLoopSections("TestStagedMultiLoopSections_4Thread_100Loop", 4, 100);
}

#ifdef _WIN32
#if WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_DESKTOP)
#pragma warning(push)
#pragma warning(disable : 6387)
TEST(ThreadPoolTest, TestStackSize) {
  ThreadOptions to;
  // For ARM, x86 and x64 machines, the default stack size is 1 MB
  // We change it to a different value to see if the setting works
  to.stack_size = 8 * 1024 * 1024;
  auto tp = std::make_unique<ThreadPool>(&onnxruntime::Env::Default(), to, nullptr, 2, true);
  typedef void(WINAPI * FnGetCurrentThreadStackLimits)(_Out_ PULONG_PTR LowLimit, _Out_ PULONG_PTR HighLimit);

  Notification n;
  ULONG_PTR low_limit, high_limit;
  bool has_thread_limit_info = false;
  ThreadPool::Schedule(tp.get(), [&]() {
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
#pragma warning(pop)
#endif
#endif

#if !defined(ORT_MINIMAL_BUILD) && !defined(ORT_EXTENDED_MINIMAL_BUILD)

#ifndef ORT_NO_EXCEPTIONS
TEST(ThreadPoolTest, TestAffinityStringMisshaped) {
  OrtThreadPoolParams tp_params;
  tp_params.thread_pool_size = 3;
  const char* wrong_formats[] = {
      ",",     //1st and 2nd processor id are empty strings
      "1,",    //2nd processor id is an empty string
      ";",     //affinity settings for both threads are empty
      ";1",    //missing the affinity setting for the 1st thread
      "a",     //invalid char, must be digit
      "a;b",   //invalid char, must be digit
      "1;a",   //invalid char, must be digit
      "0;1",   //processor string must start from 1
      "-;2",   //invalid char, must be digit
      "--",    //invalid char, must be digit
      "2-1;3", //invalid interval, "from" must be equal to or smaller than "to"
      "5;3a"   //invalid processor id containing non-digit as suffix
  };
  for (const auto* wrong_format : wrong_formats) {
    tp_params.affinity_str = wrong_format;
    ASSERT_THROW(concurrency::CreateThreadPool(&onnxruntime::Env::Default(),
                                               tp_params,
                                               concurrency::ThreadPoolType::INTRA_OP),
                 std::exception);
  }
  const char* less_than_expected_vec[] = {"1", "1,2", "1-2"};
  for (const auto* less_than_expected : less_than_expected_vec) {
    tp_params.affinity_str = less_than_expected;
    ASSERT_THROW(concurrency::CreateThreadPool(&onnxruntime::Env::Default(),
                                               tp_params,
                                               concurrency::ThreadPoolType::INTRA_OP),
                 std::exception);
  }
  const char* more_than_expected_vec[] = {"1;2;3", "1-2;2-2;3-4", "1;2;3;4;5"};
  for (const auto* more_than_expected : more_than_expected_vec) {
    tp_params.affinity_str = more_than_expected;
    ASSERT_THROW(concurrency::CreateThreadPool(&onnxruntime::Env::Default(),
                                               tp_params,
                                               concurrency::ThreadPoolType::INTRA_OP),
                 std::exception);
  }
}
#endif

TEST(ThreadPoolTest, TestAffinityStringWellShaped) {
  OrtThreadPoolParams tp_params;
  auto default_tp = concurrency::CreateThreadPool(&onnxruntime::Env::Default(),
                                                  tp_params,
                                                  concurrency::ThreadPoolType::INTRA_OP);
  if (concurrency::ThreadPool::DegreeOfParallelism(default_tp.get()) < 3) {
    return;
  }
  tp_params.thread_pool_size = 3;
  const char* good_formats[] = {"1;1",
                                "2;2",
                                "1-1;2-2",
                                "1-2;1-2"};
  for (const auto* good_format : good_formats) {
    tp_params.affinity_str = good_format;
    auto non_default_tp = concurrency::CreateThreadPool(&onnxruntime::Env::Default(),
                                                        tp_params,
                                                        concurrency::ThreadPoolType::INTRA_OP);
    ASSERT_TRUE(concurrency::ThreadPool::DegreeOfParallelism(non_default_tp.get()) == 3);
  }
}

#ifdef _WIN32
TEST(ThreadPoolTest, TestDefaultAffinity) {
  test::CpuGroup cpu_group = {{0, 1},
                              {2, 3},
                              {4, 5},
                              {6, 7}};
  // 2 logical processors per core, single group
  test::CpuInfo cpu_info_single = {cpu_group};
  test::WindowsEnvTester win_env;
  win_env.SetCpuInfo(cpu_info_single);
  auto default_affinities = win_env.GetDefaultThreadAffinities();
  ASSERT_TRUE(default_affinities.size() == 4);
  for (int i = 0; i < 4; ++i) {
    ASSERT_TRUE(default_affinities[i].size() == 2);
    for (int j = 0; j < 2; ++j) {
      ASSERT_TRUE(default_affinities[i][j] == i * 2 + j);
    }
  }
  // 2 logical processors per core, two groups
  test::CpuInfo cpu_info_double = {cpu_group, cpu_group};
  win_env.SetCpuInfo(cpu_info_double);
  default_affinities = win_env.GetDefaultThreadAffinities();
  ASSERT_TRUE(default_affinities.size() == 8);
  for (int i = 0; i < 8; ++i) {
    ASSERT_TRUE(default_affinities[i].size() == 2);
    for (int j = 0; j < 2; ++j) {
      ASSERT_TRUE(default_affinities[i][j] == i * 2 + j);
    }
  }
  // 4 logical processors per core, single group
  cpu_group = {{0, 1, 2, 3},
               {4, 5, 6, 7},
               {8, 9, 10, 11},
               {12, 13, 14, 15}};
  cpu_info_single = {cpu_group};
  win_env.SetCpuInfo(cpu_info_single);
  default_affinities = win_env.GetDefaultThreadAffinities();
  ASSERT_TRUE(default_affinities.size() == 4);
  for (int i = 0; i < 4; ++i) {
    ASSERT_TRUE(default_affinities[i].size() == 4);
    for (int j = 0; j < 4; ++j) {
      ASSERT_TRUE(default_affinities[i][j] == i * 4 + j);
    }
  }
  // 4 logical processors per core, two groups
  cpu_info_double = {cpu_group, cpu_group};
  win_env.SetCpuInfo(cpu_info_double);
  default_affinities = win_env.GetDefaultThreadAffinities();
  ASSERT_TRUE(default_affinities.size() == 8);
  for (int i = 0; i < 8; ++i) {
    ASSERT_TRUE(default_affinities[i].size() == 4);
    for (int j = 0; j < 4; ++j) {
      ASSERT_TRUE(default_affinities[i][j] == i * 4 + j);
    }
  }
}
#endif
#endif

}  // namespace onnxruntime
