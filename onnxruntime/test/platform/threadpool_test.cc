// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/threadpool.h"
#include "core/platform/EigenNonBlockingThreadPool.h"
#include <mutex>
#include "core/util/thread_utils.h"
#include "test/util/include/scoped_env_vars.h"
#ifdef _WIN32
#include "test/platform/windows/env.h"
#include <Windows.h>
#endif

#include "gtest/gtest.h"
#include <algorithm>
#include <memory>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <Windows.h>
#endif

using namespace onnxruntime::concurrency;

namespace {

struct TestData {
  explicit TestData(int num) : data(num, 0) {
  }
  std::vector<int> data;
  std::mutex mutex;
};

// This unittest tests ThreadPool function by counting the number of calls to function with each index.
// the function should be called exactly once for each element.

std::unique_ptr<TestData> CreateTestData(int num) {
  return std::make_unique<TestData>(num);
}

void IncrementElement(TestData& test_data, ptrdiff_t i) {
  std::lock_guard<std::mutex> lock(test_data.mutex);
  test_data.data[i]++;
}

void ValidateTestData(TestData& test_data, int expected = 1) {
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
      auto tp_dynamic_block_size = std::make_unique<ThreadPool>(&onnxruntime::Env::Default(), thread_options, nullptr, num_threads, onnxruntime::concurrency::kSpinDurationDefault, mock_hybrid);
      test_body(tp_dynamic_block_size.get());  // test thread pool with dynamic block size
    } else {
      auto tp_constant_block_size = std::make_unique<ThreadPool>(&onnxruntime::Env::Default(), onnxruntime::ThreadOptions{}, nullptr, num_threads, onnxruntime::concurrency::kSpinDurationDefault, mock_hybrid);
      test_body(tp_constant_block_size.get());  // test thread pool with constant block size
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
    ASSERT_TRUE(ctr == num_tasks * 2);
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
                                           onnxruntime::concurrency::kSpinDurationDefault);
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
    auto test_data1 = CreateTestData(num_threads / 2);
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
  auto tp = std::make_unique<ThreadPool>(&onnxruntime::Env::Default(), to, nullptr, 2, onnxruntime::concurrency::kSpinDurationDefault);
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
      ",",      // 1st and 2nd processor id are empty strings
      "1,",     // 2nd processor id is an empty string
      ";",      // affinity settings for both threads are empty
      ";1",     // missing the affinity setting for the 1st thread
      "a",      // invalid char, must be digit
      "a;b",    // invalid char, must be digit
      "1;a",    // invalid char, must be digit
      "0;1",    // processor string must start from 1
      "-;2",    // invalid char, must be digit
      "--",     // invalid char, must be digit
      "2-1;3",  // invalid interval, "from" must be equal to or smaller than "to"
      "5;3a"    // invalid processor id containing non-digit as suffix
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
    auto DOP = concurrency::ThreadPool::DegreeOfParallelism(non_default_tp.get());
    ASSERT_TRUE(DOP >= 3 && DOP % 3 == 0);  // for hybrid cpu, dop is a multiple of 3
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

#ifdef ORT_ENABLE_SESSION_THREADPOOL_CALLBACKS
// Test for OrtThreadPoolCallbacksConfig - validates that callbacks are invoked
// when work is scheduled to the thread pool.
namespace {

struct WorkCallbackTestContext {
  std::atomic<int> enqueue_count{0};
  std::atomic<int> start_count{0};
  std::atomic<int> stop_count{0};
  std::atomic<int> abandon_count{0};
};

void* TestOnEnqueue(void* user_context) noexcept {
  auto* ctx = static_cast<WorkCallbackTestContext*>(user_context);
  ctx->enqueue_count++;
  return reinterpret_cast<void*>(static_cast<uintptr_t>(0xCB00CB00));
}

void TestOnStart(void* user_context, void* callback_data) noexcept {
  auto* ctx = static_cast<WorkCallbackTestContext*>(user_context);
  ctx->start_count++;
  EXPECT_EQ(callback_data, reinterpret_cast<void*>(static_cast<uintptr_t>(0xCB00CB00)));
}

void TestOnStop(void* user_context, void* callback_data) noexcept {
  auto* ctx = static_cast<WorkCallbackTestContext*>(user_context);
  ctx->stop_count++;
  EXPECT_EQ(callback_data, reinterpret_cast<void*>(static_cast<uintptr_t>(0xCB00CB00)));
}

void TestOnAbandon(void* user_context, void* callback_data) noexcept {
  auto* ctx = static_cast<WorkCallbackTestContext*>(user_context);
  ctx->abandon_count++;
  EXPECT_EQ(callback_data, reinterpret_cast<void*>(static_cast<uintptr_t>(0xCB00CB00)));
}

// Helper to create a thread pool with work callbacks and run a test
void CreateThreadPoolWithCallbacksAndTest(
    int num_threads,
    WorkCallbackTestContext& ctx,
    bool enable_start_stop,
    const std::function<void(ThreadPool*)>& test_body) {
  OrtThreadPoolCallbacksConfig callbacks{};
  callbacks.on_enqueue = TestOnEnqueue;
  callbacks.on_start_work = enable_start_stop ? TestOnStart : nullptr;
  callbacks.on_stop_work = enable_start_stop ? TestOnStop : nullptr;
  callbacks.on_abandon = enable_start_stop ? TestOnAbandon : nullptr;
  callbacks.user_context = &ctx;

  onnxruntime::ThreadOptions thread_options;
  thread_options.work_callbacks = &callbacks;

  auto tp = std::make_unique<ThreadPool>(&onnxruntime::Env::Default(),
                                         thread_options,
                                         nullptr,
                                         num_threads,
                                         onnxruntime::concurrency::kSpinDurationDefault);
  test_body(tp.get());
}

}  // namespace

TEST(ThreadPoolTest, TestWorkCallbacks_Schedule) {
  WorkCallbackTestContext ctx;
  constexpr int num_tasks = 100;
  std::atomic<int> tasks_completed{0};

  CreateThreadPoolWithCallbacksAndTest(4, ctx, true, [&](ThreadPool* tp) {
    for (int i = 0; i < num_tasks; i++) {
      ThreadPool::Schedule(tp, [&]() { tasks_completed++; });
    }
  });

  ASSERT_EQ(tasks_completed.load(), num_tasks);
  ASSERT_EQ(ctx.enqueue_count.load(), num_tasks);
  ASSERT_EQ(ctx.start_count.load(), num_tasks);
  ASSERT_EQ(ctx.stop_count.load(), num_tasks);
}

TEST(ThreadPoolTest, TestWorkCallbacks_OnEnqueueOnly) {
  WorkCallbackTestContext ctx;
  constexpr int num_tasks = 50;
  std::atomic<int> tasks_completed{0};

  CreateThreadPoolWithCallbacksAndTest(2, ctx, false, [&](ThreadPool* tp) {
    for (int i = 0; i < num_tasks; i++) {
      ThreadPool::Schedule(tp, [&]() { tasks_completed++; });
    }
  });

  ASSERT_EQ(tasks_completed.load(), num_tasks);
  ASSERT_EQ(ctx.enqueue_count.load(), num_tasks);
  ASSERT_EQ(ctx.start_count.load(), 0);  // Not set
  ASSERT_EQ(ctx.stop_count.load(), 0);   // Not set
}

TEST(ThreadPoolTest, TestWorkCallbacks_NoCallbacks) {
  WorkCallbackTestContext ctx;
  constexpr int num_tasks = 50;
  std::atomic<int> tasks_completed{0};

  CreateThreadPoolAndTest("NoCallbacks", 2, [&](ThreadPool* tp) {
    for (int i = 0; i < num_tasks; i++) {
      ThreadPool::Schedule(tp, [&]() { tasks_completed++; });
    }
  });

  ASSERT_EQ(tasks_completed.load(), num_tasks);
  ASSERT_EQ(ctx.enqueue_count.load(), 0);
  ASSERT_EQ(ctx.start_count.load(), 0);
  ASSERT_EQ(ctx.stop_count.load(), 0);
}

TEST(ThreadPoolTest, TestWorkCallbacks_ParallelFor) {
  WorkCallbackTestContext ctx;
  constexpr int num_tasks = 100;
  std::atomic<int> tasks_completed{0};

  CreateThreadPoolWithCallbacksAndTest(4, ctx, true, [&](ThreadPool* tp) {
    ThreadPool::TrySimpleParallelFor(tp, num_tasks, [&](std::ptrdiff_t) {
      tasks_completed++;
    });
  });

  ASSERT_EQ(tasks_completed.load(), num_tasks);
  // Worker threads get callbacks; main thread's fn(0) does not.
  // Some enqueued tasks may be revoked before execution (work completed by other threads),
  // so enqueue_count >= start_count. Start/stop must always be balanced.
  // Every enqueued item must end with either start+stop or abandon.
  ASSERT_GE(ctx.enqueue_count.load(), ctx.start_count.load());
  ASSERT_EQ(ctx.start_count.load(), ctx.stop_count.load());
  ASSERT_EQ(ctx.enqueue_count.load(), ctx.start_count.load() + ctx.abandon_count.load());
}

TEST(ThreadPoolTest, TestWorkCallbacks_ParallelSection) {
  WorkCallbackTestContext ctx;
  constexpr int num_tasks = 50;
  constexpr int num_loops = 3;
  std::atomic<int> tasks_completed{0};

  CreateThreadPoolWithCallbacksAndTest(4, ctx, true, [&](ThreadPool* tp) {
    ThreadPool::ParallelSection ps(tp);
    for (int loop = 0; loop < num_loops; loop++) {
      ThreadPool::TrySimpleParallelFor(tp, num_tasks, [&](std::ptrdiff_t) {
        tasks_completed++;
      });
    }
  });

  ASSERT_EQ(tasks_completed.load(), num_tasks * num_loops);
  // Some enqueued tasks may be revoked before execution (work completed by other threads),
  // so enqueue_count >= start_count. Start/stop must always be balanced.
  // Every enqueued item must end with either start+stop or abandon.
  ASSERT_GE(ctx.enqueue_count.load(), ctx.start_count.load());
  ASSERT_EQ(ctx.start_count.load(), ctx.stop_count.load());
  ASSERT_EQ(ctx.enqueue_count.load(), ctx.start_count.load() + ctx.abandon_count.load());
}

TEST(ThreadPoolTest, TestWorkCallbacks_Abandon) {
  // Verify that on_abandon is called when enqueued work is revoked.
  // Block all workers so dispatch tasks sit in queues unexecuted,
  // then the main thread completes all iterations and revokes them.
  //
  // ThreadPool(num_threads) creates num_threads-1 actual worker threads
  // (the calling thread counts as one).  We must block exactly
  // num_threads-1 workers so that all pool workers are occupied.
  WorkCallbackTestContext ctx;
  constexpr int num_threads = 5;
  const int num_workers = num_threads - 1;  // actual pool worker threads

  CreateThreadPoolWithCallbacksAndTest(num_threads, ctx, true, [&](ThreadPool* tp) {
    onnxruntime::Barrier workers_ready(num_workers);
    onnxruntime::Barrier workers_released(num_workers);
    std::atomic<bool> release{false};

    for (int i = 0; i < num_workers; i++) {
      ThreadPool::Schedule(tp, [&]() {
        workers_ready.Notify();
        while (!release.load(std::memory_order_acquire)) {
          onnxruntime::concurrency::SpinPause();
        }
        workers_released.Notify();
      });
    }
    workers_ready.Wait();  // All workers are now blocked

    // Reset counters so we only measure the parallel loop below.
    ctx.enqueue_count = 0;
    ctx.start_count = 0;
    ctx.stop_count = 0;
    ctx.abandon_count = 0;

    // The parallel loop enqueues a dispatch task, but no worker can pick it up.
    // The main thread completes all iterations, then EndParallelSection revokes
    // the dispatch task, triggering on_abandon.
    std::atomic<int> tasks_done{0};
    ThreadPool::TrySimpleParallelFor(tp, 100, [&](std::ptrdiff_t) {
      tasks_done++;
    });

    ASSERT_EQ(tasks_done.load(), 100);
    ASSERT_GT(ctx.abandon_count.load(), 0);
    ASSERT_EQ(ctx.start_count.load(), ctx.stop_count.load());
    ASSERT_EQ(ctx.enqueue_count.load(), ctx.start_count.load() + ctx.abandon_count.load());

    release.store(true, std::memory_order_release);
    workers_released.Wait();
  });
}

TEST(ThreadPoolTest, TestWorkCallbacks_EnqueueReturnsNull) {
  // Verify that when on_enqueue returns nullptr, it is correctly passed
  // through to on_start_work/on_stop_work.
  WorkCallbackTestContext ctx;
  constexpr int num_tasks = 50;
  std::atomic<int> tasks_completed{0};

  OrtThreadPoolCallbacksConfig callbacks{};
  callbacks.on_enqueue = [](void* user_context) noexcept -> void* {
    auto* c = static_cast<WorkCallbackTestContext*>(user_context);
    c->enqueue_count++;
    return nullptr;
  };
  callbacks.on_start_work = [](void* user_context, void* enqueue_data) noexcept {
    auto* c = static_cast<WorkCallbackTestContext*>(user_context);
    c->start_count++;
    EXPECT_EQ(enqueue_data, nullptr);
  };
  callbacks.on_stop_work = [](void* user_context, void* enqueue_data) noexcept {
    auto* c = static_cast<WorkCallbackTestContext*>(user_context);
    c->stop_count++;
    EXPECT_EQ(enqueue_data, nullptr);
  };
  callbacks.user_context = &ctx;

  onnxruntime::ThreadOptions thread_options;
  thread_options.work_callbacks = &callbacks;

  auto tp = std::make_unique<ThreadPool>(&onnxruntime::Env::Default(),
                                         thread_options,
                                         nullptr,
                                         4,
                                         onnxruntime::concurrency::kSpinDurationDefault);

  for (int i = 0; i < num_tasks; i++) {
    ThreadPool::Schedule(tp.get(), [&]() { tasks_completed++; });
  }
  tp.reset();

  ASSERT_EQ(tasks_completed.load(), num_tasks);
  ASSERT_EQ(ctx.enqueue_count.load(), num_tasks);
  ASSERT_EQ(ctx.start_count.load(), num_tasks);
  ASSERT_EQ(ctx.stop_count.load(), num_tasks);
}

TEST(ThreadPoolTest, TestWorkCallbacks_NoEnqueueWithStartStop) {
  // Verify that on_start_work/on_stop_work are called even when
  // on_enqueue is not set. enqueue_data should be nullptr.
  WorkCallbackTestContext ctx;
  constexpr int num_tasks = 50;
  std::atomic<int> tasks_completed{0};

  OrtThreadPoolCallbacksConfig callbacks{};
  callbacks.on_enqueue = nullptr;
  callbacks.on_start_work = [](void* user_context, void* enqueue_data) noexcept {
    auto* c = static_cast<WorkCallbackTestContext*>(user_context);
    c->start_count++;
    EXPECT_EQ(enqueue_data, nullptr);
  };
  callbacks.on_stop_work = [](void* user_context, void* enqueue_data) noexcept {
    auto* c = static_cast<WorkCallbackTestContext*>(user_context);
    c->stop_count++;
    EXPECT_EQ(enqueue_data, nullptr);
  };
  callbacks.user_context = &ctx;

  onnxruntime::ThreadOptions thread_options;
  thread_options.work_callbacks = &callbacks;

  auto tp = std::make_unique<ThreadPool>(&onnxruntime::Env::Default(),
                                         thread_options,
                                         nullptr,
                                         4,
                                         onnxruntime::concurrency::kSpinDurationDefault);

  for (int i = 0; i < num_tasks; i++) {
    ThreadPool::Schedule(tp.get(), [&]() { tasks_completed++; });
  }
  tp.reset();

  ASSERT_EQ(tasks_completed.load(), num_tasks);
  ASSERT_EQ(ctx.enqueue_count.load(), 0);  // No enqueue callback
  ASSERT_EQ(ctx.start_count.load(), num_tasks);
  ASSERT_EQ(ctx.stop_count.load(), num_tasks);
}

#endif  // ORT_ENABLE_SESSION_THREADPOOL_CALLBACKS

// -------------------------------------------------------------------
// Tests for the three spin_duration_us modes (-1/0/>0)
// -------------------------------------------------------------------

// Helper: create a thread pool with the given spin_duration_us, run a parallel
// workload, and verify correctness.
void TestSpinDurationMode(int spin_duration_us) {
  constexpr int num_threads = 4;
  constexpr std::ptrdiff_t num_tasks = 1024;
  auto tp = std::make_unique<ThreadPool>(&onnxruntime::Env::Default(),
                                         onnxruntime::ThreadOptions(),
                                         nullptr,
                                         num_threads,
                                         spin_duration_us);
  std::atomic<std::ptrdiff_t> ctr{0};
  ThreadPool::TryParallelFor(tp.get(), num_tasks, 0.0,
                             [&](std::ptrdiff_t s, std::ptrdiff_t e) {
                               ctr += e - s;
                             });
  ASSERT_EQ(ctr.load(), num_tasks);
}

// Default (-1): iteration-count-based spinning (original behavior).
TEST(ThreadPoolTest, SpinDurationDefault) {
  TestSpinDurationMode(onnxruntime::concurrency::kSpinDurationDefault);
}

// Zero: no spinning — threads block immediately when idle.
TEST(ThreadPoolTest, SpinDurationZero_NoSpinning) {
  TestSpinDurationMode(0);
}

// Positive: time-based spinning with a short duration.
TEST(ThreadPoolTest, SpinDurationPositive_TimeBased) {
  TestSpinDurationMode(100);   // 100us
  TestSpinDurationMode(1000);  // 1ms
}

// Smoke test: exponential backoff (spin_backoff_max > 1) produces correct results.
void TestSpinBackoffMode(int spin_duration_us, unsigned int spin_backoff_max) {
  constexpr int num_threads = 4;
  constexpr std::ptrdiff_t num_tasks = 1024;
  auto tp = std::make_unique<ThreadPool>(&onnxruntime::Env::Default(),
                                         onnxruntime::ThreadOptions(),
                                         nullptr,
                                         num_threads,
                                         spin_duration_us,
                                         /*force_hybrid*/ false,
                                         spin_backoff_max);
  std::atomic<std::ptrdiff_t> ctr{0};
  ThreadPool::TryParallelFor(tp.get(), num_tasks, 0.0,
                             [&](std::ptrdiff_t s, std::ptrdiff_t e) {
                               ctr += e - s;
                             });
  ASSERT_EQ(ctr.load(), num_tasks);
}

TEST(ThreadPoolTest, SpinBackoffDefault_NoBackoff) {
  TestSpinBackoffMode(onnxruntime::concurrency::kSpinDurationDefault, 1U);
}

TEST(ThreadPoolTest, SpinBackoffEnabled) {
  TestSpinBackoffMode(onnxruntime::concurrency::kSpinDurationDefault, 4U);
  TestSpinBackoffMode(onnxruntime::concurrency::kSpinDurationDefault, 8U);
}

TEST(ThreadPoolTest, SpinBackoffWithTimeBoundedSpin) {
  TestSpinBackoffMode(1000, 8U);  // 1ms + backoff cap 8
}

// Tests for sizing default (thread_pool_size <= 0) pools from the environment.
// Each test pins both variables so ambient values cannot leak in.
// DegreeOfParallelism scales by a granularity factor on hybrid CPUs, so tests compare an
// env-sized pool against a reference pool of the intended explicit size rather than asserting
// absolute values.
namespace {
int PoolDegreeWithEnv(const test::EnvVarMap& env_vars, concurrency::ThreadPoolType tpool_type,
                      int thread_pool_size = 0) {
  test::ScopedEnvironmentVariables scoped_env(env_vars);
  OrtThreadPoolParams tpo;
  tpo.thread_pool_size = thread_pool_size;
  auto tp = concurrency::CreateThreadPool(&Env::Default(), tpo, tpool_type);
  return concurrency::ThreadPool::DegreeOfParallelism(tp.get());
}

const optional<std::string> kUnset{};
const test::EnvVarMap kAllUnset{{"ORT_INTRA_OP_NUM_THREADS", kUnset},
                                {"ORT_INTER_OP_NUM_THREADS", kUnset}};

int PoolDegreeForExplicitSize(int thread_pool_size) {
  return PoolDegreeWithEnv(kAllUnset, concurrency::ThreadPoolType::INTRA_OP, thread_pool_size);
}
}  // namespace

TEST(ThreadPoolTest, DefaultPoolSizeFromOrtEnvVars) {
  EXPECT_EQ(PoolDegreeWithEnv({{"ORT_INTRA_OP_NUM_THREADS", "3"},
                               {"ORT_INTER_OP_NUM_THREADS", kUnset}},
                              concurrency::ThreadPoolType::INTRA_OP),
            PoolDegreeForExplicitSize(3));
  EXPECT_EQ(PoolDegreeWithEnv({{"ORT_INTRA_OP_NUM_THREADS", kUnset},
                               {"ORT_INTER_OP_NUM_THREADS", "3"}},
                              concurrency::ThreadPoolType::INTER_OP),
            PoolDegreeForExplicitSize(3));
}

TEST(ThreadPoolTest, ExplicitPoolSizeWinsOverEnvVars) {
  EXPECT_EQ(PoolDegreeWithEnv({{"ORT_INTRA_OP_NUM_THREADS", "2"},
                               {"ORT_INTER_OP_NUM_THREADS", kUnset}},
                              concurrency::ThreadPoolType::INTRA_OP,
                              /*thread_pool_size=*/4),
            PoolDegreeForExplicitSize(4));
}

TEST(ThreadPoolTest, OrtEnvVarZeroRestoresMachineSizedDefault) {
  // An explicit 0 opts back into the machine-sized default.
  EXPECT_EQ(PoolDegreeWithEnv({{"ORT_INTRA_OP_NUM_THREADS", "0"},
                               {"ORT_INTER_OP_NUM_THREADS", kUnset}},
                              concurrency::ThreadPoolType::INTRA_OP),
            PoolDegreeWithEnv(kAllUnset, concurrency::ThreadPoolType::INTRA_OP));
}

#ifndef ORT_NO_EXCEPTIONS
TEST(ThreadPoolTest, InvalidOrtEnvVarValueThrows) {
  EXPECT_THROW(PoolDegreeWithEnv({{"ORT_INTRA_OP_NUM_THREADS", "-1"},
                                  {"ORT_INTER_OP_NUM_THREADS", kUnset}},
                                 concurrency::ThreadPoolType::INTRA_OP),
               OnnxRuntimeException);
}
#endif

// -------------------------------------------------------------------
// Accuracy guards for the intra-op threading tuning: the spin_backoff_max
// default and the ParallelFor go/no-go cost scale.
//
// Neither knob is meant to change a single numeric result. The cost scale only
// moves the point at which ParallelFor decides a loop is worth splitting (block
// sizing keeps using the unscaled cost), and spin_backoff_max only changes how
// an idle worker waits. Both, though, change *how* - and whether - a loop is
// carved into blocks, so a loop body that is not invariant under
// re-partitioning would start drifting silently. These tests pin that
// invariance down across the whole space of partitionings ParallelFor can hand
// out: thread count, dynamic_block_base_, per-iteration cost and backoff cap.
//
// Note on the cost scale itself: ParallelForCostScale() in threadpool.cc caches
// the ORT_PARALLEL_COST_SCALE lookup in a function-local static, so its value is
// fixed for the lifetime of the process and cannot be varied from inside a
// single test binary. Rather than assert on one scale, these tests assert the
// property that has to hold at *every* scale, and CostBasedSplitDecisionIsAThreshold
// checks that the scale in force behaves as a clean threshold. Running the
// binary a second time with ORT_PARALLEL_COST_SCALE=1 exercises the pre-change
// decision point against the same assertions.
// -------------------------------------------------------------------
namespace {

// Per-iteration body. Strongly index-dependent, so an index that is skipped,
// visited twice, or handed to the wrong block shows up immediately - but built
// only from exactly-representable float arithmetic (integers below 2^24 scaled
// by powers of two), so every operation is exact. That keeps the expected
// result independent of vectorization and of whether the compiler contracts a
// multiply-add into an FMA, which would otherwise make a bitwise comparison
// between the serial and parallel code paths flaky rather than meaningful.
float ElementKernel(std::ptrdiff_t i) {
  float acc = static_cast<float>(i % 97);
  for (int k = 0; k < 8; ++k) {
    acc = acc * 2.0f + static_cast<float>((i + k) % 13);
  }
  return acc * 0.25f;
}

std::vector<float> SerialReference(std::ptrdiff_t n) {
  std::vector<float> out(static_cast<size_t>(n));
  for (std::ptrdiff_t i = 0; i < n; ++i) {
    out[static_cast<size_t>(i)] = ElementKernel(i);
  }
  return out;
}

using Block = std::pair<std::ptrdiff_t, std::ptrdiff_t>;

// Runs the kernel over [0, n) through TryParallelFor with the given per-unit
// cost, filling `out` and returning the block boundaries ParallelFor handed out,
// sorted by start index.
//
// The blocks are recorded without a lock by giving each one a slot it owns
// outright: its start index. There are at most n blocks and their starts are
// distinct, so a vector sized n upfront gives every block a private destination
// and no two threads ever touch the same element. A block's end index is never
// negative, so kUnclaimed marks a slot no block wrote to. Compacting the
// claimed slots in index order then yields exactly the sorted block list,
// without a separate sort.
std::vector<Block> RunElementwise(ThreadPool* tp, std::ptrdiff_t n,
                                  const onnxruntime::TensorOpCost& cost,
                                  std::vector<float>& out) {
  constexpr std::ptrdiff_t kUnclaimed = -1;
  out.assign(static_cast<size_t>(n), 0.0f);
  std::vector<Block> blocks(static_cast<size_t>(n), Block{0, kUnclaimed});
  ThreadPool::TryParallelFor(tp, n, cost, [&](std::ptrdiff_t first, std::ptrdiff_t last) {
    for (std::ptrdiff_t i = first; i < last; ++i) {
      out[static_cast<size_t>(i)] = ElementKernel(i);
    }
    blocks[static_cast<size_t>(first)] = Block{first, last};
  });
  // TryParallelFor joins every block before returning, so those writes are
  // visible here.
  size_t claimed = 0;
  for (size_t i = 0; i < blocks.size(); ++i) {
    if (blocks[i].second != kUnclaimed) {
      blocks[claimed++] = blocks[i];
    }
  }
  blocks.resize(claimed);
  return blocks;
}

// The contract every ParallelFor caller relies on: the blocks tile [0, n)
// exactly once, in increasing order, with no gap and no overlap.
void ExpectBlocksTileRange(const std::vector<Block>& blocks, std::ptrdiff_t n,
                           const std::string& context) {
  ASSERT_FALSE(blocks.empty()) << context;
  std::ptrdiff_t expected_start = 0;
  for (const auto& block : blocks) {
    ASSERT_EQ(block.first, expected_start) << context;
    ASSERT_GT(block.second, block.first) << context;
    expected_start = block.second;
  }
  ASSERT_EQ(expected_start, n) << context;
}

// Reports the first differing index rather than dumping both vectors, which
// would be tens of thousands of values wide for the larger loop sizes.
void ExpectSameValues(const std::vector<float>& actual, const std::vector<float>& expected,
                      const std::string& context) {
  ASSERT_EQ(actual.size(), expected.size()) << context;
  size_t mismatches = 0;
  size_t first_mismatch = 0;
  for (size_t i = 0; i < actual.size(); ++i) {
    if (actual[i] != expected[i]) {
      if (mismatches == 0) {
        first_mismatch = i;
      }
      ++mismatches;
    }
  }
  ASSERT_EQ(mismatches, 0u) << context << ": " << mismatches << " of " << actual.size()
                            << " values differ; first at index " << first_mismatch << " ("
                            << actual[first_mismatch] << " vs " << expected[first_mismatch] << ")";
}

std::unique_ptr<ThreadPool> MakePool(int num_threads, int dynamic_block_base = 0,
                                     unsigned int spin_backoff_max = 1U) {
  if (num_threads <= 0) {
    return nullptr;  // exercises the no-pool path
  }
  onnxruntime::ThreadOptions thread_options;
  thread_options.dynamic_block_base_ = dynamic_block_base;
  return std::make_unique<ThreadPool>(&onnxruntime::Env::Default(),
                                      thread_options,
                                      nullptr,
                                      num_threads,
                                      onnxruntime::concurrency::kSpinDurationDefault,
                                      /*force_hybrid*/ false,
                                      spin_backoff_max);
}

// Per-iteration costs spanning both sides of Eigen's parallelization threshold
// (kStartupCycles = 100000) for the loop sizes used below, at any cost scale
// between 1 and a few hundred.
const std::vector<double> kCostSweep{0.0, 1.0, 4.0, 16.0, 20.0, 64.0, 256.0,
                                     1024.0, 4096.0, 16384.0, 262144.0};

}  // namespace

// The new default. Guards against a rebase or a merge quietly restoring 1.
TEST(ThreadPoolTest, SpinBackoffMaxDefaultIsTwo) {
  const OrtThreadPoolParams defaults;
  EXPECT_EQ(defaults.spin_backoff_max, 2U);
  EXPECT_LE(defaults.spin_backoff_max, concurrency::kSpinBackoffMaxLimit);
}

// Core accuracy guard for the cost-scale change: whatever the scale decides,
// the values produced must be bit-identical to the serial computation. Sweeping
// thread count x dynamic_block_base_ x per-iteration cost covers both the
// "stayed serial" and the "got split" outcomes, and every block size the
// scheduler can pick in between.
TEST(ThreadPoolTest, ParallelForResultsAreBitIdenticalUnderAllPartitionings) {
  for (std::ptrdiff_t n : {std::ptrdiff_t{1}, std::ptrdiff_t{2}, std::ptrdiff_t{97},
                           std::ptrdiff_t{1024}, std::ptrdiff_t{4096}, std::ptrdiff_t{65537}}) {
    const std::vector<float> reference = SerialReference(n);
    for (int num_threads : {0, 1, 2, 4, 8}) {
      for (int dynamic_block_base : {0, 1, 4}) {
        auto tp = MakePool(num_threads, dynamic_block_base);
        for (double compute_cycles : kCostSweep) {
          const onnxruntime::TensorOpCost cost{0.0, 0.0, compute_cycles};
          std::vector<float> actual;
          const auto blocks = RunElementwise(tp.get(), n, cost, actual);

          const std::string context = "n=" + std::to_string(n) +
                                      " threads=" + std::to_string(num_threads) +
                                      " dynamic_block_base=" + std::to_string(dynamic_block_base) +
                                      " compute_cycles=" + std::to_string(compute_cycles);
          ASSERT_NO_FATAL_FAILURE(ExpectBlocksTileRange(blocks, n, context));
          // Bit-identical, not merely close: re-partitioning an elementwise loop
          // must not perturb a single result.
          ASSERT_NO_FATAL_FAILURE(ExpectSameValues(actual, reference, context));
        }
      }
    }
  }
}

// Same guard for the spin_backoff_max change. Backoff only affects how idle
// workers wait, so results must be identical for every legal value, including
// values above kSpinBackoffMaxLimit (which are clamped, not rejected).
TEST(ThreadPoolTest, SpinBackoffDoesNotChangeParallelForResults) {
  constexpr std::ptrdiff_t n = 4096;
  const std::vector<float> reference = SerialReference(n);
  for (unsigned int spin_backoff_max : {1U, 2U, 4U, 8U,
                                        concurrency::kSpinBackoffMaxLimit,
                                        concurrency::kSpinBackoffMaxLimit * 4U}) {
    for (int num_threads : {1, 2, 4}) {
      auto tp = MakePool(num_threads, /*dynamic_block_base*/ 0, spin_backoff_max);
      for (double compute_cycles : {0.0, 20.0, 4096.0}) {
        std::vector<float> actual;
        const auto blocks = RunElementwise(tp.get(), n,
                                           onnxruntime::TensorOpCost{0.0, 0.0, compute_cycles},
                                           actual);
        const std::string context = "spin_backoff_max=" + std::to_string(spin_backoff_max) +
                                    " threads=" + std::to_string(num_threads) +
                                    " compute_cycles=" + std::to_string(compute_cycles);
        ASSERT_NO_FATAL_FAILURE(ExpectBlocksTileRange(blocks, n, context));
        ASSERT_NO_FATAL_FAILURE(ExpectSameValues(actual, reference, context));
      }
    }
  }
}

// The scaled cost is only allowed to move the go/no-go point, so the decision
// must stay a monotone threshold in the per-iteration cost: once a loop of a
// given size is worth splitting, a more expensive one is too. Holds at any
// ORT_PARALLEL_COST_SCALE, and would catch the scale leaking into the block
// sizing (which still uses the unscaled cost) as a non-monotone block count.
TEST(ThreadPoolTest, CostBasedSplitDecisionIsAThreshold) {
  constexpr std::ptrdiff_t n = 65537;
  auto tp = MakePool(4);
  bool seen_split = false;
  for (double compute_cycles : kCostSweep) {
    std::vector<float> actual;
    const auto blocks = RunElementwise(tp.get(), n,
                                       onnxruntime::TensorOpCost{0.0, 0.0, compute_cycles},
                                       actual);
    const bool split = blocks.size() > 1;
    const std::string context = "compute_cycles=" + std::to_string(compute_cycles);
    if (seen_split) {
      EXPECT_TRUE(split) << "split decision is not monotone in cost at " << context;
    }
    seen_split = seen_split || split;
  }
  // The sweep spans several orders of magnitude either side of Eigen's
  // threshold, so at least the top of it has to parallelize on a 4-thread pool.
  EXPECT_TRUE(seen_split) << "no cost in the sweep parallelized; the cost model or "
                             "ORT_PARALLEL_COST_SCALE is not doing anything";
}

// Pins ORT_DEFAULT_PARALLEL_COST_SCALE=8 and verifies it causes a different
// split decision than scale=1 at a known boundary.
//
// With n=65537 and compute_cycles=1.0 on a 4-thread pool:
//   scale=1 : n * cost =     65,537 < Eigen kStartupCycles (100,000) → serial
//   scale=8 : n * cost * 8 = 524,296 > kStartupCycles               → split
//
// If this fails, ORT_DEFAULT_PARALLEL_COST_SCALE was reverted to 1 or the
// scale is not reaching the split decision in ParallelFor.
TEST(ThreadPoolTest, CostScaleDefaultSplitsAtKnownBoundary) {
  constexpr std::ptrdiff_t n = 65537;
  auto tp = MakePool(4);
  std::vector<float> actual;
  const auto blocks = RunElementwise(tp.get(), n, onnxruntime::TensorOpCost{0.0, 0.0, 1.0}, actual);
  EXPECT_GT(blocks.size(), 1u)
      << "n=" << n << " compute_cycles=1.0 must split at ORT_DEFAULT_PARALLEL_COST_SCALE=8 "
         "(would stay serial at scale=1); constant was reverted or scale is being bypassed";
}

}  // namespace onnxruntime
