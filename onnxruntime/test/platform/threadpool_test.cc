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
#include "test/platform/windows/env.h"
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

void ValidateTestData(TestData& test_data, int expected=1) {
  ASSERT_TRUE(std::count_if(test_data.data.cbegin(), test_data.data.cend(), [&](int i) { return i != expected; }) == 0);
}

// Run a test with a new thread pool created with num_threads threads
// in total (including the main thread).  If num_threads is 0 then we
// test the function with a null pointer, reflecting scenarios where we
// run with just the main thread.  Note that the thread pool API uses
// static methods and should operate across all of these cases.
void CreateThreadPoolAndTest(const std::string&, int num_threads, const std::function<void(ThreadPool*)>& test_body) {
  if (num_threads > 0) {
    auto tp = onnxruntime::make_unique<ThreadPool>(&onnxruntime::Env::Default(), onnxruntime::ThreadOptions(), nullptr,
                                                   num_threads, true);
    test_body(tp.get());
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

void TestConcurrentParallelFor(const std::string& name, int num_threads, int num_concurrent, int num_tasks) {
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
    auto tp = onnxruntime::make_unique<ThreadPool>(&onnxruntime::Env::Default(),
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

void TestMultiLoopSections(const std::string& name, int num_threads, int num_loops) {
  for (int rep = 0; rep < 5; rep++) {
    const int num_tasks = 1024;
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

TEST(ThreadPoolTest, TestConcurrentParallelFor_4Thread_4Conc_1Tasks) {
  TestConcurrentParallelFor("TestConcurrentParallelFor_4Thread_4Conc_1Tasks", 4, 4, 1);
}

TEST(ThreadPoolTest, TestConcurrentParallelFor_4Thread_4Conc_8Tasks) {
  TestConcurrentParallelFor("TestConcurrentParallelFor_4Thread_4Conc_8Tasks", 4, 4, 8);
}

TEST(ThreadPoolTest, TestConcurrentParallelFor_4Thread_4Conc_1MTasks) {
  TestConcurrentParallelFor("TestConcurrentParallelFor_4Thread_4Conc_1MTasks", 4, 4, 1000000);
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
#endif

#ifdef _WIN32
void TestAffinityString(const Env& env, int32_t num_group, int32_t num_logical_processor) {
  std::vector<uint64_t> default_affinities;
  int default_threadpool_size = static_cast<int>(env.GetDefaultThreadpoolSetting(default_affinities));
  ASSERT_TRUE(default_threadpool_size > 0);
  int num_non_default_num_threads;
  std::vector<uint64_t> non_default_affinities;
  for (int32_t num_threads = num_group; num_threads <= num_logical_processor; num_threads += 1) {
    if (num_logical_processor % num_threads) {
      continue;
    }
    int32_t num_logical_processors_per_thread = num_logical_processor / num_threads;
    std::stringstream affinity_stream_comma;
    std::stringstream affinity_stream_hyphen;
    for (int32_t processor_id_start = 1; processor_id_start <= num_logical_processor; processor_id_start += num_logical_processors_per_thread) {
      int32_t processor_id_end = processor_id_start + num_logical_processors_per_thread - 1;
      if (processor_id_start > 1) {
        affinity_stream_comma << ";";
        affinity_stream_hyphen << ";";
      }
      for (int32_t i = processor_id_start; i <= processor_id_end; ++i) {
        affinity_stream_comma << (i == processor_id_start ? "" : ",") << i;
      }
      affinity_stream_hyphen << processor_id_start << "-" << processor_id_end;
    }
    non_default_affinities.clear();
    num_non_default_num_threads = static_cast<int>(env.ReadThreadAffinityConfig(affinity_stream_comma.str(), non_default_affinities));
    ASSERT_TRUE(num_non_default_num_threads == num_threads);
    non_default_affinities.clear();
    num_non_default_num_threads = static_cast<int>(env.ReadThreadAffinityConfig(affinity_stream_hyphen.str(), non_default_affinities));
    ASSERT_TRUE(num_non_default_num_threads == num_threads);
  }
}

void TestAffinityStringMisshaped(const Env& env) {
  std::vector<uint64_t> default_affinities;
  int default_threadpool_size = static_cast<int>(env.GetDefaultThreadpoolSetting(default_affinities));
  ASSERT_TRUE(default_threadpool_size > 0);
  // test only one machines has more than 8 logical processors
  if (default_threadpool_size >= 4) {
    std::vector<uint64_t> affinities;
    // cases - test processor id that's out of boundary
    ASSERT_TRUE(0 == env.ReadThreadAffinityConfig("0", affinities));
    ASSERT_TRUE(0 == env.ReadThreadAffinityConfig("1000000", affinities));
    // cases - test empty strings
    ASSERT_TRUE(0 == env.ReadThreadAffinityConfig(",", affinities));
    ASSERT_TRUE(0 == env.ReadThreadAffinityConfig(",1", affinities));
    ASSERT_TRUE(0 == env.ReadThreadAffinityConfig(";", affinities));
    ASSERT_TRUE(0 == env.ReadThreadAffinityConfig(";1", affinities));
    // cases - test invalid chars
    ASSERT_TRUE(0 == env.ReadThreadAffinityConfig("2,a,1", affinities));
    ASSERT_TRUE(0 == env.ReadThreadAffinityConfig("*4,3,)", affinities));
    ASSERT_TRUE(0 == env.ReadThreadAffinityConfig("3,5,1;^.2", affinities));
    // cases - test comma wrongly coupled with hyphen
    ASSERT_TRUE(0 == env.ReadThreadAffinityConfig("1,-5", affinities));
    ASSERT_TRUE(0 == env.ReadThreadAffinityConfig("2,3,-", affinities));
    ASSERT_TRUE(0 == env.ReadThreadAffinityConfig("3,5-1;6,8", affinities));
    ASSERT_TRUE(0 == env.ReadThreadAffinityConfig("-,-", affinities));
    ASSERT_TRUE(0 == env.ReadThreadAffinityConfig("1-2;3-4,", affinities));
    // cases - test hyphen of invalid intervals
    ASSERT_TRUE(0 == env.ReadThreadAffinityConfig("6-2", affinities));
    ASSERT_TRUE(0 == env.ReadThreadAffinityConfig("-2", affinities));
    ASSERT_TRUE(0 == env.ReadThreadAffinityConfig("1-2;4-3", affinities));
    ASSERT_TRUE(0 == env.ReadThreadAffinityConfig("1--2;3-4", affinities));
    ASSERT_TRUE(0 == env.ReadThreadAffinityConfig("1-3-4", affinities));
  }
}

test::CpuInfo MockCpuInfo(int32_t num_group, int32_t core_per_group, int32_t logical_processor_per_core) {
  ORT_ENFORCE(num_group > 0 && core_per_group > 0 && logical_processor_per_core > 0);
  test::CpuInfo cpu_info;
  for (int32_t group = 0; group < num_group; group++) {
    test::CpuCores cpu_cores;
    int32_t logical_processor_start = 0;
    for (int32_t core = 0; core < core_per_group; core++) {
      test::CpuCore cpu_core;
      int32_t logical_processor_end = logical_processor_start + logical_processor_per_core;
      for (int32_t processor = logical_processor_start; processor < logical_processor_end; processor++) {
        cpu_core.push_back(processor);
      }
      cpu_cores.push_back(std::move(cpu_core));
      logical_processor_start = logical_processor_end;
    }
    cpu_info.push_back(std::move(cpu_cores));
  }
  return cpu_info;
}

void TestWindowsNuma(int32_t num_group, int32_t core_per_group, int32_t logical_processor_per_core) {
  ORT_ENFORCE(num_group > 0 &&
              core_per_group > 0 &&
              logical_processor_per_core > 0 &&
              (logical_processor_per_core & (logical_processor_per_core - 1)) == 0);

  test::WindowsEnvTester env;
  env.SetCpuInfo(MockCpuInfo(num_group, core_per_group, logical_processor_per_core));

  std::vector<uint64_t> default_affinities;
  int default_threadpool_size = static_cast<int>(env.GetDefaultThreadpoolSetting(default_affinities));

  int32_t total_core = core_per_group * num_group;
  int32_t total_processors = total_core * logical_processor_per_core;
  ASSERT_TRUE(default_threadpool_size == total_core);
  ASSERT_TRUE(default_affinities.size() == default_threadpool_size * 2);

  uint64_t bits = (1ULL << logical_processor_per_core) - 1;
  int32_t num_bit = logical_processor_per_core;
  
  for (int32_t group = 0; group < num_group; ++group) {
    for (int32_t core = 0; core < core_per_group; ++core) {
      int32_t thread_id = group * core_per_group + core;
      int32_t affinity_index = thread_id * 2;

      ASSERT_TRUE(default_affinities[affinity_index] == static_cast<uint64_t>(group));
      ASSERT_TRUE(default_affinities[affinity_index + 1] == bits << (core * num_bit));
    }
  }

  // test non-default settings
  TestAffinityString(env, num_group, total_processors);
  TestAffinityStringMisshaped(env);
}

TEST(NumaTest, 1_Group_16_core_64_Processor) {
  TestWindowsNuma(1, 16, 4);
}

TEST(NumaTest, 1_Group_32_core_32_Processor) {
  TestWindowsNuma(1, 32, 1);
}

TEST(NumaTest, 1_Group_32_core_64_Processor) {
  TestWindowsNuma(1, 32, 2);
}

TEST(NumaTest, 1_Group_64_core_64_Processor) {
  TestWindowsNuma(1, 64, 1);
}

TEST(NumaTest, 2_Group_32_core_128_Processor) {
  TestWindowsNuma(2, 16, 4);
}

TEST(NumaTest, 2_Group_64_core_128_Processor) {
  TestWindowsNuma(2, 32, 2);
}

TEST(NumaTest, 2_Group_128_core_128_Processor) {
  TestWindowsNuma(2, 64, 1);
}

TEST(NumaTest, 4_Group_128_core_256_Processor) {
  TestWindowsNuma(4, 32, 2);
}

TEST(NumaTest, 4_Group_256_core_256_Processor) {
  TestWindowsNuma(4, 64, 1);
}

#endif

}  // namespace onnxruntime
