// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/Barrier.h"
#include "core/platform/threadpool.h"


#include "gtest/gtest.h"

#include <atomic>
#include <thread>

namespace {

static void TestBarrier(int num_threads, uint64_t per_thread_count, bool spin) {
  std::atomic<uint64_t> counter{0};
  onnxruntime::Barrier barrier(num_threads, spin);

  std::vector<std::thread> threads;
  for (auto i = 0; i < num_threads + 1; i++) {
    threads.push_back(std::thread([&, i] {
      if (i > 0) {
        // Worker thread; increment the shared counter then
        // notify the barrier.
        for (uint64_t j = 0; j < per_thread_count; j++) {
          counter++;
        }
        barrier.Notify();
      }  else {
        // Main thread; wait on the barrier, and then check the count seen.
        barrier.Wait();
        ASSERT_EQ(counter, per_thread_count * num_threads);
      } 
    }));
  }

  // Wait for the threads to finish
  for (auto &t : threads) {
    t.join();
  }
}

}  // namespace

namespace onnxruntime {

constexpr uint64_t count = 1000000ull;

TEST(BarrierTest, TestBarrier_0Workers_Spin) {
  TestBarrier(0, count, true);
}

TEST(BarrierTest, TestBarrier_0Workers_Block) {
  TestBarrier(0, count, false);
}

TEST(BarrierTest, TestBarrier_1Worker_Spin) {
  TestBarrier(1, count, true);
}

TEST(BarrierTest, TestBarrier_1Worker_Block) {
  TestBarrier(1, count, false);
}

TEST(BarrierTest, TestBarrier_4Workers_Spin) {
  TestBarrier(4, count, true);
}

TEST(BarrierTest, TestBarrier_4Workers_Block) {
  TestBarrier(4, count, false);
}

}  // namespace onnxruntime
