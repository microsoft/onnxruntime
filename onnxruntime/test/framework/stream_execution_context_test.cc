// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <atomic>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

#include "core/framework/run_options.h"
#include "core/framework/stream_execution_context.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

using namespace std::chrono_literals;

TEST(StreamExecutionContextTest, CountDownBarrierZeroInitialCount) {
  StreamExecutionContext::CountDownBarrier barrier;
  barrier.Wait();
  EXPECT_EQ(barrier.Get(), 0);
}

TEST(StreamExecutionContextTest, CountDownBarrierDelayedFinalDecrementPublishesWrites) {
  StreamExecutionContext::CountDownBarrier barrier;
  barrier.Set(1);
  int published_value = 0;

  std::thread worker([&]() {
    std::this_thread::sleep_for(10ms);
    published_value = 42;
    EXPECT_TRUE(barrier.Dec());
  });

  barrier.Wait();
  EXPECT_EQ(published_value, 42);
  worker.join();
}

TEST(StreamExecutionContextTest, CountDownBarrierMultipleDecrementsPublishWrites) {
  StreamExecutionContext::CountDownBarrier barrier;
  constexpr size_t kWorkerCount = 4;
  barrier.Set(static_cast<int32_t>(kWorkerCount));
  std::array<int, kWorkerCount> published_values{};
  std::vector<std::thread> workers;
  workers.reserve(kWorkerCount);

  for (size_t i = 0; i < kWorkerCount; ++i) {
    workers.emplace_back([&, i]() {
      published_values[i] = static_cast<int>(i + 1);
      barrier.Dec();
    });
  }

  barrier.Wait();
  for (size_t i = 0; i < kWorkerCount; ++i) {
    EXPECT_EQ(published_values[i], static_cast<int>(i + 1));
  }

  for (auto& worker : workers) {
    worker.join();
  }
}

TEST(StreamExecutionContextTest, CountDownBarrierMultipleWaitersHaveNoLostWakeup) {
  StreamExecutionContext::CountDownBarrier barrier;
  constexpr size_t kWaiterCount = 8;
  barrier.Set(1);
  std::atomic<size_t> ready_count{0};
  std::atomic<size_t> completed_count{0};
  std::vector<std::thread> waiters;
  waiters.reserve(kWaiterCount);

  for (size_t i = 0; i < kWaiterCount; ++i) {
    waiters.emplace_back([&]() {
      ready_count.fetch_add(1, std::memory_order_release);
      ready_count.notify_one();
      barrier.Wait();
      completed_count.fetch_add(1, std::memory_order_relaxed);
    });
  }

  size_t ready = ready_count.load(std::memory_order_acquire);
  while (ready != kWaiterCount) {
    ready_count.wait(ready, std::memory_order_acquire);
    ready = ready_count.load(std::memory_order_acquire);
  }

  barrier.Dec();
  for (auto& waiter : waiters) {
    waiter.join();
  }

  EXPECT_EQ(completed_count.load(std::memory_order_relaxed), kWaiterCount);
}

TEST(StreamExecutionContextTest, CountDownBarrierSupportsDynamicTaskAddition) {
  StreamExecutionContext::CountDownBarrier barrier;
  barrier.Set(1);
  barrier.Inc();

  std::thread first([&]() { barrier.Dec(); });
  std::thread second([&]() { barrier.Dec(); });
  barrier.Wait();
  first.join();
  second.join();

  EXPECT_EQ(barrier.Get(), 0);
}

TEST(StreamExecutionContextTest, FirstFailureStatusPublishesOneCompetingFailure) {
  StreamExecutionContext::FirstFailureStatus first_failure;
  constexpr size_t kWorkerCount = 16;
  std::atomic<bool> start{false};
  std::vector<std::thread> workers;
  workers.reserve(kWorkerCount);

  for (size_t i = 0; i < kWorkerCount; ++i) {
    workers.emplace_back([&, i]() {
      start.wait(false, std::memory_order_acquire);
      first_failure.SetStatus(
          ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "worker failure ", i));
    });
  }

  start.store(true, std::memory_order_release);
  start.notify_all();
  for (auto& worker : workers) {
    worker.join();
  }

  const Status status = first_failure.GetStatus();
  ASSERT_FALSE(status.IsOK());
  EXPECT_EQ(status.ErrorMessage().find("worker failure "), 0);

  first_failure.SetStatus(ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "late failure"));
  EXPECT_EQ(first_failure.GetStatus().ErrorMessage(), status.ErrorMessage());
}

TEST(RunOptionsTerminationTest, SnapshotFanoutAndReset) {
  RunOptions run_options;
  const auto first_token = run_options.GetTerminateToken();
  const auto second_token = run_options.GetTerminateToken();

  run_options.RequestTerminate();
  EXPECT_TRUE(first_token.stop_requested());
  EXPECT_TRUE(second_token.stop_requested());

  run_options.ResetTerminate();
  const auto reset_token = run_options.GetTerminateToken();
  EXPECT_FALSE(reset_token.stop_requested());
  EXPECT_TRUE(first_token.stop_requested());
  EXPECT_TRUE(second_token.stop_requested());

  RunOptions copied_options = run_options;
  copied_options.ResetTerminate();
  const auto copied_token = copied_options.GetTerminateToken();
  EXPECT_FALSE(copied_token.stop_requested());
  copied_options.RequestTerminate();
  EXPECT_TRUE(copied_token.stop_requested());
  EXPECT_FALSE(reset_token.stop_requested());
  EXPECT_TRUE(first_token.stop_requested());
}

}  // namespace test
}  // namespace onnxruntime
