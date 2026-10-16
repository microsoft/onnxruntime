// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <functional>
#include <future>
#include <memory>
#include <thread>
#include <vector>

#include "core/framework/cancellation.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

using namespace std::chrono_literals;

TEST(CancellationTest, RegisteringAfterStopInvokesOnRegisteringThread) {
  CancellationSource source;
  ASSERT_TRUE(source.request_stop());

  std::thread::id callback_thread;
  CancellationCallback callback(source.get_token(), [&]() noexcept {
    callback_thread = std::this_thread::get_id();
  });

  EXPECT_EQ(callback_thread, std::this_thread::get_id());
}

TEST(CancellationTest, ConcurrentStopRequestsInvokeCallbackExactlyOnce) {
  CancellationSource source;
  std::atomic<size_t> callback_count{0};
  CancellationCallback callback(source.get_token(), [&]() noexcept {
    callback_count.fetch_add(1, std::memory_order_relaxed);
  });

  constexpr size_t kThreadCount = 16;
  std::atomic<size_t> ready_count{0};
  std::atomic<bool> start{false};
  std::array<bool, kThreadCount> stopped{};
  std::vector<std::thread> threads;
  threads.reserve(kThreadCount);
  for (size_t i = 0; i < kThreadCount; ++i) {
    threads.emplace_back([&, i, source]() mutable {
      ready_count.fetch_add(1, std::memory_order_release);
      ready_count.notify_one();
      start.wait(false, std::memory_order_acquire);
      stopped[i] = source.request_stop();
    });
  }

  size_t ready = ready_count.load(std::memory_order_acquire);
  while (ready != kThreadCount) {
    ready_count.wait(ready, std::memory_order_acquire);
    ready = ready_count.load(std::memory_order_acquire);
  }
  start.store(true, std::memory_order_release);
  start.notify_all();

  for (auto& thread : threads) {
    thread.join();
  }

  EXPECT_EQ(callback_count.load(std::memory_order_relaxed), 1U);
  EXPECT_EQ(std::count(stopped.begin(), stopped.end(), true), 1);
}

TEST(CancellationTest, CallbackDestructionWaitsForConcurrentInvocation) {
  CancellationSource source;
  std::atomic<bool> callback_started{false};
  std::atomic<bool> release_callback{false};
  using Callback = CancellationCallback<std::function<void()>>;
  auto callback = std::make_unique<Callback>(source.get_token(), [&]() noexcept {
    callback_started.store(true, std::memory_order_release);
    callback_started.notify_one();
    release_callback.wait(false, std::memory_order_acquire);
  });

  std::thread requester([&]() { source.request_stop(); });
  callback_started.wait(false, std::memory_order_acquire);

  std::atomic<bool> destroy_started{false};
  auto destroy_result = std::async(std::launch::async, [&]() {
    destroy_started.store(true, std::memory_order_release);
    destroy_started.notify_one();
    callback.reset();
  });
  destroy_started.wait(false, std::memory_order_acquire);
  EXPECT_EQ(destroy_result.wait_for(20ms), std::future_status::timeout);

  release_callback.store(true, std::memory_order_release);
  release_callback.notify_one();
  EXPECT_EQ(destroy_result.wait_for(1s), std::future_status::ready);
  destroy_result.get();
  requester.join();
}

TEST(CancellationTest, CallbackCanReleaseEveryStateOwnerDuringRequestStop) {
  auto source = std::make_unique<CancellationSource>();
  auto token = source->get_token();
  using Callback = CancellationCallback<std::function<void()>>;
  std::unique_ptr<Callback> callback;
  bool invoked = false;
  callback = std::make_unique<Callback>(token, [&]() noexcept {
    source.reset();
    invoked = true;
    // Destroy the executing closure last; accessing captures after this is undefined.
    callback.reset();
  });
  token = {};

  auto* source_ptr = source.get();
  EXPECT_TRUE(source_ptr->request_stop());
  EXPECT_TRUE(invoked);
  EXPECT_EQ(source, nullptr);
  EXPECT_EQ(callback, nullptr);
}

}  // namespace test
}  // namespace onnxruntime
