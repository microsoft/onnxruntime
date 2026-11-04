// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <atomic>
#include <chrono>
#include <future>
#include <latch>
#include <memory>
#include <thread>

#include "core/providers/cuda/cuda_external_data_loader_gds.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {
namespace {

struct DriverState {
  std::atomic<size_t> initialize_attempts{0};
  std::atomic<size_t> opens{0};
  std::atomic<size_t> closes{0};
  std::atomic<size_t> active{0};
  bool fail_initialization{false};
  bool block_first_close{false};
  std::latch close_started{1};
  std::latch allow_close{1};
};

class TestDriver {
 public:
  explicit TestDriver(DriverState& state) : state_(state) {}

  ~TestDriver() {
    if (initialized_) {
      if (state_.block_first_close && state_.closes.load() == 0) {
        state_.close_started.count_down();
        state_.allow_close.wait();
      }
      --state_.active;
      ++state_.closes;
    }
  }

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(TestDriver);

  Status Initialize() {
    ++state_.initialize_attempts;
    ORT_RETURN_IF(state_.fail_initialization, "Driver initialization failed");
    ORT_RETURN_IF(state_.active.exchange(1) != 0, "Previous driver is still open");
    ++state_.opens;
    initialized_ = true;
    return Status::OK();
  }

 private:
  DriverState& state_;
  bool initialized_{false};
};

using DriverHandle = cuda::GdsDriverHandle<TestDriver>;

TEST(CudaGdsDriverTest, SharesDriverUntilLastHandleIsReleased) {
  DriverState state;
  {
    auto first = std::make_unique<DriverHandle>();
    auto second = std::make_unique<DriverHandle>();
    ASSERT_TRUE(first->Acquire(state).IsOK());
    ASSERT_TRUE(first->Acquire(state).IsOK());
    ASSERT_TRUE(second->Acquire(state).IsOK());
    EXPECT_EQ(first->operator->(), second->operator->());
    EXPECT_EQ(state.initialize_attempts.load(), 1U);

    first.reset();
    EXPECT_EQ(state.closes.load(), 0U);
    second.reset();
    EXPECT_EQ(state.closes.load(), 1U);
    EXPECT_EQ(state.active.load(), 0U);

    DriverHandle next;
    ASSERT_TRUE(next.Acquire(state).IsOK());
    EXPECT_EQ(state.opens.load(), 2U);
  }
  EXPECT_EQ(state.closes.load(), 2U);
}

TEST(CudaGdsDriverTest, FailedInitializationCanBeRetried) {
  DriverState state;
  state.fail_initialization = true;
  {
    DriverHandle handle;
    EXPECT_FALSE(handle.Acquire(state).IsOK());
    EXPECT_EQ(state.closes.load(), 0U);
    EXPECT_EQ(state.active.load(), 0U);

    state.fail_initialization = false;
    ASSERT_TRUE(handle.Acquire(state).IsOK());
    EXPECT_EQ(state.initialize_attempts.load(), 2U);
    EXPECT_EQ(state.opens.load(), 1U);
  }
  EXPECT_EQ(state.closes.load(), 1U);
}

TEST(CudaGdsDriverTest, ReacquireWaitsForFinalClose) {
  DriverState state;
  state.block_first_close = true;
  auto first = std::make_unique<DriverHandle>();
  ASSERT_TRUE(first->Acquire(state).IsOK());

  std::jthread releasing([handle = std::move(first)]() mutable { handle.reset(); });
  state.close_started.wait();

  std::latch acquire_started{1};
  std::promise<Status> acquired;
  auto result = acquired.get_future();
  std::jthread acquiring([&]() {
    DriverHandle next;
    acquire_started.count_down();
    acquired.set_value(next.Acquire(state));
  });
  acquire_started.wait();

  const auto waiting = result.wait_for(std::chrono::milliseconds(100));
  state.allow_close.count_down();
  releasing.join();
  acquiring.join();

  EXPECT_EQ(waiting, std::future_status::timeout);
  EXPECT_TRUE(result.get().IsOK());
  EXPECT_EQ(state.initialize_attempts.load(), 2U);
  EXPECT_EQ(state.opens.load(), 2U);
  EXPECT_EQ(state.closes.load(), 2U);
  EXPECT_EQ(state.active.load(), 0U);
}

}  // namespace
}  // namespace test
}  // namespace onnxruntime
