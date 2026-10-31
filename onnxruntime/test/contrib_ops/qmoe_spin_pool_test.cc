// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

#if defined(__linux__)
#include <pthread.h>

#include <filesystem>
#include <fstream>
#include <system_error>
#endif

#include "contrib_ops/cuda/moe/qmoe_spin_pool.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

namespace qmoe = contrib::cuda::qmoe;

namespace {

constexpr int kSpinUs = 50;

std::unique_ptr<qmoe::SpinPool> MakePool(int threads) {
  return std::make_unique<qmoe::SpinPool>(threads, kSpinUs);
}

int64_t SumWithPool(qmoe::SpinPool& pool, ptrdiff_t count) {
  std::atomic<int64_t> sum{0};
  auto add = [&sum](ptrdiff_t index) { sum.fetch_add(index, std::memory_order_relaxed); };
  pool.ParallelFor(count, add);
  return sum.load();
}

#if defined(__linux__)
constexpr const char* kWorkerThreadName = "qmoe-spin";

// Worker threads are only observable through /proc, so tests that need the real thread count are Linux-only.
int CountThreadsNamed(const std::string& name) {
  int count = 0;
  std::error_code ec;
  std::filesystem::directory_iterator end;
  for (std::filesystem::directory_iterator it("/proc/self/task", ec); !ec && it != end; it.increment(ec)) {
    std::ifstream comm(it->path() / "comm");
    std::string value;
    if (comm && std::getline(comm, value) && value == name) ++count;
  }
  return count;
}

// Thread names are set by the thread itself and a joined thread can linger in /proc for a moment, so poll.
bool WaitForThreadCount(const std::string& name, int expected) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (CountThreadsNamed(name) != expected) {
    if (std::chrono::steady_clock::now() > deadline) return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  return true;
}
#endif

}  // namespace

TEST(QMoESpinPoolRegistryTest, BuildsEachThreadCountOnce) {
  qmoe::SpinPoolRegistry registry;
  int builds = 0;
  auto make = [&builds](int threads) {
    ++builds;
    return MakePool(threads);
  };

  qmoe::SpinPool& first = registry.GetOrCreate(2, make);
  qmoe::SpinPool& again = registry.GetOrCreate(2, make);
  qmoe::SpinPool& other = registry.GetOrCreate(3, make);

  EXPECT_EQ(&first, &again);
  EXPECT_NE(&first, &other);
  EXPECT_EQ(first.ThreadCount(), 2);
  EXPECT_EQ(other.ThreadCount(), 3);
  EXPECT_EQ(builds, 2);
  EXPECT_EQ(registry.PoolCount(), size_t{2});
  EXPECT_EQ(SumWithPool(first, 1000), 999 * 1000 / 2);
  EXPECT_EQ(SumWithPool(other, 1000), 999 * 1000 / 2);
}

TEST(QMoESpinPoolRegistryTest, FailedFactoryLeavesRegistryUnchanged) {
  qmoe::SpinPoolRegistry registry;
  auto throwing = [](int) -> std::unique_ptr<qmoe::SpinPool> { throw std::runtime_error("no pool"); };
  auto null_pool = [](int) -> std::unique_ptr<qmoe::SpinPool> { return nullptr; };

  EXPECT_THROW(registry.GetOrCreate(2, throwing), std::runtime_error);
  EXPECT_THROW(registry.GetOrCreate(2, null_pool), std::logic_error);
  EXPECT_EQ(registry.PoolCount(), size_t{0});

  qmoe::SpinPool& pool = registry.GetOrCreate(2, MakePool);
  EXPECT_EQ(registry.PoolCount(), size_t{1});
  EXPECT_EQ(SumWithPool(pool, 100), 99 * 100 / 2);
}

TEST(QMoESpinPoolRegistryTest, ShutdownOfEmptyRegistryIsANoOp) {
  qmoe::SpinPoolRegistry registry;
  registry.Shutdown();
  registry.Shutdown();
  EXPECT_EQ(registry.PoolCount(), size_t{0});
}

TEST(QMoESpinPoolRegistryTest, ShutdownIsIdempotentAndPoolsAreRebuiltOnDemand) {
  qmoe::SpinPoolRegistry registry;
  int builds = 0;
  auto make = [&builds](int threads) {
    ++builds;
    return MakePool(threads);
  };

  EXPECT_EQ(SumWithPool(registry.GetOrCreate(3, make), 500), 499 * 500 / 2);
  EXPECT_EQ(registry.PoolCount(), size_t{1});

  registry.Shutdown();
  registry.Shutdown();
  EXPECT_EQ(registry.PoolCount(), size_t{0});

  qmoe::SpinPool& rebuilt = registry.GetOrCreate(3, make);
  EXPECT_EQ(builds, 2);
  EXPECT_EQ(rebuilt.ThreadCount(), 3);
  EXPECT_EQ(SumWithPool(rebuilt, 500), 499 * 500 / 2);
}

TEST(QMoESpinPoolRegistryTest, ShutdownRightAfterWorkDoesNotHang) {
  qmoe::SpinPoolRegistry registry;
  for (int round = 0; round < 8; ++round) {
    qmoe::SpinPool& pool = registry.GetOrCreate(4, MakePool);
    EXPECT_EQ(SumWithPool(pool, 2000), 1999 * 2000 / 2);
    registry.Shutdown();
  }
  EXPECT_EQ(registry.PoolCount(), size_t{0});
}

#if defined(__linux__)
TEST(QMoESpinPoolRegistryTest, NameCurrentThreadSetsTheCallingThreadName) {
  char name[16] = {};
  std::thread worker([&name] {
    qmoe::NameCurrentThread("qmoe-test-name");
    ASSERT_EQ(pthread_getname_np(pthread_self(), name, sizeof(name)), 0);
  });
  worker.join();
  EXPECT_STREQ(name, "qmoe-test-name");
}

TEST(QMoESpinPoolRegistryTest, ShutdownJoinsWorkerThreads) {
  const int baseline = CountThreadsNamed(kWorkerThreadName);
  qmoe::SpinPoolRegistry registry;

  // A pool with N participants owns N - 1 worker threads; the caller is the remaining participant.
  qmoe::SpinPool& pool = registry.GetOrCreate(5, MakePool);
  ASSERT_TRUE(WaitForThreadCount(kWorkerThreadName, baseline + 4));
  EXPECT_EQ(SumWithPool(pool, 1000), 999 * 1000 / 2);

  registry.GetOrCreate(3, MakePool);
  ASSERT_TRUE(WaitForThreadCount(kWorkerThreadName, baseline + 4 + 2));

  registry.Shutdown();
  ASSERT_TRUE(WaitForThreadCount(kWorkerThreadName, baseline));

  // The threads come back with the pools.
  qmoe::SpinPool& rebuilt = registry.GetOrCreate(5, MakePool);
  ASSERT_TRUE(WaitForThreadCount(kWorkerThreadName, baseline + 4));
  EXPECT_EQ(SumWithPool(rebuilt, 1000), 999 * 1000 / 2);

  registry.Shutdown();
  ASSERT_TRUE(WaitForThreadCount(kWorkerThreadName, baseline));
}

TEST(QMoESpinPoolRegistryTest, DestroyingTheRegistryJoinsWorkerThreads) {
  const int baseline = CountThreadsNamed(kWorkerThreadName);
  {
    qmoe::SpinPoolRegistry registry;
    registry.GetOrCreate(4, MakePool);
    ASSERT_TRUE(WaitForThreadCount(kWorkerThreadName, baseline + 3));
  }
  ASSERT_TRUE(WaitForThreadCount(kWorkerThreadName, baseline));
}
#endif

}  // namespace test
}  // namespace onnxruntime
