// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <exception>
#include <future>
#include <memory>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

#include "contrib_ops/cuda/llm/common/cuda_runtime_utils.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_gemm_profiler.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_kernels.h"
#include "core/platform/env_var_utils.h"
#include "core/providers/cuda/cuda_allocator.h"

namespace onnxruntime::test {
namespace {

using llm::kernels::cutlass_kernels::ActivationType;
using llm::kernels::cutlass_kernels::CutlassMoeFCRunner;
using llm::kernels::cutlass_kernels::MoeGemmId;
using llm::kernels::cutlass_kernels::MoeGemmProfiler;
using llm::kernels::cutlass_kernels::MOEParallelismConfig;
using llm::kernels::weight_only::GemmDims;
using Runner = CutlassMoeFCRunner<half, cutlass::uint4b_t>;

// Deliberately unusual shapes, and a different intermediate size per test, so no other test in the process profiles
// the same key first whatever the test order. The shared cache also outlives each test, so SetUp gives every test
// invocation (including --gtest_repeat iterations) a new expert count, which makes all of its keys new.
std::atomic<int> g_num_experts{0};
constexpr int kTopK = 2;
constexpr int64_t kHidden = 256;
constexpr int64_t kGroupSize = 128;

int CurrentDevice() {
  int device = 0;
  ORT_ENFORCE(cudaGetDevice(&device) == cudaSuccess);
  return device;
}

void Configure(MoeGemmProfiler& profiler, int sm, int64_t inter, AllocatorPtr allocator = nullptr) {
  profiler.setAllocator(allocator ? std::move(allocator) : std::make_shared<CUDAAllocator>(CurrentDevice(), CUDA));
  profiler.setProfilerParams(g_num_experts.load(), kTopK, kHidden, inter, kGroupSize, ActivationType::Swiglu,
                             /*bias=*/false, /*need_weights=*/true, MOEParallelismConfig{}, sm);
}

MoeGemmId Fc1(int64_t inter) {
  return MoeGemmId(static_cast<int>(2 * inter), static_cast<int>(kHidden), llm::nvinfer::DataType::kHALF,
                   llm::nvinfer::DataType::kINT4, MoeGemmId::GemmType::Gemm1);
}

GemmDims Dims(int64_t m, int64_t inter) {
  return GemmDims(m, m, 2 * inter, kHidden);
}

class ScopedStream {
 public:
  ScopedStream() { ORT_ENFORCE(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) == cudaSuccess); }
  ~ScopedStream() {
    cudaStreamSynchronize(stream_);
    cudaStreamDestroy(stream_);
  }
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ScopedStream);
  cudaStream_t Get() const { return stream_; }

 private:
  cudaStream_t stream_ = nullptr;
};

// Profiles one problem with a profiler, runner and stream of its own, as one MoE node on its own thread would.
std::optional<MoeGemmProfiler::Config> ProfileFc1(int sm, int64_t inter, int64_t m, AllocatorPtr allocator = nullptr) {
  Runner runner(sm, ActivationType::Swiglu, /*normalize_routing_weights=*/true, /*use_sparse_mixer=*/false);
  MoeGemmProfiler profiler;
  Configure(profiler, sm, inter, std::move(allocator));
  ScopedStream stream;
  profiler.profileTactics(&runner, Dims(m, inter), Fc1(inter), stream.Get());
  return profiler.getBestConfig(static_cast<int>(m), Fc1(inter));
}

// Fails every allocation, as a transient out-of-memory would.
class RefusingAllocator : public CUDAAllocator {
 public:
  RefusingAllocator() : CUDAAllocator(CurrentDevice(), CUDA) {}
  void* Alloc(size_t /*size*/) override { return nullptr; }
};

// Throws from every allocation, as the built-in CUDA EP allocator does on out-of-memory.
class ThrowingAllocator : public CUDAAllocator {
 public:
  ThrowingAllocator() : CUDAAllocator(CurrentDevice(), CUDA) {}
  void* Alloc(size_t size) override { ORT_THROW("simulated CUDA out of memory allocating ", size, " bytes"); }
};

// Holds its first allocation until `open` is ready, which keeps that profiling run in flight.
class GatedAllocator : public CUDAAllocator {
 public:
  GatedAllocator(std::promise<void>& entered, std::shared_future<void> open)
      : CUDAAllocator(CurrentDevice(), CUDA), entered_(entered), open_(std::move(open)) {}

  void* Alloc(size_t size) override {
    if (!first_call_done_.exchange(true)) {
      entered_.set_value();
      opened_in_time_ = open_.wait_for(std::chrono::seconds(30)) == std::future_status::ready;
    }
    return CUDAAllocator::Alloc(size);
  }

  bool OpenedInTime() const { return opened_in_time_; }

 private:
  std::promise<void>& entered_;
  std::shared_future<void> open_;
  std::atomic<bool> first_call_done_{false};
  std::atomic<bool> opened_in_time_{false};
};

class MoeGemmProfilerSharedCacheTest : public ::testing::Test {
 protected:
  void SetUp() override {
    static int next_num_experts = 4;
    g_num_experts.store(next_num_experts++);
    sm_ = llm::common::getSMVersion();
    if (sm_ < 80) {
      GTEST_SKIP() << "MoE GEMM requires SM80 or later";
    }
    if (ParseEnvironmentVariableWithDefault<int>("ORT_DISABLE_MOE_SHARED_TACTIC_CACHE", 0) == 1) {
      GTEST_SKIP() << "Shared MoE tactic cache disabled by environment";
    }
  }

  int sm_ = 0;
};

}  // namespace

TEST_F(MoeGemmProfilerSharedCacheTest, IdenticalProblemsAreProfiledOncePerProcess) {
  constexpr int64_t kInter = 192;
  Runner runner(sm_, ActivationType::Swiglu, /*normalize_routing_weights=*/true, /*use_sparse_mixer=*/false);
  ScopedStream stream;
  MoeGemmProfiler first;
  MoeGemmProfiler second;
  Configure(first, sm_, kInter);
  Configure(second, sm_, kInter);
  const size_t before = MoeGemmProfiler::ProfilingRunCountForTesting();

  first.profileTactics(&runner, Dims(64, kInter), Fc1(kInter), stream.Get());
  second.profileTactics(&runner, Dims(64, kInter), Fc1(kInter), stream.Get());
  EXPECT_EQ(MoeGemmProfiler::ProfilingRunCountForTesting() - before, 1u);

  const auto first_config = first.getBestConfig(64, Fc1(kInter));
  const auto second_config = second.getBestConfig(64, Fc1(kInter));
  ASSERT_TRUE(first_config.has_value());
  ASSERT_TRUE(second_config.has_value());
  EXPECT_EQ(first_config->toString(), second_config->toString());

  // A different M bucket is a different problem and is still profiled on first use.
  second.profileTactics(&runner, Dims(1, kInter), Fc1(kInter), stream.Get());
  EXPECT_EQ(MoeGemmProfiler::ProfilingRunCountForTesting() - before, 2u);
}

// Runner state that filters the candidate tactics (SM80 FP4 family) or changes the profiled GEMM2 epilogue
// (deterministic Hopper reduce) must keep results apart even when the problem is identical.
TEST_F(MoeGemmProfilerSharedCacheTest, RunnerStateThatChangesProfilingIsPartOfTheKey) {
  constexpr int64_t kInter = 192;
  Runner plain(sm_, ActivationType::Swiglu, /*normalize_routing_weights=*/true, /*use_sparse_mixer=*/false);
  Runner plain_again(sm_, ActivationType::Swiglu, /*normalize_routing_weights=*/true, /*use_sparse_mixer=*/false);
  Runner sm80_fp4(sm_, ActivationType::Swiglu, /*normalize_routing_weights=*/true, /*use_sparse_mixer=*/false);
  sm80_fp4.setUseSm80Fp4(true);
  Runner deterministic(sm_, ActivationType::Swiglu, /*normalize_routing_weights=*/true, /*use_sparse_mixer=*/false);
  deterministic.use_deterministic_hopper_reduce_ = true;
  ScopedStream stream;

  // An M bucket the test above does not use, so the two tests are independent of their order.
  const size_t before = MoeGemmProfiler::ProfilingRunCountForTesting();
  for (Runner* runner : {&plain, &sm80_fp4, &deterministic, &plain_again}) {
    MoeGemmProfiler profiler;
    Configure(profiler, sm_, kInter);
    profiler.profileTactics(runner, Dims(128, kInter), Fc1(kInter), stream.Get());
  }
  // plain_again reuses plain's result; the other two runners differ in tactic-relevant state.
  EXPECT_EQ(MoeGemmProfiler::ProfilingRunCountForTesting() - before, 3u);
}

// A failed run (here an allocation failure, as a transient out-of-memory gives) must not be shared: the next
// profiler of the same problem profiles again, and its result is the one that is shared. An allocator that throws
// instead of returning nullptr must take the same no-config fallback rather than fail the caller.
TEST_F(MoeGemmProfilerSharedCacheTest, FailedProfilingIsNotShared) {
  constexpr int64_t kInter = 256;
  const size_t before = MoeGemmProfiler::ProfilingRunCountForTesting();

  EXPECT_FALSE(ProfileFc1(sm_, kInter, 64, std::make_shared<RefusingAllocator>()).has_value());
  std::optional<MoeGemmProfiler::Config> after_throw;
  EXPECT_NO_THROW(after_throw = ProfileFc1(sm_, kInter, 64, std::make_shared<ThrowingAllocator>()));
  EXPECT_FALSE(after_throw.has_value());
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
  const auto retried = ProfileFc1(sm_, kInter, 64);
  EXPECT_EQ(MoeGemmProfiler::ProfilingRunCountForTesting() - before, 3u);
  ASSERT_TRUE(retried.has_value());

  const auto reused = ProfileFc1(sm_, kInter, 64);
  EXPECT_EQ(MoeGemmProfiler::ProfilingRunCountForTesting() - before, 3u);
  ASSERT_TRUE(reused.has_value());
  EXPECT_EQ(reused->toString(), retried->toString());
}

// Profilers that reach one key at the same time profile it once and all get its result.
TEST_F(MoeGemmProfilerSharedCacheTest, ConcurrentFirstUsesOfOneKeyProfileOnce) {
  constexpr int64_t kInter = 320;
  constexpr int kThreads = 4;
  std::vector<std::optional<MoeGemmProfiler::Config>> configs(kThreads);
  std::vector<std::exception_ptr> errors(kThreads);
  std::atomic<int> arrived{0};
  const size_t before = MoeGemmProfiler::ProfilingRunCountForTesting();

  std::vector<std::thread> threads;
  for (int i = 0; i < kThreads; ++i) {
    threads.emplace_back([&, i] {
      arrived.fetch_add(1);
      while (arrived.load() < kThreads) {
        std::this_thread::yield();
      }
      try {
        configs[i] = ProfileFc1(sm_, kInter, 64);
      } catch (...) {
        errors[i] = std::current_exception();
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  for (const auto& error : errors) {
    if (error) {
      std::rethrow_exception(error);
    }
  }

  EXPECT_EQ(MoeGemmProfiler::ProfilingRunCountForTesting() - before, 1u);
  for (const auto& config : configs) {
    ASSERT_TRUE(config.has_value());
    EXPECT_EQ(config->toString(), configs[0]->toString());
  }
}

// A key being profiled must not hold up other keys (other problems, M buckets or devices): a second problem is
// profiled to completion while the first run is held in flight inside its workspace allocation.
TEST_F(MoeGemmProfilerSharedCacheTest, OtherKeysProfileWhileOneIsInFlight) {
  constexpr int64_t kHeldInter = 384;
  constexpr int64_t kOtherInter = 448;
  std::promise<void> entered;
  std::promise<void> open;
  std::future<void> held_entered = entered.get_future();
  auto gated = std::make_shared<GatedAllocator>(entered, open.get_future().share());

  std::exception_ptr held_error;
  std::thread held([&] {
    try {
      ProfileFc1(sm_, kHeldInter, 64, gated);
    } catch (...) {
      held_error = std::current_exception();
    }
  });
  const bool held_in_flight = held_entered.wait_for(std::chrono::seconds(60)) == std::future_status::ready;
  std::optional<MoeGemmProfiler::Config> other;
  std::exception_ptr other_error;
  if (held_in_flight) {
    try {
      other = ProfileFc1(sm_, kOtherInter, 64);
    } catch (...) {
      other_error = std::current_exception();  // Rethrown after the worker is released and joined.
    }
  }
  open.set_value();
  held.join();
  for (const std::exception_ptr& error : {held_error, other_error}) {
    if (error) {
      std::rethrow_exception(error);
    }
  }

  ASSERT_TRUE(held_in_flight);
  EXPECT_TRUE(other.has_value());
  EXPECT_TRUE(gated->OpenedInTime()) << "profiling another key waited for the key in flight";
}

}  // namespace onnxruntime::test
