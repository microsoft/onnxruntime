// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <memory>

#include "contrib_ops/cuda/llm/common/cuda_runtime_utils.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_gemm_profiler.h"
#include "core/framework/allocator_utils.h"
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

// A shape no other test in this binary profiles. Every profiler also takes an expert count no earlier profiler used,
// so a profiling result shared across profilers (or kept from an earlier --gtest_repeat iteration) cannot skip a run.
constexpr int kTopK = 2;
constexpr int64_t kHidden = 256;
constexpr int64_t kInter = 160;
constexpr int64_t kGroupSize = 128;

void Configure(MoeGemmProfiler& profiler, const AllocatorPtr& arena, int sm) {
  static int next_num_experts = 6;
  profiler.setAllocator(arena);
  profiler.setProfilerParams(next_num_experts++, kTopK, kHidden, kInter, kGroupSize, ActivationType::Swiglu,
                             /*bias=*/false, /*need_weights=*/true, MOEParallelismConfig{}, sm);
}

MoeGemmId Fc1() {
  return MoeGemmId(static_cast<int>(2 * kInter), static_cast<int>(kHidden), llm::nvinfer::DataType::kHALF,
                   llm::nvinfer::DataType::kINT4, MoeGemmId::GemmType::Gemm1);
}

AllocatorStats Stats(IAllocator& allocator) {
  AllocatorStats stats;
  allocator.GetStats(&stats);
  return stats;
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

// Once failing, throws as CUDAAllocator does when cudaMalloc runs out of memory.
class FailingCUDAAllocator : public CUDAAllocator {
 public:
  explicit FailingCUDAAllocator(OrtDevice::DeviceId device) : CUDAAllocator(device, CUDA) {}
  void* Alloc(size_t size) override {
    if (failing_) {
      ORT_THROW("simulated cudaMalloc out of memory allocating ", size, " bytes");
    }
    return CUDAAllocator::Alloc(size);
  }
  void StartFailing() { failing_ = true; }

 private:
  bool failing_ = false;
};

}  // namespace

// Profiling scratch is needed once per (node, M bucket). It must be returned to the device when profiling ends instead
// of growing the session arena, which keeps every region it grows for the life of the session.
TEST(MoeGemmProfilerTest, ProfilingScratchDoesNotGrowArena) {
  const int sm = llm::common::getSMVersion();
  if (sm < 80) {
    GTEST_SKIP() << "MoE GEMM requires SM80 or later";
  }
  int device = 0;
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  AllocatorCreationInfo arena_info({[](OrtDevice::DeviceId id) { return std::make_unique<CUDAAllocator>(id, CUDA); },
                                    static_cast<OrtDevice::DeviceId>(device)});
  AllocatorPtr arena = CreateAllocator(arena_info);
  ASSERT_EQ(arena->Info().alloc_type, OrtArenaAllocator);

  Runner runner(sm, ActivationType::Swiglu, /*normalize_routing_weights=*/true, /*use_sparse_mixer=*/false);
  ScopedStream stream;

  const AllocatorStats before = Stats(*arena);
  int64_t profiled = 0;
  // Several nodes and M buckets profile one after another, as the MoE layers of one model do.
  for (int node = 0; node < 3; ++node) {
    MoeGemmProfiler profiler;
    Configure(profiler, arena, sm);
    for (const int64_t m : {int64_t{16}, int64_t{256}}) {
      profiler.profileTactics(&runner, GemmDims(m, m, 2 * kInter, kHidden), Fc1(), stream.Get());
      ASSERT_TRUE(profiler.getBestConfig(static_cast<int>(m), Fc1()).has_value());
      ++profiled;
    }
  }
  const AllocatorStats after = Stats(*arena);

  EXPECT_EQ(after.num_reserves - before.num_reserves, profiled) << "each node and M bucket reserves its own scratch";
  EXPECT_EQ(after.num_arena_extensions, before.num_arena_extensions);
  EXPECT_EQ(after.total_allocated_bytes, before.total_allocated_bytes) << "profiling grew the arena";
  EXPECT_EQ(after.bytes_in_use, before.bytes_in_use);
  EXPECT_EQ(cudaStreamSynchronize(stream.Get()), cudaSuccess);
}

// On the built-in CUDA EP, BFCArena::Reserve() goes straight to the device allocator, which throws when cudaMalloc runs
// out of memory even while the arena still holds free chunks. Profiling must then fall back to no config instead of
// failing the node, and leave the arena as it was.
TEST(MoeGemmProfilerTest, ReserveFailureFallsBackWithoutGrowingArena) {
  const int sm = llm::common::getSMVersion();
  if (sm < 80) {
    GTEST_SKIP() << "MoE GEMM requires SM80 or later";
  }
  int device = 0;
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  FailingCUDAAllocator* device_allocator = nullptr;
  AllocatorCreationInfo arena_info({[&device_allocator](OrtDevice::DeviceId id) {
                                      auto allocator = std::make_unique<FailingCUDAAllocator>(id);
                                      device_allocator = allocator.get();
                                      return allocator;
                                    },
                                    static_cast<OrtDevice::DeviceId>(device)});
  AllocatorPtr arena = CreateAllocator(arena_info);
  ASSERT_EQ(arena->Info().alloc_type, OrtArenaAllocator);
  ASSERT_NE(device_allocator, nullptr);
  void* held = arena->Alloc(1 << 20);  // Memory the session already holds in the arena.
  ASSERT_NE(held, nullptr);
  device_allocator->StartFailing();

  Runner runner(sm, ActivationType::Swiglu, /*normalize_routing_weights=*/true, /*use_sparse_mixer=*/false);
  ScopedStream stream;
  MoeGemmProfiler profiler;
  Configure(profiler, arena, sm);

  const AllocatorStats before = Stats(*arena);
  EXPECT_NO_THROW(profiler.profileTactics(&runner, GemmDims(16, 16, 2 * kInter, kHidden), Fc1(), stream.Get()));
  EXPECT_FALSE(profiler.getBestConfig(16, Fc1()).has_value());
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
  const AllocatorStats after = Stats(*arena);

  EXPECT_EQ(after.num_reserves, before.num_reserves);
  EXPECT_EQ(after.num_arena_extensions, before.num_arena_extensions);
  EXPECT_EQ(after.total_allocated_bytes, before.total_allocated_bytes);
  EXPECT_EQ(after.bytes_in_use, before.bytes_in_use);
  arena->Free(held);
}

}  // namespace onnxruntime::test
