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

// A shape no other test in this binary profiles, so a profiling result shared across profilers cannot skip the runs.
constexpr int kExperts = 6;
constexpr int kTopK = 2;
constexpr int64_t kHidden = 256;
constexpr int64_t kInter = 160;
constexpr int64_t kGroupSize = 128;

AllocatorStats Stats(IAllocator& allocator) {
  AllocatorStats stats;
  allocator.GetStats(&stats);
  return stats;
}

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

  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
  CutlassMoeFCRunner<half, cutlass::uint4b_t> runner(sm, ActivationType::Swiglu, /*normalize_routing_weights=*/true,
                                                     /*use_sparse_mixer=*/false);
  const MoeGemmId fc1(static_cast<int>(2 * kInter), static_cast<int>(kHidden), llm::nvinfer::DataType::kHALF,
                      llm::nvinfer::DataType::kINT4, MoeGemmId::GemmType::Gemm1);

  const AllocatorStats before = Stats(*arena);
  // Several nodes and M buckets profile one after another, as the MoE layers of one model do.
  for (int node = 0; node < 3; ++node) {
    MoeGemmProfiler profiler;
    profiler.setAllocator(arena);
    profiler.setProfilerParams(kExperts, kTopK, kHidden, kInter, kGroupSize, ActivationType::Swiglu,
                               /*bias=*/false, /*need_weights=*/true, MOEParallelismConfig{}, sm);
    for (const int64_t m : {int64_t{16}, int64_t{256}}) {
      profiler.profileTactics(&runner, GemmDims(m, m, 2 * kInter, kHidden), fc1, stream);
      ASSERT_TRUE(profiler.getBestConfig(static_cast<int>(m), fc1).has_value());
    }
  }
  const AllocatorStats after = Stats(*arena);

  EXPECT_GT(after.num_reserves, before.num_reserves);
  EXPECT_EQ(after.num_arena_extensions, before.num_arena_extensions);
  EXPECT_EQ(after.total_allocated_bytes, before.total_allocated_bytes) << "profiling grew the arena";
  EXPECT_EQ(after.bytes_in_use, before.bytes_in_use);

  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

}  // namespace onnxruntime::test
