// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
#include "test/common/tensor_op_test_utils.h"

namespace onnxruntime {
namespace test {

#ifndef ENABLE_TRAINING
#ifdef USE_WEBGPU
namespace {

const onnxruntime::RunOptions run_options = []() {
  onnxruntime::RunOptions options{};
  ORT_THROW_IF_ERROR(options.config_options.AddConfigEntry(kOpTesterRunOptionsConfigTestTunableOp, "true"));
  return options;
}();

const constexpr auto run_with_tunable_op = &run_options;

}  // namespace

// f16 MatMul cases that exercise the Intel 8x16x16 subgroup-matrix impl.
// The host picks the tile shape adaptively (TileM in {8,16,32,64}, TileN in
// {16,32,64}); M and N may be any size and K must be a multiple of 16. When the
// output has few tiles and K is large, the host also splits K across multiple
// cooperating subgroups (split_k in {1,2,4,8}) that reduce in shared memory. B
// is a constant initializer. On non-Intel hardware these fall back to the
// default impl but still validate correctness.
static void RunSubgroupMatrixMatMulTest(const std::vector<int64_t>& a_dims, int64_t K, int64_t N) {
  int64_t M = 1;
  for (size_t i = 0; i + 1 < a_dims.size(); ++i) {
    M *= a_dims[i];
  }

  std::vector<float> A_f32(M * K);
  std::vector<float> B_f32(K * N);
  for (int64_t i = 0; i < M * K; ++i) {
    A_f32[i] = static_cast<float>((i % 7) - 3) * 0.1f;
  }
  for (int64_t i = 0; i < K * N; ++i) {
    B_f32[i] = static_cast<float>((i % 5) - 2) * 0.1f;
  }

  std::vector<float> Y_f32(M * N, 0.0f);
  for (int64_t m = 0; m < M; ++m) {
    for (int64_t n = 0; n < N; ++n) {
      float sum = 0.0f;
      for (int64_t k = 0; k < K; ++k) {
        sum += A_f32[m * K + k] * B_f32[k * N + n];
      }
      Y_f32[m * N + n] = sum;
    }
  }

  std::vector<MLFloat16> f_A(A_f32.size());
  std::vector<MLFloat16> f_B(B_f32.size());
  std::vector<MLFloat16> f_Y(Y_f32.size());
  ConvertFloatToMLFloat16(A_f32.data(), f_A.data(), static_cast<int>(A_f32.size()));
  ConvertFloatToMLFloat16(B_f32.data(), f_B.data(), static_cast<int>(B_f32.size()));
  ConvertFloatToMLFloat16(Y_f32.data(), f_Y.data(), static_cast<int>(Y_f32.size()));

  std::vector<int64_t> y_dims = a_dims;
  y_dims.back() = N;

  OpTester test("MatMul", 14);
  test.AddInput<MLFloat16>("A", a_dims, f_A);
  test.AddInput<MLFloat16>("B", {K, N}, f_B, /*is_initializer=*/true);
  test.AddOutput<MLFloat16>("Y", y_dims, f_Y);
  test.SetOutputTolerance(0.02f);
  test.ConfigExcludeEps({kTensorrtExecutionProvider})
      .Config(run_with_tunable_op)
      .RunWithConfig();
}

// A single test with many sub-cases that just barely cover the tile/split-K
// candidate space: TileM in {8,16,32,64}, TileN in {16,32,64}, split_k in
// {1,2,4,8}, plus the partial-edge, batched, and scratch-cap boundaries. Each
// sub-case runs under a SCOPED_TRACE so a failure names the exact shape.
TEST(MathOpTest, MatMulSubgroupMatrix) {
  struct Case {
    const char* name;
    std::vector<int64_t> a_dims;
    int64_t K;
    int64_t N;
  };

  const std::vector<Case> cases = {
      // TileM coverage: M selects the largest TileM candidate <= M.
      {"TileM8 (M=8 -> TileM=8)", {8, 64}, 64, 64},
      {"TileM16 (M=16 -> TileM=16)", {16, 64}, 64, 32},
      {"TileM32 (M=32 -> TileM=32)", {32, 64}, 64, 16},
      {"TileM64 (M=64 -> TileM=64)", {64, 64}, 64, 64},
      // TileN coverage: N selects the largest TileN candidate <= N.
      {"TileN16 (N=16 -> TileN=16)", {32, 64}, 64, 16},
      {"TileN32 (N=32 -> TileN=32)", {16, 64}, 64, 32},
      {"TileN64 (N=64 -> TileN=64)", {64, 128}, 128, 64},
      // Partial edges: M and N not multiples of the tile; bounds-checked stores.
      {"PartialMN (M=40,N=48)", {40, 64}, 64, 48},
      {"LargeNonAligned (M=100,N=96)", {100, 80}, 80, 96},
      // Below-minimum and minimum boundaries.
      {"TinyM (M=4 -> TileM=8 partial)", {4, 64}, 64, 32},
      {"MinK (single K block)", {32, 16}, 16, 32},
      // Batched A folds leading dims into M.
      {"Batched (2*32 -> M=64)", {2, 32, 64}, 64, 64},
      // Split-K coverage: one small output tile with growing K drives split_k.
      // K=64 (4 blocks) -> 2, K=128 (8 blocks) -> 4, K=256 (16 blocks) -> 8.
      {"SplitK2 (K=64 -> split_k=2)", {8, 64}, 64, 16},
      {"SplitK4 (K=128 -> split_k=4)", {8, 128}, 128, 16},
      {"SplitK8 (K=256 -> split_k=8)", {8, 256}, 256, 16},
      // K=144 -> 9 K-blocks, not divisible by split_k=4; round-robin K split.
      {"SplitKUnevenBlocks (K=144)", {8, 144}, 144, 16},
      // Split-K with a partial N edge tile (N=24 -> two TileN=16 tiles).
      {"SplitKPartialN (N=24)", {8, 256}, 256, 24},
      // Largest tile (64x64) with K=256: split_k capped at 4 by scratch budget.
      {"SplitKScratchCap (64x64,K=256)", {64, 256}, 256, 64},
      // Batched A folds to M=8; one tile + K=256 -> split_k=8.
      {"SplitKBatched (2*4 -> M=8)", {2, 4, 256}, 256, 16},
      // Odd N: the subgroup f16 load needs an even B row stride, so a constant odd-N
      // weight is padded once to N+1 (even) and cached; output is still written at
      // the real, odd N. Covers small/large odd N, min K, partial M, batched-A fold,
      // and split-K, all with odd N.
      {"OddN15 (N=15)", {32, 64}, 64, 15},
      {"OddN33 (N=33)", {16, 64}, 64, 33},
      {"OddN MinK (K=16,N=17)", {8, 16}, 16, 17},
      {"OddN PartialM (M=40,N=31)", {40, 64}, 64, 31},
      {"OddN BatchedA (2*32 -> M=64,N=63)", {2, 32, 64}, 64, 63},
      {"OddN SplitK (K=256,N=17)", {8, 256}, 256, 17},
  };

  for (const auto& c : cases) {
    SCOPED_TRACE(c.name);
    RunSubgroupMatrixMatMulTest(c.a_dims, c.K, c.N);
  }
}
#endif  // defined(USE_WEBGPU)
#endif

}  // namespace test
}  // namespace onnxruntime
