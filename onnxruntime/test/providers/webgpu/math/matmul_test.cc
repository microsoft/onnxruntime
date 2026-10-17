// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdint>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "core/providers/webgpu/webgpu_provider_options.h"
#if !defined(ORT_USE_EP_API_ADAPTERS)
#include "core/providers/webgpu/webgpu_context.h"
#endif
#include "core/session/onnxruntime_session_options_config_keys.h"
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

TEST(MathOpTest, MatMulPackedFp16LongReductionUsesFloat32Accumulator) {
  ConfigOptions provider_options;
  ASSERT_STATUS_OK(provider_options.AddConfigEntry(webgpu::options::kEnableMatmulFp32Accumulation, "1"));
  auto webgpu_ep = WebGpuExecutionProviderWithOptions(provider_options);
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU execution provider is not available";
  }

  constexpr int64_t M = 3;
  constexpr int64_t K = 900;
  constexpr int64_t N = 4;
  const MLFloat16 value{0.1f};
  const MLFloat16 expected{value.ToFloat() * value.ToFloat() * static_cast<float>(K)};

  SessionOptions session_options;
  ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));

  OpTester test("MatMul", 14);
  test.AddInput<MLFloat16>("A", {M, K}, std::vector<MLFloat16>(M * K, value));
  test.AddInput<MLFloat16>("B", {K, N}, std::vector<MLFloat16>(K * N, value), /*is_initializer=*/true);
  test.AddOutput<MLFloat16>("Y", {M, N}, std::vector<MLFloat16>(M * N, expected));
  test.SetOutputAbsErr("Y", 0.01f);
  test.Config(session_options).ConfigEp(std::move(webgpu_ep)).RunWithConfig();
}

TEST(MathOpTest, MatMulIntelEligibleFp16LongReductionCancellation) {
  ConfigOptions provider_options;
  ASSERT_STATUS_OK(provider_options.AddConfigEntry(webgpu::options::kEnableMatmulFp32Accumulation, "1"));
  auto webgpu_ep = WebGpuExecutionProviderWithOptions(provider_options);
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU execution provider is not available";
  }

  constexpr int64_t rows = 64;
  constexpr int64_t reduction_size = 1024;
  constexpr int64_t output_columns = 512;
  std::vector<MLFloat16> weights(reduction_size * output_columns);
  for (int64_t reduction_index = 0; reduction_index < reduction_size; ++reduction_index) {
    const float value = reduction_index < reduction_size / 2 ? 512.0f : -512.0f;
    for (int64_t column = 0; column < output_columns; ++column) {
      weights[reduction_index * output_columns + column] = MLFloat16(value);
    }
  }
  for (int64_t column = 0; column < output_columns; ++column) {
    weights[column] = MLFloat16(513.0f);
  }

  SessionOptions session_options;
  ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  session_options.session_log_severity_level = 0;
  session_options.session_log_verbosity_level = 1;

  OpTester test("MatMul", 14);
  test.AddInput<MLFloat16>("A", {rows, reduction_size},
                           std::vector<MLFloat16>(rows * reduction_size, MLFloat16(1.0f)));
  test.AddInput<MLFloat16>("B", {reduction_size, output_columns}, weights, true);
  test.AddOutput<MLFloat16>("Y", {rows, output_columns},
                            std::vector<MLFloat16>(rows * output_columns, MLFloat16(1.0f)));
  test.Config(session_options).ConfigEp(std::move(webgpu_ep)).RunWithConfig();
}

TEST(MathOpTest, MatMulPackedFp16SplitKEligibleCancellationKeepsFloat32Precision) {
  constexpr int64_t rows = 3;
  constexpr int64_t reduction_size = 1028;
  constexpr int64_t output_columns = 4;
  constexpr int64_t split_size = 256;
  std::vector<MLFloat16> weights(reduction_size * output_columns, MLFloat16(0.0f));
  for (int64_t reduction_index = 0; reduction_index < split_size; ++reduction_index) {
    for (int64_t column = 0; column < output_columns; ++column) {
      weights[reduction_index * output_columns + column] = MLFloat16(512.0f);
      weights[(reduction_index + split_size) * output_columns + column] = MLFloat16(-512.0f);
    }
  }
  for (int64_t column = 0; column < output_columns; ++column) {
    weights[(reduction_size - 1) * output_columns + column] = MLFloat16(1.0f);
  }

  for (const bool deterministic : {false, true}) {
    for (const int64_t batch_size : {1, 2}) {
      for (const bool batched_weights : {false, true}) {
        SCOPED_TRACE(deterministic);
        SCOPED_TRACE(batch_size);
        SCOPED_TRACE(batched_weights);
        ConfigOptions provider_options;
        ASSERT_STATUS_OK(provider_options.AddConfigEntry(webgpu::options::kEnableMatmulFp32Accumulation, "1"));
        auto webgpu_ep = WebGpuExecutionProviderWithOptions(provider_options);
        if (!webgpu_ep) {
          GTEST_SKIP() << "WebGPU execution provider is not available";
        }

        std::vector<int64_t> a_shape{rows, reduction_size};
        std::vector<int64_t> b_shape{reduction_size, output_columns};
        std::vector<int64_t> y_shape{rows, output_columns};
        if (batch_size > 1 || batched_weights) {
          a_shape.insert(a_shape.begin(), batch_size);
          y_shape.insert(y_shape.begin(), batch_size);
        }
        std::vector<MLFloat16> batched_b = weights;
        if (batched_weights) {
          b_shape.insert(b_shape.begin(), batch_size);
          for (int64_t batch = 1; batch < batch_size; ++batch) {
            batched_b.insert(batched_b.end(), weights.begin(), weights.end());
          }
        }

        SessionOptions session_options;
        session_options.use_deterministic_compute = deterministic;
        ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));

        OpTester test("MatMul", 14);
        test.AddInput<MLFloat16>("A", a_shape,
                                 std::vector<MLFloat16>(batch_size * rows * reduction_size, MLFloat16(1.0f)));
        test.AddInput<MLFloat16>("B", b_shape, batched_b);
        test.AddOutput<MLFloat16>("Y", y_shape,
                                  std::vector<MLFloat16>(batch_size * rows * output_columns, MLFloat16(1.0f)));
        test.Config(session_options).ConfigEp(std::move(webgpu_ep)).RunWithConfig();
      }
    }
  }
}

TEST(MathOpTest, MatMulPackedScalarFp16LongReductionUsesFloat32Accumulator) {
  for (const auto& dimensions : {std::pair<int64_t, int64_t>{895, 4}, {896, 3}}) {
    const auto [reduction_size, output_columns] = dimensions;
    SCOPED_TRACE(reduction_size);
    SCOPED_TRACE(output_columns);
    ConfigOptions provider_options;
    ASSERT_STATUS_OK(provider_options.AddConfigEntry(webgpu::options::kEnableMatmulFp32Accumulation, "1"));
    auto webgpu_ep = WebGpuExecutionProviderWithOptions(provider_options);
    if (!webgpu_ep) {
      GTEST_SKIP() << "WebGPU execution provider is not available";
    }

    constexpr int64_t rows = 3;
    const MLFloat16 value{0.1f};
    const MLFloat16 expected{value.ToFloat() * value.ToFloat() * static_cast<float>(reduction_size)};

    SessionOptions session_options;
    ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));

    OpTester test("MatMul", 14);
    test.AddInput<MLFloat16>("A", {rows, reduction_size}, std::vector<MLFloat16>(rows * reduction_size, value));
    test.AddInput<MLFloat16>("B", {reduction_size, output_columns},
                             std::vector<MLFloat16>(reduction_size * output_columns, value));
    test.AddOutput<MLFloat16>("Y", {rows, output_columns}, std::vector<MLFloat16>(rows * output_columns, expected));
    test.SetOutputAbsErr("Y", 0.01f);
    test.Config(session_options).ConfigEp(std::move(webgpu_ep)).RunWithConfig();
  }
}

// f16 MatMul cases that exercise the 8x16x16 and 16x16x16 subgroup-matrix impl.
// The host picks the tile shape adaptively (TileM in {8,16,32,64}, TileN in
// {16,32,64}); M and N may be any size and K must be a multiple of 16. When the
// output has few tiles and K is large, the host also splits K across multiple
// cooperating subgroups (split_k in {1,2,4,8}) that reduce in shared memory. B
// is a constant initializer except in the last case. On hardware that reports
// neither config these fall back to the default impl but still validate correctness.
static void RunSubgroupMatrixMatMulTest(const std::vector<int64_t>& a_dims, int64_t K, int64_t N,
                                        bool b_is_initializer = true) {
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
  test.AddInput<MLFloat16>("B", {K, N}, f_B, b_is_initializer);
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
      // weight is padded once to an even width and cached. Output is still written at
      // the real, odd N. Covers small/large odd N, min K, partial M, batched-A fold,
      // and split-K, all with odd N.
      {"OddN15 (N=15)", {32, 64}, 64, 15},
      {"OddN33 (N=33)", {16, 64}, 64, 33},
      {"OddN MinK (K=16,N=17)", {8, 16}, 16, 17},
      {"OddN PartialM (M=40,N=31)", {40, 64}, 64, 31},
      {"OddN BatchedA (2*32 -> M=64,N=63)", {2, 32, 64}, 64, 63},
      {"OddN SplitK (K=256,N=17)", {8, 256}, 256, 17},
      // N is even but not a multiple of 8 so only the 16x16x16 config pads B.
      {"UnalignedN36 (N=36)", {64, 128}, 128, 36},
      {"UnalignedN44 PartialM (M=40,N=44)", {40, 128}, 128, 44},
      // Odd N and partial M past 64 so the default tiling takes its 64x64 tile on padded B.
      {"OddN LargeTile (M=72,N=67)", {72, 96}, 96, 67},
  };

  for (const auto& c : cases) {
    SCOPED_TRACE(c.name);
    RunSubgroupMatrixMatMulTest(c.a_dims, c.K, c.N);
  }

  // B is a runtime input here so its padded copy could not be cached and the
  // 16x16x16 config falls back to the default impl for this width.
  SCOPED_TRACE("RuntimeB UnalignedN36 (N=36)");
  RunSubgroupMatrixMatMulTest({64, 128}, 128, 36, /*b_is_initializer=*/false);
}

class WebGpuMatMulGemvTest : public ::testing::Test {
 protected:
  void SetUp() override {
    WebGpuExecutionProviderTestOptions options{};
    options.forced_matmul_algorithm = webgpu::MatMulAlgorithm::Gemv;
    try {
      auto ep = WebGpuExecutionProviderWithTestOptions(options);
      if (!ep) {
        GTEST_SKIP() << "WebGPU execution provider test options are not available.";
      }
#if !defined(ORT_USE_EP_API_ADAPTERS)
      auto& context = webgpu::WebGpuContextFactory::GetContext(ep->GetDeviceId());
      if (!context.HasDevice() || !context.DeviceHasFeature(wgpu::FeatureName::ShaderF16)) {
        GTEST_SKIP() << "A usable FP16 WebGPU adapter is not available.";
      }
#endif
    } catch (const OnnxRuntimeException& error) {
      if (std::string_view{error.what()}.find("Failed to get a WebGPU adapter") != std::string_view::npos) {
        GTEST_SKIP() << error.what();
      }
      throw;
    }
  }

  void Run(const std::vector<int64_t>& a_dims, int64_t K, int64_t N, bool expect_gemv,
           bool force_gemv = false, bool batched_weights = false) {
    SCOPED_TRACE(::testing::PrintToString(a_dims));
    SCOPED_TRACE(K);
    SCOPED_TRACE(N);
    SCOPED_TRACE(batched_weights);
    const int64_t M = TensorShape{a_dims}.SizeToDimension(a_dims.size() - 1);
    const int64_t rows_per_batch = a_dims[a_dims.size() - 2];
    const int64_t weight_batches = batched_weights ? M / rows_per_batch : 1;
    std::vector<MLFloat16> a(M * K), b(weight_batches * K * N), y(M * N);
    for (int64_t i = 0; i < M * K; ++i) {
      a[i] = MLFloat16(static_cast<float>((i % 7) - 3) * 0.125f);
    }
    for (int64_t i = 0; i < weight_batches * K * N; ++i) {
      const int64_t batch = i / (K * N);
      b[i] = MLFloat16(static_cast<float>((i % 5) - 2) * 0.125f + static_cast<float>(batch) * 0.25f);
    }
    for (int64_t m = 0; m < M; ++m) {
      const int64_t b_offset = batched_weights ? (m / rows_per_batch) * K * N : 0;
      for (int64_t n = 0; n < N; ++n) {
        float sum = 0.0f;
        for (int64_t k = 0; k < K; ++k) {
          sum += a[m * K + k].ToFloat() * b[b_offset + k * N + n].ToFloat();
        }
        y[m * N + n] = MLFloat16(sum);
      }
    }

    struct DispatchCounts {
      size_t total = 0;
      size_t gemv = 0;
    } dispatches;
    SessionOptions session_options;
    ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
    session_options.session_log_severity_level = ORT_LOGGING_LEVEL_INFO;
    session_options.user_logging_param = &dispatches;
    session_options.user_logging_function = [](void* param, OrtLoggingLevel, const char*, const char*,
                                               const char*, const char* message) {
      auto& counts = *static_cast<DispatchCounts*>(param);
      const std::string_view text{message};
      if (text.find("Starting program \"") == 0) {
        ++counts.total;
        if (text.find("Starting program \"MatMulGemv") == 0) {
          ++counts.gemv;
        }
      }
    };
    WebGpuExecutionProviderTestOptions options{};
    options.forced_matmul_algorithm = webgpu::MatMulAlgorithm::Gemv;
    auto ep = force_gemv ? WebGpuExecutionProviderWithTestOptions(options) : DefaultWebGpuExecutionProvider();
    ASSERT_NE(ep, nullptr);

    auto y_dims = a_dims;
    y_dims.back() = N;
    auto b_dims = batched_weights ? a_dims : std::vector<int64_t>{K, N};
    b_dims[b_dims.size() - 2] = K;
    b_dims.back() = N;
    OpTester test("MatMul", 14);
    test.AddInput<MLFloat16>("A", a_dims, a);
    test.AddInput<MLFloat16>("B", b_dims, b);
    test.AddOutput<MLFloat16>("Y", y_dims, y);
    test.SetOutputAbsErr("Y", 0.02f);
    test.Config(session_options).ConfigEp(std::move(ep)).RunWithConfig();
    ASSERT_GT(dispatches.total, 0u);
    if (expect_gemv) {
      EXPECT_EQ(dispatches.gemv, dispatches.total);
    } else {
      EXPECT_EQ(dispatches.gemv, 0u);
    }
  }
};

TEST_F(WebGpuMatMulGemvTest, SmallRows) {
  for (int64_t m = 1; m <= 8; ++m) {
    Run({m, 5120}, 5120, 48, true, true);
  }
  Run({4, 2048}, 2048, 16, true, true);
  Run({4, 8192}, 8192, 64, true, true);
  Run({3, 2049}, 2049, 20, true, true);
  Run({2, 2, 5120}, 5120, 48, true, true);
}

TEST_F(WebGpuMatMulGemvTest, AutomaticSelection) {
  // Non-aligned K excludes subgroup-matrix kernels without forcing GEMV.
  for (int64_t m = 1; m <= 8; ++m) {
    Run({m, 2049}, 2049, 48, true);
  }
}

TEST_F(WebGpuMatMulGemvTest, GuardFallbacks) {
  Run({9, 2049}, 2049, 48, false);
  Run({4, 2047}, 2047, 48, false);
  Run({4, 8193}, 8193, 48, false);
  Run({4, 5120}, 5120, 12, false);
  Run({4, 5120}, 5120, 68, false);
  Run({4, 5120}, 5120, 49, false);
}

TEST_F(WebGpuMatMulGemvTest, BatchedWeightsFallback) {
  Run({2, 2, 2049}, 2049, 48, false, /*force_gemv=*/false, /*batched_weights=*/true);
}

TEST_F(WebGpuMatMulGemvTest, ForcedGemvRejectsTooManyRows) {
  constexpr int64_t M = 9;
  constexpr int64_t K = 2049;
  constexpr int64_t N = 48;
  WebGpuExecutionProviderTestOptions options{};
  options.forced_matmul_algorithm = webgpu::MatMulAlgorithm::Gemv;
  auto ep = WebGpuExecutionProviderWithTestOptions(options);
  ASSERT_NE(ep, nullptr);

  SessionOptions session_options;
  ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));

  OpTester test("MatMul", 14);
  test.AddInput<MLFloat16>("A", {M, K}, std::vector<MLFloat16>(M * K, MLFloat16{1.0f}));
  test.AddInput<MLFloat16>("B", {K, N}, std::vector<MLFloat16>(K * N, MLFloat16{1.0f}));
  test.AddOutput<MLFloat16>("Y", {M, N}, std::vector<MLFloat16>(M * N, MLFloat16{static_cast<float>(K)}));
  test.Config(session_options)
      .ConfigEp(std::move(ep))
      .Config(OpTester::ExpectResult::kExpectFailure,
              "MatMul algorithm gemv does not support these inputs or this device.")
      .RunWithConfig();
}
#endif  // defined(USE_WEBGPU)
#endif

}  // namespace test
}  // namespace onnxruntime
