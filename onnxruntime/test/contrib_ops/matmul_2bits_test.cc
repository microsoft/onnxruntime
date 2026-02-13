// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef ORT_MINIMAL_BUILD

#include <optional>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "core/common/narrow.h"
#include "core/common/span_utils.h"
#include "core/framework/tensor.h"
#include "core/mlas/inc/mlas_qnbit.h"
#include "core/mlas/inc/mlas_q4.h"
#include "core/mlas/inc/mlas.h"
#include "core/session/inference_session.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/unittest_util/graph_transform_test_builder.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/scoped_env_vars.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/ort_env.h"
#include "core/util/qmath.h"
#include "core/providers/webgpu/webgpu_provider_options.h"

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {

namespace test {

namespace {

constexpr int QBits = 2;

struct TestOptions2Bits {
  int64_t M{1};
  int64_t N{1};
  int64_t K{1};
  int64_t block_size{32};
  int64_t accuracy_level{0};

  bool has_zero_point{false};
  bool has_g_idx{false};
  bool has_bias{false};

  std::optional<float> output_abs_error{};
  std::optional<float> output_rel_error{};
};

[[maybe_unused]] std::ostream& operator<<(std::ostream& os, const TestOptions2Bits& opts) {
  return os << "M:" << opts.M << ", N:" << opts.N << ", K:" << opts.K
            << ", block_size:" << opts.block_size
            << ", accuracy_level:" << opts.accuracy_level
            << ", has_zero_point:" << opts.has_zero_point
            << ", has_g_idx:" << opts.has_g_idx
            << ", has_bias:" << opts.has_bias;
}

template <typename T1>
void RunTest2Bits(const TestOptions2Bits& opts) {
  SCOPED_TRACE(opts);

  const int64_t M = opts.M,
                K = opts.K,
                N = opts.N;

  RandomValueGenerator random{1234};
  std::vector<float> input0_fp32_vals(random.Gaussian<float>(AsSpan({M, K}), 0.0f, 0.25f));
  std::vector<float> input1_fp32_vals(random.Gaussian<float>(AsSpan({K, N}), 0.0f, 0.25f));

  int q_rows, q_cols;
  MlasBlockwiseQuantizedShape<float, QBits>(static_cast<int>(opts.block_size), /* columnwise */ true,
                                            static_cast<int>(K), static_cast<int>(N),
                                            q_rows, q_cols);

  size_t q_data_size_in_bytes, q_scale_size, q_zp_size_in_bytes;
  MlasBlockwiseQuantizedBufferSizes<QBits>(static_cast<int>(opts.block_size), /* columnwise */ true,
                                           static_cast<int>(K), static_cast<int>(N),
                                           q_data_size_in_bytes, q_scale_size, &q_zp_size_in_bytes);

  std::vector<uint8_t> input1_vals(q_data_size_in_bytes);
  std::vector<float> scales(q_scale_size);
  std::vector<uint8_t> zp(q_zp_size_in_bytes);

  auto& ortenv = **ort_env.get();
  onnxruntime::concurrency::ThreadPool* tp = ortenv.GetEnvironment().GetIntraOpThreadPool();

  MlasQuantizeBlockwise<float, QBits>(
      input1_vals.data(),
      scales.data(),
      opts.has_zero_point ? zp.data() : nullptr,
      input1_fp32_vals.data(),
      static_cast<int32_t>(opts.block_size),
      true,
      static_cast<int32_t>(K),
      static_cast<int32_t>(N),
      static_cast<int32_t>(N),
      tp);

  // Note that raw_vals is NxK after dequant
  MlasDequantizeBlockwise<float, QBits>(
      input1_fp32_vals.data(),
      input1_vals.data(),
      scales.data(),
      opts.has_zero_point ? zp.data() : nullptr,
      static_cast<int32_t>(opts.block_size),
      true,
      static_cast<int32_t>(K),
      static_cast<int32_t>(N),
      tp);

  const std::vector<int64_t> bias_shape = {N};
  const auto bias = [&]() -> std::optional<std::vector<float>> {
    if (opts.has_bias) {
      return random.Uniform(bias_shape, 1.0f, 5.0f);
    }
    return std::nullopt;
  }();

  std::vector<float> expected_vals(M * N);
  for (int64_t m = 0; m < M; m++) {
    for (int64_t n = 0; n < N; n++) {
      float sum = 0.0f;
      for (int64_t k = 0; k < K; k++) {
        sum += input0_fp32_vals[m * K + k] * input1_fp32_vals[n * K + k];
      }
      expected_vals[m * N + n] = sum + (bias.has_value() ? (*bias)[n] : 0.0f);
    }
  }

  OpTester test("MatMulNBits", 1, kMSDomain);
  test.AddAttribute<int64_t>("K", K);
  test.AddAttribute<int64_t>("N", N);
  test.AddAttribute<int64_t>("block_size", opts.block_size);
  test.AddAttribute<int64_t>("bits", QBits);
  test.AddAttribute<int64_t>("accuracy_level", opts.accuracy_level);

  if constexpr (std::is_same<T1, float>::value) {
    test.AddInput<T1>("A", {M, K}, input0_fp32_vals, false);
  } else if constexpr (std::is_same<T1, MLFloat16>::value) {
    test.AddInput<T1>("A", {M, K}, FloatsToMLFloat16s(input0_fp32_vals), false);
  } else if constexpr (std::is_same<T1, BFloat16>::value) {
    test.AddInput<T1>("A", {M, K}, FloatsToBFloat16s(input0_fp32_vals), false);
  }

  int64_t k_blocks = (K + opts.block_size - 1) / opts.block_size;
  test.AddInput<uint8_t>("B", {q_cols, k_blocks, q_rows / k_blocks}, input1_vals, true);

  if constexpr (std::is_same<T1, float>::value) {
    test.AddInput<T1>("scales", {N, static_cast<int64_t>(q_scale_size) / N}, scales, true);
  } else if constexpr (std::is_same<T1, MLFloat16>::value) {
    test.AddInput<T1>("scales", {N, static_cast<int64_t>(q_scale_size) / N}, FloatsToMLFloat16s(scales), true);
  } else if constexpr (std::is_same<T1, BFloat16>::value) {
    test.AddInput<T1>("scales", {N, static_cast<int64_t>(q_scale_size) / N}, FloatsToBFloat16s(scales), true);
  }

  if (opts.has_zero_point) {
    test.AddInput<uint8_t>("zero_points", {N, static_cast<int64_t>(q_zp_size_in_bytes) / N}, zp, true);
  } else {
    test.AddOptionalInputEdge<uint8_t>();
  }

  // Account for deprecated "g_idx" input
  test.AddOptionalInputEdge<int32_t>();

  if (bias.has_value()) {
    if constexpr (std::is_same<T1, float>::value) {
      test.AddInput<T1>("bias", bias_shape, *bias, true);
    } else if constexpr (std::is_same<T1, MLFloat16>::value) {
      test.AddInput<T1>("bias", bias_shape, FloatsToMLFloat16s(*bias), true);
    } else if constexpr (std::is_same<T1, BFloat16>::value) {
      test.AddInput<T1>("bias", bias_shape, FloatsToBFloat16s(*bias), true);
    }
  } else {
    test.AddOptionalInputEdge<T1>();
  }

  if constexpr (std::is_same<T1, float>::value) {
    test.AddOutput<T1>("Y", {M, N}, expected_vals);
  } else if constexpr (std::is_same<T1, MLFloat16>::value) {
    test.AddOutput<T1>("Y", {M, N}, FloatsToMLFloat16s(expected_vals));
  } else if constexpr (std::is_same<T1, BFloat16>::value) {
    test.AddOutput<T1>("Y", {M, N}, FloatsToBFloat16s(expected_vals));
  }

  if (opts.output_abs_error.has_value()) {
    test.SetOutputAbsErr("Y", *opts.output_abs_error);
  }

  if (opts.output_rel_error.has_value()) {
    test.SetOutputRelErr("Y", *opts.output_rel_error);
  }

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  if constexpr (std::is_same<T1, float>::value) {
#ifdef USE_WEBGPU
    execution_providers.push_back(DefaultWebGpuExecutionProvider());
#endif
    execution_providers.emplace_back(DefaultCpuExecutionProvider());
    test.ConfigEps(std::move(execution_providers));
    test.RunWithConfig();
  }
}

template <typename AType, int M, int N, int K, int block_size, int accuracy_level = 0>
void TestMatMul2BitsTyped(float abs_error = 0.1f, float rel_error = 0.02f) {
  TestOptions2Bits base_opts{};
  base_opts.M = M, base_opts.N = N, base_opts.K = K;
  base_opts.block_size = block_size;
  base_opts.accuracy_level = accuracy_level;

  base_opts.output_abs_error = abs_error;
  base_opts.output_rel_error = rel_error;

  {
    TestOptions2Bits opts = base_opts;
    opts.has_zero_point = false;
    opts.has_bias = false;
    RunTest2Bits<AType>(opts);
  }

  {
    TestOptions2Bits opts = base_opts;
    opts.has_zero_point = true;
    opts.has_bias = false;
    RunTest2Bits<AType>(opts);
  }

  {
    TestOptions2Bits opts = base_opts;
    opts.has_zero_point = false;
    opts.has_bias = true;
    RunTest2Bits<AType>(opts);
  }

  {
    TestOptions2Bits opts = base_opts;
    opts.has_zero_point = true;
    opts.has_bias = true;
    RunTest2Bits<AType>(opts);
  }
}

}  // namespace

template <typename AType>
void TestMatMul2BitsLutGemm(int64_t M, int64_t N, int64_t K, int64_t block_size,
                            bool has_zero_point, float abs_error = 0.15f, float rel_error = 0.05f) {
  if (K % 32 != 0 || N % 128 != 0 || block_size % 32 != 0) {
    GTEST_SKIP() << "LUT GEMM requires K multiple of 32, N multiple of 128, block_size multiple of 32";
  }

  if (!MlasIsLutGemmAvailable(static_cast<size_t>(N), static_cast<size_t>(K), 2, static_cast<size_t>(block_size))) {
    GTEST_SKIP() << "LUT GEMM not available on this platform";
  }

  RandomValueGenerator random{1234};
  std::vector<float> input0_fp32_vals(random.Gaussian<float>(AsSpan({M, K}), 0.0f, 0.25f));
  std::vector<float> input1_fp32_vals(random.Gaussian<float>(AsSpan({K, N}), 0.0f, 0.25f));

  int q_rows, q_cols;
  MlasBlockwiseQuantizedShape<float, QBits>(static_cast<int>(block_size), /* columnwise */ true,
                                            static_cast<int>(K), static_cast<int>(N),
                                            q_rows, q_cols);

  size_t q_data_size_in_bytes, q_scale_size, q_zp_size_in_bytes;
  MlasBlockwiseQuantizedBufferSizes<QBits>(static_cast<int>(block_size), /* columnwise */ true,
                                           static_cast<int>(K), static_cast<int>(N),
                                           q_data_size_in_bytes, q_scale_size, &q_zp_size_in_bytes);

  std::vector<uint8_t> input1_vals(q_data_size_in_bytes);
  std::vector<float> scales(q_scale_size);
  std::vector<uint8_t> zp(q_zp_size_in_bytes);

  auto& ortenv = **ort_env.get();
  onnxruntime::concurrency::ThreadPool* tp = ortenv.GetEnvironment().GetIntraOpThreadPool();

  MlasQuantizeBlockwise<float, QBits>(
      input1_vals.data(),
      scales.data(),
      has_zero_point ? zp.data() : nullptr,
      input1_fp32_vals.data(),
      static_cast<int32_t>(block_size),
      true,
      static_cast<int32_t>(K),
      static_cast<int32_t>(N),
      static_cast<int32_t>(N),
      tp);

  // Dequantize for reference computation
  MlasDequantizeBlockwise<float, QBits>(
      input1_fp32_vals.data(),
      input1_vals.data(),
      scales.data(),
      has_zero_point ? zp.data() : nullptr,
      static_cast<int32_t>(block_size),
      true,
      static_cast<int32_t>(K),
      static_cast<int32_t>(N),
      tp);

  std::vector<float> expected_vals(M * N);
  for (int64_t m = 0; m < M; m++) {
    for (int64_t n = 0; n < N; n++) {
      float sum = 0.0f;
      for (int64_t k = 0; k < K; k++) {
        sum += input0_fp32_vals[m * K + k] * input1_fp32_vals[n * K + k];
      }
      expected_vals[m * N + n] = sum;
    }
  }

  OpTester test("MatMulNBits", 1, kMSDomain);
  test.AddAttribute<int64_t>("K", K);
  test.AddAttribute<int64_t>("N", N);
  test.AddAttribute<int64_t>("block_size", block_size);
  test.AddAttribute<int64_t>("bits", QBits);
  test.AddAttribute<int64_t>("accuracy_level", static_cast<int64_t>(0));

  if constexpr (std::is_same<AType, float>::value) {
    test.AddInput<AType>("A", {M, K}, input0_fp32_vals, false);
  }

  int64_t k_blocks = (K + block_size - 1) / block_size;
  test.AddInput<uint8_t>("B", {q_cols, k_blocks, q_rows / k_blocks}, input1_vals, true);

  if constexpr (std::is_same<AType, float>::value) {
    test.AddInput<AType>("scales", {N, static_cast<int64_t>(q_scale_size) / N}, scales, true);
  }

  if (has_zero_point) {
    test.AddInput<uint8_t>("zero_points", {N, static_cast<int64_t>(q_zp_size_in_bytes) / N}, zp, true);
  } else {
    test.AddOptionalInputEdge<uint8_t>();
  }

  test.AddOptionalInputEdge<int32_t>();
  test.AddOptionalInputEdge<AType>();

  if constexpr (std::is_same<AType, float>::value) {
    test.AddOutput<AType>("Y", {M, N}, expected_vals);
  }

  test.SetOutputAbsErr("Y", abs_error);
  test.SetOutputRelErr("Y", rel_error);

  SessionOptions so;
  ASSERT_STATUS_OK(so.config_options.AddConfigEntry(kOrtSessionOptionsMlasLutGemm, "1"));

  test.Config(so)
      .ConfigEp(DefaultCpuExecutionProvider())
      .RunWithConfig();
}

TEST(MatMulNBitsLutGemm, Float32_2Bits_Symmetric_128x128) {
  TestMatMul2BitsLutGemm<float>(1, 128, 128, 32, false);
}

TEST(MatMulNBitsLutGemm, Float32_2Bits_Asymmetric_128x128) {
  TestMatMul2BitsLutGemm<float>(1, 128, 128, 32, true);
}

TEST(MatMulNBitsLutGemm, Float32_2Bits_Symmetric_256x256) {
  TestMatMul2BitsLutGemm<float>(1, 256, 256, 32, false);
}

// This test was previously disabled due to accuracy issues related to non-deterministic
// gather operations. It is now re-enabled after replacing gather with deterministic
// load+shuffle operations to improve determinism and stability.
TEST(MatMulNBitsLutGemm, Float32_2Bits_Asymmetric_256x256) {
  TestMatMul2BitsLutGemm<float>(1, 256, 256, 32, true);
}

TEST(MatMulNBitsLutGemm, Float32_2Bits_Symmetric_256x256_BlkLen64) {
  TestMatMul2BitsLutGemm<float>(1, 256, 256, 64, false);
}

TEST(MatMulNBitsLutGemm, Float32_2Bits_Asymmetric_256x256_BlkLen64) {
  TestMatMul2BitsLutGemm<float>(1, 256, 256, 64, true);
}

TEST(MatMulNBitsLutGemm, Float32_2Bits_Symmetric_128x256_BlkLen128) {
  TestMatMul2BitsLutGemm<float>(1, 128, 256, 128, false);
}

TEST(MatMulNBitsLutGemm, Float32_2Bits_Asymmetric_128x256_BlkLen128) {
  TestMatMul2BitsLutGemm<float>(1, 128, 256, 128, true);
}

// Batch tests (M > 1)
TEST(MatMulNBitsLutGemm, Float32_2Bits_Symmetric_Batch32_128x128) {
  TestMatMul2BitsLutGemm<float>(32, 128, 128, 32, false);
}

TEST(MatMulNBitsLutGemm, Float32_2Bits_Asymmetric_Batch32_256x256) {
  TestMatMul2BitsLutGemm<float>(32, 256, 256, 32, true);
}

TEST(MatMul2Bits, Float32_2b_Accuracy0) {
  TestMatMul2BitsTyped<float, 1, 1, 16, 16, 0>();
  TestMatMul2BitsTyped<float, 1, 2, 16, 16, 0>();
  TestMatMul2BitsTyped<float, 1, 32, 16, 16, 0>();
  TestMatMul2BitsTyped<float, 1, 32, 32, 16, 0>();
  TestMatMul2BitsTyped<float, 1, 32, 16, 128, 0>();
  TestMatMul2BitsTyped<float, 1, 288, 16, 16, 0>();
  TestMatMul2BitsTyped<float, 2, 1, 16, 16, 0>();
  TestMatMul2BitsTyped<float, 2, 2, 16, 16, 0>();
  TestMatMul2BitsTyped<float, 4, 1, 16, 16, 0>();
  TestMatMul2BitsTyped<float, 4, 2, 16, 16, 0>();
  TestMatMul2BitsTyped<float, 4, 32, 16, 16, 0>();
  TestMatMul2BitsTyped<float, 4, 32, 32, 16, 0>();
  TestMatMul2BitsTyped<float, 4, 32, 16, 128, 0>();
  TestMatMul2BitsTyped<float, 4, 288, 16, 16, 0>();
  TestMatMul2BitsTyped<float, 100, 1, 16, 16, 0>();
  TestMatMul2BitsTyped<float, 100, 2, 16, 16, 0>();
  TestMatMul2BitsTyped<float, 100, 32, 16, 16, 0>();
  TestMatMul2BitsTyped<float, 100, 32, 32, 16, 0>();
  TestMatMul2BitsTyped<float, 100, 32, 16, 128, 0>();
  TestMatMul2BitsTyped<float, 100, 288, 16, 16, 0>();
}

TEST(MatMul2Bits, Float32_2b_Accuracy4) {
  TestMatMul2BitsTyped<float, 1, 1, 16, 16, 4>();
  TestMatMul2BitsTyped<float, 1, 2, 16, 16, 4>();
  TestMatMul2BitsTyped<float, 1, 32, 16, 16, 4>();
  TestMatMul2BitsTyped<float, 1, 32, 32, 16, 4>();
  TestMatMul2BitsTyped<float, 1, 32, 16, 128, 4>();
  TestMatMul2BitsTyped<float, 1, 288, 16, 16, 4>();
  TestMatMul2BitsTyped<float, 2, 1, 16, 16, 4>();
  TestMatMul2BitsTyped<float, 2, 2, 16, 16, 4>();
  TestMatMul2BitsTyped<float, 4, 1, 16, 16, 4>();
  TestMatMul2BitsTyped<float, 4, 2, 16, 16, 4>();
  TestMatMul2BitsTyped<float, 4, 32, 16, 16, 4>();
  TestMatMul2BitsTyped<float, 4, 32, 32, 16, 4>();
  TestMatMul2BitsTyped<float, 4, 32, 16, 128, 4>();
  TestMatMul2BitsTyped<float, 4, 288, 16, 16, 4>();
  TestMatMul2BitsTyped<float, 100, 1, 16, 16, 4>();
  TestMatMul2BitsTyped<float, 100, 2, 16, 16, 4>();
  TestMatMul2BitsTyped<float, 100, 32, 16, 16, 4>();
  TestMatMul2BitsTyped<float, 100, 32, 32, 16, 4>();
  TestMatMul2BitsTyped<float, 100, 32, 16, 128, 4>();
  TestMatMul2BitsTyped<float, 100, 288, 16, 16, 4>();
}

#ifdef USE_WEBGPU

namespace {

// Runs a 2-bit MatMulNBits test on WebGPU EP with CPU as baseline.
// The test quantizes random weights to 2 bits, dequantizes to compute
// expected output via matmul on CPU, then compares WebGPU output.
template <typename T>
void RunWebGpu2BitsTest(int64_t M, int64_t N, int64_t K, int64_t block_size,
                        bool has_zero_point, float abs_error = 0.1f, float rel_error = 0.02f) {
  TestOptions2Bits opts{};
  opts.M = M;
  opts.N = N;
  opts.K = K;
  opts.block_size = block_size;
  opts.has_zero_point = has_zero_point;
  opts.output_abs_error = abs_error;
  opts.output_rel_error = rel_error;

  RunTest2Bits<T>(opts);
}

}  // namespace

// WebGPU 2-bit tests: symmetric (no zero points)
TEST(MatMul2BitsWebGpu, Float32_Symmetric_Small) {
  RunWebGpu2BitsTest<float>(1, 32, 32, 16, false);
  RunWebGpu2BitsTest<float>(1, 32, 32, 32, false);
  RunWebGpu2BitsTest<float>(1, 32, 16, 16, false);
}

TEST(MatMul2BitsWebGpu, Float32_Symmetric_Medium) {
  RunWebGpu2BitsTest<float>(1, 288, 16, 16, false);
  RunWebGpu2BitsTest<float>(4, 32, 32, 16, false);
  RunWebGpu2BitsTest<float>(4, 288, 16, 16, false);
  RunWebGpu2BitsTest<float>(100, 32, 32, 16, false);
  RunWebGpu2BitsTest<float>(100, 288, 16, 16, false);
}

// WebGPU 2-bit tests: asymmetric (with zero points) — the primary accuracy concern
TEST(MatMul2BitsWebGpu, Float32_ZeroPoint_Small) {
  RunWebGpu2BitsTest<float>(1, 1, 16, 16, true);
  RunWebGpu2BitsTest<float>(1, 2, 16, 16, true);
  RunWebGpu2BitsTest<float>(1, 32, 16, 16, true);
  RunWebGpu2BitsTest<float>(1, 32, 32, 16, true);
  RunWebGpu2BitsTest<float>(1, 32, 32, 32, true);
}

TEST(MatMul2BitsWebGpu, Float32_ZeroPoint_Medium) {
  RunWebGpu2BitsTest<float>(1, 288, 16, 16, true);
  RunWebGpu2BitsTest<float>(4, 32, 32, 16, true);
  RunWebGpu2BitsTest<float>(4, 288, 16, 16, true);
  RunWebGpu2BitsTest<float>(100, 32, 32, 16, true);
  RunWebGpu2BitsTest<float>(100, 288, 16, 16, true);
}

TEST(MatMul2BitsWebGpu, Float32_ZeroPoint_BlockSize32) {
  // blockSize=32 triggers the Intel Gen12 optimized path on matching hardware.
  RunWebGpu2BitsTest<float>(1, 32, 32, 32, true);
  RunWebGpu2BitsTest<float>(4, 32, 32, 32, true);
  RunWebGpu2BitsTest<float>(100, 32, 32, 32, true);
}

TEST(MatMul2BitsWebGpu, Float32_ZeroPoint_BlockSize128) {
  RunWebGpu2BitsTest<float>(1, 32, 16, 128, true);
  RunWebGpu2BitsTest<float>(4, 32, 16, 128, true);
  RunWebGpu2BitsTest<float>(100, 32, 16, 128, true);
}

// BlockSize=64 tests — covers nBlocksPerCol not a multiple of 4 (padding edge case).
// These match configurations found in real 2-bit quantized transformer models.
TEST(MatMul2BitsWebGpu, Float32_ZeroPoint_BlockSize64) {
  RunWebGpu2BitsTest<float>(1, 32, 64, 64, true);
  RunWebGpu2BitsTest<float>(1, 32, 128, 64, true);
  RunWebGpu2BitsTest<float>(1, 192, 384, 64, true, 0.3f, 0.05f);
  RunWebGpu2BitsTest<float>(1, 384, 1024, 64, true, 0.5f, 0.05f);
}

TEST(MatMul2BitsWebGpu, Float32_Symmetric_BlockSize64) {
  RunWebGpu2BitsTest<float>(1, 32, 64, 64, false);
  RunWebGpu2BitsTest<float>(1, 32, 128, 64, false);
  RunWebGpu2BitsTest<float>(1, 192, 384, 64, false, 0.3f, 0.05f);
}

// Larger K tests — exercises multi-word (multiple u32) extraction per block,
// verifying the Q2 nibble-spread and A-offset tracking across passes.
TEST(MatMul2BitsWebGpu, Float32_ZeroPoint_LargerK) {
  RunWebGpu2BitsTest<float>(1, 32, 64, 32, true);
  RunWebGpu2BitsTest<float>(1, 32, 128, 32, true);
  RunWebGpu2BitsTest<float>(1, 32, 256, 32, true, 0.3f, 0.05f);
}

#endif  // USE_WEBGPU

}  // namespace test
}  // namespace onnxruntime

#endif  // ORT_MINIMAL_BUILD
