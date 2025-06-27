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
#include "test/common/tensor_op_test_utils.h"
#include "test/framework/test_utils.h"
#include "test/optimizer/graph_transform_test_builder.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/scoped_env_vars.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/ort_env.h"
#include "core/util/qmath.h"

#if ((defined(MLAS_TARGET_AMD64_IX86) || defined(MLAS_TARGET_ARM64)) && !defined(USE_DML) && !defined(USE_WEBGPU) && !defined(USE_COREML)) || defined(USE_CUDA) || defined(USE_WEBGPU)

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {

namespace test {

namespace {

constexpr int QBits = 8;

struct TestOptions8Bits {
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

[[maybe_unused]] std::ostream& operator<<(std::ostream& os, const TestOptions8Bits& opts) {
  return os << "M:" << opts.M << ", N:" << opts.N << ", K:" << opts.K
            << ", block_size:" << opts.block_size
            << ", accuracy_level:" << opts.accuracy_level
            << ", has_zero_point:" << opts.has_zero_point
            << ", has_g_idx:" << opts.has_g_idx
            << ", has_bias:" << opts.has_bias;
}

template <typename T1>
void RunTest8Bits(const TestOptions8Bits& opts) {
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
  } else {
    test.AddInput<T1>("A", {M, K}, FloatsToMLFloat16s(input0_fp32_vals), false);
  }

  int64_t k_blocks = (K + opts.block_size - 1) / opts.block_size;
  test.AddInput<uint8_t>("B", {q_cols, k_blocks, q_rows / k_blocks}, input1_vals, true);

  if constexpr (std::is_same<T1, float>::value) {
    test.AddInput<T1>("scales", {N, static_cast<int64_t>(q_scale_size) / N}, scales, true);
  } else {
    test.AddInput<T1>("scales", {N, static_cast<int64_t>(q_scale_size) / N}, FloatsToMLFloat16s(scales), true);
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
    } else {
      test.AddInput<T1>("bias", bias_shape, FloatsToMLFloat16s(*bias), true);
    }
  } else {
    test.AddOptionalInputEdge<T1>();
  }

  if constexpr (std::is_same<T1, float>::value) {
    test.AddOutput<T1>("Y", {M, N}, expected_vals);
  } else {
    test.AddOutput<T1>("Y", {M, N}, FloatsToMLFloat16s(expected_vals));
  }

  if (opts.output_abs_error.has_value()) {
    test.SetOutputAbsErr("Y", *opts.output_abs_error);
  }

  if (opts.output_rel_error.has_value()) {
    test.SetOutputRelErr("Y", *opts.output_rel_error);
  }

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#ifdef USE_CUDA
  execution_providers.emplace_back(DefaultCudaExecutionProvider());
#elif USE_WEBGPU
  execution_providers.emplace_back(DefaultWebGpuExecutionProvider());
#endif
#if defined(USE_CUDA) || defined(USE_WEBGPU)
  test.ConfigEps(std::move(execution_providers));
  test.RunWithConfig();
  execution_providers.clear();
#else
  if constexpr (std::is_same<T1, float>::value) {
    if (MlasIsQNBitGemmAvailable(8, 32, SQNBIT_CompInt8)) {
      execution_providers.emplace_back(DefaultCpuExecutionProvider());
      test.ConfigEps(std::move(execution_providers));
      test.RunWithConfig();
    }
  }
#endif
}

template <typename AType, int M, int N, int K, int block_size, int accuracy_level>
void TestMatMul8BitsTyped(float abs_error = 0.1f, float rel_error = 0.02f) {
  TestOptions8Bits base_opts{};
  base_opts.M = M, base_opts.N = N, base_opts.K = K;
  base_opts.block_size = block_size;
  base_opts.accuracy_level = accuracy_level;

  base_opts.output_abs_error = abs_error;
  base_opts.output_rel_error = rel_error;

// TODO: Re-enable on ARM64 after fixing the MLAS kernel when has_zero_point = false
#if !defined(MLAS_TARGET_ARM64)
  {
    TestOptions8Bits opts = base_opts;
    opts.has_zero_point = false;
    opts.has_bias = false;
    RunTest8Bits<AType>(opts);
  }
#endif

  {
    TestOptions8Bits opts = base_opts;
    opts.has_zero_point = true;
    opts.has_bias = false;
    RunTest8Bits<AType>(opts);
  }

// CUDA/WEBGPU does not support bias for MatMulNBits
#if !defined(USE_CUDA) && !defined(USE_WEBGPU)

// TODO: Re-enable on ARM64 after fixing the MLAS kernel when has_zero_point = false
#if !defined(MLAS_TARGET_ARM64)
  {
    TestOptions8Bits opts = base_opts;
    opts.has_zero_point = false;
    opts.has_bias = true;
    RunTest8Bits<AType>(opts);
  }
#endif

  {
    TestOptions8Bits opts = base_opts;
    opts.has_zero_point = true;
    opts.has_bias = true;
    RunTest8Bits<AType>(opts);
  }
#endif
}
}  // namespace

TEST(MatMulNBits, Float32_8b_AccuracyLevel4) {
#if defined(__ANDROID__) && defined(MLAS_TARGET_AMD64_IX86)
  // Fails on Android CI build on Linux host machine:
  //   [ RUN      ] MatMulNBits.Float32_8b_AccuracyLevel4
  //   Test was not executed.
  //   Trap
  // TODO investigate failure
  GTEST_SKIP() << "Skipping test on Android x86_64 (emulator).";
#endif
  TestMatMul8BitsTyped<float, 1, 1, 16, 16, 4>();
  TestMatMul8BitsTyped<float, 1, 8, 16, 16, 4>();
  TestMatMul8BitsTyped<float, 1, 2, 16, 16, 4>();
  TestMatMul8BitsTyped<float, 1, 32, 16, 16, 4>();
  TestMatMul8BitsTyped<float, 1, 4, 32, 16, 4>();
  TestMatMul8BitsTyped<float, 2, 4, 32, 16, 4>();
  TestMatMul8BitsTyped<float, 1, 4, 64, 16, 4>();
  TestMatMul8BitsTyped<float, 1, 32, 16, 128, 4>();
  TestMatMul8BitsTyped<float, 1, 256, 32, 16, 4>();
  TestMatMul8BitsTyped<float, 1, 288, 16, 16, 4>();
  TestMatMul8BitsTyped<float, 1, 288, 1024, 16, 4>();
  TestMatMul8BitsTyped<float, 1, 288, 1024, 128, 4>();
  TestMatMul8BitsTyped<float, 2, 288, 1024, 128, 4>();
  TestMatMul8BitsTyped<float, 1, 40, 576, 32, 4>();
  TestMatMul8BitsTyped<float, 2, 40, 576, 32, 4>();
  TestMatMul8BitsTyped<float, 1, 288, 93, 32, 4>();
  TestMatMul8BitsTyped<float, 1, 288, 93, 128, 4>();
  TestMatMul8BitsTyped<float, 1, 288, 1234, 16, 4>();
  TestMatMul8BitsTyped<float, 2, 1, 16, 16, 4>();
  TestMatMul8BitsTyped<float, 2, 2, 16, 16, 4>();
  TestMatMul8BitsTyped<float, 100, 1, 16, 16, 4>();
  TestMatMul8BitsTyped<float, 100, 2, 16, 16, 4>();
  TestMatMul8BitsTyped<float, 100, 32, 16, 16, 4>();
  TestMatMul8BitsTyped<float, 100, 32, 32, 16, 4>();
  TestMatMul8BitsTyped<float, 100, 32, 16, 128, 4>();
  TestMatMul8BitsTyped<float, 100, 288, 16, 16, 4>();
  TestMatMul8BitsTyped<float, 100, 288, 1024, 16, 4>();
  TestMatMul8BitsTyped<float, 100, 288, 1024, 128, 4>();
  TestMatMul8BitsTyped<float, 100, 288, 192, 64, 4>();
  TestMatMul8BitsTyped<float, 100, 288, 93, 32, 4>();
  TestMatMul8BitsTyped<float, 100, 288, 93, 128, 4>();
  TestMatMul8BitsTyped<float, 100, 288, 1234, 16, 4>();
  TestMatMul8BitsTyped<float, 2, 5120, 3072, 32, 4>();
}

TEST(MatMulNBits, Float32_8b_AccuracyLevel1) {
#if defined(__ANDROID__) && defined(MLAS_TARGET_AMD64_IX86)
  // Fails on Android CI build on Linux host machine:
  //   [ RUN      ] MatMulNBits.Float32_8b_AccuracyLevel1
  //   Trap
  // TODO investigate failure
  GTEST_SKIP() << "Skipping test on Android x86_64 (emulator).";
#endif

  // At the time of writing these tests, Fp32 activations + 8 bit weights + Accuracy level 1
  // do not have MLAS optimized kernels on any platform and hence this will use the "unpacked"
  // compute mode (i.e.) de-quantize the 8 bit weights to fp32 and invoke vanilla fp32 Gemm
  // in MLAS. This test helps keep that path tested.
  TestMatMul8BitsTyped<float, 1, 1, 16, 16, 1>();
  TestMatMul8BitsTyped<float, 1, 2, 16, 16, 1>();
  TestMatMul8BitsTyped<float, 1, 32, 16, 16, 1>();
  TestMatMul8BitsTyped<float, 1, 4, 32, 16, 1>();
  TestMatMul8BitsTyped<float, 2, 4, 32, 16, 1>();
  TestMatMul8BitsTyped<float, 1, 4, 64, 16, 1>();
  TestMatMul8BitsTyped<float, 1, 32, 16, 128, 1>();
  TestMatMul8BitsTyped<float, 1, 256, 32, 16, 1>();
  TestMatMul8BitsTyped<float, 1, 288, 16, 16, 1>();
  TestMatMul8BitsTyped<float, 1, 288, 1024, 16, 1>();
  TestMatMul8BitsTyped<float, 1, 288, 1024, 128, 1>();
  TestMatMul8BitsTyped<float, 2, 288, 1024, 128, 1>();
  TestMatMul8BitsTyped<float, 1, 40, 576, 32, 1>();
  TestMatMul8BitsTyped<float, 2, 40, 576, 32, 1>();
  TestMatMul8BitsTyped<float, 1, 288, 93, 32, 1>();
  TestMatMul8BitsTyped<float, 1, 288, 93, 128, 1>();
  TestMatMul8BitsTyped<float, 1, 288, 1234, 16, 1>();
  TestMatMul8BitsTyped<float, 2, 1, 16, 16, 1>();
  TestMatMul8BitsTyped<float, 2, 2, 16, 16, 1>();
  TestMatMul8BitsTyped<float, 100, 1, 16, 16, 1>();
  TestMatMul8BitsTyped<float, 100, 2, 16, 16, 1>();
  TestMatMul8BitsTyped<float, 100, 32, 16, 16, 1>();
  TestMatMul8BitsTyped<float, 100, 32, 32, 16, 1>();
  TestMatMul8BitsTyped<float, 100, 32, 16, 128, 1>();
  TestMatMul8BitsTyped<float, 100, 288, 16, 16, 1>();
  TestMatMul8BitsTyped<float, 100, 288, 1024, 16, 1>();
  TestMatMul8BitsTyped<float, 100, 288, 1024, 128, 1>();
  TestMatMul8BitsTyped<float, 100, 288, 192, 64, 1>();
  TestMatMul8BitsTyped<float, 100, 288, 93, 32, 1>();
  TestMatMul8BitsTyped<float, 100, 288, 93, 128, 1>();
  TestMatMul8BitsTyped<float, 100, 288, 1234, 16, 1>();
  TestMatMul8BitsTyped<float, 2, 5120, 3072, 32, 1>();
}

#if defined(USE_CUDA) || defined(USE_WEBGPU)
TEST(MatMulNBits, Float16_8b_AccuracyLevel4) {
  constexpr float abs_error = 0.055f;
  constexpr float rel_error = 0.02f;
  TestMatMul8BitsTyped<MLFloat16, 2, 4, 32, 16, 4>(abs_error, rel_error);
  TestMatMul8BitsTyped<MLFloat16, 100, 64, 32, 32, 4>(abs_error, rel_error);
  TestMatMul8BitsTyped<MLFloat16, 100, 128, 128, 32, 4>(abs_error, rel_error);
  TestMatMul8BitsTyped<MLFloat16, 199, 40, 576, 32, 4>(abs_error, rel_error);
}
#endif

#if defined(USE_CUDA)
TEST(MatMulNBits, Fp16_Int8_Cuda) {
  constexpr float abs_error = 0.5f;
  constexpr float rel_error = 0.05f;

  ScopedEnvironmentVariables scoped_env_vars{EnvVarMap{{"ORT_FPA_INTB_GEMM", "1"}}};

  TestMatMul8BitsTyped<MLFloat16, 1, 256, 256, 64, 4>(abs_error, rel_error);
  TestMatMul8BitsTyped<MLFloat16, 32, 512, 512, 64, 4>(abs_error, rel_error);
  TestMatMul8BitsTyped<MLFloat16, 1, 256, 256, 128, 4>(abs_error, rel_error);
  TestMatMul8BitsTyped<MLFloat16, 32, 1024, 1024, 128, 4>(abs_error, rel_error);
}
#endif

}  // namespace test
}  // namespace onnxruntime

#endif
#endif  // ORT_MINIMAL_BUILD
