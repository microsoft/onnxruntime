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
#include "test/common/cuda_op_test_utils.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/framework/test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/graph_transform_test_builder.h"
#include "test/util/include/scoped_env_vars.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/ort_env.h"
#include "core/util/qmath.h"

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

TEST(MatMulNBits, Float32_2Bits_Accuracy0) {
  // Currently, only fallback option enabled for 2bit datatypes
  // where the 2bits are dequantized to fp32
  TestMatMul2BitsTyped<float, 1, 1, 16, 16, 0>();
  TestMatMul2BitsTyped<float, 1, 2, 16, 16, 0>();
  TestMatMul2BitsTyped<float, 1, 32, 16, 16, 0>();
  TestMatMul2BitsTyped<float, 1, 32, 32, 16, 0>();
  TestMatMul2BitsTyped<float, 1, 32, 16, 128, 0>();
  TestMatMul2BitsTyped<float, 1, 288, 16, 16, 0>();
  TestMatMul2BitsTyped<float, 1, 1, 16, 16, 0>();
  TestMatMul2BitsTyped<float, 4, 2, 16, 16, 0>();
  TestMatMul2BitsTyped<float, 4, 32, 16, 16, 0>();
  TestMatMul2BitsTyped<float, 4, 32, 32, 16, 0>();
  TestMatMul2BitsTyped<float, 4, 32, 16, 128, 0>();
  TestMatMul2BitsTyped<float, 4, 288, 16, 16, 0>();
}
}  // namespace test
}  // namespace onnxruntime

#endif  // ORT_MINIMAL_BUILD
