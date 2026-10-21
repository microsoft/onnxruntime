// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <ostream>
#include <type_traits>
#include <vector>

#include "core/common/span_utils.h"
#include "core/mlas/inc/mlas_q4.h"
#include "core/session/ort_env.h"
#include "onnxruntime_cxx_api.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime::test {

struct MatMul2BitsTestParams {
  int64_t M{1};
  int64_t N{1};
  int64_t K{1};
  int64_t block_size{32};
  int64_t accuracy_level{0};

  bool has_zero_point{false};
  bool has_typed_zero_point{false};
  bool has_g_idx{false};
  bool has_bias{false};

  std::optional<float> output_abs_error{};
  std::optional<float> output_rel_error{};
};

inline std::ostream& operator<<(std::ostream& os, const MatMul2BitsTestParams& opts) {
  return os << "M:" << opts.M << ", N:" << opts.N << ", K:" << opts.K
            << ", block_size:" << opts.block_size
            << ", accuracy_level:" << opts.accuracy_level
            << ", has_zero_point:" << opts.has_zero_point
            << ", has_typed_zero_point:" << opts.has_typed_zero_point
            << ", has_g_idx:" << opts.has_g_idx
            << ", has_bias:" << opts.has_bias;
}

// Populate the numerical case without choosing an execution provider.
template <typename T1>
void PopulateMatMul2BitsTest(OpTester& test, const MatMul2BitsTestParams& opts) {
  constexpr int kQuantBits = 2;
  const int64_t M = opts.M,
                K = opts.K,
                N = opts.N;

  RandomValueGenerator random{1234};
  std::vector<float> input0_fp32_vals(random.Gaussian<float>(AsSpan({M, K}), 0.0f, 0.25f));
  std::vector<float> input1_fp32_vals(random.Gaussian<float>(AsSpan({K, N}), 0.0f, 0.25f));

  int q_rows, q_cols;
  MlasBlockwiseQuantizedShape<float, kQuantBits>(static_cast<int>(opts.block_size), /* columnwise */ true,
                                                 static_cast<int>(K), static_cast<int>(N),
                                                 q_rows, q_cols);

  size_t q_data_size_in_bytes, q_scale_size, q_zp_size_in_bytes;
  MlasBlockwiseQuantizedBufferSizes<kQuantBits>(static_cast<int>(opts.block_size), /* columnwise */ true,
                                                static_cast<int>(K), static_cast<int>(N),
                                                q_data_size_in_bytes, q_scale_size, &q_zp_size_in_bytes);

  std::vector<uint8_t> input1_vals(q_data_size_in_bytes);
  std::vector<float> scales(q_scale_size);
  std::vector<uint8_t> zp(q_zp_size_in_bytes);

  auto& ortenv = **ort_env.get();
  onnxruntime::concurrency::ThreadPool* tp = ortenv.GetEnvironment().GetIntraOpThreadPool();

  MlasQuantizeBlockwise<float, kQuantBits>(
      input1_vals.data(),
      scales.data(),
      opts.has_zero_point && !opts.has_typed_zero_point ? zp.data() : nullptr,
      input1_fp32_vals.data(),
      static_cast<int32_t>(opts.block_size),
      true,
      static_cast<int32_t>(K),
      static_cast<int32_t>(N),
      static_cast<int32_t>(N),
      tp);

  // Note that raw_vals is NxK after dequant
  MlasDequantizeBlockwise<float, kQuantBits>(
      input1_fp32_vals.data(),
      input1_vals.data(),
      scales.data(),
      opts.has_zero_point && !opts.has_typed_zero_point ? zp.data() : nullptr,
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

  test.AddAttribute<int64_t>("K", K);
  test.AddAttribute<int64_t>("N", N);
  test.AddAttribute<int64_t>("block_size", opts.block_size);
  test.AddAttribute<int64_t>("bits", kQuantBits);
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
    if (opts.has_typed_zero_point) {
      const std::vector<float> typed_zp(q_scale_size, 2.0f);
      if constexpr (std::is_same<T1, float>::value) {
        test.AddInput<T1>("zero_points", {N, static_cast<int64_t>(q_scale_size) / N}, typed_zp, true);
      } else if constexpr (std::is_same<T1, MLFloat16>::value) {
        test.AddInput<T1>("zero_points", {N, static_cast<int64_t>(q_scale_size) / N},
                          FloatsToMLFloat16s(typed_zp), true);
      } else if constexpr (std::is_same<T1, BFloat16>::value) {
        test.AddInput<T1>("zero_points", {N, static_cast<int64_t>(q_scale_size) / N},
                          FloatsToBFloat16s(typed_zp), true);
      }
    } else {
      test.AddInput<uint8_t>("zero_points", {N, static_cast<int64_t>(q_zp_size_in_bytes) / N}, zp, true);
    }
  } else {
    test.AddOptionalInputEdge<uint8_t>();
  }

  if (opts.has_g_idx) {
    test.AddInput<int32_t>("g_idx", {K}, std::vector<int32_t>(static_cast<size_t>(K), 0), false);
  } else {
    test.AddOptionalInputEdge<int32_t>();
  }

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
}

enum class MatMul2BitsTestGroup { Basic,
                                  Block32,
                                  Block64,
                                  Block128 };

struct MatMul2BitsTestShape {
  int64_t M, N, K, block_size;
};

inline gsl::span<const MatMul2BitsTestShape> GetMatMul2BitsTestShapes(MatMul2BitsTestGroup group) {
  static constexpr std::array<MatMul2BitsTestShape, 20> kBasic{{
      {1, 1, 16, 16},
      {1, 2, 16, 16},
      {1, 32, 16, 16},
      {1, 32, 32, 16},
      {1, 32, 16, 128},
      {1, 288, 16, 16},
      {2, 1, 16, 16},
      {2, 2, 16, 16},
      {4, 1, 16, 16},
      {4, 2, 16, 16},
      {4, 32, 16, 16},
      {4, 32, 32, 16},
      {4, 32, 16, 128},
      {4, 288, 16, 16},
      {100, 1, 16, 16},
      {100, 2, 16, 16},
      {100, 32, 16, 16},
      {100, 32, 32, 16},
      {100, 32, 16, 128},
      {100, 288, 16, 16},
  }};
  static constexpr std::array<MatMul2BitsTestShape, 13> kBlock32{{
      {1, 16, 32, 32},
      {1, 32, 32, 32},
      {2, 16, 32, 32},
      {4, 32, 32, 32},
      {1, 32, 64, 32},
      {1, 32, 96, 32},
      {2, 16, 64, 32},
      {1, 32, 128, 32},
      {1, 288, 256, 32},
      {4, 32, 512, 32},
      {1, 288, 192, 32},
      {100, 32, 128, 32},
      {100, 288, 256, 32},
  }};
  static constexpr std::array<MatMul2BitsTestShape, 13> kBlock64{{
      {1, 16, 64, 64},
      {1, 32, 64, 64},
      {2, 16, 64, 64},
      {4, 32, 64, 64},
      {1, 32, 128, 64},
      {1, 32, 192, 64},
      {2, 16, 128, 64},
      {1, 32, 256, 64},
      {1, 288, 512, 64},
      {4, 32, 1024, 64},
      {1, 288, 384, 64},
      {100, 32, 256, 64},
      {100, 288, 1024, 64},
  }};
  static constexpr std::array<MatMul2BitsTestShape, 13> kBlock128{{
      {1, 16, 128, 128},
      {1, 32, 128, 128},
      {2, 16, 128, 128},
      {4, 32, 128, 128},
      {1, 32, 256, 128},
      {1, 32, 384, 128},
      {2, 16, 256, 128},
      {1, 32, 512, 128},
      {1, 288, 1024, 128},
      {4, 32, 2048, 128},
      {1, 288, 768, 128},
      {100, 32, 512, 128},
      {100, 288, 1024, 128},
  }};
  switch (group) {
    case MatMul2BitsTestGroup::Basic:
      return kBasic;
    case MatMul2BitsTestGroup::Block32:
      return kBlock32;
    case MatMul2BitsTestGroup::Block64:
      return kBlock64;
    case MatMul2BitsTestGroup::Block128:
      return kBlock128;
  }
  ORT_THROW("Unknown MatMul2Bits test group");
}

// Keep the original shape order and zero-point/bias combinations shared by callers.
template <typename RunCase>
void ForEachMatMul2BitsTestCase(MatMul2BitsTestGroup group, int64_t accuracy_level, RunCase&& run_case) {
  for (const auto& shape : GetMatMul2BitsTestShapes(group)) {
    MatMul2BitsTestParams params;
    params.M = shape.M;
    params.N = shape.N;
    params.K = shape.K;
    params.block_size = shape.block_size;
    params.accuracy_level = accuracy_level;
    params.output_abs_error = 0.1f;
    params.output_rel_error = 0.02f;
    for (bool has_bias : {false, true}) {
      for (bool has_zero_point : {false, true}) {
        params.has_bias = has_bias;
        params.has_zero_point = has_zero_point;
        run_case(params);
      }
    }
  }
}

}  // namespace onnxruntime::test
