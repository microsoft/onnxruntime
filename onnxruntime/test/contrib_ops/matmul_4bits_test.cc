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
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/ort_env.h"
#include "core/util/qmath.h"

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {

namespace test {

namespace {

constexpr int QBits = 4;

void QuantizeDequantize(std::vector<float>& raw_vals,
                        std::vector<uint8_t>& quant_vals,
                        std::vector<float>& scales,
                        std::vector<uint8_t>* zp,
                        int32_t N,
                        int32_t K,
                        int32_t block_size) {
  auto& ortenv = **ort_env.get();
  onnxruntime::concurrency::ThreadPool* tp = ortenv.GetEnvironment().GetIntraOpThreadPool();

  MlasQuantizeBlockwise<float, QBits>(
      quant_vals.data(),
      scales.data(),
      zp != nullptr ? zp->data() : nullptr,
      raw_vals.data(),
      block_size,
      true,
      K,
      N,
      N,
      tp);

  // Note that raw_vals is NxK after dequant
  MlasDequantizeBlockwise<float, QBits>(
      raw_vals.data(),                       // dequantized output
      quant_vals.data(),                     // quantized input
      scales.data(),                         // quantization scales
      zp != nullptr ? zp->data() : nullptr,  // quantization zero points
      block_size,                            // quantization block size
      true,                                  // columnwise quantization
      K,                                     // number of rows
      N,                                     // number of columns
      tp);
}

struct TestOptions {
  int64_t M{1};
  int64_t N{1};
  int64_t K{1};
  int64_t block_size{32};
  int64_t accuracy_level{0};

  bool has_zero_point{false};
  bool zp_is_4bit{true};
  bool has_g_idx{false};
  bool has_bias{false};

  std::optional<float> output_abs_error{};
  std::optional<float> output_rel_error{};
};

std::ostream& operator<<(std::ostream& os, const TestOptions& opts) {
  return os << "M:" << opts.M << ", N:" << opts.N << ", K:" << opts.K
            << ", block_size:" << opts.block_size
            << ", accuracy_level:" << opts.accuracy_level
            << ", has_zero_point:" << opts.has_zero_point
            << ", zp_is_4bit:" << opts.zp_is_4bit
            << ", has_g_idx:" << opts.has_g_idx
            << ", has_bias:" << opts.has_bias;
}

template <typename T1>
void RunTest(const TestOptions& opts,
             std::vector<std::unique_ptr<IExecutionProvider>>&& explicit_eps = {}) {
  SCOPED_TRACE(opts);

  static_assert(std::is_same_v<T1, float> || std::is_same_v<T1, MLFloat16>,
                "unexpected type for T1");

  constexpr bool use_float16 = std::is_same_v<T1, MLFloat16>;

  const bool zp_is_4bit = opts.zp_is_4bit || opts.has_g_idx;

  const int64_t M = opts.M,
                K = opts.K,
                N = opts.N;

  RandomValueGenerator random{1234};
  std::vector<float> input0_vals(random.Gaussian<float>(AsSpan({M, K}), 0.0f, 0.25f));
  std::vector<float> input1_f_vals(random.Gaussian<float>(AsSpan({K, N}), 0.0f, 0.25f));

#if 0  // for Debugging
  std::vector<float> input1_f_vals_trans(N * K);
  MlasTranspose(input1_f_vals.data(), input1_f_vals_trans.data(), K, N);
#endif

  int q_rows, q_cols;
  MlasBlockwiseQuantizedShape<float, QBits>(static_cast<int>(opts.block_size), /* columnwise */ true,
                                            static_cast<int>(K), static_cast<int>(N),
                                            q_rows, q_cols);

  size_t q_data_size_in_bytes, q_scale_size, q_zp_size_in_bytes;
  MlasBlockwiseQuantizedBufferSizes(QBits, static_cast<int>(opts.block_size), /* columnwise */ true,
                                    static_cast<int>(K), static_cast<int>(N),
                                    q_data_size_in_bytes, q_scale_size, &q_zp_size_in_bytes);

  std::vector<uint8_t> input1_vals(q_data_size_in_bytes);
  std::vector<float> scales(q_scale_size);
  std::vector<uint8_t> zp(q_zp_size_in_bytes);

  QuantizeDequantize(input1_f_vals,
                     input1_vals,
                     scales,
                     opts.has_zero_point ? &zp : nullptr,
                     static_cast<int32_t>(N),
                     static_cast<int32_t>(K),
                     static_cast<int32_t>(opts.block_size));

#if 0
  for (int i = 0; i < input1_vals.size(); i++)
  {
    uint8_t byte = input1_vals[i];
    uint8_t val_lo = byte & 0x0f;
    uint8_t val_hi = byte >> 4;
    std::cout << (int)val_lo << ", " << (int)val_hi << ", ";
  }
#endif

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
        sum += input0_vals[m * K + k] * input1_f_vals[n * K + k];
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

  if constexpr (use_float16) {
    test.AddInput<T1>("A", {M, K}, ToFloat16(input0_vals), false);
  } else {
    test.AddInput<T1>("A", {M, K}, input0_vals, false);
  }

  test.AddInput<uint8_t>("B", {q_cols, q_rows}, input1_vals, true);

  if constexpr (use_float16) {
    test.AddInput<T1>("scales", {static_cast<int64_t>(q_scale_size)}, ToFloat16(scales), true);
  } else {
    test.AddInput<T1>("scales", {static_cast<int64_t>(q_scale_size)}, scales, true);
  }

  if (opts.has_zero_point) {
    if (zp_is_4bit) {
      test.AddInput<uint8_t>("zero_points", {static_cast<int64_t>(q_zp_size_in_bytes)}, zp, true);
    } else {
      std::vector<float> zp_f;
      zp_f.reserve(q_zp_size_in_bytes * 2);
      for (size_t i = 0; i < zp.size(); i++) {
        zp_f.push_back(static_cast<float>(zp[i] & 0xf));
        zp_f.push_back(static_cast<float>((zp[i] >> 4) & 0xf));
      }
      size_t ind = zp_f.size() - 1;
      while (zp_f.size() != q_scale_size) {
        zp_f.erase(zp_f.begin() + ind);
        ind -= q_scale_size / N + 1;
      }

      if constexpr (use_float16) {
        test.AddInput<T1>("zero_points", {static_cast<int64_t>(q_scale_size)}, ToFloat16(zp_f), true);
      } else {
        test.AddInput<T1>("zero_points", {static_cast<int64_t>(q_scale_size)}, zp_f, true);
      }
    }
  } else {
    if (zp_is_4bit) {
      test.AddOptionalInputEdge<uint8_t>();
    } else {
      test.AddOptionalInputEdge<T1>();
    }
  }

  if (opts.has_g_idx) {
    auto ceildiv = [](int64_t a, int64_t b) { return (a + b - 1) / b; };
    int K_pad = narrow<int32_t>(ceildiv(K, opts.block_size) * opts.block_size);
    std::vector<int32_t> g_idx(K_pad);
    for (int64_t i = 0; i < K_pad; i++) {
      g_idx[i] = narrow<int32_t>(i / opts.block_size);
    }
    test.AddInput<int32_t>("g_idx", {static_cast<int64_t>(K_pad)}, g_idx, true);
  } else {
    test.AddOptionalInputEdge<int32_t>();
  }

  if (bias.has_value()) {
    if constexpr (use_float16) {
      test.AddInput<T1>("bias", bias_shape, ToFloat16(*bias), true);
    } else {
      test.AddInput<T1>("bias", bias_shape, *bias, true);
    }
  } else {
    test.AddOptionalInputEdge<T1>();
  }

  if constexpr (use_float16) {
    test.AddOutput<T1>("Y", {M, N}, ToFloat16(expected_vals));
  } else {
    test.AddOutput<T1>("Y", {M, N}, expected_vals);
  }

  if (opts.output_abs_error.has_value()) {
    test.SetOutputAbsErr("Y", *opts.output_abs_error);
  }

  if (opts.output_rel_error.has_value()) {
    test.SetOutputRelErr("Y", *opts.output_rel_error);
  }

  if (!explicit_eps.empty()) {
    test.ConfigEps(std::move(explicit_eps));
  }

  test.RunWithConfig();
}

}  // namespace

template <typename AType, int M, int N, int K, int block_size, int accuracy_level>
void TestMatMulNBitsTyped() {
  TestOptions base_opts{};
  base_opts.M = M, base_opts.N = N, base_opts.K = K;
  base_opts.block_size = block_size;
  base_opts.accuracy_level = accuracy_level;

  if (base_opts.accuracy_level == 4) {
    base_opts.output_abs_error = 0.1f;
    base_opts.output_rel_error = 0.02f;
  } else if constexpr (std::is_same<AType, MLFloat16>::value) {
    base_opts.output_abs_error = 0.055f;
    base_opts.output_rel_error = 0.02f;
  }

  {
    TestOptions opts = base_opts;
    RunTest<AType>(opts);
  }

  {
    TestOptions opts = base_opts;
    opts.has_zero_point = true;
    RunTest<AType>(opts);
  }

#if !defined(USE_DML) && !defined(USE_WEBGPU)
  {
    TestOptions opts = base_opts;
    opts.has_g_idx = true;
    RunTest<AType>(opts);
  }

  {
    TestOptions opts = base_opts;
    opts.has_g_idx = true;
    opts.has_bias = true;
    if constexpr (std::is_same<AType, float>::value) {
      if (opts.accuracy_level == 0 || opts.accuracy_level == 1) {
        // CI failure (not able to repro on either local machines):
        // M:100, N:288, K:1234, block_size:16, accuracy_level:0, has_zero_point:0, zp_is_4bit:1, has_g_idx:1, has_bias:1
        // The difference between cur_expected[i] and cur_actual[i] is 1.0401010513305664e-05, which exceeds tolerance,
        // tolerance evaluates to 1.006456386676291e-05.
        opts.output_abs_error = 0.0001f;
      }
    }
    // only enabled for CPU EP for now
    std::vector<std::unique_ptr<IExecutionProvider>> explicit_eps;
    explicit_eps.emplace_back(DefaultCpuExecutionProvider());
    RunTest<AType>(opts, std::move(explicit_eps));
  }

  {
    TestOptions opts = base_opts;
    opts.has_zero_point = true, opts.zp_is_4bit = false;
    RunTest<AType>(opts);
  }
#endif  // !defined(USE_DML) && !defined(USE_WEBGPU)
}

TEST(MatMulNBits, Float32_Accuracy0) {
  TestMatMulNBitsTyped<float, 1, 1, 16, 16, 0>();
  TestMatMulNBitsTyped<float, 1, 2, 16, 16, 0>();
  TestMatMulNBitsTyped<float, 1, 32, 16, 16, 0>();
  TestMatMulNBitsTyped<float, 1, 32, 32, 16, 0>();
  TestMatMulNBitsTyped<float, 1, 32, 16, 128, 0>();
  TestMatMulNBitsTyped<float, 1, 288, 16, 16, 0>();
  TestMatMulNBitsTyped<float, 1, 288, 1024, 16, 0>();
  TestMatMulNBitsTyped<float, 1, 288, 1024, 128, 0>();
  TestMatMulNBitsTyped<float, 1, 288, 93, 32, 0>();
  TestMatMulNBitsTyped<float, 1, 288, 93, 128, 0>();
  TestMatMulNBitsTyped<float, 1, 288, 1234, 16, 0>();
  TestMatMulNBitsTyped<float, 2, 1, 16, 16, 0>();
  TestMatMulNBitsTyped<float, 2, 2, 16, 16, 0>();
  TestMatMulNBitsTyped<float, 100, 1, 16, 16, 0>();
  TestMatMulNBitsTyped<float, 100, 2, 16, 16, 0>();
  TestMatMulNBitsTyped<float, 100, 32, 16, 16, 0>();
  TestMatMulNBitsTyped<float, 100, 32, 32, 16, 0>();
  TestMatMulNBitsTyped<float, 100, 32, 16, 128, 0>();
  TestMatMulNBitsTyped<float, 100, 288, 16, 16, 0>();
  TestMatMulNBitsTyped<float, 100, 288, 1024, 16, 0>();
  TestMatMulNBitsTyped<float, 100, 288, 1024, 128, 0>();
  TestMatMulNBitsTyped<float, 100, 288, 93, 32, 0>();
  TestMatMulNBitsTyped<float, 100, 288, 93, 128, 0>();
  TestMatMulNBitsTyped<float, 100, 288, 1234, 16, 0>();
}

TEST(MatMulNBits, Float32_Accuracy1) {
  TestMatMulNBitsTyped<float, 1, 1, 16, 16, 1>();
  TestMatMulNBitsTyped<float, 1, 288, 1024, 128, 1>();
  TestMatMulNBitsTyped<float, 1, 288, 93, 32, 1>();
  TestMatMulNBitsTyped<float, 1, 288, 1234, 16, 1>();
  TestMatMulNBitsTyped<float, 100, 32, 16, 128, 1>();
  TestMatMulNBitsTyped<float, 100, 288, 1024, 128, 1>();
  TestMatMulNBitsTyped<float, 100, 288, 93, 128, 1>();
  TestMatMulNBitsTyped<float, 100, 288, 1234, 16, 1>();
}

TEST(MatMulNBits, Float32_Accuracy4) {
  TestMatMulNBitsTyped<float, 1, 1, 16, 16, 4>();
  TestMatMulNBitsTyped<float, 1, 2, 16, 16, 4>();
  TestMatMulNBitsTyped<float, 1, 32, 16, 16, 4>();
  TestMatMulNBitsTyped<float, 1, 32, 32, 16, 4>();
  TestMatMulNBitsTyped<float, 1, 32, 16, 128, 4>();
  TestMatMulNBitsTyped<float, 1, 288, 16, 16, 4>();
  TestMatMulNBitsTyped<float, 1, 288, 1024, 16, 4>();
  TestMatMulNBitsTyped<float, 1, 288, 1024, 128, 4>();
  TestMatMulNBitsTyped<float, 1, 288, 93, 32, 4>();
  TestMatMulNBitsTyped<float, 1, 288, 93, 128, 4>();
  TestMatMulNBitsTyped<float, 1, 288, 1234, 16, 4>();
  TestMatMulNBitsTyped<float, 2, 1, 16, 16, 4>();
  TestMatMulNBitsTyped<float, 2, 2, 16, 16, 4>();
  TestMatMulNBitsTyped<float, 100, 1, 16, 16, 4>();
  TestMatMulNBitsTyped<float, 100, 2, 16, 16, 4>();
  TestMatMulNBitsTyped<float, 100, 32, 16, 16, 4>();
  TestMatMulNBitsTyped<float, 100, 32, 32, 16, 4>();
  TestMatMulNBitsTyped<float, 100, 32, 16, 128, 4>();
  TestMatMulNBitsTyped<float, 100, 288, 16, 16, 4>();
  TestMatMulNBitsTyped<float, 100, 288, 1024, 16, 4>();
  TestMatMulNBitsTyped<float, 100, 288, 1024, 128, 4>();
  TestMatMulNBitsTyped<float, 100, 288, 93, 32, 4>();
  TestMatMulNBitsTyped<float, 100, 288, 93, 128, 4>();
  TestMatMulNBitsTyped<float, 100, 288, 1234, 16, 4>();
}

#if defined(MLAS_TARGET_AMD64_IX86) || defined(MLAS_TARGET_ARM64)
#if !defined(USE_DML)
// Actual and expected difference is over 0.01 with DmlExecutionProvider.
// Skip the tests instead of raising the tolerance to make is pass.
TEST(MatMulNBits, Float16_Accuracy2) {
  TestMatMulNBitsTyped<MLFloat16, 1, 1, 16, 16, 2>();
  TestMatMulNBitsTyped<MLFloat16, 1, 2, 16, 16, 2>();
  TestMatMulNBitsTyped<MLFloat16, 1, 32, 16, 16, 2>();
  TestMatMulNBitsTyped<MLFloat16, 1, 32, 32, 16, 2>();
  TestMatMulNBitsTyped<MLFloat16, 1, 32, 16, 128, 2>();
  TestMatMulNBitsTyped<MLFloat16, 1, 288, 16, 16, 2>();
  TestMatMulNBitsTyped<MLFloat16, 1, 288, 1024, 16, 2>();
  TestMatMulNBitsTyped<MLFloat16, 1, 288, 1024, 128, 2>();
  TestMatMulNBitsTyped<MLFloat16, 1, 288, 93, 32, 2>();
  TestMatMulNBitsTyped<MLFloat16, 1, 288, 93, 128, 2>();
  TestMatMulNBitsTyped<MLFloat16, 1, 288, 1234, 16, 2>();
  TestMatMulNBitsTyped<MLFloat16, 2, 1, 16, 16, 2>();
  TestMatMulNBitsTyped<MLFloat16, 2, 2, 16, 16, 2>();
  TestMatMulNBitsTyped<MLFloat16, 100, 1, 16, 16, 2>();
  TestMatMulNBitsTyped<MLFloat16, 100, 2, 16, 16, 2>();
  TestMatMulNBitsTyped<MLFloat16, 100, 32, 16, 16, 2>();
  TestMatMulNBitsTyped<MLFloat16, 100, 32, 32, 16, 2>();
  TestMatMulNBitsTyped<MLFloat16, 100, 32, 16, 128, 2>();
  TestMatMulNBitsTyped<MLFloat16, 100, 288, 16, 16, 2>();
  TestMatMulNBitsTyped<MLFloat16, 100, 288, 1024, 16, 2>();
  TestMatMulNBitsTyped<MLFloat16, 100, 288, 1024, 128, 2>();
  TestMatMulNBitsTyped<MLFloat16, 100, 288, 93, 32, 2>();
  TestMatMulNBitsTyped<MLFloat16, 100, 288, 93, 128, 2>();
  TestMatMulNBitsTyped<MLFloat16, 100, 288, 1234, 16, 2>();
}

TEST(MatMulNBits, Float16_Accuracy0) {
  TestMatMulNBitsTyped<MLFloat16, 1, 1, 16, 16, 0>();
  TestMatMulNBitsTyped<MLFloat16, 1, 288, 93, 32, 0>();
  TestMatMulNBitsTyped<MLFloat16, 1, 288, 1234, 16, 0>();
  TestMatMulNBitsTyped<MLFloat16, 2, 1, 16, 16, 0>();
  TestMatMulNBitsTyped<MLFloat16, 100, 2, 16, 16, 0>();
  TestMatMulNBitsTyped<MLFloat16, 100, 288, 1024, 128, 0>();
  TestMatMulNBitsTyped<MLFloat16, 100, 288, 93, 32, 0>();
  TestMatMulNBitsTyped<MLFloat16, 100, 288, 1234, 16, 0>();
}

TEST(MatMulNBits, Float16_Accuracy4) {
  TestMatMulNBitsTyped<MLFloat16, 1, 1, 16, 16, 4>();
  TestMatMulNBitsTyped<MLFloat16, 1, 2, 16, 16, 4>();
  TestMatMulNBitsTyped<MLFloat16, 1, 32, 16, 16, 4>();
  TestMatMulNBitsTyped<MLFloat16, 1, 32, 32, 16, 4>();
  TestMatMulNBitsTyped<MLFloat16, 1, 32, 16, 128, 4>();
  TestMatMulNBitsTyped<MLFloat16, 1, 288, 16, 16, 4>();
  TestMatMulNBitsTyped<MLFloat16, 1, 288, 1024, 16, 4>();
  TestMatMulNBitsTyped<MLFloat16, 1, 288, 1024, 128, 4>();
  TestMatMulNBitsTyped<MLFloat16, 1, 288, 93, 32, 4>();
  TestMatMulNBitsTyped<MLFloat16, 1, 288, 93, 128, 4>();
  TestMatMulNBitsTyped<MLFloat16, 1, 288, 1234, 16, 4>();
  TestMatMulNBitsTyped<MLFloat16, 2, 1, 16, 16, 4>();
  TestMatMulNBitsTyped<MLFloat16, 2, 2, 16, 16, 4>();
  TestMatMulNBitsTyped<MLFloat16, 100, 1, 16, 16, 4>();
  TestMatMulNBitsTyped<MLFloat16, 100, 2, 16, 16, 4>();
  TestMatMulNBitsTyped<MLFloat16, 100, 32, 16, 16, 4>();
  TestMatMulNBitsTyped<MLFloat16, 100, 32, 32, 16, 4>();
  TestMatMulNBitsTyped<MLFloat16, 100, 32, 16, 128, 4>();
  TestMatMulNBitsTyped<MLFloat16, 100, 288, 16, 16, 4>();
  TestMatMulNBitsTyped<MLFloat16, 100, 288, 1024, 16, 4>();
  TestMatMulNBitsTyped<MLFloat16, 100, 288, 1024, 128, 4>();
  TestMatMulNBitsTyped<MLFloat16, 100, 288, 93, 32, 4>();
  TestMatMulNBitsTyped<MLFloat16, 100, 288, 93, 128, 4>();
  TestMatMulNBitsTyped<MLFloat16, 100, 288, 1234, 16, 4>();
}
#endif
#endif

#if defined(USE_CUDA) || defined(USE_ROCM) || defined(USE_DML) || defined(USE_WEBGPU)

namespace {
// Legacy test function.
// This has too many parameters of the same type that must be specified in the correct order.
// Consider using the overload with a TestOptions parameter.
void RunTest(int64_t M, int64_t N, int64_t K, int64_t block_size, int64_t accuracy_level,
             bool has_zeropoint, bool use_float16, bool has_g_idx = false,
             bool zp_is_4bit = true, float fp16_abs_error = 0.02f, bool has_bias = false) {
  TestOptions opts{};
  opts.M = M;
  opts.N = N;
  opts.K = K;
  opts.block_size = block_size;
  opts.accuracy_level = accuracy_level;
  opts.has_zero_point = has_zeropoint;
  opts.zp_is_4bit = zp_is_4bit;
  opts.has_g_idx = has_g_idx;
  opts.has_bias = has_bias;

  if (use_float16) {
    opts.output_abs_error = fp16_abs_error;
  }

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  if (use_float16) {
#ifdef USE_CUDA
    if (DefaultCudaExecutionProvider() != nullptr) {
      execution_providers.push_back(DefaultCudaExecutionProvider());
    }
#endif
#ifdef USE_ROCM
    execution_providers.push_back(DefaultRocmExecutionProvider());
#endif
#ifdef USE_DML
    if (DefaultDmlExecutionProvider() != nullptr) {
      execution_providers.push_back(DefaultDmlExecutionProvider());
    }
#endif
#ifdef USE_WEBGPU
    execution_providers.push_back(DefaultWebGpuExecutionProvider());
#endif

    RunTest<MLFloat16>(opts, std::move(execution_providers));
  } else {
#ifdef USE_ROCM
    execution_providers.push_back(DefaultRocmExecutionProvider());
#endif

    RunTest<float>(opts, std::move(execution_providers));
  }
}
}  // namespace

TEST(MatMulNBits, Float16Cuda) {
#if defined(USE_CUDA) || defined(USE_ROCM) || defined(USE_DML)
  std::vector<bool> has_gidx_options = {true, false};
  if (DefaultDmlExecutionProvider() != nullptr) {
    has_gidx_options.assign(1, false);
  }
#else
  auto has_gidx_options = {false};
#endif

  for (auto M : {1, 2, 100}) {
    for (auto N : {1, 2, 32, 288}) {
      for (auto K : {16, 32, 64, 128, 256, 1024, 93, 1234}) {
        for (auto block_size : {16, 32, 64, 128}) {
          for (auto has_gidx : has_gidx_options) {
#ifdef USE_DML
            if (DefaultDmlExecutionProvider() != nullptr) {
              RunTest(M, N, K, block_size, 0, false, true, has_gidx, true, 0.04f);
            }
#else
            RunTest(M, N, K, block_size, 0, false, true, has_gidx);
            RunTest(M, N, K, block_size, 0, true, true, has_gidx, false);
#endif
          }
        }
      }
    }
  }
}

TEST(MatMulNBits, Float16Large) {
#if defined(USE_CUDA) || defined(USE_DML)
  // For some reason, the A10 machine that runs these tests during CI has a much bigger error than all retail
  // machines we tested on. All consumer-grade machines from Nvidia/AMD/Intel seem to pass these tests with an
  // absolute error of 0.08, but the A10 has errors going as high as 0.22. Ultimately, given the large number
  // of elements in this test, ULPs should probably be used instead of absolute/relative tolerances.
  float abs_error = 0.05f;
  if (DefaultDmlExecutionProvider() != nullptr) {
    // it means the ep is dml in runtime, the abs_error is changed to 0.3f
    abs_error = 0.3f;
  }
#elif USE_WEBGPU
  // See Intel A770 to pass these tests with an absolute error of 0.08.
  float abs_error = 0.08f;
#else
  float abs_error = 0.05f;
#endif

  for (auto block_size : {16, 32, 64, 128}) {
    for (auto symmetric : {false, true}) {
      RunTest(1, 4096, 4096, block_size, 0, symmetric, true, false, true, abs_error);
      RunTest(1, 4096, 11008, block_size, 0, symmetric, true, false, true, abs_error);
      RunTest(1, 11008, 4096, block_size, 0, symmetric, true, false, true, abs_error);
    }
  }
}
#endif  // defined(USE_CUDA) || defined(USE_ROCM) || defined(USE_DML)
}  // namespace test
}  // namespace onnxruntime

#endif  // ORT_MINIMAL_BUILD
