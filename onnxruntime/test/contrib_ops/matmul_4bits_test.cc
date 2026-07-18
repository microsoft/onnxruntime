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
#include "test/unittest_util/framework_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/unittest_util/graph_transform_test_builder.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/scoped_env_vars.h"
#include "test/contrib_ops/matmul_nbits_prepack_sharing_test_util.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/ort_env.h"
#include "core/util/qmath.h"
#include "core/providers/webgpu/webgpu_provider_options.h"
#include "core/framework/prepacked_weights_container.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/util/include/test/test_environment.h"

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
  int64_t batch_count{1};
  int64_t M{1};
  int64_t N{1};
  int64_t K{1};
  int64_t block_size{32};
  int64_t accuracy_level{0};

  bool disable_cpu_ep_fallback{false};

  bool has_zero_point{false};
  bool zp_is_4bit{true};
  bool has_g_idx{false};
  bool has_bias{false};

  bool legacy_shape{false};  // for backward compatibility

  // When set, RunTest validates cross-session sharing of the pre-packed weights instead of doing a
  // single run. The model is run in two sessions that use the same pre-packed weights container.
  std::optional<PrepackSharingMode> prepack_sharing_mode{};

  std::optional<int64_t> weight_prepacked{};
  std::optional<std::string> expected_failure{};

  std::optional<float> output_abs_error{};
  std::optional<float> output_rel_error{};
};

[[maybe_unused]] std::ostream& operator<<(std::ostream& os, const TestOptions& opts) {
  return os << "batch_count:" << opts.batch_count
            << ", M:" << opts.M << ", N:" << opts.N << ", K:" << opts.K
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

  static_assert(std::is_same_v<T1, float> || std::is_same_v<T1, MLFloat16> || std::is_same_v<T1, BFloat16>,
                "unexpected type for T1");

#ifdef USE_CUDA
  if (opts.accuracy_level != 0 && !opts.legacy_shape) {
    return;  // CUDA EP does not handle accuracy level, so only test one level to avoid unnecessary tests.
  }
#endif

  const bool zp_is_4bit = opts.zp_is_4bit || opts.has_g_idx;

  const int64_t batch_count = opts.batch_count;
  const int64_t M = opts.M;
  const int64_t K = opts.K;
  const int64_t N = opts.N;

  RandomValueGenerator random{1234};
  std::vector<float> input0_vals(random.Gaussian<float>(AsSpan({batch_count, M, K}), 0.0f, 0.25f));
  std::vector<float> input1_f_vals(random.Gaussian<float>(AsSpan({K, N}), 0.0f, 0.25f));

  int64_t k_blocks = (K + opts.block_size - 1) / opts.block_size;
  int64_t blob_size = (opts.block_size * QBits + 7) / 8;
  size_t q_scale_size = static_cast<size_t>(N * k_blocks);
  size_t q_data_size_in_bytes = static_cast<size_t>(N * k_blocks * blob_size);  // packed as UInt4x2
  const int64_t zero_point_blob_size = (k_blocks * QBits + 7) / 8;
  size_t q_zp_size_in_bytes = static_cast<size_t>(N * zero_point_blob_size);  // packed as UInt4x2

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

  const std::vector<int64_t> bias_shape = {N};
  const auto bias = [&]() -> std::optional<std::vector<float>> {
    if (opts.has_bias) {
      return random.Uniform(bias_shape, 1.0f, 5.0f);
    }
    return std::nullopt;
  }();

  std::vector<float> expected_vals(batch_count * M * N);
  for (int64_t b = 0; b < batch_count; b++) {
    for (int64_t m = 0; m < M; m++) {
      for (int64_t n = 0; n < N; n++) {
        float sum = 0.0f;
        for (int64_t k = 0; k < K; k++) {
          sum += input0_vals[b * M * K + m * K + k] * input1_f_vals[n * K + k];
        }
        expected_vals[b * M * N + m * N + n] = sum + (bias.has_value() ? (*bias)[n] : 0.0f);
      }
    }
  }

  OpTester test("MatMulNBits", 1, kMSDomain);
  test.AddAttribute<int64_t>("K", K);
  test.AddAttribute<int64_t>("N", N);
  test.AddAttribute<int64_t>("block_size", opts.block_size);
  test.AddAttribute<int64_t>("bits", QBits);
  test.AddAttribute<int64_t>("accuracy_level", opts.accuracy_level);
  if (opts.weight_prepacked.has_value()) {
    test.AddAttribute<int64_t>("weight_prepacked", *opts.weight_prepacked);
  }

  if constexpr (std::is_same_v<T1, float>) {
    test.AddInput<T1>("A", {batch_count, M, K}, input0_vals, false);
  } else if constexpr (std::is_same<T1, MLFloat16>::value) {
    test.AddInput<T1>("A", {batch_count, M, K}, FloatsToMLFloat16s(input0_vals), false);
  } else if constexpr (std::is_same<T1, BFloat16>::value) {
    test.AddInput<T1>("A", {batch_count, M, K}, FloatsToBFloat16s(input0_vals), false);
  }

  test.AddInput<uint8_t>("B", {N, k_blocks, blob_size}, input1_vals, true);

  auto scales_shape = opts.legacy_shape ? std::vector<int64_t>{N * k_blocks}
                                        : std::vector<int64_t>{N, k_blocks};

  if constexpr (std::is_same<T1, float>::value) {
    test.AddInput<T1>("scales", scales_shape, scales, true);
  } else if constexpr (std::is_same<T1, MLFloat16>::value) {
    test.AddInput<T1>("scales", scales_shape, FloatsToMLFloat16s(scales), true);
  } else if constexpr (std::is_same<T1, BFloat16>::value) {
    test.AddInput<T1>("scales", scales_shape, FloatsToBFloat16s(scales), true);
  }

  if (opts.has_zero_point) {
    if (zp_is_4bit) {
      auto zp_shape = opts.legacy_shape ? std::vector<int64_t>{N * zero_point_blob_size}
                                        : std::vector<int64_t>{N, zero_point_blob_size};
      test.AddInput<uint8_t>("zero_points", zp_shape, zp, true);
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

      if constexpr (std::is_same_v<T1, float>) {
        test.AddInput<T1>("zero_points", scales_shape, zp_f, true);
      } else if constexpr (std::is_same_v<T1, MLFloat16>) {
        test.AddInput<T1>("zero_points", scales_shape, FloatsToMLFloat16s(zp_f), true);
      } else if constexpr (std::is_same_v<T1, BFloat16>) {
        test.AddInput<T1>("zero_points", scales_shape, FloatsToBFloat16s(zp_f), true);
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
    test.AddOutput<T1>("Y", {batch_count, M, N}, expected_vals);
  } else if constexpr (std::is_same<T1, MLFloat16>::value) {
    test.AddOutput<T1>("Y", {batch_count, M, N}, FloatsToMLFloat16s(expected_vals));
  } else if constexpr (std::is_same<T1, BFloat16>::value) {
    test.AddOutput<T1>("Y", {batch_count, M, N}, FloatsToBFloat16s(expected_vals));
  }

  if (opts.output_abs_error.has_value()) {
    test.SetOutputAbsErr("Y", *opts.output_abs_error);
  }

  if (opts.output_rel_error.has_value()) {
    test.SetOutputRelErr("Y", *opts.output_rel_error);
  }

  if (opts.prepack_sharing_mode.has_value()) {
    // Pre-packed weight sharing is a CPU-EP-only feature; the helper runs the model on the CPU EP
    // in two sessions and validates the sharing counters.
    CheckSharedPrepackedWeights(test, *opts.prepack_sharing_mode, {N, k_blocks, blob_size}, input1_vals);
    return;
  }

  if (!explicit_eps.empty()) {
    test.ConfigEps(std::move(explicit_eps));
  }

  if (opts.disable_cpu_ep_fallback) {
    SessionOptions session_options;
    session_options.use_per_session_threads = false;
    ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
    test.Config(session_options);
  }

  if (opts.expected_failure.has_value()) {
    test.Config(OpTester::ExpectResult::kExpectFailure, *opts.expected_failure);
    test.RunWithConfig();
    return;
  }

  test.RunWithConfig();
}

}  // namespace

template <typename AType, int M, int N, int K, int block_size, int accuracy_level, bool legacy_shape = false>
void TestMatMulNBitsTyped(std::optional<float> abs_error = std::nullopt,
                          std::optional<float> rel_error = std::nullopt) {
  TestOptions base_opts{};
  base_opts.M = M, base_opts.N = N, base_opts.K = K;
  base_opts.block_size = block_size;
  base_opts.accuracy_level = accuracy_level;
  base_opts.legacy_shape = legacy_shape;

  if (abs_error.has_value()) {
    base_opts.output_abs_error = *abs_error;
  } else if (base_opts.accuracy_level == 4) {
    base_opts.output_abs_error = 0.1f;
  } else if constexpr (std::is_same<AType, MLFloat16>::value) {
    // The fp16 provider paths compare against a float reference while native kernels may accumulate
    // in fp16 (for example native HGEMM on SME; see PR #28786), so allow slightly wider drift.
#if defined(USE_WEBGPU)
    // WebGPU's fp16 path has additional provider-specific rounding drift for these quantized matmul cases.
    base_opts.output_abs_error = 0.1f;
#else
    base_opts.output_abs_error = 0.065f;
#endif
  } else {
    base_opts.output_abs_error = 0.05f;
  }

  if (rel_error.has_value()) {
    base_opts.output_rel_error = *rel_error;
  } else if (base_opts.accuracy_level == 4) {
    base_opts.output_rel_error = 0.02f;
  } else if constexpr (std::is_same<AType, MLFloat16>::value) {
    base_opts.output_rel_error = 0.02f;
  } else {
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
    opts.has_zero_point = true;
    opts.zp_is_4bit = false;
    RunTest<AType>(opts);
  }
#endif  // !defined(USE_DML) && !defined(USE_WEBGPU)
#if defined(USE_WEBGPU)
  {
    // WebGPU does support bias but no g_idx
    TestOptions opts = base_opts;
    opts.has_bias = true;
    RunTest<AType>(opts);
  }
#endif
}

TEST(MatMulNBits, Float32_4b_Accuracy0) {
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

TEST(MatMulNBits, Float32_4b_Accuracy1) {
  TestMatMulNBitsTyped<float, 1, 1, 16, 16, 1>();
  TestMatMulNBitsTyped<float, 1, 288, 1024, 128, 1>();
  TestMatMulNBitsTyped<float, 1, 288, 93, 32, 1>();
  TestMatMulNBitsTyped<float, 1, 288, 1234, 16, 1>();
  TestMatMulNBitsTyped<float, 100, 32, 16, 128, 1>();
  TestMatMulNBitsTyped<float, 100, 288, 1024, 128, 1>();
  TestMatMulNBitsTyped<float, 100, 288, 93, 128, 1>();
  TestMatMulNBitsTyped<float, 100, 288, 1234, 16, 1>();
}

TEST(MatMulNBits, Float32_4b_Accuracy4) {
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
  TestMatMulNBitsTyped<float, 100, 288, 192, 64, 4>();
  TestMatMulNBitsTyped<float, 100, 288, 93, 32, 4>();
  TestMatMulNBitsTyped<float, 100, 288, 93, 128, 4>();
  TestMatMulNBitsTyped<float, 100, 288, 1234, 16, 4>();
}

#if defined(MLAS_TARGET_AMD64_IX86) || defined(MLAS_TARGET_ARM64)
#if !defined(USE_DML)
// Actual and expected difference is over 0.01 with DmlExecutionProvider.
// Skip the tests instead of raising the tolerance to make is pass.
TEST(MatMulNBits, Float16_4b_Accuracy2) {
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

TEST(MatMulNBits, Float16_4b_Accuracy0) {
  TestMatMulNBitsTyped<MLFloat16, 1, 1, 16, 16, 0>();
  TestMatMulNBitsTyped<MLFloat16, 1, 288, 93, 32, 0>();
  TestMatMulNBitsTyped<MLFloat16, 1, 288, 1234, 16, 0>();
  TestMatMulNBitsTyped<MLFloat16, 2, 1, 16, 16, 0>();
  TestMatMulNBitsTyped<MLFloat16, 100, 2, 16, 16, 0>();
  TestMatMulNBitsTyped<MLFloat16, 100, 288, 1024, 128, 0>();
  TestMatMulNBitsTyped<MLFloat16, 100, 288, 93, 32, 0>();
  TestMatMulNBitsTyped<MLFloat16, 100, 288, 1234, 16, 0>();
  TestMatMulNBitsTyped<MLFloat16, 100, 256, 128, 32, 0>();
  TestMatMulNBitsTyped<MLFloat16, 100, 192, 128, 32, 0>();
}

TEST(MatMulNBits, Float16_4b_Accuracy4) {
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
  TestMatMulNBitsTyped<MLFloat16, 100, 64, 32, 32, 4>();
  TestMatMulNBitsTyped<MLFloat16, 100, 128, 128, 32, 4>();
  TestMatMulNBitsTyped<MLFloat16, 100, 288, 16, 16, 4>();
  TestMatMulNBitsTyped<MLFloat16, 100, 288, 1024, 16, 4>();
  TestMatMulNBitsTyped<MLFloat16, 100, 288, 1024, 128, 4>();
  TestMatMulNBitsTyped<MLFloat16, 100, 288, 192, 64, 4>();
  TestMatMulNBitsTyped<MLFloat16, 100, 288, 93, 32, 4>();
  TestMatMulNBitsTyped<MLFloat16, 100, 288, 93, 128, 4>();
  TestMatMulNBitsTyped<MLFloat16, 100, 288, 1234, 16, 4>();
  TestMatMulNBitsTyped<MLFloat16, 100, 256, 128, 32, 4>();
  TestMatMulNBitsTyped<MLFloat16, 100, 192, 128, 32, 4>();

  // See PR #27412 for details on the following test case,
  // which is added to cover a specific failure case in the past.
  // 6144, 2048

  // Since K is larger (more chance of larger error),
  // and N is larger (more chance of having a value with larger error),
  // we set a higher tolerance for this case to avoid false positives
  // and flaky failures.
  TestMatMulNBitsTyped<MLFloat16, 369, 6144, 2048, 32, 4>(0.2f, 0.03f);
}

TEST(MatMulNBits, LegacyShape_4b) {
  constexpr bool legacy_shape = true;
  TestMatMulNBitsTyped<float, 4, 2, 16, 16, 4, legacy_shape>();
  TestMatMulNBitsTyped<MLFloat16, 1, 2, 16, 16, 4, legacy_shape>();
}

// Batch tests for DP4A path (accuracy_level 4)
TEST(MatMulNBits, Float32_4b_Accuracy4_Batch) {
  // Test batch support with DP4A requirements:
  // - accuracy_level == 4
  // - block_size % 32 == 0
  // - K % 128 == 0
  // - N % 16 == 0

  // Small batch tests
  TestOptions opts{};
  opts.accuracy_level = 4;
  opts.output_abs_error = 0.1f;
  opts.output_rel_error = 0.02f;

  // Batch=2 tests
  opts.batch_count = 2;
  opts.M = 1;
  opts.N = 16;
  opts.K = 128;
  opts.block_size = 32;
  RunTest<float>(opts);

  opts.M = 2;
  opts.N = 32;
  opts.K = 128;
  opts.block_size = 32;
  RunTest<float>(opts);

  opts.M = 32;
  opts.N = 64;
  opts.K = 256;
  opts.block_size = 64;
  RunTest<float>(opts);

  opts.M = 100;
  opts.N = 288;
  opts.K = 1024;
  opts.block_size = 128;
  RunTest<float>(opts);

  // Batch=4 tests
  opts.batch_count = 4;
  opts.M = 1;
  opts.N = 16;
  opts.K = 128;
  opts.block_size = 32;
  RunTest<float>(opts);

  opts.M = 32;
  opts.N = 64;
  opts.K = 256;
  opts.block_size = 64;
  RunTest<float>(opts);

  opts.M = 100;
  opts.N = 288;
  opts.K = 1024;
  opts.block_size = 128;
  RunTest<float>(opts);

  // Batch=8 test
  opts.batch_count = 8;
  opts.M = 32;
  opts.N = 128;
  opts.K = 256;
  opts.block_size = 64;
  RunTest<float>(opts);

  // Test with bias
  opts.batch_count = 2;
  opts.M = 32;
  opts.N = 64;
  opts.K = 256;
  opts.block_size = 64;
  opts.has_bias = true;
  RunTest<float>(opts);
}

#ifndef ENABLE_TRAINING
// Pre-packing (and therefore cross-session sharing of pre-packed weights) is disabled in a full
// training build, so there is nothing to exercise there.

namespace {
// Builds a representative MatMulNBits TestOptions for the pre-packed weight sharing tests.
TestOptions MakeSharingTestOptions(int64_t N, int64_t K, int64_t block_size, int64_t accuracy_level,
                                   bool has_zero_point, bool has_bias, PrepackSharingMode mode) {
  TestOptions opts{};
  opts.M = 8;
  opts.N = N;
  opts.K = K;
  opts.block_size = block_size;
  opts.accuracy_level = accuracy_level;
  opts.has_zero_point = has_zero_point;
  opts.zp_is_4bit = true;
  opts.has_bias = has_bias;
  opts.prepack_sharing_mode = mode;
  opts.output_abs_error = 0.1f;
  opts.output_rel_error = 0.02f;
  return opts;
}
}  // namespace

// Legacy sharing path: the weight B is registered as a shared initializer via
// SessionOptions::AddInitializer. Covers float and float16 activations, symmetric/asymmetric, +/- bias.
TEST(MatMulNBits, SharedPrepackedWeights_AddInitializer) {
  for (bool has_zero_point : {false, true}) {
    for (bool has_bias : {false, true}) {
      RunTest<float>(MakeSharingTestOptions(32, 256, /*block_size*/ 32, /*accuracy_level*/ 0, has_zero_point,
                                            has_bias, PrepackSharingMode::kAddInitializer));
      RunTest<MLFloat16>(MakeSharingTestOptions(32, 256, /*block_size*/ 32, /*accuracy_level*/ 0, has_zero_point,
                                                has_bias, PrepackSharingMode::kAddInitializer));
    }
  }
}

// Negative control: with the shared container present but neither opt-in mechanism enabled, no
// pre-packed weights are shared across sessions.
TEST(MatMulNBits, SharedPrepackedWeights_NotSharedWithoutOptIn) {
  RunTest<float>(MakeSharingTestOptions(32, 256, /*block_size*/ 32, /*accuracy_level*/ 0, /*has_zero_point*/ true,
                                        /*has_bias*/ true, PrepackSharingMode::kNoSharing));
  RunTest<MLFloat16>(MakeSharingTestOptions(32, 256, /*block_size*/ 32, /*accuracy_level*/ 0,
                                            /*has_zero_point*/ false, /*has_bias*/ false,
                                            PrepackSharingMode::kNoSharing));
}

#endif  // !ENABLE_TRAINING

#endif
#endif

#if defined(USE_CUDA) || defined(USE_DML) || defined(USE_WEBGPU)

namespace {
// Legacy test function.
// This has too many parameters of the same type that must be specified in the correct order.
// Consider using the overload with a TestOptions parameter.
template <typename T>
void RunTest(int64_t M, int64_t N, int64_t K, int64_t block_size, bool has_zeropoint, bool zp_is_4bit = true,
             float abs_error = 0.f, bool has_g_idx = false, bool has_bias = false) {
  TestOptions opts{};
  opts.M = M;
  opts.N = N;
  opts.K = K;
  opts.block_size = block_size;
  opts.accuracy_level = 0;
  opts.has_zero_point = has_zeropoint;
  opts.zp_is_4bit = zp_is_4bit;
  opts.has_g_idx = has_g_idx;
  opts.has_bias = has_bias;

  if (abs_error > 0.f) {
    opts.output_abs_error = abs_error;
  }

  if (std::is_same_v<T, MLFloat16>) {
    opts.output_rel_error = 0.001f;
  } else if (std::is_same_v<T, float>) {
    opts.output_rel_error = 0.0005f;
  }

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  if (std::is_same_v<T, MLFloat16>) {
#ifdef USE_CUDA
    execution_providers.push_back(DefaultCudaExecutionProvider());
    RunTest<MLFloat16>(opts, std::move(execution_providers));
#endif
#ifdef USE_DML
    execution_providers.push_back(DefaultDmlExecutionProvider());
    RunTest<MLFloat16>(opts, std::move(execution_providers));
#endif
#ifdef USE_WEBGPU
    execution_providers.push_back(DefaultWebGpuExecutionProvider());
    RunTest<MLFloat16>(opts, std::move(execution_providers));
#endif
  } else {
#ifdef USE_WEBGPU
    ConfigOptions config_options{};
    ORT_ENFORCE(config_options.AddConfigEntry(webgpu::options::kMaxStorageBufferBindingSize, "134217728").IsOK());
    execution_providers.push_back(WebGpuExecutionProviderWithOptions(config_options));
#endif
    RunTest<float>(opts, std::move(execution_providers));
  }
}

constexpr bool kPipelineMode = true;  // CI pipeline?
}  // namespace

TEST(MatMulNBits, Float16_Comprehensive) {
  if constexpr (kPipelineMode) {
    GTEST_SKIP() << "Skipping in pipeline mode";  // This test has too many combinations. Skip it in CI pipeline.
  } else {
    constexpr float abs_error = 0.02f;

    for (auto M : {1, 2, 100}) {
      for (auto N : {1, 2, 32, 288}) {
        for (auto K : {16, 32, 64, 128, 256, 1024, 93, 1234}) {
          for (auto block_size : {16, 32, 64, 128}) {
            for (auto has_g_idx : {false, true}) {
              for (auto has_zero_point : {false, true}) {
                for (auto is_zero_point_4bit : {false, true}) {
                  RunTest<MLFloat16>(M, N, K, block_size, has_zero_point, is_zero_point_4bit, abs_error, has_g_idx);
                }
              }
            }
          }
        }
      }
    }
  }
}

TEST(MatMulNBits, Float16_Large) {
#ifdef USE_DML
  // For some reason, the A10 machine that runs these tests during CI has a much bigger error than all retail
  // machines we tested on. All consumer-grade machines from Nvidia/AMD/Intel seem to pass these tests with an
  // absolute error of 0.08, but the A10 has errors going as high as 0.22. Ultimately, given the large number
  // of elements in this test, ULPs should probably be used instead of absolute/relative tolerances.
  constexpr float abs_error = 0.3f;
#else
  constexpr float abs_error = 0.1f;
#endif

  constexpr bool zp_is_4bit = true;

  for (auto block_size : {16, 32, 64, 128}) {
    for (auto has_zeropoint : {false, true}) {
      RunTest<MLFloat16>(1, 4096, 4096, block_size, has_zeropoint, zp_is_4bit, abs_error);
      RunTest<MLFloat16>(1, 4096, 11008, block_size, has_zeropoint, zp_is_4bit, abs_error);
      RunTest<MLFloat16>(1, 11008, 4096, block_size, has_zeropoint, zp_is_4bit, abs_error);
    }
  }
}

#ifdef USE_WEBGPU
// Similar to Float16_Large but for float32 and crafted so that the input_b and output buffer size exceeds
// maxStorageBufferBindingSize (128MB) so it must be split into 2 segments internally (~128.00006MB).
//
// input_b size(4-bits): N * K / 2 = 8388612 * 32 / 2 = 134217792 bytes > 134217728 bytes (128MB)
// output size(float32): M * N * 4 = 4 * 8388612 * 4 = 134217792 bytes > 134217728 bytes (128MB)
TEST(MatMulNBits, Float32_Large) {
  // Keep tolerance similar to Float16_Large (float path typically equal or better numerically).
  constexpr float abs_error = 0.1f;
  constexpr bool zp_is_4bit = true;
  constexpr bool has_zeropoint = false;
  constexpr auto block_size = 16;

  RunTest<float>(4 /*M*/, 8388612 /*N*/, 32 /*K*/, block_size, has_zeropoint, zp_is_4bit, abs_error);
}
#endif

#ifdef USE_CUDA
TEST(MatMulNBits, Fp16_Int4_Int4ZeroPoint) {
  constexpr float abs_error = 0.1f;
  constexpr bool zp_is_4bit = true;
  constexpr bool has_zeropoint = true;

  for (auto block_size : {64, 128}) {
    RunTest<MLFloat16>(1, 256, 1024, block_size, has_zeropoint, zp_is_4bit, abs_error);
    RunTest<MLFloat16>(32, 1024, 2048, block_size, has_zeropoint, zp_is_4bit, abs_error);
  }

  ScopedEnvironmentVariables scoped_env_vars{EnvVarMap{{"ORT_FPA_INTB_GEMM", "1"}}};

  for (auto block_size : {64, 128}) {
    RunTest<MLFloat16>(1, 256, 1024, block_size, has_zeropoint, zp_is_4bit, abs_error);
    RunTest<MLFloat16>(32, 1024, 2048, block_size, has_zeropoint, zp_is_4bit, abs_error);
  }
}

TEST(MatMulNBits, Fp16_Int4_Fp16ZeroPoint) {
  constexpr float abs_error = 0.1f;
  constexpr bool zp_is_4bit = false;
  constexpr bool has_zeropoint = true;

  ScopedEnvironmentVariables scoped_env_vars{EnvVarMap{{"ORT_FPA_INTB_GEMM", "1"}}};

  for (auto block_size : {64, 128}) {
    RunTest<MLFloat16>(1, 256, 1024, block_size, has_zeropoint, zp_is_4bit, abs_error);
    RunTest<MLFloat16>(32, 1024, 2048, block_size, has_zeropoint, zp_is_4bit, abs_error);
  }
}

TEST(MatMulNBits, BFloat16_Int4_Int4ZeroPoint) {
  constexpr float abs_error = 0.1f;
  constexpr bool zp_is_4bit = true;
  constexpr bool has_zeropoint = true;

  for (auto block_size : {64, 128}) {
    RunTest<BFloat16>(1, 256, 1024, block_size, has_zeropoint, zp_is_4bit, abs_error);
    RunTest<BFloat16>(32, 1024, 2048, block_size, has_zeropoint, zp_is_4bit, abs_error);
  }

  ScopedEnvironmentVariables scoped_env_vars{EnvVarMap{{"ORT_FPA_INTB_GEMM", "1"}}};

  for (auto block_size : {64, 128}) {
    RunTest<BFloat16>(1, 256, 1024, block_size, has_zeropoint, zp_is_4bit, abs_error);
    RunTest<BFloat16>(32, 1024, 2048, block_size, has_zeropoint, zp_is_4bit, abs_error);
  }
}

TEST(MatMulNBits, BFloat16_Int4_BFloat16ZeroPoint) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "Skipping BFloat16 8-bit MatMul tests on CUDA < 8.0";
  }

  constexpr float abs_error = 0.1f;
  constexpr bool zp_is_4bit = false;
  constexpr bool has_zeropoint = true;

  for (auto block_size : {64, 128}) {
    RunTest<BFloat16>(1, 256, 1024, block_size, has_zeropoint, zp_is_4bit, abs_error);
    RunTest<BFloat16>(32, 1024, 2048, block_size, has_zeropoint, zp_is_4bit, abs_error);
  }

  ScopedEnvironmentVariables scoped_env_vars{EnvVarMap{{"ORT_FPA_INTB_GEMM", "1"}}};

  for (auto block_size : {64, 128}) {
    RunTest<BFloat16>(1, 256, 1024, block_size, has_zeropoint, zp_is_4bit, abs_error);
    RunTest<BFloat16>(32, 1024, 2048, block_size, has_zeropoint, zp_is_4bit, abs_error);
  }
}

TEST(MatMulNBits, Fp16_Int4_NoZeroPoint) {
  constexpr float abs_error = 0.1f;
  constexpr bool zp_is_4bit = true;
  constexpr bool has_zeropoint = false;

  ScopedEnvironmentVariables scoped_env_vars{EnvVarMap{{"ORT_FPA_INTB_GEMM", "1"}}};

  for (auto block_size : {64, 128}) {
    RunTest<MLFloat16>(1, 256, 1024, block_size, has_zeropoint, zp_is_4bit, abs_error);
    RunTest<MLFloat16>(32, 1024, 2048, block_size, has_zeropoint, zp_is_4bit, abs_error);
  }
}

// block_size=32 with the fpA_intB path. Production rc2/rc3 int4 models are quantized with
// block_size=32. The fpA_intB kernels support group_size=32: the GEMV select_gs dispatches
// GroupSize==32, and the SM80/Ampere fine-grained CUTLASS GEMM uses kMinFinegrainedGroupSize=32
// (two scale rows per 64-element K tile). Exercises M=1 (GEMV) and M=32 (CUTLASS), with and
// without zero-points, for fp16 and bf16.
TEST(MatMulNBits, Fp16_Int4_BlockSize32_FpAIntB) {
  constexpr float abs_error = 0.1f;
  constexpr bool zp_is_4bit = true;

  ScopedEnvironmentVariables scoped_env_vars{EnvVarMap{{"ORT_FPA_INTB_GEMM", "1"}}};

  for (auto has_zeropoint : {false, true}) {
    RunTest<MLFloat16>(1, 256, 1024, 32, has_zeropoint, zp_is_4bit, abs_error);
    RunTest<MLFloat16>(32, 1024, 2048, 32, has_zeropoint, zp_is_4bit, abs_error);
  }
}

TEST(MatMulNBits, BFloat16_Int4_BlockSize32_FpAIntB) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "Skipping BFloat16 MatMul tests on CUDA < 8.0";
  }

  constexpr float abs_error = 0.5f;
  constexpr bool zp_is_4bit = true;

  ScopedEnvironmentVariables scoped_env_vars{EnvVarMap{{"ORT_FPA_INTB_GEMM", "1"}}};

  for (auto has_zeropoint : {false, true}) {
    RunTest<BFloat16>(1, 256, 1024, 32, has_zeropoint, zp_is_4bit, abs_error);
    RunTest<BFloat16>(32, 1024, 2048, 32, has_zeropoint, zp_is_4bit, abs_error);
  }
}

// Fused bias with the fpA_intB path. Exercises both the GEMV path (M=1) and the CUTLASS GEMM path
// (M=32), for fp16 and bf16, with block_size 64/128. This is the gpt-oss qkv_proj/o_proj scenario
// where MatMulNBitsFusion folds the Add(bias) into MatMulNBits input[5].
TEST(MatMulNBits, Fp16_Int4_NoZeroPoint_Bias) {
  constexpr float abs_error = 0.1f;
  constexpr bool zp_is_4bit = true;
  constexpr bool has_zeropoint = false;
  constexpr bool has_g_idx = false;
  constexpr bool has_bias = true;

  ScopedEnvironmentVariables scoped_env_vars{EnvVarMap{{"ORT_FPA_INTB_GEMM", "1"}}};

  for (auto block_size : {64, 128}) {
    RunTest<MLFloat16>(1, 256, 1024, block_size, has_zeropoint, zp_is_4bit, abs_error, has_g_idx, has_bias);
    RunTest<MLFloat16>(32, 1024, 2048, block_size, has_zeropoint, zp_is_4bit, abs_error, has_g_idx, has_bias);
  }
}

TEST(MatMulNBits, BFloat16_Int4_NoZeroPoint_Bias) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "Skipping BFloat16 MatMul tests on CUDA < 8.0";
  }

  constexpr float abs_error = 0.5f;
  constexpr bool zp_is_4bit = true;
  constexpr bool has_zeropoint = false;
  constexpr bool has_g_idx = false;
  constexpr bool has_bias = true;

  ScopedEnvironmentVariables scoped_env_vars{EnvVarMap{{"ORT_FPA_INTB_GEMM", "1"}}};

  for (auto block_size : {64, 128}) {
    RunTest<BFloat16>(1, 256, 1024, block_size, has_zeropoint, zp_is_4bit, abs_error, has_g_idx, has_bias);
    RunTest<BFloat16>(32, 1024, 2048, block_size, has_zeropoint, zp_is_4bit, abs_error, has_g_idx, has_bias);
  }
}

TEST(MatMulNBits, Fp16_Int4_NoZeroPoint_Bias_Prepacked) {
  ScopedEnvironmentVariables scoped_env_vars{EnvVarMap{{"ORT_FPA_INTB_GEMM", "1"}}};

  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA execution provider is unavailable";
  }

  // Bias-bearing node with runtime prepacking (weight_prepacked=0): the kernel's PrePack transforms
  // the raw weight into the fpA_intB layout at session init and the fused bias flows through the
  // CUTLASS/GEMV epilogue. Offline weight_prepacked=1 parity for bias is covered by the Python test
  // test_op_matmulnbits_prepacked_cuda.py.
  TestOptions opts{};
  opts.M = 32, opts.N = 1024, opts.K = 2048;
  opts.block_size = 64;
  opts.has_zero_point = false;
  opts.has_bias = true;
  opts.output_abs_error = 0.1f;
  opts.output_rel_error = 0.02f;
  std::vector<std::unique_ptr<IExecutionProvider>> eps;
  eps.push_back(std::move(cuda_ep));
  RunTest<MLFloat16>(opts, std::move(eps));
}

// A prepacked weight (weight_prepacked!=0) forces the fpA_intB path on regardless of the enable
// flag, and the constructor ORT_ENFORCEs that the path is actually supported for the node. Here the
// block_size (256) is outside the fpA_intB-supported set {32, 64, 128}, so kernel construction is
// rejected up front with "weight_prepacked requires the fpA_intB path, but it is unsupported ...",
// even though ORT_FPA_INTB_GEMM is enabled.
TEST(MatMulNBits, Fp16_Int4_PrepackedWeightRejectedWhenFpAIntBUnsupported) {
  ScopedEnvironmentVariables scoped_env_vars{EnvVarMap{{"ORT_FPA_INTB_GEMM", "1"}}};

  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA execution provider is unavailable";
  }

  TestOptions opts{};
  opts.M = 1, opts.N = 256, opts.K = 1024;
  opts.block_size = 256;
  opts.disable_cpu_ep_fallback = true;
  opts.weight_prepacked = 1;
  opts.expected_failure = "weight_prepacked requires";
  std::vector<std::unique_ptr<IExecutionProvider>> eps;
  eps.push_back(std::move(cuda_ep));
  RunTest<MLFloat16>(opts, std::move(eps));
}

// weight_prepacked=2 selects the native SM90 (Hopper) mixed-GEMM layout. It is rejected up front
// unless the device is SM90 and block_size is 64 or 128 (the SM90 TMA kernel requires group_size to
// be a multiple of the 64-element Hopper K tile, so block_size=32 is SM80-only). When the fpA_intB
// path is compiled in, both rejection messages begin with "weight_prepacked=2 (SM90 layout)", so the
// check is device-independent: non-Hopper hits the compute-capability guard, Hopper hits the
// block_size guard. In a build without onnxruntime_USE_FPA_INTB_GEMM the kernel rejects any
// weight_prepacked!=0 up front with a different ("weight_prepacked requires ...") message.
TEST(MatMulNBits, Fp16_Int4_PrepackedSm90BlockSize32Rejected) {
  ScopedEnvironmentVariables scoped_env_vars{EnvVarMap{{"ORT_FPA_INTB_GEMM", "1"}}};

  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA execution provider is unavailable";
  }

  TestOptions opts{};
  opts.M = 1, opts.N = 256, opts.K = 1024;
  opts.block_size = 32;
  opts.disable_cpu_ep_fallback = true;
  opts.weight_prepacked = 2;
#if USE_FPA_INTB_GEMM
  opts.expected_failure = "weight_prepacked=2 (SM90 layout)";
#else
  opts.expected_failure = "weight_prepacked requires";
#endif
  std::vector<std::unique_ptr<IExecutionProvider>> eps;
  eps.push_back(std::move(cuda_ep));
  RunTest<MLFloat16>(opts, std::move(eps));
}

// Exercises the CUDA small-M batched GEMV tiles: CtaM in {2,4,8,16} (with M values that are not a
// multiple of CtaM so the row-skip path runs) and CtaN in {1,2} (N divisible / not divisible by 16).
TEST(MatMulNBits, Fp16_Int4_SmallMBatchedTiles) {
  constexpr float abs_error = 0.1f;
  constexpr bool zp_is_4bit = true;
  for (auto block_size : {32, 128}) {
    for (auto m : {3, 4, 5, 8, 12, 16}) {
      for (auto has_zeropoint : {false, true}) {
        RunTest<MLFloat16>(m, 256, 1024, block_size, has_zeropoint, zp_is_4bit, abs_error);  // N % 16 == 0 -> CtaN=2
        RunTest<MLFloat16>(m, 24, 1024, block_size, has_zeropoint, zp_is_4bit, abs_error);   // N % 16 != 0 -> CtaN=1
      }
    }
  }
}

TEST(MatMulNBits, BFloat16_Int4_SmallMBatchedTiles) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "Skipping BFloat16 tests on CUDA < 8.0";
  }

  constexpr float abs_error = 0.1f;
  for (auto block_size : {32, 128}) {
    for (auto m : {3, 4, 5, 8, 12, 16}) {
      for (auto n : {256, 24}) {  // N=256 -> CtaN=2, N=24 -> CtaN=1
        for (auto has_zeropoint : {false, true}) {
          TestOptions opts{};
          opts.M = m, opts.N = n, opts.K = 1024;
          opts.block_size = block_size;
          opts.has_zero_point = has_zeropoint;
          opts.zp_is_4bit = true;
          opts.output_abs_error = abs_error;
          opts.output_rel_error = 0.02f;
          std::vector<std::unique_ptr<IExecutionProvider>> eps;
          eps.push_back(DefaultCudaExecutionProvider());
          RunTest<BFloat16>(opts, std::move(eps));
        }
      }
    }
  }
}

TEST(MatMulNBits, Fp16_Int4_GptOssRouterShapeNoZeroPoint) {
  constexpr float abs_error = 0.1f;

  // IsSupportedRouterGemvShape enables the specialization for block_size 32 and 64, so exercise both
  // to keep the MatMulFloatInt4RouterKernel<T, 32> and <T, 64> instantiations covered.
  for (auto block_size : {32, 64}) {
    TestOptions opts{};
    opts.M = 1, opts.N = 32, opts.K = 2880;
    opts.block_size = block_size;
    opts.has_zero_point = false;
    opts.zp_is_4bit = true;
    opts.output_abs_error = abs_error;
    opts.output_rel_error = 0.02f;

    {
      ScopedEnvironmentVariables scoped_env_vars{
          EnvVarMap{{"ORT_DISABLE_QMOE_ROUTER_GEMV_SPECIALIZATION", std::optional<std::string>{}}}};
      std::vector<std::unique_ptr<IExecutionProvider>> eps;
      eps.push_back(DefaultCudaExecutionProvider());
      RunTest<MLFloat16>(opts, std::move(eps));
    }

    {
      ScopedEnvironmentVariables scoped_env_vars{EnvVarMap{{"ORT_DISABLE_QMOE_ROUTER_GEMV_SPECIALIZATION", "1"}}};
      std::vector<std::unique_ptr<IExecutionProvider>> eps;
      eps.push_back(DefaultCudaExecutionProvider());
      RunTest<MLFloat16>(opts, std::move(eps));
    }

    {
      opts.has_bias = true;
      ScopedEnvironmentVariables scoped_env_vars{
          EnvVarMap{{"ORT_DISABLE_QMOE_ROUTER_GEMV_SPECIALIZATION", std::optional<std::string>{}}}};
      std::vector<std::unique_ptr<IExecutionProvider>> eps;
      eps.push_back(DefaultCudaExecutionProvider());
      RunTest<MLFloat16>(opts, std::move(eps));
    }
  }
}

TEST(MatMulNBits, BFloat16_Int4_NoZeroPoint) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "Skipping BFloat16 8-bit MatMul tests on CUDA < 8.0";
  }

  constexpr float abs_error = 0.5f;
  constexpr bool zp_is_4bit = true;
  constexpr bool has_zeropoint = false;

  for (auto block_size : {64, 128}) {
    RunTest<MLFloat16>(1, 256, 1024, block_size, has_zeropoint, zp_is_4bit, abs_error);
    RunTest<MLFloat16>(32, 1024, 2048, block_size, has_zeropoint, zp_is_4bit, abs_error);
  }

  ScopedEnvironmentVariables scoped_env_vars{EnvVarMap{{"ORT_FPA_INTB_GEMM", "1"}}};

  for (auto block_size : {64, 128}) {
    RunTest<BFloat16>(1, 256, 1024, block_size, has_zeropoint, zp_is_4bit, abs_error);
    RunTest<BFloat16>(32, 1024, 2048, block_size, has_zeropoint, zp_is_4bit, abs_error);
  }
}

// Chunked dequant+GEMM path tests.
// Force the chunked path with a small chunk size to exercise per-chunk pointer
// arithmetic (blob, scales, zero_points offsets) and strided cuBLAS output
// with manageable tensor sizes.

TEST(MatMulNBits, Fp16_Int4_Chunked_Uint8ZeroPoint) {
  constexpr float abs_error = 0.1f;

  ScopedEnvironmentVariables scoped_env_vars{EnvVarMap{
      {"ORT_MATMULNBITS_FORCE_CHUNKED", "1"},
      {"ORT_MATMULNBITS_CHUNK_SIZE", "64"}}};

  for (auto block_size : {32, 64, 128}) {
    for (auto M : {1, 2}) {
      TestOptions opts{};
      opts.M = M, opts.N = 256, opts.K = 1024;
      opts.block_size = block_size;
      opts.has_zero_point = true;
      opts.zp_is_4bit = true;
      opts.output_abs_error = abs_error;
      opts.output_rel_error = 0.001f;
      std::vector<std::unique_ptr<IExecutionProvider>> eps;
      eps.push_back(DefaultCudaExecutionProvider());
      RunTest<MLFloat16>(opts, std::move(eps));
    }
  }
  // Odd blocks_per_col (K=96, block_size=32 → blocks_per_col=3) exercises the
  // (blocks_per_col + 1) / 2 rounding in the 4-bit packed ZP offset.
  {
    TestOptions opts{};
    opts.M = 1, opts.N = 256, opts.K = 96;
    opts.block_size = 32;
    opts.has_zero_point = true;
    opts.zp_is_4bit = true;
    opts.output_abs_error = abs_error;
    opts.output_rel_error = 0.001f;
    std::vector<std::unique_ptr<IExecutionProvider>> eps;
    eps.push_back(DefaultCudaExecutionProvider());
    RunTest<MLFloat16>(opts, std::move(eps));
  }
}

TEST(MatMulNBits, Fp16_Int4_Chunked_Fp16ZeroPoint) {
  constexpr float abs_error = 0.1f;

  ScopedEnvironmentVariables scoped_env_vars{EnvVarMap{
      {"ORT_MATMULNBITS_FORCE_CHUNKED", "1"},
      {"ORT_MATMULNBITS_CHUNK_SIZE", "64"}}};

  for (auto block_size : {32, 64, 128}) {
    for (auto M : {1, 2}) {
      TestOptions opts{};
      opts.M = M, opts.N = 256, opts.K = 1024;
      opts.block_size = block_size;
      opts.has_zero_point = true;
      opts.zp_is_4bit = false;
      opts.output_abs_error = abs_error;
      opts.output_rel_error = 0.001f;
      std::vector<std::unique_ptr<IExecutionProvider>> eps;
      eps.push_back(DefaultCudaExecutionProvider());
      RunTest<MLFloat16>(opts, std::move(eps));
    }
  }
}

TEST(MatMulNBits, Fp16_Int4_Chunked_NoZeroPoint) {
  constexpr float abs_error = 0.1f;

  ScopedEnvironmentVariables scoped_env_vars{EnvVarMap{
      {"ORT_MATMULNBITS_FORCE_CHUNKED", "1"},
      {"ORT_MATMULNBITS_CHUNK_SIZE", "64"}}};

  for (auto block_size : {32, 64, 128}) {
    for (auto M : {1, 2}) {
      TestOptions opts{};
      opts.M = M, opts.N = 256, opts.K = 1024;
      opts.block_size = block_size;
      opts.has_zero_point = false;
      opts.zp_is_4bit = true;
      opts.output_abs_error = abs_error;
      opts.output_rel_error = 0.001f;
      std::vector<std::unique_ptr<IExecutionProvider>> eps;
      eps.push_back(DefaultCudaExecutionProvider());
      RunTest<MLFloat16>(opts, std::move(eps));
    }
  }
}

TEST(MatMulNBits, BFloat16_Int4_Chunked_Uint8ZeroPoint) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "Skipping BFloat16 tests on CUDA < 8.0";
  }

  constexpr float abs_error = 0.1f;
  constexpr bool zp_is_4bit = true;
  constexpr bool has_zeropoint = true;

  ScopedEnvironmentVariables scoped_env_vars{EnvVarMap{
      {"ORT_MATMULNBITS_FORCE_CHUNKED", "1"},
      {"ORT_MATMULNBITS_CHUNK_SIZE", "64"}}};

  for (auto block_size : {32, 64, 128}) {
    for (auto M : {1, 2}) {
      TestOptions opts{};
      opts.M = M, opts.N = 256, opts.K = 1024;
      opts.block_size = block_size;
      opts.has_zero_point = has_zeropoint;
      opts.zp_is_4bit = zp_is_4bit;
      opts.output_abs_error = abs_error;
      opts.output_rel_error = 0.001f;
      std::vector<std::unique_ptr<IExecutionProvider>> eps;
      eps.push_back(DefaultCudaExecutionProvider());
      RunTest<BFloat16>(opts, std::move(eps));
    }
  }
  // Odd blocks_per_col (K=96, block_size=32 → blocks_per_col=3) exercises the
  // (blocks_per_col + 1) / 2 rounding in the 4-bit packed ZP offset.
  {
    TestOptions opts{};
    opts.M = 1, opts.N = 256, opts.K = 96;
    opts.block_size = 32;
    opts.has_zero_point = has_zeropoint;
    opts.zp_is_4bit = zp_is_4bit;
    opts.output_abs_error = abs_error;
    opts.output_rel_error = 0.001f;
    std::vector<std::unique_ptr<IExecutionProvider>> eps;
    eps.push_back(DefaultCudaExecutionProvider());
    RunTest<BFloat16>(opts, std::move(eps));
  }
}

TEST(MatMulNBits, BFloat16_Int4_Chunked_BFloat16ZeroPoint) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "Skipping BFloat16 tests on CUDA < 8.0";
  }

  constexpr float abs_error = 0.1f;
  constexpr bool zp_is_4bit = false;
  constexpr bool has_zeropoint = true;

  ScopedEnvironmentVariables scoped_env_vars{EnvVarMap{
      {"ORT_MATMULNBITS_FORCE_CHUNKED", "1"},
      {"ORT_MATMULNBITS_CHUNK_SIZE", "64"}}};

  for (auto block_size : {32, 64, 128}) {
    for (auto M : {1, 2}) {
      TestOptions opts{};
      opts.M = M, opts.N = 256, opts.K = 1024;
      opts.block_size = block_size;
      opts.has_zero_point = has_zeropoint;
      opts.zp_is_4bit = zp_is_4bit;
      opts.output_abs_error = abs_error;
      opts.output_rel_error = 0.001f;
      std::vector<std::unique_ptr<IExecutionProvider>> eps;
      eps.push_back(DefaultCudaExecutionProvider());
      RunTest<BFloat16>(opts, std::move(eps));
    }
  }
}
#endif

#endif  // defined(USE_CUDA) || defined(USE_DML)

#if defined(USE_QNN) && defined(_M_ARM64)

namespace {
// QNN-EP Test Function
// This has too many parameters of the same type that must be specified in the correct order.
// Consider using the overload with a TestOptions parameter.
void RunQnnEpTest(int64_t M, int64_t N, int64_t K, bool has_zeropoint = true, float abs_error = 0.05f) {
  TestOptions opts{};
  opts.M = M;
  opts.N = N;
  opts.K = K;
  opts.block_size = 32;
  opts.accuracy_level = 4;
  opts.has_zero_point = has_zeropoint;
  opts.zp_is_4bit = true;
  opts.has_g_idx = false;
  opts.has_bias = false;

  if (abs_error > 0.f) {
    opts.output_abs_error = abs_error;
  }

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  ProviderOptions provider_options;
  provider_options["backend_type"] = "gpu";
  provider_options["offload_graph_io_quantization"] = "0";
  execution_providers.push_back(QnnExecutionProviderWithOptions(provider_options));

  RunTest<float>(opts, std::move(execution_providers));
}
}  // namespace

// QNN GPU Only support FP16 activations and Q4_0 weights, with zero_points = 8
// Accumulation with larger channel accumulates more error. Set higher abs_error with respect to K.
TEST(MatMulNBits, Basic_M1_N128_K512_withZp) {
  constexpr float abs_error = 0.05f;
  RunQnnEpTest(1, 128, 512, true, abs_error);
}

TEST(MatMulNBits, Basic_M1_N128_K512) {
  constexpr float abs_error = 0.05f;
  RunQnnEpTest(1, 128, 512, false, abs_error);
}

TEST(MatMulNBits, Basic_M10_N128_K512_withZp) {
  constexpr float abs_error = 0.05f;
  RunQnnEpTest(10, 128, 512, true, abs_error);
}

TEST(MatMulNBits, Basic_M10_N128_K512) {
  constexpr float abs_error = 0.05f;
  RunQnnEpTest(10, 128, 512, false, abs_error);
}
#endif

// Test that out-of-range g_idx values are rejected with INVALID_ARGUMENT.
// CUDA EP is excluded from these tests, so no risk of hitting CUDA_KERNEL_ASSERT.
TEST(MatMulNBits, InvalidGIdx_OutOfRange) {
  constexpr int64_t M = 2, N = 4, K = 32, block_size = 16;
  constexpr int64_t k_blocks = (K + block_size - 1) / block_size;  // 2
  constexpr int64_t blob_size = block_size * QBits / 8;            // 8

  OpTester test("MatMulNBits", 1, kMSDomain);
  test.AddAttribute<int64_t>("K", K);
  test.AddAttribute<int64_t>("N", N);
  test.AddAttribute<int64_t>("block_size", block_size);
  test.AddAttribute<int64_t>("bits", QBits);
  test.AddAttribute<int64_t>("accuracy_level", int64_t{0});

  // A: [M, K]
  std::vector<float> a_data(M * K, 1.0f);
  test.AddInput<float>("A", {M, K}, a_data, false);

  // B: [N, k_blocks, blob_size]
  std::vector<uint8_t> b_data(N * k_blocks * blob_size, 0);
  test.AddInput<uint8_t>("B", {N, k_blocks, blob_size}, b_data, true);

  // scales: [N, k_blocks]
  std::vector<float> scales(N * k_blocks, 1.0f);
  test.AddInput<float>("scales", {N, k_blocks}, scales, true);

  // zero_points: optional (skip)
  test.AddOptionalInputEdge<uint8_t>();

  // g_idx with out-of-range values (valid range is [0, k_blocks) = [0, 2))
  std::vector<int32_t> g_idx(K);
  for (int64_t i = 0; i < K; i++) {
    g_idx[i] = 99999;  // way out of range
  }
  test.AddInput<int32_t>("g_idx", {K}, g_idx, true);

  // bias: optional (skip)
  test.AddOptionalInputEdge<float>();

  // Output placeholder (won't actually be compared since we expect failure)
  std::vector<float> y_data(M * N, 0.0f);
  test.AddOutput<float>("Y", {M, N}, y_data);

  test.Run(OpTester::ExpectResult::kExpectFailure, "group_index value",
           {kCudaExecutionProvider, kCudaNHWCExecutionProvider, kDmlExecutionProvider, kWebGpuExecutionProvider,
            kOpenVINOExecutionProvider});
}

// Test that negative g_idx values are rejected.
TEST(MatMulNBits, InvalidGIdx_Negative) {
  constexpr int64_t M = 2, N = 4, K = 32, block_size = 16;
  constexpr int64_t k_blocks = (K + block_size - 1) / block_size;
  constexpr int64_t blob_size = block_size * QBits / 8;

  OpTester test("MatMulNBits", 1, kMSDomain);
  test.AddAttribute<int64_t>("K", K);
  test.AddAttribute<int64_t>("N", N);
  test.AddAttribute<int64_t>("block_size", block_size);
  test.AddAttribute<int64_t>("bits", QBits);
  test.AddAttribute<int64_t>("accuracy_level", int64_t{0});

  std::vector<float> a_data(M * K, 1.0f);
  test.AddInput<float>("A", {M, K}, a_data, false);

  std::vector<uint8_t> b_data(N * k_blocks * blob_size, 0);
  test.AddInput<uint8_t>("B", {N, k_blocks, blob_size}, b_data, true);

  std::vector<float> scales(N * k_blocks, 1.0f);
  test.AddInput<float>("scales", {N, k_blocks}, scales, true);

  test.AddOptionalInputEdge<uint8_t>();

  // g_idx with negative values
  std::vector<int32_t> g_idx(K);
  for (int64_t i = 0; i < K; i++) {
    g_idx[i] = -1;
  }
  test.AddInput<int32_t>("g_idx", {K}, g_idx, true);

  test.AddOptionalInputEdge<float>();

  std::vector<float> y_data(M * N, 0.0f);
  test.AddOutput<float>("Y", {M, N}, y_data);

  test.Run(OpTester::ExpectResult::kExpectFailure, "group_index value",
           {kCudaExecutionProvider, kCudaNHWCExecutionProvider, kDmlExecutionProvider, kWebGpuExecutionProvider,
            kOpenVINOExecutionProvider});
}

// Test that block_size=512 (unsupported) is rejected at kernel creation.
TEST(MatMulNBits, UnsupportedBlockSize_512) {
  constexpr int64_t M = 1, N = 1, K = 512, block_size = 512;
  constexpr int64_t k_blocks = (K + block_size - 1) / block_size;
  constexpr int64_t blob_size = block_size * QBits / 8;

  OpTester test("MatMulNBits", 1, kMSDomain);
  test.AddAttribute<int64_t>("K", K);
  test.AddAttribute<int64_t>("N", N);
  test.AddAttribute<int64_t>("bits", int64_t{QBits});
  test.AddAttribute<int64_t>("block_size", block_size);
  test.AddAttribute<int64_t>("accuracy_level", int64_t{0});

  std::vector<float> a_data(M * K, 1.0f);
  test.AddInput<float>("A", {M, K}, a_data, false);

  std::vector<uint8_t> b_data(N * k_blocks * blob_size, 0);
  test.AddInput<uint8_t>("B", {N, k_blocks, blob_size}, b_data, true);

  std::vector<float> scales(N * k_blocks, 1.0f);
  test.AddInput<float>("scales", {N, k_blocks}, scales, true);

  std::vector<float> y_data(M * N, 0.0f);
  test.AddOutput<float>("Y", {M, N}, y_data);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectFailure, "Only block sizes 16, 32, 64, 128, and 256 are supported",
           {}, nullptr, &execution_providers);
}

// The following tests cover the shape-validation guard added at the top of
// MatMulNBits<T1>::PrePack. The guard rejects initializer shapes that do not
// match the attribute-derived shape so that a crafted model whose (N, K, bits,
// block_size) attributes overstate the real tensor extents cannot trigger an
// out-of-bounds READ inside the MLAS pack routines during session
// initialization. Each test passes a B/scales/zero_points initializer whose
// declared shape (and matching data buffer size) is inconsistent with the
// attribute-derived shape, and expects session creation to fail with
// "MatMulNBits PrePack:" (i.e. before Compute() is ever invoked).

// B shape mismatches the (N, k_blocks, blob_size) shape derived from attributes.
TEST(MatMulNBits, PrePack_InvalidBShape_RejectsAtSessionInit) {
  constexpr int64_t M = 1, N = 4, K = 32, block_size = 32;
  constexpr int64_t k_blocks = (K + block_size - 1) / block_size;  // 1
  constexpr int64_t blob_size = block_size * QBits / 8;            // 16

  OpTester test("MatMulNBits", 1, kMSDomain);
  test.AddAttribute<int64_t>("K", K);
  test.AddAttribute<int64_t>("N", N);
  test.AddAttribute<int64_t>("block_size", block_size);
  test.AddAttribute<int64_t>("bits", QBits);
  test.AddAttribute<int64_t>("accuracy_level", int64_t{0});

  std::vector<float> a_data(M * K, 1.0f);
  test.AddInput<float>("A", {M, K}, a_data, false);

  // Declare B with one fewer row than attributes claim. The data buffer matches
  // the smaller declared shape, exactly mirroring the crafted-model scenario in
  // which the attributes overstate the tensor's real extent.
  constexpr int64_t bad_N = N - 1;
  std::vector<uint8_t> b_data(bad_N * k_blocks * blob_size, 0);
  test.AddInput<uint8_t>("B", {bad_N, k_blocks, blob_size}, b_data, true);

  std::vector<float> scales(N * k_blocks, 1.0f);
  test.AddInput<float>("scales", {N, k_blocks}, scales, true);

  std::vector<float> y_data(M * N, 0.0f);
  test.AddOutput<float>("Y", {M, N}, y_data);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "MatMulNBits PrePack: B initializer shape",
           {}, nullptr, &execution_providers);
}

// B shape has the wrong rank (2D instead of (N, k_blocks, blob_size)).
TEST(MatMulNBits, PrePack_InvalidBRank_RejectsAtSessionInit) {
  constexpr int64_t M = 1, N = 4, K = 32, block_size = 32;
  constexpr int64_t k_blocks = (K + block_size - 1) / block_size;
  constexpr int64_t blob_size = block_size * QBits / 8;

  OpTester test("MatMulNBits", 1, kMSDomain);
  test.AddAttribute<int64_t>("K", K);
  test.AddAttribute<int64_t>("N", N);
  test.AddAttribute<int64_t>("block_size", block_size);
  test.AddAttribute<int64_t>("bits", QBits);
  test.AddAttribute<int64_t>("accuracy_level", int64_t{0});

  std::vector<float> a_data(M * K, 1.0f);
  test.AddInput<float>("A", {M, K}, a_data, false);

  // Flatten the trailing k_blocks/blob_size dims into a single dimension.
  // The total element count still matches, but the rank differs from the
  // attribute-derived (N, k_blocks, blob_size) shape.
  std::vector<uint8_t> b_data(N * k_blocks * blob_size, 0);
  test.AddInput<uint8_t>("B", {N, k_blocks * blob_size}, b_data, true);

  std::vector<float> scales(N * k_blocks, 1.0f);
  test.AddInput<float>("scales", {N, k_blocks}, scales, true);

  std::vector<float> y_data(M * N, 0.0f);
  test.AddOutput<float>("Y", {M, N}, y_data);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "MatMulNBits PrePack: B initializer shape",
           {}, nullptr, &execution_providers);
}

// scales shape does not match either of the accepted layouts
// ([N * k_blocks] or [N, k_blocks]).
TEST(MatMulNBits, PrePack_InvalidScalesShape_RejectsAtSessionInit) {
  constexpr int64_t M = 1, N = 4, K = 32, block_size = 32;
  constexpr int64_t k_blocks = (K + block_size - 1) / block_size;
  constexpr int64_t blob_size = block_size * QBits / 8;

  OpTester test("MatMulNBits", 1, kMSDomain);
  test.AddAttribute<int64_t>("K", K);
  test.AddAttribute<int64_t>("N", N);
  test.AddAttribute<int64_t>("block_size", block_size);
  test.AddAttribute<int64_t>("bits", QBits);
  test.AddAttribute<int64_t>("accuracy_level", int64_t{0});

  std::vector<float> a_data(M * K, 1.0f);
  test.AddInput<float>("A", {M, K}, a_data, false);

  std::vector<uint8_t> b_data(N * k_blocks * blob_size, 0);
  test.AddInput<uint8_t>("B", {N, k_blocks, blob_size}, b_data, true);

  // Declare scales with one fewer row than the attribute-derived layout.
  constexpr int64_t bad_N = N - 1;
  std::vector<float> scales(bad_N * k_blocks, 1.0f);
  test.AddInput<float>("scales", {bad_N, k_blocks}, scales, true);

  std::vector<float> y_data(M * N, 0.0f);
  test.AddOutput<float>("Y", {M, N}, y_data);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "MatMulNBits PrePack: scales initializer shape",
           {}, nullptr, &execution_providers);
}

// uint8 (packed) zero_points shape does not match the
// [N * zp_blob_size] / [N, zp_blob_size] layout derived from attributes.
TEST(MatMulNBits, PrePack_InvalidUInt8ZeroPointsShape_RejectsAtSessionInit) {
  constexpr int64_t M = 1, N = 4, K = 32, block_size = 32;
  constexpr int64_t k_blocks = (K + block_size - 1) / block_size;
  constexpr int64_t blob_size = block_size * QBits / 8;
  constexpr int64_t zp_blob_size = (k_blocks * QBits + 7) / 8;

  OpTester test("MatMulNBits", 1, kMSDomain);
  test.AddAttribute<int64_t>("K", K);
  test.AddAttribute<int64_t>("N", N);
  test.AddAttribute<int64_t>("block_size", block_size);
  test.AddAttribute<int64_t>("bits", QBits);
  test.AddAttribute<int64_t>("accuracy_level", int64_t{0});

  std::vector<float> a_data(M * K, 1.0f);
  test.AddInput<float>("A", {M, K}, a_data, false);

  std::vector<uint8_t> b_data(N * k_blocks * blob_size, 0);
  test.AddInput<uint8_t>("B", {N, k_blocks, blob_size}, b_data, true);

  std::vector<float> scales(N * k_blocks, 1.0f);
  test.AddInput<float>("scales", {N, k_blocks}, scales, true);

  // Declare uint8 zero_points with one fewer row than the attribute-derived
  // layout. zp_blob_size==1 here, so this is also distinguishable from any
  // legacy 1D layout that would otherwise be accepted.
  constexpr int64_t bad_N = N - 1;
  std::vector<uint8_t> zp(bad_N * zp_blob_size, 0);
  test.AddInput<uint8_t>("zero_points", {bad_N, zp_blob_size}, zp, true);

  std::vector<float> y_data(M * N, 0.0f);
  test.AddOutput<float>("Y", {M, N}, y_data);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "MatMulNBits PrePack: zero_points initializer shape",
           {}, nullptr, &execution_providers);
}

// Sanity check: the legacy 1D layouts for scales and uint8 zero_points are
// still accepted by the new shape validation guard (i.e. the guard only
// rejects truly mismatched shapes and does not regress backward
// compatibility for existing models).
TEST(MatMulNBits, PrePack_LegacyFlattenedShapes_Accepted) {
  constexpr int64_t M = 1, N = 4, K = 32, block_size = 32;
  constexpr int64_t k_blocks = (K + block_size - 1) / block_size;
  constexpr int64_t blob_size = block_size * QBits / 8;
  constexpr int64_t zp_blob_size = (k_blocks * QBits + 7) / 8;

  RandomValueGenerator random{1234};
  std::vector<float> a_vals(random.Gaussian<float>(AsSpan({M, K}), 0.0f, 0.25f));
  std::vector<float> b_f_vals(random.Gaussian<float>(AsSpan({K, N}), 0.0f, 0.25f));

  std::vector<uint8_t> b_data(N * k_blocks * blob_size);
  std::vector<float> scales(N * k_blocks);
  std::vector<uint8_t> zp(N * zp_blob_size);
  QuantizeDequantize(b_f_vals, b_data, scales, &zp,
                     static_cast<int32_t>(N), static_cast<int32_t>(K),
                     static_cast<int32_t>(block_size));

  std::vector<float> expected(M * N);
  for (int64_t m = 0; m < M; ++m) {
    for (int64_t n = 0; n < N; ++n) {
      float sum = 0.0f;
      for (int64_t k = 0; k < K; ++k) {
        sum += a_vals[m * K + k] * b_f_vals[n * K + k];
      }
      expected[m * N + n] = sum;
    }
  }

  OpTester test("MatMulNBits", 1, kMSDomain);
  test.AddAttribute<int64_t>("K", K);
  test.AddAttribute<int64_t>("N", N);
  test.AddAttribute<int64_t>("block_size", block_size);
  test.AddAttribute<int64_t>("bits", QBits);
  test.AddAttribute<int64_t>("accuracy_level", int64_t{0});

  test.AddInput<float>("A", {M, K}, a_vals, false);
  test.AddInput<uint8_t>("B", {N, k_blocks, blob_size}, b_data, true);
  // Legacy flattened 1D layouts for scales and zero_points.
  test.AddInput<float>("scales", {N * k_blocks}, scales, true);
  test.AddInput<uint8_t>("zero_points", {N * zp_blob_size}, zp, true);

  test.AddOutput<float>("Y", {M, N}, expected);
  test.SetOutputAbsErr("Y", 0.1f);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {}, nullptr, &execution_providers);
}

}  // namespace test
}  // namespace onnxruntime

#endif  // ORT_MINIMAL_BUILD
