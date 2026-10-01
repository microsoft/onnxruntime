// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <type_traits>
#include <vector>

#include "core/common/float16.h"
#include "core/framework/session_options.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/scoped_env_vars.h"

namespace onnxruntime {
namespace test {
namespace {

enum class DispatchMode {
  kForceSmallN,  // ORT_ENABLE_SMALL_N_GEMV=1
  kAutoTune,     // ep.cuda.enable_gemm_auto_tune=1
};

template <typename T>
void RunMatMulOperatorCase(int m, int n, int k, DispatchMode mode) {
  SCOPED_TRACE(std::string(std::is_same_v<T, MLFloat16> ? "fp16" : "bf16") + " m=" + std::to_string(m) +
               " n=" + std::to_string(n) + " k=" + std::to_string(k) +
               (mode == DispatchMode::kAutoTune ? " auto-tune" : " forced"));
  std::vector<T> a(static_cast<size_t>(m) * k);
  std::vector<T> b(static_cast<size_t>(k) * n);
  std::vector<T> expected(static_cast<size_t>(m) * n);
  for (size_t index = 0; index < a.size(); ++index) {
    a[index] = T(static_cast<float>((index * 7 + 1) % 23) / 16.0f - 0.6875f);
  }
  for (size_t index = 0; index < b.size(); ++index) {
    b[index] = T(static_cast<float>((index * 11 + 2) % 19) / 16.0f - 0.5625f);
  }
  float max_abs_expected = 0.0f;
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      float sum = 0.0f;
      for (int kk = 0; kk < k; ++kk) {
        sum += a[static_cast<size_t>(row) * k + kk].ToFloat() *
               b[static_cast<size_t>(kk) * n + col].ToFloat();
      }
      expected[static_cast<size_t>(row) * n + col] = T(sum);
      max_abs_expected = std::max(max_abs_expected, std::fabs(sum));
    }
  }

  ScopedEnvironmentVariables scoped_env_vars{
      EnvVarMap{{"ORT_ENABLE_SMALL_N_GEMV",
                 mode == DispatchMode::kForceSmallN ? optional<std::string>{"1"} : std::nullopt},
                {"ORT_CUDA_GEMM_AUTO_TUNE", std::nullopt}}};
  SessionOptions session_options;
  if (mode == DispatchMode::kAutoTune) {
    ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(kOrtSessionOptionsCudaEnableGemmAutoTune, "1"));
  }

  OpTester test("MatMul", 14);
  test.AddInput<T>("A", {m, k}, a);
  test.AddInput<T>("B", {k, n}, b);
  test.AddOutput<T>("Y", {m, n}, expected);
  // Kernels accumulate in different orders; allow two output ulps at the largest magnitude.
  const float ulp = std::is_same_v<T, MLFloat16> ? 1.0f / 1024.0f : 1.0f / 128.0f;
  test.SetOutputAbsErr("Y", 0.05f + 2.0f * ulp * max_abs_expected);
  test.Config(session_options).ConfigEp(DefaultCudaExecutionProvider()).RunWithConfig();
}

template <typename T>
void RunEligibleShapes(DispatchMode mode) {
  RunMatMulOperatorCase<T>(1, 48, 5120, mode);
  RunMatMulOperatorCase<T>(8, 1, 128, mode);
  RunMatMulOperatorCase<T>(8, 1024, 128, mode);
  RunMatMulOperatorCase<T>(9, 48, 5120, mode);
  RunMatMulOperatorCase<T>(33, 37, 1032, mode);
  RunMatMulOperatorCase<T>(64, 48, 5120, mode);
}

TEST(MatMulSmallNGemvOpTest, DispatchesEligibleShapesWhenForced) {
  RunEligibleShapes<MLFloat16>(DispatchMode::kForceSmallN);
  RunEligibleShapes<BFloat16>(DispatchMode::kForceSmallN);
}

TEST(MatMulSmallNGemvOpTest, AutoTunesEligibleShapes) {
  RunEligibleShapes<MLFloat16>(DispatchMode::kAutoTune);
  RunEligibleShapes<BFloat16>(DispatchMode::kAutoTune);
}

TEST(MatMulSmallNGemvOpTest, FallsBackForIneligibleShapes) {
  for (const DispatchMode mode : {DispatchMode::kForceSmallN, DispatchMode::kAutoTune}) {
    RunMatMulOperatorCase<MLFloat16>(8, 32, 127, mode);
    RunMatMulOperatorCase<BFloat16>(65, 48, 256, mode);
    RunMatMulOperatorCase<MLFloat16>(4, 1025, 256, mode);
  }
}

}  // namespace
}  // namespace test
}  // namespace onnxruntime
