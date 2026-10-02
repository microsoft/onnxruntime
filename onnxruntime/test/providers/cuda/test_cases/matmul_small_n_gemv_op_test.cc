// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include <vector>

#include "core/common/float16.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/scoped_env_vars.h"

namespace onnxruntime {
namespace test {
namespace {

void RunMatMulOperatorCase(int m, int n, int k) {
  std::vector<MLFloat16> a(static_cast<size_t>(m) * k);
  std::vector<MLFloat16> b(static_cast<size_t>(k) * n);
  std::vector<MLFloat16> expected(static_cast<size_t>(m) * n);
  for (size_t index = 0; index < a.size(); ++index) {
    a[index] = MLFloat16(static_cast<float>((index * 7 + 1) % 23) / 16.0f - 0.6875f);
  }
  for (size_t index = 0; index < b.size(); ++index) {
    b[index] = MLFloat16(static_cast<float>((index * 11 + 2) % 19) / 16.0f - 0.5625f);
  }
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      float sum = 0.0f;
      for (int kk = 0; kk < k; ++kk) {
        sum += a[static_cast<size_t>(row) * k + kk].ToFloat() *
               b[static_cast<size_t>(kk) * n + col].ToFloat();
      }
      expected[static_cast<size_t>(row) * n + col] = MLFloat16(sum);
    }
  }

  ScopedEnvironmentVariables scoped_env_vars{EnvVarMap{{"ORT_ENABLE_SMALL_N_GEMV", "1"}}};
  OpTester test("MatMul", 14);
  test.AddInput<MLFloat16>("A", {m, k}, a);
  test.AddInput<MLFloat16>("B", {k, n}, b);
  test.AddOutput<MLFloat16>("Y", {m, n}, expected);
  test.SetOutputAbsErr("Y", 0.05f);
  test.ConfigEp(DefaultCudaExecutionProvider()).RunWithConfig();
}

TEST(MatMulSmallNGemvOpTest, DispatchesEligibleShapesWhenEnabled) {
  RunMatMulOperatorCase(8, 1, 128);
  RunMatMulOperatorCase(8, 1024, 128);
  RunMatMulOperatorCase(9, 48, 5120);
  RunMatMulOperatorCase(33, 48, 5120);
  RunMatMulOperatorCase(64, 48, 5120);
}

TEST(MatMulSmallNGemvOpTest, FallsBackForIneligibleShapeWhenEnabled) {
  RunMatMulOperatorCase(8, 32, 127);
}

}  // namespace
}  // namespace test
}  // namespace onnxruntime