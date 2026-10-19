// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "core/common/float8.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

#if defined(USE_CUDA) && !defined(DISABLE_FLOAT8_TYPES)
static void RunFp8Bf16Case(const std::vector<int64_t>& shape, int64_t axis,
                           const std::vector<float>& values, const std::vector<int64_t>& scale_shape,
                           const std::vector<float>& scales, const std::vector<float>& zeros,
                           const std::vector<float>& expected,
                           const std::string& error = "") {
  auto provider = DefaultCudaExecutionProvider();
  if (provider == nullptr) {
    GTEST_SKIP() << "CUDA execution provider is not available";
  }
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(std::move(provider));

  OpTester test("DequantizeLinear", 1, kMSDomain);
  test.AddAttribute<int64_t>("axis", axis);
  test.AddAttribute<int64_t>("output_dtype", ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16);
  std::vector<Float8E4M3FN> input;
  for (float value : values) {
    input.emplace_back(value);
  }
  test.AddInput<Float8E4M3FN>("x", shape, input);
  test.AddInput<float>("x_scale", scale_shape, scales);
  if (zeros.empty()) {
    test.AddOptionalInputEdge<Float8E4M3FN>();
  } else {
    std::vector<Float8E4M3FN> zero_point;
    for (float value : zeros) {
      zero_point.emplace_back(value);
    }
    test.AddInput<Float8E4M3FN>("x_zero_point", scale_shape, zero_point);
  }
  test.AddOutput<BFloat16>("y", shape, ToBFloat16(expected));
  test.SetOutputTolerance(0.0f, 0.0f);
  test.Run(error.empty() ? OpTester::ExpectResult::kExpectSuccess : OpTester::ExpectResult::kExpectFailure,
           error, {}, nullptr, &providers);
}

TEST(DequantizeLinearTest, Fp8ToBf16) {
  auto provider = DefaultCudaExecutionProvider();
  if (provider == nullptr) {
    GTEST_SKIP() << "CUDA execution provider is not available";
  }
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(std::move(provider));

  OpTester test("DequantizeLinear", 1, kMSDomain);
  test.AddAttribute<int64_t>("axis", 1);
  test.AddAttribute<int64_t>("output_dtype", ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16);
  test.AddInput<Float8E4M3FN>("weight", {1, 8}, std::vector<Float8E4M3FN>(8, Float8E4M3FN(1.0f)));
  test.AddInput<float>("scale", {8}, std::vector<float>(8, 2.0f));
  test.AddOptionalInputEdge<Float8E4M3FN>();
  test.AddOutput<BFloat16>("weight_bf16", {1, 8}, ToBFloat16(std::vector<float>(8, 2.0f)));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

TEST(DequantizeLinearTest, Fp8ToBf16ScalarFp32Scale) {
  // 1.00390625 rounds to 1 in BF16. Multiplication must use the original FP32 scale.
  RunFp8Bf16Case({4}, 0, {1.0f, 3.0f, -3.0f, 0.0f}, {}, {1.00390625f}, {},
                 {1.00390625f, 3.01171875f, -3.01171875f, 0.0f});
}

TEST(DequantizeLinearTest, Fp8ToBf16NegativeNonLastAxisAndZeroPoint) {
  RunFp8Bf16Case({2, 2, 2}, -2, {1, 2, 3, 4, -1, -2, -3, -4}, {2}, {2, 3}, {1, -1},
                 {0, 2, 12, 15, -4, -6, -6, -9});
}

TEST(DequantizeLinearTest, Fp8ToBf16RankOnePerAxis) {
  RunFp8Bf16Case({3}, 0, {1, -2, 4}, {3}, {0.5f, 2, 3}, {},
                 {0.5f, -4, 12});
}

TEST(DequantizeLinearTest, Fp8ToBf16EmptyInput) {
  RunFp8Bf16Case({0, 2}, 1, {}, {2}, {1, 2}, {}, {});
}

TEST(DequantizeLinearTest, Fp8ToBf16RejectInvalidScales) {
  RunFp8Bf16Case({2, 2}, 1, {1, 2, 3, 4}, {3}, {1, 2, 3}, {}, {1, 2, 3, 4},
                 "x_scale must match the input dimension along axis");
  RunFp8Bf16Case({2, 2}, 1, {1, 2, 3, 4}, {0}, {}, {}, {1, 2, 3, 4},
                 "x_scale must not be empty");
  RunFp8Bf16Case({2, 2}, 1, {1, 2, 3, 4}, {1, 2}, {1, 2}, {}, {1, 2, 3, 4},
                 "x_scale must be a scalar or 1D tensor");
}
#endif

}  // namespace test
}  // namespace onnxruntime
