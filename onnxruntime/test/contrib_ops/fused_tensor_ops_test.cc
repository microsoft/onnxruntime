// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "core/common/float8.h"
#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime::test {
namespace {

template <typename T>
std::vector<T> ToTensorType(const std::vector<float>& data) {
  if constexpr (std::is_same_v<T, MLFloat16>) {
    return ToFloat16(data);
  } else if constexpr (std::is_same_v<T, BFloat16>) {
    return ToBFloat16(data);
  } else {
    return data;
  }
}

std::vector<std::unique_ptr<IExecutionProvider>> CudaProvider() {
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  if (auto cuda = DefaultCudaExecutionProvider()) {
    providers.push_back(std::move(cuda));
  }
  return providers;
}

TEST(FusedTensorOpsTest, ArgMaxLastAxisWithoutKeepdims) {
  auto providers = CudaProvider();
  if (providers.empty()) GTEST_SKIP();

  OpTester test("ArgMaxLastAxis", 1, kMSDomain);
  test.AddInput<float>("logits", {2, 4}, {1.0f, 5.0f, 5.0f, -1.0f, 4.0f, 3.0f, 2.0f, 1.0f});
  test.AddOutput<int64_t>("indices", {2}, {1, 0});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

TEST(FusedTensorOpsTest, ArgMaxLastAxisWithKeepdims) {
  auto providers = CudaProvider();
  if (providers.empty()) GTEST_SKIP();

  OpTester test("ArgMaxLastAxis", 1, kMSDomain);
  test.AddAttribute<int64_t>("keepdims", 1);
  test.AddInput<BFloat16>("logits", {2, 4},
                          ToTensorType<BFloat16>({1.0f, 5.0f, 5.0f, -1.0f, 4.0f, 3.0f, 2.0f, 1.0f}));
  test.AddOutput<int64_t>("indices", {2, 1}, {1, 0});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

TEST(FusedTensorOpsTest, ArgMaxLastAxisPartitionedVocabulary) {
  constexpr int64_t width = 248321;
  for (bool keepdims : {false, true}) {
    auto providers = CudaProvider();
    if (providers.empty()) GTEST_SKIP();
    std::vector<float> values(3 * width, -std::numeric_limits<float>::infinity());
    values[4095] = values[4096] = values[width - 1] = 5.0f;
    values[2 * width - 1] = 7.0f;
    OpTester test("ArgMaxLastAxis", 1, kMSDomain);
    test.AddAttribute<int64_t>("keepdims", keepdims ? 1 : 0);
    test.AddInput<BFloat16>("logits", {3, width}, ToBFloat16(values));
    test.AddOutput<int64_t>("indices", keepdims ? std::vector<int64_t>{3, 1} : std::vector<int64_t>{3},
                            {4095, width - 1, 0});
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
  }
}

TEST(FusedTensorOpsTest, EmbeddingRMSNorm) {
  auto providers = CudaProvider();
  if (providers.empty()) GTEST_SKIP();

  const std::vector<float> table = {3.0f, 4.0f, 0.0f, 0.0f, 0.0f, 0.0f, 2.0f, 0.0f};
  const float inverse_first = 1.0f / std::sqrt(6.25f + 1e-6f);
  const float inverse_second = 1.0f / std::sqrt(1.0f + 1e-6f);
  const std::vector<float> expected = {
      3.0f * inverse_first, 4.0f * inverse_first, 0.0f, 0.0f,
      0.0f, 0.0f, 2.0f * inverse_second, 0.0f};

  OpTester test("EmbeddingRMSNorm", 1, kMSDomain);
  test.AddAttribute<float>("epsilon", 1e-6f);
  test.AddInput<int64_t>("ids", {2}, {0, 1});
  test.AddInput<BFloat16>("table", {2, 4}, ToTensorType<BFloat16>(table));
  test.AddInput<float>("weight", {4}, {1.0f, 1.0f, 1.0f, 1.0f});
  test.AddOutput<BFloat16>("normalized", {2, 4}, ToTensorType<BFloat16>(expected));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

TEST(FusedTensorOpsTest, RowSelect) {
  auto providers = CudaProvider();
  if (providers.empty()) GTEST_SKIP();

  OpTester test("RowSelect", 1, kMSDomain);
  test.AddInput<int64_t>("selection", {}, {1});
  for (int i = 0; i < 3; ++i) {
    const std::string input_name = "x" + std::to_string(i);
    const std::string output_name = "y" + std::to_string(i);
    test.AddInput<BFloat16>(input_name.c_str(), {1, 2, 2},
                            ToTensorType<BFloat16>({1.0f, 2.0f, 3.0f, 4.0f}));
    test.AddOutput<BFloat16>(output_name.c_str(), {1, 1, 2},
                             ToTensorType<BFloat16>({3.0f, 4.0f}));
  }
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

TEST(FusedTensorOpsTest, SigmoidGateMul) {
  auto providers = CudaProvider();
  if (providers.empty()) GTEST_SKIP();

  OpTester test("SigmoidGateMul", 1, kMSDomain);
  test.AddAttribute<int64_t>("head_dim", 2);
  test.AddAttribute<int64_t>("num_heads", 1);
  test.AddAttribute<int64_t>("q_stride", 4);
  test.AddAttribute<int64_t>("tok_stride", 4);
  test.AddAttribute<int64_t>("q_offset", 0);
  test.AddInput<BFloat16>("attention", {1, 1, 2}, ToTensorType<BFloat16>({2.0f, 4.0f}));
  test.AddInput<BFloat16>("projection", {1, 4}, ToTensorType<BFloat16>({1.0f, 2.0f, 0.0f, 0.0f}));
  test.AddOutput<BFloat16>("output", {1, 1, 2}, ToTensorType<BFloat16>({1.0f, 2.0f}));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

void RunSigmoidGateMulBroadcastResidualRounding(bool sigmoid_in_fp32, bool silu_gate = false) {
  for (bool residual : {false, true}) {
    for (int64_t width : {1, 257, 2560}) {
      auto providers = CudaProvider();
      if (providers.empty()) GTEST_SKIP();
      const std::vector<BFloat16> base_gates = ToBFloat16({-100.0f, -5.5f, -0.0f, 0.3f, 5.5f, 100.0f});
      std::vector<BFloat16> gates(silu_gate ? 6 * width : 6);
      for (size_t i = 0; i < gates.size(); ++i) gates[i] = base_gates[i % 6];
      std::vector<BFloat16> values(6 * width), residuals(6 * width), expected(6 * width);
      for (size_t i = 0; i < values.size(); ++i) {
        values[i] = BFloat16(static_cast<float>(static_cast<int>(i % 23) - 11) * 0.27f);
        residuals[i] = BFloat16(static_cast<float>(static_cast<int>(i % 17) - 8) * 0.13f);
        const float gate = gates[silu_gate ? i : i / width].ToFloat();
        const float exp_value = std::exp(gate >= 0 ? -gate : gate);
        const float sigmoid = gate >= 0 ? 1.0f / (1.0f + exp_value) : exp_value / (1.0f + exp_value);
        const BFloat16 denominator(1.0f + BFloat16(exp_value).ToFloat());
        const BFloat16 reciprocal(1.0f / denominator.ToFloat());
        const BFloat16 rounded_sigmoid = sigmoid_in_fp32
                                             ? BFloat16(sigmoid)
                                             : (gate > 0 ? reciprocal : BFloat16(1.0f - reciprocal.ToFloat()));
        const BFloat16 activation = silu_gate ? BFloat16(gate * rounded_sigmoid.ToFloat()) : rounded_sigmoid;
        const BFloat16 product(values[i].ToFloat() * activation.ToFloat());
        expected[i] = residual ? BFloat16(product.ToFloat() + residuals[i].ToFloat()) : product;
      }
      OpTester test("SigmoidGateMul", 1, kMSDomain);
      test.AddAttribute<int64_t>(silu_gate ? "silu_gate" : "broadcast_gate", 1);
      test.AddAttribute<int64_t>("sigmoid_in_fp32", sigmoid_in_fp32 ? 1 : 0);
      test.AddInput<BFloat16>("attention", {2, 3, width}, values);
      test.AddInput<BFloat16>("projection", {2, 3, silu_gate ? width : 1}, gates);
      if (residual) test.AddInput<BFloat16>("residual", {2, 3, width}, residuals);
      test.AddOutput<BFloat16>("output", {2, 3, width}, expected);
      test.SetOutputTolerance(0.0f, 0.0f);
      test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
    }
  }
}

TEST(FusedTensorOpsTest, SigmoidGateMulBroadcastResidualRounding) {
  RunSigmoidGateMulBroadcastResidualRounding(true);
}

TEST(FusedTensorOpsTest, SigmoidGateMulBroadcastInputPrecision) {
  RunSigmoidGateMulBroadcastResidualRounding(false);
}

TEST(FusedTensorOpsTest, SigmoidGateMulSiluRounding) {
  RunSigmoidGateMulBroadcastResidualRounding(true, true);
  RunSigmoidGateMulBroadcastResidualRounding(false, true);
}

TEST(FusedTensorOpsTest, SigmoidGateMulRejectsBroadcastAndResidualShapes) {
  for (bool bad_residual : {false, true}) {
    auto providers = CudaProvider();
    if (providers.empty()) GTEST_SKIP();
    OpTester test("SigmoidGateMul", 1, kMSDomain);
    test.AddAttribute<int64_t>("broadcast_gate", 1);
    test.AddInput<BFloat16>("attention", {2, 3}, ToBFloat16(std::vector<float>(6, 1.0f)));
    test.AddInput<BFloat16>("projection", bad_residual ? std::vector<int64_t>{2, 1} : std::vector<int64_t>{1, 2},
                            ToBFloat16({1.0f, 1.0f}));
    if (bad_residual) test.AddInput<BFloat16>("residual", {1}, ToBFloat16({1.0f}));
    test.AddOutput<BFloat16>("output", {2, 3}, ToBFloat16(std::vector<float>(6, 0.0f)));
    test.Run(OpTester::ExpectResult::kExpectFailure,
             bad_residual ? "residual must have the same shape" : "singleton final dimension",
             {}, nullptr, &providers);
  }
}

TEST(FusedTensorOpsTest, SigmoidGateMulBroadcastEmptyAndInvalidAttributes) {
  for (int mode = 0; mode < 4; ++mode) {
    auto providers = CudaProvider();
    if (providers.empty()) GTEST_SKIP();
    OpTester test("SigmoidGateMul", 1, kMSDomain);
    test.AddAttribute<int64_t>("broadcast_gate", mode == 1 ? 2 : 1);
    if (mode == 2) test.AddAttribute<int64_t>("sigmoid_in_fp32", 2);
    if (mode == 3) test.AddAttribute<int64_t>("head_dim", 2);
    test.AddInput<BFloat16>("attention", {0, 3}, {});
    test.AddInput<BFloat16>("projection", {0, 1}, {});
    test.AddInput<BFloat16>("residual", {0, 3}, {});
    test.AddOutput<BFloat16>("output", {0, 3}, {});
    const std::vector<std::string> errors = {"", "broadcast_gate must be zero or one",
                                             "sigmoid_in_fp32 must be zero or one",
                                             "cannot be combined with packed layout attributes"};
    test.Run(mode == 0 ? OpTester::ExpectResult::kExpectSuccess : OpTester::ExpectResult::kExpectFailure,
             errors[mode], {}, nullptr, &providers);
  }
}

TEST(FusedTensorOpsTest, SigmoidGateMulSiluEmptyAndInvalidAttributes) {
  for (int mode = 0; mode < 5; ++mode) {
    auto providers = CudaProvider();
    if (providers.empty()) GTEST_SKIP();
    OpTester test("SigmoidGateMul", 1, kMSDomain);
    test.AddAttribute<int64_t>("silu_gate", mode == 1 ? 2 : 1);
    if (mode == 2) test.AddAttribute<int64_t>("broadcast_gate", 1);
    if (mode == 3) test.AddAttribute<int64_t>("head_dim", 2);
    test.AddInput<BFloat16>("attention", {0, 3}, {});
    test.AddInput<BFloat16>("projection", {0, mode == 4 ? 1 : 3}, {});
    test.AddOutput<BFloat16>("output", {0, 3}, {});
    const std::vector<std::string> errors = {"", "silu_gate must be zero or one", "are exclusive",
                                             "cannot be combined with packed layout attributes",
                                             "silu gate must have the same shape"};
    test.Run(mode == 0 ? OpTester::ExpectResult::kExpectSuccess : OpTester::ExpectResult::kExpectFailure,
             errors[mode], {}, nullptr, &providers);
  }
}

#if !defined(DISABLE_FLOAT8_TYPES)
TEST(FusedTensorOpsTest, DequantizeLinearFp8ToBf16) {
  auto providers = CudaProvider();
  if (providers.empty()) GTEST_SKIP();

  OpTester test("DequantizeLinear", 1, kMSDomain);
  test.AddAttribute<int64_t>("axis", 1);
  test.AddAttribute<int64_t>("output_dtype", ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16);
  test.AddInput<Float8E4M3FN>("weight", {1, 8}, std::vector<Float8E4M3FN>(8, Float8E4M3FN(1.0f)));
  test.AddInput<float>("scale", {8}, std::vector<float>(8, 2.0f));
  test.AddOptionalInputEdge<Float8E4M3FN>();
  test.AddOutput<BFloat16>("weight_bf16", {1, 8}, ToTensorType<BFloat16>(std::vector<float>(8, 2.0f)));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}
#endif

TEST(FusedTensorOpsTest, UntileHyperConnectionWeights) {
  constexpr int64_t kInputElements = 160ll * 20480;
  constexpr int64_t kVectorCount = kInputElements / 16;
  for (const int64_t kind : {0, 1}) {
    auto providers = CudaProvider();
    if (providers.empty()) GTEST_SKIP();
    OpTester test("UntileHyperConnectionWeights", 1, kMSDomain);
    test.AddAttribute<int64_t>("kind", kind);
    std::vector<BFloat16> input(kInputElements);
    std::vector<BFloat16> expected(kInputElements);
    for (int64_t i = 0; i < kInputElements; ++i) {
      input[i] = BFloat16(static_cast<float>(i % 127));
    }
    for (int64_t vector_index = 0; vector_index < kVectorCount; ++vector_index) {
      int64_t destination_vector;
      if (kind == 0) {
        const int64_t rp = vector_index & 127;
        const int64_t t1 = vector_index >> 7;
        const int64_t i = t1 % 10;
        const int64_t t2 = t1 / 10;
        const int64_t j = t2 % 20;
        const int64_t s = t2 / 20;
        destination_vector = (((s * 10 + i) * 128 + rp) * 20 + j);
      } else {
        const int64_t c = vector_index & 3;
        const int64_t t1 = vector_index >> 2;
        const int64_t rp = t1 & 31;
        const int64_t t2 = t1 >> 5;
        const int64_t i = t2 % 10;
        const int64_t hb = t2 / 10;
        destination_vector = (((i * 32 + rp) * 4 + c) * 160 + hb);
      }
      for (int64_t lane = 0; lane < 16; ++lane) {
        expected[destination_vector * 16 + lane] = input[vector_index * 16 + lane];
      }
    }
    test.AddInput<BFloat16>("tiled", {160, 20480}, input);
    const std::vector<int64_t> output_shape = kind == 0
                                                  ? std::vector<int64_t>{10240, 320}
                                                  : std::vector<int64_t>{320, 10240};
    test.AddOutput<BFloat16>("untiled", output_shape, expected);
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
  }
}

}  // namespace
}  // namespace onnxruntime::test
