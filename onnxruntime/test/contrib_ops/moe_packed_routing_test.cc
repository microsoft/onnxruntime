// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

#if defined(USE_CUDA) && !defined(USE_CUDA_MINIMAL)
namespace onnxruntime::test {
namespace {

void RunMoeCuda(OpTester& test) {
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCudaExecutionProvider());
  test.SetOutputTolerance(0, 0);
  test.Run(SessionOptions{}, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

void TestPreselectedReluSquared(int rows, int topk, int malformed = 0) {
  constexpr int experts = 32;
  constexpr int hidden_size = 1024;
  constexpr int inter_size = 256;
  std::vector<uint8_t> gate(experts * inter_size * 264);
  std::vector<uint8_t> down(experts * hidden_size * 66);
  std::vector<BFloat16> input(rows * hidden_size), expected(rows * hidden_size);
  std::vector<int32_t> indices(rows * topk);
  std::vector<float> weights(rows * topk);
  std::vector<float> sum8(experts);
  for (int expert = 0; expert < experts; ++expert) {
    const MLFloat16 d(expert & 1 ? .023f : .019f);
    sum8[expert] = BFloat16(d.ToFloat() * .125f * 43.f).ToFloat() +
                   7.f * BFloat16(d.ToFloat()).ToFloat();
    for (int stage = 0; stage < 2; ++stage) {
      const int n = stage == 0 ? inter_size : hidden_size;
      const int blocks = stage == 0 ? hidden_size / 256 : inter_size / 256;
      auto& bank = stage == 0 ? gate : down;
      for (int col = 0; col < n; ++col) {
        for (int block = 0; block < blocks; ++block) {
          auto* bytes = bank.data() + ((expert * n + col) * blocks + block) * 66;
          bytes[0] = static_cast<uint8_t>(d.val);
          bytes[1] = static_cast<uint8_t>(d.val >> 8);
          for (int record = 0; record < 8; ++record) {
            for (int group = 0; group < 4; ++group) bytes[2 + record * 8 + group] = 1;
            if (stage == 0 && expert % 5 == 4) {
              for (int byte = 0; byte < 4; ++byte) {
                bytes[2 + record * 8 + 4 + byte] = static_cast<uint8_t>(0x0fffffffu >> (8 * byte));
              }
            }
          }
        }
      }
    }
  }
  for (int row = 0; row < rows; ++row) {
    const float factor = static_cast<float>(1 + (row & 1));
    for (int col = 0; col < hidden_size; ++col) input[row * hidden_size + col] = BFloat16(factor / 32.f);
    float mixed = 0.f;
    for (int rank = 0; rank < topk; ++rank) {
      const int expert = row & 1 ? (experts - 1 - rank) : rank;
      indices[row * topk + rank] = expert;
      const float weight = rank == 0 ? .333333f : .5001f;
      weights[row * topk + rank] = weight;
      const float fc1 = expert % 5 == 4 ? 0.f : BFloat16(4.f * factor * sum8[expert]).ToFloat();
      const float activation = BFloat16(fc1 * fc1).ToFloat();
      const float fc2 = BFloat16(32.f * activation * sum8[expert]).ToFloat();
      mixed = std::fma(weight, fc2, mixed);
    }
    for (int col = 0; col < hidden_size; ++col) expected[row * hidden_size + col] = BFloat16(mixed);
  }
  if (malformed) {
    if (malformed == 1) indices[0] = -1;
    if (malformed == 2) indices[1] = indices[0];
    if (malformed == 3) weights[0] = std::numeric_limits<float>::infinity();
    for (int col = 0; col < hidden_size; ++col) {
      expected[col] = BFloat16(std::numeric_limits<float>::quiet_NaN());
    }
  }
  OpTester test("MoEPacked", 1, kMSDomain);
  test.AddAttribute("k", static_cast<int64_t>(topk));
  test.AddAttribute("normalize_routing_weights", int64_t{0});
  test.AddAttribute("activation_type", std::string("relu_squared"));
  test.AddAttribute("gate_type", int64_t{16});
  test.AddAttribute("down_type", int64_t{16});
  test.AddInput<BFloat16>("input", {1, rows, hidden_size}, input);
  test.AddOptionalInputEdge<BFloat16>();
  test.AddInput<uint8_t>("gate", {experts, inter_size, 264}, gate, true);
  test.AddOptionalInputEdge<uint8_t>();
  test.AddInput<uint8_t>("down", {experts, hidden_size, 66}, down, true);
  test.AddInput<int32_t>("expert_indices", {rows, topk}, indices);
  test.AddInput<float>("expert_weights", {rows, topk}, weights);
  test.AddOutput<BFloat16>("output", {1, rows, hidden_size}, expected);
  RunMoeCuda(test);
}

TEST(MoEPackedRoutingTest, ReluSquaredPreservesPreselectedFp32Routing) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  for (int rows : {1, 8, 9, 17}) {
    for (int topk : {2, 32}) {
      SCOPED_TRACE(testing::Message() << "rows=" << rows << " k=" << topk);
      TestPreselectedReluSquared(rows, topk);
    }
  }
}

TEST(MoEPackedRoutingTest, MalformedRoutesPoisonOnlyTheirToken) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  for (int malformed : {1, 2, 3}) {
    SCOPED_TRACE(testing::Message() << "malformed=" << malformed);
    TestPreselectedReluSquared(2, 2, malformed);
  }
}

TEST(MoEPackedRoutingTest, ExistingSwigluRoutingDefaultsRemainUnchanged) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  constexpr int size = 32;
  constexpr int experts = 2;
  const BFloat16 hidden(.125f * (.25f / (1.f + std::exp(-.25f))));
  const BFloat16 output(16.f * hidden.ToFloat());
  for (int rows : {4, 48}) {
    const auto bank = [](float value) {
      std::vector<uint8_t> bytes(experts * size * size * 2);
      const BFloat16 element(value);
      for (size_t index = 0; index < bytes.size(); index += 2) {
        bytes[index] = static_cast<uint8_t>(element.val);
        bytes[index + 1] = static_cast<uint8_t>(element.val >> 8);
      }
      return bytes;
    };
    OpTester test("MoEPacked", 1, kMSDomain);
    test.AddAttribute("k", int64_t{1});
    test.AddAttribute("gate_type", int64_t{30});
    test.AddAttribute("up_type", int64_t{30});
    test.AddAttribute("down_type", int64_t{30});
    test.AddInput<BFloat16>("input", {rows, size}, std::vector<BFloat16>(rows * size, BFloat16(.03125f)));
    test.AddInput<BFloat16>("router_probs", {rows, experts}, std::vector<BFloat16>(rows * experts, BFloat16(0.f)));
    test.AddInput<uint8_t>("gate", {experts, size, size * 2}, bank(.25f), true);
    test.AddInput<uint8_t>("up", {experts, size, size * 2}, bank(.125f), true);
    test.AddInput<uint8_t>("down", {experts, size, size * 2}, bank(.5f), true);
    test.AddOutput<BFloat16>("output", {rows, size}, std::vector<BFloat16>(rows * size, output));
    RunMoeCuda(test);
  }
}

}  // namespace
}  // namespace onnxruntime::test
#endif
