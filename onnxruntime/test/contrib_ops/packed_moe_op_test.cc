// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "gtest/gtest.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
#include <algorithm>

namespace onnxruntime::test {
#if defined(USE_CUDA)
namespace {
template <typename Scale>
void TestReusedExperts(int rows) {
  constexpr int experts = 2, intermediate = 256, topk = 2;
  const int pairs = rows * topk;
  OpTester tester("MoEPackedIQ2XXS", 1, kMSDomain);
  tester.AddAttribute("topk", int64_t{topk});
  tester.AddInput<BFloat16>("X", {rows, 1024}, std::vector<BFloat16>(rows * 1024, BFloat16(0.03125f)));
  std::vector<int32_t> indices(pairs);
  std::vector<float> weights(pairs);
  for (int r = 0; r < rows; ++r) {
    indices[r * 2] = r & 1;
    indices[r * 2 + 1] = 1 - (r & 1);
    weights[r * 2] = (r & 1) ? 2.f : 0.5f;
    weights[r * 2 + 1] = (r & 1) ? 0.5f : 2.f;
  }
  tester.AddInput<int32_t>("indices", {rows, topk}, indices);
  tester.AddInput<float>("weights", {rows, topk}, weights);
  // Grid one is [43,8,8,8,8,8,8,8]. Scale 0.019 distinguishes full-operand from scale-first rounding.
  auto make_packed = [](int bytes) {
    std::vector<uint8_t> data(bytes, 0);
    for (int record = 0; record < bytes; record += 8) {
      for (int i = 0; i < 4; ++i) data[record + i] = 1;
    }
    return data;
  };
  tester.AddInput<uint8_t>("fc1", {experts, intermediate, 32, 8}, make_packed(experts * intermediate * 256));
  auto make_scale = [](int elements) {
    std::vector<Scale> data;
    data.reserve(elements * experts);
    for (int e = 0; e < experts; ++e) {
      const auto bits = MLFloat16(e == 0 ? 0.019f : 0.023f).val;
      data.insert(data.end(), elements, Scale::FromBits(bits));
    }
    return data;
  };
  tester.AddInput<Scale>("scale1", {experts, intermediate, 4}, make_scale(intermediate * 4));
  tester.AddInput<uint8_t>("fc2", {experts, 1024, 8, 8}, make_packed(experts * 1024 * 64));
  tester.AddInput<Scale>("scale2", {experts, 1024, 1}, make_scale(1024));
  std::vector<BFloat16> hidden(pairs * intermediate);
  std::vector<float> partial(pairs * 1024);
  float expert_hidden[experts], expert_partial[experts];
  for (int e = 0; e < experts; ++e) {
    const float d = MLFloat16(e == 0 ? 0.019f : 0.023f).ToFloat();
    const float sum8 = BFloat16(d * 0.125f * 43.f).ToFloat() + 7.f * BFloat16(d).ToFloat();
    const float fc1 = BFloat16(4.f * sum8).ToFloat();
    expert_hidden[e] = BFloat16(fc1 * fc1).ToFloat();
    expert_partial[e] = BFloat16(32.f * expert_hidden[e] * sum8).ToFloat();
  }
  for (int p = 0; p < pairs; ++p) {
    const int e = indices[p];
    std::fill_n(hidden.begin() + p * intermediate, intermediate, BFloat16(expert_hidden[e]));
    std::fill_n(partial.begin() + p * 1024, 1024, expert_partial[e]);
  }
  std::vector<int32_t> metadata(1 + 9 * pairs, -1);
  metadata[0] = 2;
  metadata[1] = 0;
  metadata[2] = 1;
  for (int p = 0; p < pairs; ++p) metadata[1 + pairs + indices[p] * 8 + p / topk] = p;
  const BFloat16 sum(0.5f * expert_partial[0] + 2.f * expert_partial[1]);
  tester.AddOutput<BFloat16>("Y", {rows, 1024}, std::vector<BFloat16>(rows * 1024, sum));
  tester.AddOutput<BFloat16>("hidden", {pairs, intermediate}, hidden);
  tester.AddOutput<float>("expert_output", {pairs, 1024}, partial);
  tester.AddOutput<int32_t>("metadata", {1 + 9 * pairs}, metadata);
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCudaExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}
}  // namespace

TEST(PackedMoETest, ReusedExpertsWeightedSumAndFP16ScaleBits) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  TestReusedExperts<BFloat16>(1);
  TestReusedExperts<BFloat16>(8);
  TestReusedExperts<MLFloat16>(3);
}
#endif
}  // namespace onnxruntime::test
