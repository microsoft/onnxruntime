// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "gtest/gtest.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace onnxruntime::test {
#if defined(USE_CUDA)
namespace {
void RunCuda(OpTester& tester) {
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCudaExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}
float Rounded(float x) { return BFloat16(x).ToFloat(); }

void TestMambaSnapshots(int rows, int selected, int heads = 2) {
  const int di = heads * 64, channels = di + 256, stride = di + channels + heads;
  const int slots = std::max(rows, selected + 1) + 1;
  std::vector<BFloat16> proj(rows * stride), conv(slots * channels * 3), cw(channels * 4);
  std::vector<float> state(static_cast<size_t>(slots) * heads * 128 * 64);
  for (int r = 0; r < rows; ++r) {
    for (int j = 0; j < di; ++j) proj[r * stride + j] = BFloat16(0.5f);
    for (int j = 0; j < channels; ++j) proj[r * stride + di + j] = BFloat16(0.25f + r * 0.03125f);
    for (int h = 0; h < heads; ++h) proj[r * stride + di + channels + h] = BFloat16(-0.25f + r * 0.03125f);
  }
  for (int s = 0; s < slots; ++s) {
    for (int j = 0; j < channels; ++j) {
      for (int k = 0; k < 3; ++k) conv[(s * channels + j) * 3 + k] = BFloat16((s + 1) * (k + 1) * 0.03125f);
    }
    std::fill_n(state.begin() + static_cast<size_t>(s) * heads * 128 * 64, heads * 128 * 64, (s + 1) * 0.015625f);
  }
  for (int j = 0; j < channels; ++j) {
    cw[j * 4] = BFloat16(0.125f);
    cw[j * 4 + 1] = BFloat16(0.25f);
    cw[j * 4 + 2] = BFloat16(0.5f);
    cw[j * 4 + 3] = BFloat16(1.f);
  }
  auto conv_expected = conv;
  auto state_expected = state;
  std::vector<BFloat16> xbc(rows * channels), y(rows * di);
  std::vector<float> readout(rows * di), st(heads, (selected + 1) * 0.015625f);
  float history[3] = {(selected + 1) * 0.03125f, (selected + 1) * 0.0625f, (selected + 1) * 0.09375f};
  for (int r = 0; r < rows; ++r) {
    const float input = Rounded(0.25f + r * 0.03125f);
    const float value = -0.125f + 0.125f * history[0] + 0.25f * history[1] + 0.5f * history[2] + input;
    const float x = Rounded(value / (1.f + std::exp(-value)));
    std::fill_n(xbc.begin() + r * channels, channels, BFloat16(x));
    history[0] = history[1];
    history[1] = history[2];
    history[2] = input;
    for (int j = 0; j < channels; ++j) {
      for (int k = 0; k < 3; ++k) conv_expected[(r * channels + j) * 3 + k] = BFloat16(history[k]);
    }
    const float dt = std::max(std::log1p(std::exp(-0.125f + r * 0.03125f)), 0.01f);
    float mean_square = 0.f;
    for (int h = 0; h < heads; ++h) {
      st[h] = std::fma(st[h], std::exp((-0.125f - h * 0.0078125f) * dt), (dt * x) * x);
      std::fill_n(state_expected.begin() + (static_cast<size_t>(r) * heads + h) * 128 * 64, 128 * 64, st[h]);
      float partial = 0.f;
      const int parts = rows == 1 ? 2 : 4;
      for (int n = 0; n < 128 / parts; ++n) partial = std::fma(x, st[h], partial);
      float result = 0.25f * x;
      for (int q = 0; q < parts; ++q) result += partial;
      std::fill_n(readout.begin() + r * di + h * 64, 64, result);
      const float gated = result * (0.5f / (1.f + std::exp(-0.5f)));
      mean_square += gated * gated / heads;
    }
    for (int j = 0; j < di; ++j) {
      const float gated = readout[r * di + j] * (0.5f / (1.f + std::exp(-0.5f)));
      y[r * di + j] = BFloat16(gated / std::sqrt(mean_square + 1e-5f) * 1.25f);
    }
  }
  OpTester tester("Mamba2Verify", 1, kMSDomain);
  tester.AddAttribute("heads", static_cast<int64_t>(heads));
  tester.AddAttribute("n_groups", int64_t{1});
  tester.AddAttribute("time_step_min", 0.01f);
  tester.AddInput<BFloat16>("projection", {rows, stride}, proj);
  tester.AddInput<BFloat16>("conv_state", {slots, channels, 3}, conv);
  tester.AddInput<float>("ssm_state", {slots, heads, 128, 64}, state);
  tester.AddInput<BFloat16>("conv_weight", {channels, 1, 4}, cw);
  tester.AddInput<BFloat16>("conv_bias", {channels}, std::vector<BFloat16>(channels, BFloat16(-0.125f)));
  tester.AddInput<float>("dt_bias", {heads}, std::vector<float>(heads, 0.125f));
  std::vector<float> decay(heads);
  for (int h = 0; h < heads; ++h) decay[h] = -0.125f - h * 0.0078125f;
  tester.AddInput<float>("A_negative", {heads}, decay);
  tester.AddInput<float>("D", {heads}, std::vector<float>(heads, 0.25f));
  tester.AddInput<float>("norm_weight", {di}, std::vector<float>(di, 1.25f));
  tester.AddInput<int32_t>("state_row", {1}, {selected});
  tester.AddOutput<BFloat16>("Y", {rows, di}, y);
  tester.AddOutput<BFloat16>("conv_snapshots", {slots, channels, 3}, conv_expected);
  tester.AddOutput<float>("ssm_snapshots", {slots, heads, 128, 64}, state_expected);
  tester.AddOutput<BFloat16>("xbc", {rows, channels}, xbc);
  tester.AddOutput<float>("readout", {rows, di}, readout);
  tester.SetOutputAbsErr("ssm_snapshots", 2e-5f);
  tester.SetOutputAbsErr("readout", 2e-4f);
  RunCuda(tester);
}
}  // namespace

TEST(FusedStateSpaceTest, MambaRecurrenceAndAcceptedSnapshot) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  TestMambaSnapshots(1, 3);
  TestMambaSnapshots(3, 1);
  TestMambaSnapshots(8, 7);
  TestMambaSnapshots(16, 5);
}

TEST(FusedStateSpaceTest, MambaEightRows1024WideNorm) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  TestMambaSnapshots(8, 3, 16);
}

TEST(FusedStateSpaceTest, RouterBiasOnlySelectionAndNonunitScale) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  OpTester tester("BiasCorrectedMoERoute", 1, kMSDomain);
  tester.AddAttribute("topk", int64_t{2});
  tester.AddAttribute("scale", 5.f);
  tester.AddInput<BFloat16>("X", {2, 256}, std::vector<BFloat16>(512, BFloat16(1.f)));
  std::vector<BFloat16> weight(4 * 256, BFloat16(0.f));
  weight[256] = BFloat16(1.f);
  weight[512] = BFloat16(-1.f);
  tester.AddInput<BFloat16>("W", {4, 256}, weight);
  tester.AddInput<float>("bias", {4}, {0.f, 0.f, 2.f, 0.f});
  const float high = 1.f / (1.f + std::exp(-1.f)), low = 1.f / (1.f + std::exp(1.f));
  tester.AddOutput<int32_t>("indices", {2, 2}, {2, 1, 2, 1});
  tester.AddOutput<float>("weights", {2, 2}, {low * 5.f, high * 5.f, low * 5.f, high * 5.f});
  tester.AddOutput<float>("scores", {2, 4}, {0.5f, high, low, 0.5f, 0.5f, high, low, 0.5f});
  RunCuda(tester);
}

TEST(FusedStateSpaceTest, RouterTiesPreferLowerExpert) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  OpTester tester("BiasCorrectedMoERoute", 1, kMSDomain);
  tester.AddAttribute("topk", int64_t{3});
  tester.AddInput<BFloat16>("X", {1, 256}, std::vector<BFloat16>(256, BFloat16(0.f)));
  tester.AddInput<BFloat16>("W", {40, 256}, std::vector<BFloat16>(40 * 256, BFloat16(0.f)));
  tester.AddInput<float>("bias", {40}, std::vector<float>(40, 0.f));
  tester.AddOutput<int32_t>("indices", {1, 3}, {0, 1, 2});
  tester.AddOutput<float>("weights", {1, 3}, {1.f / 3, 1.f / 3, 1.f / 3});
  tester.AddOutput<float>("scores", {1, 40}, std::vector<float>(40, 0.5f));
  RunCuda(tester);
}

TEST(FusedStateSpaceTest, ReluSquaredSingleRoundingAndEmpty) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  OpTester tester("ReluSquared", 1, kMSDomain);
  tester.AddInput<BFloat16>("X", {5}, {BFloat16(-2.f), BFloat16(0.f), BFloat16(1.125f), BFloat16(2.5f), BFloat16(std::numeric_limits<float>::quiet_NaN())});
  tester.AddOutput<BFloat16>("Y", {5}, {BFloat16(0.f), BFloat16(0.f), BFloat16(1.125f * 1.125f), BFloat16(6.25f), BFloat16(0.f)});
  RunCuda(tester);
  OpTester empty("ReluSquared", 1, kMSDomain);
  empty.AddInput<BFloat16>("X", {0}, {});
  empty.AddOutput<BFloat16>("Y", {0}, {});
  RunCuda(empty);
}
#endif
}  // namespace onnxruntime::test
