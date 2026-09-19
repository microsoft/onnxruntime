// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <memory>
#include <type_traits>
#include <vector>

#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime::test {
namespace {

std::vector<std::unique_ptr<IExecutionProvider>> AvailableProviders() {
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCpuExecutionProvider());
  if (auto cuda = DefaultCudaExecutionProvider()) {
    providers.push_back(std::move(cuda));
  }
  if (auto webgpu = DefaultWebGpuExecutionProvider()) {
    providers.push_back(std::move(webgpu));
  }
  return providers;
}

template <typename AddInputs>
void RunOnAvailableProviders(const char* op_name, AddInputs add_inputs) {
  for (auto& provider : AvailableProviders()) {
    SCOPED_TRACE(provider->Type());
    OpTester tester(op_name, 1, kMSDomain);
    add_inputs(tester);
    std::vector<std::unique_ptr<IExecutionProvider>> providers;
    providers.push_back(std::move(provider));
    tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr,
               &providers);
  }
}

template <typename T>
float RoundToType(float value) {
  if constexpr (std::is_same_v<T, MLFloat16>) {
    return MLFloat16(value).ToFloat();
  } else {
    return value;
  }
}

TEST(HyperConnectionOpsTest, BranchwiseRMSNormGroupedSharedScale) {
  const std::vector<float> x = {3.0f, 4.0f, 0.0f, 2.0f};
  const float inv_first = 1.0f / std::sqrt(12.5f + 1e-5f);
  const float inv_second = 1.0f / std::sqrt(2.0f + 1e-5f);
  const std::vector<float> expected = {
      3.0f * inv_first * 2.0f, 4.0f * inv_first * 0.5f,
      0.0f, 2.0f * inv_second * 0.5f};
  RunOnAvailableProviders("BranchwiseRMSNorm", [&](OpTester& tester) {
    tester.AddInput<float>("X", {1, 2, 2}, x);
    tester.AddInput<float>("scale", {2}, {2.0f, 0.5f});
    tester.AddOutput<float>("Y", {1, 2, 2}, expected, false, 1e-5f, 1e-5f);
  });
}

TEST(HyperConnectionOpsTest, BranchwiseRMSNormFlattenedWithoutScale) {
  const std::vector<float> x = {3.0f, 4.0f, 0.0f, 2.0f};
  const float inv_first = 1.0f / std::sqrt(12.5f + 1e-5f);
  const float inv_second = 1.0f / std::sqrt(2.0f + 1e-5f);
  RunOnAvailableProviders("BranchwiseRMSNorm", [&](OpTester& tester) {
    tester.AddAttribute<int64_t>("num_branches", 2);
    tester.AddInput<float>("X", {1, 4}, x);
    tester.AddOptionalInputEdge<float>();
    tester.AddOutput<float>(
        "Y", {1, 4},
        {3.0f * inv_first, 4.0f * inv_first, 0.0f, 2.0f * inv_second},
        false, 1e-5f, 1e-5f);
  });
}

TEST(HyperConnectionOpsTest, BranchwiseRMSNormGroupedBranchScale) {
  const std::vector<float> x = {3.0f, 4.0f, 0.0f, 2.0f};
  const float inv_first = 1.0f / std::sqrt(12.5f + 1e-5f);
  const float inv_second = 1.0f / std::sqrt(2.0f + 1e-5f);
  RunOnAvailableProviders("BranchwiseRMSNorm", [&](OpTester& tester) {
    tester.AddInput<float>("X", {1, 2, 2}, x);
    tester.AddInput<float>("scale", {2, 2}, {2.0f, 0.5f, 0.25f, 3.0f});
    tester.AddOutput<float>(
        "Y", {1, 2, 2},
        {3.0f * inv_first * 2.0f, 4.0f * inv_first * 0.5f,
         0.0f, 2.0f * inv_second * 3.0f},
        false, 1e-5f, 1e-5f);
  });
}

TEST(HyperConnectionOpsTest, BranchwiseRMSNormMixedScaleType) {
  const std::vector<float> x = {3.0f, 4.0f, 0.0f, 2.0f};
  const float inv_first = 1.0f / std::sqrt(12.5f + 1e-5f);
  const float inv_second = 1.0f / std::sqrt(2.0f + 1e-5f);
  RunOnAvailableProviders("BranchwiseRMSNorm", [&](OpTester& tester) {
    tester.AddInput<MLFloat16>("X", {1, 2, 2}, ToFloat16(x));
    tester.AddInput<float>("scale", {4}, {2.0f, 0.5f, 0.25f, 3.0f});
    tester.AddOutput<MLFloat16>(
        "Y", {1, 2, 2},
        ToFloat16({3.0f * inv_first * 2.0f, 4.0f * inv_first * 0.5f,
                   0.0f, 2.0f * inv_second * 3.0f}));
  });
}

TEST(HyperConnectionOpsTest, ScaledSiLU) {
  RunOnAvailableProviders("ScaledSiLU", [&](OpTester& tester) {
    tester.AddAttribute<float>("alpha", 0.5f);
    tester.AddInput<float>("X", {3}, {-2.0f, 0.0f, 2.0f});
    tester.AddInput<float>("scale", {}, {0.25f});
    tester.AddOutput<float>(
        "Y", {3},
        {-0.5f / (1.0f + std::exp(0.5f)), 0.0f,
         0.5f / (1.0f + std::exp(-0.5f))},
        false, 1e-6f, 1e-6f);
  });
}

TEST(HyperConnectionOpsTest, ScaledSiLUDefaultScaleAndAlpha) {
  RunOnAvailableProviders("ScaledSiLU", [&](OpTester& tester) {
    tester.AddAttribute<float>("alpha", 0.5f);
    tester.AddInput<float>("X", {2}, {-1.0f, 2.0f});
    tester.AddOptionalInputEdge<float>();
    tester.AddOutput<float>(
        "Y", {2},
        {-0.5f / (1.0f + std::exp(0.5f)),
         1.0f / (1.0f + std::exp(-1.0f))},
        false, 1e-6f, 1e-6f);
  });
}

TEST(HyperConnectionOpsTest, ScaledSiLUFloat16Rounding) {
  constexpr float alpha = 0.7f;
  const std::vector<float> x = {-2.125f, 0.3333f, 1.337f};
  const float scale = RoundToType<MLFloat16>(0.731f);
  std::vector<float> expected;
  for (const float value : x) {
    const float input = RoundToType<MLFloat16>(value);
    const float z = RoundToType<MLFloat16>(input * scale);
    const float sigmoid =
        RoundToType<MLFloat16>(1.0f / (1.0f + std::exp(-z)));
    expected.push_back(RoundToType<MLFloat16>(z * sigmoid));
  }
  RunOnAvailableProviders("ScaledSiLU", [&](OpTester& tester) {
    tester.AddAttribute<float>("alpha", alpha);
    tester.AddInput<MLFloat16>("X", {3}, ToFloat16(x));
    tester.AddInput<MLFloat16>("scale", {}, ToFloat16({scale}));
    tester.AddOutput<MLFloat16>("Y", {3}, ToFloat16(expected));
  });
}

TEST(HyperConnectionOpsTest, PreMixGroupedBranchGates) {
  RunOnAvailableProviders("HyperConnectionPreMix", [&](OpTester& tester) {
    tester.AddAttribute<float>("reduction_scale", 0.5f);
    tester.AddInput<float>("streams", {1, 2, 2},
                           {1.0f, 2.0f, 3.0f, 4.0f});
    tester.AddInput<float>("pre_mix", {1, 2}, {2.0f, -1.0f});
    tester.AddOutput<float>("output", {1, 2}, {-0.5f, 0.0f});
  });
}

TEST(HyperConnectionOpsTest, PreMixFlattenedFeatureGates) {
  RunOnAvailableProviders("HyperConnectionPreMix", [&](OpTester& tester) {
    tester.AddAttribute<int64_t>("num_branches", 2);
    tester.AddInput<float>("streams", {1, 4},
                           {1.0f, 2.0f, 3.0f, 4.0f});
    tester.AddInput<float>("pre_mix", {1, 4},
                           {1.0f, 0.5f, -1.0f, 2.0f});
    tester.AddOutput<float>("output", {1, 2}, {-2.0f, 9.0f});
  });
}

TEST(HyperConnectionOpsTest, PreMixMixedInputTypes) {
  RunOnAvailableProviders("HyperConnectionPreMix", [&](OpTester& tester) {
    tester.AddInput<MLFloat16>("streams", {1, 2, 2},
                               ToFloat16({1.0f, 2.0f, 3.0f, 4.0f}));
    tester.AddInput<float>("pre_mix", {1, 2}, {2.0f, -1.0f});
    tester.AddOutput<MLFloat16>("output", {1, 2}, ToFloat16({-1.0f, 0.0f}));
  });
}

TEST(HyperConnectionOpsTest, PostMixIdentityAndBranchGates) {
  RunOnAvailableProviders("HyperConnectionPostMix", [&](OpTester& tester) {
    tester.AddInput<float>("streams", {1, 2, 2},
                           {1.0f, 2.0f, 3.0f, 4.0f});
    tester.AddInput<float>("block_output", {1, 2}, {10.0f, 20.0f});
    tester.AddInput<float>("post_mix", {1, 2, 1}, {0.5f, -1.0f});
    tester.AddOptionalInputEdge<float>();
    tester.AddOutput<float>("output", {1, 2, 2},
                            {6.0f, 12.0f, -7.0f, -16.0f});
  });
}

TEST(HyperConnectionOpsTest, PostMixFeatureGatesAndStreamMatrix) {
  RunOnAvailableProviders("HyperConnectionPostMix", [&](OpTester& tester) {
    tester.AddInput<float>("streams", {1, 2, 2},
                           {1.0f, 2.0f, 3.0f, 4.0f});
    tester.AddInput<float>("block_output", {1, 2}, {10.0f, 20.0f});
    tester.AddInput<float>("post_mix", {1, 2, 2},
                           {0.1f, 0.2f, 0.3f, 0.4f});
    tester.AddInput<float>("stream_mix", {1, 2, 2},
                           {1.0f, 2.0f, 3.0f, 4.0f});
    tester.AddOutput<float>("output", {1, 2, 2},
                            {11.0f, 18.0f, 17.0f, 28.0f});
  });
}

TEST(HyperConnectionOpsTest, PostMixFlattenedBranchGate) {
  RunOnAvailableProviders("HyperConnectionPostMix", [&](OpTester& tester) {
    tester.AddAttribute<int64_t>("num_branches", 2);
    tester.AddInput<float>("streams", {1, 4},
                           {1.0f, 2.0f, 3.0f, 4.0f});
    tester.AddInput<float>("block_output", {1, 2}, {10.0f, 20.0f});
    tester.AddInput<float>("post_mix", {1, 2}, {0.5f, 0.5f});
    tester.AddOptionalInputEdge<float>();
    tester.AddOutput<float>("output", {1, 4}, {6.0f, 12.0f, 8.0f, 14.0f});
  });
}

}  // namespace
}  // namespace onnxruntime::test
