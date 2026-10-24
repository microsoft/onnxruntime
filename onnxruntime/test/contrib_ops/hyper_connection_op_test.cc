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

std::vector<std::unique_ptr<IExecutionProvider>> AvailableProviders(bool include_webgpu = true) {
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCpuExecutionProvider());
  if (auto cuda = DefaultCudaExecutionProvider()) {
    providers.push_back(std::move(cuda));
  }
  if (include_webgpu) {
    if (auto webgpu = DefaultWebGpuExecutionProvider()) {
      providers.push_back(std::move(webgpu));
    }
  }
  return providers;
}

template <typename AddInputs>
void RunOnAvailableProviders(const char* op_name, AddInputs add_inputs, bool include_webgpu = true) {
  for (auto& provider : AvailableProviders(include_webgpu)) {
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
  } else if constexpr (std::is_same_v<T, BFloat16>) {
    return BFloat16(value).ToFloat();
  } else {
    return value;
  }
}

template <typename T>
std::vector<T> ConvertVector(const std::vector<float>& values) {
  if constexpr (std::is_same_v<T, MLFloat16>) {
    return ToFloat16(values);
  } else if constexpr (std::is_same_v<T, BFloat16>) {
    return ToBFloat16(values);
  } else {
    return values;
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

TEST(HyperConnectionOpsTest, BranchwiseRMSNormBranchScaleFloat32Overflow) {
  RunOnAvailableProviders("BranchwiseRMSNorm", [&](OpTester& tester) {
    tester.AddInput<float>("X", {1, 2, 1}, {1e20f, 1e20f});
    tester.AddInput<float>("scale", {2, 1}, {1.0f, 1.0f});
    tester.AddOutput<float>("Y", {1, 2, 1}, {0.0f, 0.0f});
  });
}

TEST(HyperConnectionOpsTest, BranchwiseRMSNormSharedScaleFloat16Precision) {
  constexpr float small_value = 1.0f / 16384.0f;
  constexpr float large_value = 65504.0f;
  const float inv_rms =
      1.0f / std::sqrt((small_value * small_value + large_value * large_value) / 2.0f + 1e-5f);
  RunOnAvailableProviders("BranchwiseRMSNorm", [&](OpTester& tester) {
    tester.AddInput<MLFloat16>("X", {1, 1, 2}, ToFloat16({small_value, large_value}));
    tester.AddInput<MLFloat16>("scale", {2}, ToFloat16({large_value, 0.0f}));
    tester.AddOutput<MLFloat16>("Y", {1, 1, 2},
                                ToFloat16({small_value * inv_rms * large_value, 0.0f}));
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

template <typename StreamT, typename MixT>
void RunPreMixVecCase(bool feature_gates, bool flattened, float reduction_scale) {
  constexpr int64_t rows = 3;
  constexpr int64_t branches = 3;
  constexpr int64_t hidden = 16;  // Divisible by 8 -> exercises PreMixVecKernel (2 vecs/row).
  constexpr bool kIncludeWebGpu = !std::is_same_v<StreamT, BFloat16> && !std::is_same_v<MixT, BFloat16>;

  std::vector<float> streams(rows * branches * hidden);
  for (size_t i = 0; i < streams.size(); ++i) {
    const float raw = 0.125f * static_cast<float>((static_cast<int>(i * 7 + 3) % 17) - 8);
    streams[i] = RoundToType<StreamT>(raw);
  }

  const size_t mix_size = feature_gates ? (rows * branches * hidden) : (rows * branches);
  std::vector<float> pre_mix(mix_size);
  for (size_t i = 0; i < pre_mix.size(); ++i) {
    const float raw = 0.25f * static_cast<float>((static_cast<int>(i * 5 + 1) % 11) - 5);
    pre_mix[i] = RoundToType<MixT>(raw);
  }

  std::vector<float> expected(rows * hidden, 0.0f);
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t h = 0; h < hidden; ++h) {
      float sum = 0.0f;
      for (int64_t c = 0; c < branches; ++c) {
        const size_t x_idx = static_cast<size_t>((r * branches + c) * hidden + h);
        const size_t g_idx = feature_gates ? x_idx : static_cast<size_t>(r * branches + c);
        sum += streams[x_idx] * pre_mix[g_idx];
      }
      expected[static_cast<size_t>(r * hidden + h)] = RoundToType<StreamT>(sum * reduction_scale);
    }
  }

  RunOnAvailableProviders(
      "HyperConnectionPreMix",
      [&](OpTester& tester) {
        if (flattened) {
          tester.AddAttribute<int64_t>("num_branches", branches);
        }
        if (reduction_scale != 1.0f) {
          tester.AddAttribute<float>("reduction_scale", reduction_scale);
        }
        const std::vector<int64_t> stream_shape =
            flattened ? std::vector<int64_t>{rows, branches * hidden}
                      : std::vector<int64_t>{rows, branches, hidden};
        std::vector<int64_t> mix_shape;
        if (feature_gates) {
          mix_shape = flattened ? std::vector<int64_t>{rows, branches * hidden}
                                : std::vector<int64_t>{rows, branches, hidden};
        } else {
          mix_shape = {rows, branches};
        }
        tester.AddInput<StreamT>("streams", stream_shape, ConvertVector<StreamT>(streams));
        tester.AddInput<MixT>("pre_mix", mix_shape, ConvertVector<MixT>(pre_mix));
        tester.AddOutput<StreamT>("output", {rows, hidden}, ConvertVector<StreamT>(expected));
      },
      kIncludeWebGpu);
}

TEST(HyperConnectionOpsTest, PreMixVectorizedFloat16AndBFloat16) {
  RunPreMixVecCase<MLFloat16, MLFloat16>(/*feature_gates=*/false, /*flattened=*/false, /*reduction_scale=*/0.375f);
  RunPreMixVecCase<MLFloat16, float>(/*feature_gates=*/true, /*flattened=*/false, /*reduction_scale=*/0.25f);
  RunPreMixVecCase<MLFloat16, BFloat16>(/*feature_gates=*/true, /*flattened=*/true, /*reduction_scale=*/0.5f);
  RunPreMixVecCase<BFloat16, BFloat16>(/*feature_gates=*/false, /*flattened=*/false, /*reduction_scale=*/0.375f);
  RunPreMixVecCase<BFloat16, float>(/*feature_gates=*/true, /*flattened=*/false, /*reduction_scale=*/0.25f);
  RunPreMixVecCase<BFloat16, MLFloat16>(/*feature_gates=*/true, /*flattened=*/true, /*reduction_scale=*/1.0f);
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

template <typename StreamT, typename MixT>
void RunPostMixVecCase(int64_t branches, bool feature_gates, bool with_stream_mix, bool flattened) {
  constexpr int64_t rows = 3;
  constexpr int64_t hidden = 16;  // Divisible by 8 -> exercises PostMixVecKernel (2 vecs/row).
  constexpr bool kIncludeWebGpu = !std::is_same_v<StreamT, BFloat16> && !std::is_same_v<MixT, BFloat16>;

  std::vector<float> streams(rows * branches * hidden);
  for (size_t i = 0; i < streams.size(); ++i) {
    const float raw = 0.125f * static_cast<float>((static_cast<int>(i * 11 + 5) % 19) - 9);
    streams[i] = RoundToType<StreamT>(raw);
  }

  std::vector<float> block_output(rows * hidden);
  for (size_t i = 0; i < block_output.size(); ++i) {
    const float raw = 0.25f * static_cast<float>((static_cast<int>(i * 3 + 2) % 13) - 6);
    block_output[i] = RoundToType<StreamT>(raw);
  }

  const size_t post_mix_size = feature_gates ? (rows * branches * hidden) : (rows * branches);
  std::vector<float> post_mix(post_mix_size);
  for (size_t i = 0; i < post_mix.size(); ++i) {
    const float raw = 0.125f * static_cast<float>((static_cast<int>(i * 7 + 1) % 15) - 7);
    post_mix[i] = RoundToType<MixT>(raw);
  }

  std::vector<float> stream_mix(rows * branches * branches);
  for (size_t i = 0; i < stream_mix.size(); ++i) {
    // Asymmetric per-(row, src, dst) weights to verify source reuse and output-branch indexing.
    const float raw = 0.25f * static_cast<float>((static_cast<int>(i * 9 + 4) % 11) - 5);
    stream_mix[i] = RoundToType<MixT>(raw);
  }

  std::vector<float> expected(rows * branches * hidden, 0.0f);
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t c = 0; c < branches; ++c) {
      for (int64_t h = 0; h < hidden; ++h) {
        const size_t out_idx = static_cast<size_t>((r * branches + c) * hidden + h);
        float value = with_stream_mix ? 0.0f : streams[out_idx];
        if (with_stream_mix) {
          for (int64_t src = 0; src < branches; ++src) {
            const size_t mix_idx = static_cast<size_t>((r * branches + src) * branches + c);
            const size_t src_idx = static_cast<size_t>((r * branches + src) * hidden + h);
            value += stream_mix[mix_idx] * streams[src_idx];
          }
        }
        const size_t g_idx = feature_gates ? out_idx : static_cast<size_t>(r * branches + c);
        value += post_mix[g_idx] * block_output[static_cast<size_t>(r * hidden + h)];
        expected[out_idx] = RoundToType<StreamT>(value);
      }
    }
  }

  RunOnAvailableProviders(
      "HyperConnectionPostMix",
      [&](OpTester& tester) {
        if (flattened) {
          tester.AddAttribute<int64_t>("num_branches", branches);
        }
        const std::vector<int64_t> stream_shape =
            flattened ? std::vector<int64_t>{rows, branches * hidden}
                      : std::vector<int64_t>{rows, branches, hidden};
        const std::vector<int64_t> post_mix_shape =
            feature_gates ? std::vector<int64_t>{rows, branches, hidden}
                          : (flattened ? std::vector<int64_t>{rows, branches}
                                       : std::vector<int64_t>{rows, branches, 1});
        tester.AddInput<StreamT>("streams", stream_shape, ConvertVector<StreamT>(streams));
        tester.AddInput<StreamT>("block_output", {rows, hidden}, ConvertVector<StreamT>(block_output));
        tester.AddInput<MixT>("post_mix", post_mix_shape, ConvertVector<MixT>(post_mix));
        if (with_stream_mix) {
          tester.AddInput<MixT>("stream_mix", {rows, branches, branches}, ConvertVector<MixT>(stream_mix));
        } else {
          tester.AddOptionalInputEdge<MixT>();
        }
        tester.AddOutput<StreamT>("output", stream_shape, ConvertVector<StreamT>(expected));
      },
      kIncludeWebGpu);
}

TEST(HyperConnectionOpsTest, PostMixVectorizedFloat16AndBFloat16) {
  RunPostMixVecCase<MLFloat16, MLFloat16>(/*branches=*/3, /*feature_gates=*/false, /*with_stream_mix=*/false, /*flattened=*/false);
  RunPostMixVecCase<MLFloat16, float>(/*branches=*/4, /*feature_gates=*/true, /*with_stream_mix=*/true, /*flattened=*/false);
  RunPostMixVecCase<MLFloat16, BFloat16>(/*branches=*/3, /*feature_gates=*/false, /*with_stream_mix=*/true, /*flattened=*/true);
  RunPostMixVecCase<BFloat16, BFloat16>(/*branches=*/3, /*feature_gates=*/false, /*with_stream_mix=*/false, /*flattened=*/false);
  RunPostMixVecCase<BFloat16, float>(/*branches=*/4, /*feature_gates=*/true, /*with_stream_mix=*/true, /*flattened=*/false);
  RunPostMixVecCase<BFloat16, MLFloat16>(/*branches=*/3, /*feature_gates=*/true, /*with_stream_mix=*/false, /*flattened=*/true);
}

}  // namespace
}  // namespace onnxruntime::test
