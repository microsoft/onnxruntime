// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <type_traits>

#include "gtest/gtest.h"

#include "core/common/inlined_containers.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

#if !defined(DISABLE_CONTRIB_OPS)

namespace onnxruntime {
namespace test {
namespace {

template <typename T, typename V>
void RunLayerNormTest(bool simplified, bool has_bias) {
  // Cover scalar, vec2, vec4, multiple elements per thread, and split normalization.
  for (const auto& dims : {TensorShapeVector{2, 3}, {2, 6}, {2, 128}, {2, 260}, {2, 1024}, {1, 1024}}) {
    SCOPED_TRACE(MakeString("shape=", TensorShape(dims), ", simplified=", simplified, ", bias=", has_bias));
    auto webgpu_ep = DefaultWebGpuExecutionProvider();
    if (!webgpu_ep) {
      GTEST_SKIP() << "WebGPU execution provider is not available.";
    }

    constexpr float epsilon = 1e-5f;
    const int64_t norm_count = dims[0];
    const int64_t norm_size = dims[1];
    InlinedVector<T> input;
    InlinedVector<V> scale;
    InlinedVector<V> bias;
    InlinedVector<V> expected;
    InlinedVector<float> mean;
    InlinedVector<float> inv_std_dev;
    input.reserve(norm_count * norm_size);
    expected.reserve(norm_count * norm_size);
    scale.reserve(norm_size);
    bias.reserve(norm_size);
    mean.reserve(norm_count);
    inv_std_dev.reserve(norm_count);

    for (int64_t j = 0; j < norm_size; ++j) {
      scale.emplace_back(0.5f * static_cast<float>(1 + j % 3));
      bias.emplace_back(0.125f * static_cast<float>(j % 5 - 2));
    }

    for (int64_t n = 0; n < norm_count; ++n) {
      double sum = 0.0;
      for (int64_t j = 0; j < norm_size; ++j) {
        input.emplace_back(0.25f * static_cast<float>(j % 17 - 8) + 0.5f * static_cast<float>(n));
        sum += static_cast<float>(input.back());
      }
      const double row_mean = simplified ? 0.0 : sum / norm_size;
      double squared_deviation = 0.0;
      for (int64_t j = 0; j < norm_size; ++j) {
        const double deviation = static_cast<float>(input[n * norm_size + j]) - row_mean;
        squared_deviation += deviation * deviation;
      }
      const double row_inv_std_dev = 1.0 / std::sqrt(squared_deviation / norm_size + epsilon);
      mean.push_back(static_cast<float>(row_mean));
      inv_std_dev.push_back(static_cast<float>(row_inv_std_dev));
      for (int64_t j = 0; j < norm_size; ++j) {
        const V normalized(static_cast<float>((static_cast<float>(input[n * norm_size + j]) - row_mean) * row_inv_std_dev));
        expected.emplace_back(static_cast<float>(normalized) * static_cast<float>(scale[j]) +
                              (has_bias ? static_cast<float>(bias[j]) : 0.0f));
      }
    }

    OpTester test(simplified ? "SimplifiedLayerNormalization" : "LayerNormalization", 15);
    test.AddAttribute("epsilon", epsilon);
    test.AddInput<T>("X", dims, input.data(), input.size());
    test.AddInput<V>("Scale", {norm_size}, scale.data(), scale.size(), true);
    if (has_bias) {
      test.AddInput<V>("B", {norm_size}, bias.data(), bias.size(), true);
    }
    test.AddOutput<V>("Y", dims, expected.data(), expected.size());
    constexpr float tolerance = std::is_same_v<V, MLFloat16> ? 0.002f : 1e-5f;
    test.SetOutputAbsErr("Y", tolerance);
    test.SetOutputRelErr("Y", tolerance);
    if (!simplified) {
      test.AddOutput<float>("Mean", {norm_count, 1}, mean.data(), mean.size());
      test.AddOutput<float>("InvStdDev", {norm_count, 1}, inv_std_dev.data(), inv_std_dev.size());
    }
    SessionOptions session_options;
    ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
    test.Config(session_options).ConfigEp(std::move(webgpu_ep)).RunWithConfig();
    ASSERT_EQ(test.GetFetches().size(), simplified ? 1u : 3u);
  }
}

template <typename T, typename V>
void RunLayerNormTests() {
  RunLayerNormTest<T, V>(false, false);
  RunLayerNormTest<T, V>(false, true);
  RunLayerNormTest<T, V>(true, false);
}

}  // namespace

TEST(LayerNorm_WebGPU, Float16InputFloat32Output) {
  RunLayerNormTests<MLFloat16, float>();
}

TEST(LayerNorm_WebGPU, Float32InputFloat16Output) {
  RunLayerNormTests<float, MLFloat16>();
}

TEST(LayerNorm_WebGPU, Float16) {
  RunLayerNormTests<MLFloat16, MLFloat16>();
}

TEST(LayerNorm_WebGPU, Float32) {
  RunLayerNormTests<float, float>();
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(DISABLE_CONTRIB_OPS)
