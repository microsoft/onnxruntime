// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <type_traits>
#include <vector>

#include "gtest/gtest.h"
#include "core/providers/webgpu/webgpu_context.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {
namespace {

template <typename T>
float AsFloat(T value) {
  if constexpr (std::is_same_v<T, MLFloat16>) {
    return value.ToFloat();
  } else {
    return value;
  }
}

template <typename T>
void RunPrefill(int batch, int sequence, int q_heads, int v_heads, int k_heads,
                int dk, int dv, bool with_initial_state) {
  auto ep = DefaultWebGpuExecutionProvider();
  if (!ep) {
    GTEST_SKIP() << "WebGPU is unavailable";
  }
  if constexpr (std::is_same_v<T, MLFloat16>) {
    const auto& context = webgpu::WebGpuContextFactory::DefaultContext();
    const bool supported = context.DeviceHasFeature(wgpu::FeatureName::ShaderF16);
    webgpu::WebGpuContextFactory::ReleaseContext(0);
    if (!supported) {
      GTEST_SKIP() << "Requires shader-f16";
    }
  }

  const auto values = [](int count, float amplitude, float bias, float frequency) {
    std::vector<T> result;
    result.reserve(count);
    for (int i = 0; i < count; ++i) {
      result.emplace_back(bias + amplitude * std::sin(static_cast<float>(i + 1) * frequency));
    }
    return result;
  };
  auto q = values(batch * sequence * q_heads * dk, 0.1f, 0.0f, 0.137f);
  auto k = values(batch * sequence * k_heads * dk, 0.1f, 0.0f, 0.173f);
  auto v = values(batch * sequence * v_heads * dv, 0.5f, 0.0f, 0.191f);
  auto initial = values(batch * v_heads * dk * dv, 0.02f, 0.0f, 0.233f);
  auto decay = values(batch * sequence * v_heads, 0.1f, -0.12f, 0.311f);
  auto beta = values(batch * sequence * v_heads, 0.4f, 0.5f, 0.373f);
  const float scale = 1.0f / std::sqrt(static_cast<float>(dk));

  // Scalar reference on the actual rounded inputs. Compute q^T*S after the
  // update, independently of the shader's fused pre-update output reduction.
  std::vector<double> state(initial.size(), 0.0);
  if (with_initial_state) {
    for (size_t i = 0; i < initial.size(); ++i) state[i] = AsFloat(initial[i]);
  }
  std::vector<T> output(batch * sequence * v_heads * dv);
  for (int b = 0; b < batch; ++b) {
    for (int h = 0; h < v_heads; ++h) {
      const int qh = h * q_heads / v_heads;
      const int kh = h * k_heads / v_heads;
      for (int t = 0; t < sequence; ++t) {
        const int bt = b * sequence + t;
        const int gate = bt * v_heads + h;
        const double factor = std::exp(static_cast<double>(AsFloat(decay[gate])));
        for (int col = 0; col < dv; ++col) {
          double retrieved = 0.0;
          for (int row = 0; row < dk; ++row) {
            auto& s = state[((b * v_heads + h) * dk + row) * dv + col];
            s *= factor;
            retrieved += s * AsFloat(k[(bt * k_heads + kh) * dk + row]);
          }
          const double delta = AsFloat(beta[gate]) * (AsFloat(v[gate * dv + col]) - retrieved);
          double projected = 0.0;
          for (int row = 0; row < dk; ++row) {
            auto& s = state[((b * v_heads + h) * dk + row) * dv + col];
            s += AsFloat(k[(bt * k_heads + kh) * dk + row]) * delta;
            projected += s * AsFloat(q[(bt * q_heads + qh) * dk + row]);
          }
          output[gate * dv + col] = T(static_cast<float>(projected * scale));
        }
      }
    }
  }
  std::vector<T> final_state;
  final_state.reserve(state.size());
  for (double value : state) final_state.emplace_back(static_cast<float>(value));

  OpTester test("LinearAttention", 1, kMSDomain);
  test.AddAttribute<std::string>("update_rule", "gated_delta");
  test.AddAttribute<int64_t>("q_num_heads", q_heads);
  test.AddAttribute<int64_t>("kv_num_heads", v_heads);
  test.AddAttribute<float>("scale", scale);
  test.AddInput<T>("query", {batch, sequence, q_heads * dk}, q);
  test.AddInput<T>("key", {batch, sequence, k_heads * dk}, k);
  test.AddInput<T>("value", {batch, sequence, v_heads * dv}, v);
  if (with_initial_state) {
    test.AddInput<T>("past_state", {batch, v_heads, dk, dv}, initial);
  } else {
    test.AddOptionalInputEdge<T>();
  }
  test.AddInput<T>("decay", {batch, sequence, v_heads}, decay);
  test.AddInput<T>("beta", {batch, sequence, v_heads}, beta);
  test.AddOutput<T>("output", {batch, sequence, v_heads * dv}, output, false, 0.0001f, 0.002f);
  test.AddOutput<T>("present_state", {batch, v_heads, dk, dv}, final_state, false, 0.0001f, 0.002f);
  SessionOptions options;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(std::move(ep));
  test.Run(options, OpTester::ExpectResult::kExpectSuccess, {}, {}, nullptr, &providers);
}

}  // namespace

TEST(WebGpuLinearAttentionTest, PrefillFloat16WithState) {
  RunPrefill<MLFloat16>(2, 129, 2, 2, 2, 128, 128, true);
}

TEST(WebGpuLinearAttentionTest, PrefillFloat16WithoutState) {
  RunPrefill<MLFloat16>(1, 33, 2, 2, 2, 128, 128, false);
}

TEST(WebGpuLinearAttentionTest, PrefillFloat32InverseGQA) {
  RunPrefill<float>(1, 16, 2, 4, 1, 128, 128, true);
}

}  // namespace test
}  // namespace onnxruntime
