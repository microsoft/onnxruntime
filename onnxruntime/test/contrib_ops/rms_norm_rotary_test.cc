// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
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

TEST(RMSNormRotaryTest, RMSNormRotaryExactUnfusedParity) {
  for (const int64_t width : {64, 256, 320}) {
    for (const int64_t sequence : {0, 1, 2, 7}) {
      for (const bool packed : {false, true}) {
        for (const int32_t dense : {0, 1}) {
          for (const bool fp32 : {false, true}) {
            const int64_t batch = packed || sequence == 7 ? 2 : 1;
            const int64_t heads = sequence == 1 && !packed ? 1 : 2;
            const int64_t stride = packed ? width * 2 : width;
            const int64_t start = packed ? 3 : 0;
            const int64_t rotary = packed ? 64 : width;
            constexpr int64_t cache_rows = 64;
            const int64_t output_batch = packed ? 1 : batch;
            const int64_t output_sequence = packed ? batch * sequence : sequence;
            SCOPED_TRACE("width=" + std::to_string(width) + " S=" + std::to_string(sequence) +
                         " packed=" + std::to_string(packed) + " dense=" + std::to_string(dense) +
                         " fp32=" + std::to_string(fp32));
            std::vector<BFloat16> input(batch * sequence * heads * stride);
            std::vector<float> scale(width), normalized(batch * sequence * heads * width);
            std::vector<BFloat16> norm(normalized.size()), expected(normalized.size());
            std::vector<BFloat16> cosine(cache_rows * rotary / 2), sine(cosine.size());
            for (size_t i = 0; i < input.size(); ++i) {
              input[i] = BFloat16(std::sin(static_cast<float>(i) * .731f) *
                                  (i % 19 == 0 ? 64.f : .03f * static_cast<float>(i % 11)));
            }
            for (int64_t i = 0; i < width; ++i) scale[i] = .8f + .0013f * static_cast<float>(i);
            for (size_t i = 0; i < cosine.size(); ++i) {
              cosine[i] = BFloat16(std::cos(static_cast<float>(i) * .13f));
              sine[i] = BFloat16(std::sin(static_cast<float>(i) * .13f));
            }
            const std::vector<int64_t> input_shape{batch, sequence, heads, stride};
            const std::vector<int64_t> norm_shape{batch, sequence, heads, width};
            const std::vector<int64_t> output_shape{output_batch, output_sequence, heads * width};
            const bool scalar = !packed;
            std::vector<int64_t> positions(scalar ? 1 : output_batch * output_sequence,
                                           cache_rows - output_sequence);
            if (!scalar) {
              for (size_t i = 0; i < positions.size(); ++i) positions[i] = static_cast<int64_t>((i * 7 + 3) % cache_rows);
              if (!positions.empty()) positions.back() = cache_rows - 1;
            }
            const auto position_shape = scalar ? std::vector<int64_t>{1} : std::vector<int64_t>{output_batch, output_sequence};
            const auto run = [](OpTester& test) {
              auto providers = CudaProvider();
              ASSERT_FALSE(providers.empty());
              test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
            };
            if (CudaProvider().empty()) GTEST_SKIP();
            const int64_t groups = batch * sequence * heads;
            for (int64_t g = 0; g < groups; ++g) {
              const auto* x = input.data() + g * stride + start;
              float partial[2] = {0.f, 0.f};
              for (int w = 0; w < 2; ++w) {
                float warp_sum[32] = {};
                for (int lane = 0; lane < 32; ++lane) {
                  const int tid = w * 32 + lane;
                  float s = 0.f;
                  for (int col = tid; col < width; col += 64) {
                    const float v = x[col].ToFloat();
                    s += v * v;
                  }
                  warp_sum[lane] = s;
                }
                for (int step = 16; step > 0; step >>= 1) {
                  for (int lane = 0; lane < step; ++lane) warp_sum[lane] += warp_sum[lane + step];
                }
                partial[w] = warp_sum[0];
              }
              const float total = partial[0] + partial[1];
              const float rms = std::sqrt((total / static_cast<float>(width)) + 1e-6f);
              const float inv_rms = 1.f / rms;
              for (int64_t col = 0; col < width; ++col) {
                const float v = x[col].ToFloat();
                const float n = dense ? (v * inv_rms) : (v / rms);
                const BFloat16 rounded(n * scale[col]);
                norm[g * width + col] = rounded;
                normalized[g * width + col] = rounded.ToFloat();
              }
              const int64_t token = g / heads;
              const int64_t base = positions[scalar ? 0 : token];
              const bool valid = scalar ? (base >= 0 && base <= cache_rows - output_sequence)
                                        : (base >= 0 && base < cache_rows);
              const int64_t pos = valid && scalar ? base + token % output_sequence : base;
              const int64_t half = rotary / 2;
              for (int64_t col = 0; col < width; ++col) {
                BFloat16 val = norm[g * width + col];
                if (col < rotary && valid) {
                  const int64_t partner = (col + half) % rotary;
                  const int64_t cache = pos * half + col % half;
                  const float a = val.ToFloat();
                  const float b = norm[g * width + partner].ToFloat();
                  const float c = cosine[cache].ToFloat();
                  const float s = sine[cache].ToFloat();
                  if (!dense && fp32) {
                    val = BFloat16((a * c) + ((col < half ? -b : b) * s));
                  } else {
                    const float p1 = BFloat16(a * c).ToFloat();
                    const float signed_b = BFloat16((col < half ? -1.f : 1.f) * b).ToFloat();
                    const float p2 = BFloat16(signed_b * s).ToFloat();
                    val = BFloat16(p1 + p2);
                  }
                }
                expected[g * width + col] = val;
              }
            }
            OpTester fused("RMSNormRotary", 1, kMSDomain);
            fused.AddAttribute<int64_t>("input_width", width);
            fused.AddAttribute<int64_t>("input_start", start);
            fused.AddAttribute<int64_t>("rotary_embedding_dim", rotary);
            fused.AddAttribute<int64_t>("use_fp32_arithmetic", fp32 ? 1 : 0);
            fused.AddAttribute<int64_t>("flatten_batch", packed ? 1 : 0);
            fused.AddAttribute<float>("epsilon", 1e-6f);
            fused.AddInput<BFloat16>("X", input_shape, input);
            fused.AddInput<float>("scale", {width}, scale);
            fused.AddInput<int64_t>("position", position_shape, positions);
            fused.AddInput<BFloat16>("cos", {cache_rows, rotary / 2}, cosine);
            fused.AddInput<BFloat16>("sin", {cache_rows, rotary / 2}, sine);
            fused.AddInput<int32_t>("mode", {1}, {dense});
            fused.AddOutput<BFloat16>("Y", output_shape, expected);
            fused.AddOutput<float>("normalized", norm_shape, normalized);
            fused.SetOutputTolerance(0.f, 0.f);
            run(fused);
          }
        }
      }
    }
  }
}

TEST(RMSNormRotaryTest, RMSNormRotaryRejectsInvalidContracts) {
  for (int mode = 0; mode < 8; ++mode) {
    SCOPED_TRACE(mode);
    auto providers = CudaProvider();
    if (providers.empty()) GTEST_SKIP();
    OpTester test("RMSNormRotary", 1, kMSDomain);
    test.AddAttribute<int64_t>("input_width", 256);
    test.AddAttribute<int64_t>("input_start", mode == 0 ? 1 : 0);
    test.AddAttribute<int64_t>("rotary_embedding_dim", mode == 1 ? 33 : 32);
    if (mode == 2) test.AddAttribute<float>("epsilon", -1.f);
    if (mode == 3) test.AddAttribute<int64_t>("use_fp32_arithmetic", 2);
    test.AddInput<BFloat16>("X", {1, 2, 2, 256}, std::vector<BFloat16>(1024));
    test.AddInput<float>("scale", {mode == 4 ? 128 : 256}, std::vector<float>(mode == 4 ? 128 : 256, 1.f));
    test.AddInput<int64_t>("pos", {mode == 5 ? 3 : 1}, std::vector<int64_t>(mode == 5 ? 3 : 1));
    test.AddInput<BFloat16>("cos", {8, 16}, std::vector<BFloat16>(128));
    test.AddInput<BFloat16>("sin", {mode == 6 ? 7 : 8, 16}, std::vector<BFloat16>(mode == 6 ? 112 : 128));
    test.AddInput<int32_t>("mode", {1}, {mode == 7 ? 2 : 0});
    test.AddOutput<BFloat16>("Y", {1, 2, 512}, std::vector<BFloat16>(1024));
    test.Run(OpTester::ExpectResult::kExpectFailure,
             mode == 0 ? "slice" : mode == 1 ? "rotary"
                               : mode == 2   ? "requires finite"
                               : mode == 3   ? "invalid slice"
                               : mode == 4   ? "scale"
                               : mode == 5   ? "positions"
                               : mode == 6   ? "cosine/sine"
                                             : "prefill_numerics",
             {}, nullptr, &providers);
  }
}

}  // namespace
}  // namespace onnxruntime::test
