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
            const auto position_shape = scalar ? std::vector<int64_t>{1} :
                                                 std::vector<int64_t>{output_batch, output_sequence};
            const auto run = [](OpTester& test) {
              auto providers = CudaProvider();
              ASSERT_FALSE(providers.empty());
              test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
            };
            if (CudaProvider().empty()) GTEST_SKIP();
            OpTester original("BranchwiseRMSNorm", 2, kMSDomain);
            original.AddAttribute<int64_t>("use_onnx_arithmetic", 1);
            original.AddAttribute<int64_t>("input_width", width);
            original.AddAttribute<int64_t>("input_start", start);
            original.AddAttribute<float>("epsilon", 1e-6f);
            original.AddInput<BFloat16>("X", input_shape, input);
            original.AddInput<float>("scale", {width}, scale);
            original.AddInput<int32_t>("mode", {1}, {dense});
            original.AddOutput<BFloat16>("Y", norm_shape, norm);
            original.AddOutput<float>("Y_fp32", norm_shape, normalized);
            original.SetCustomOutputVerifier([&](const std::vector<OrtValue>& values, const std::string&) {
              if (norm.empty()) return;
              const auto& y = values[0].Get<Tensor>();
              norm.assign(y.Data<BFloat16>(), y.Data<BFloat16>() + norm.size());
              const auto& widened = values[1].Get<Tensor>();
              normalized.assign(widened.Data<float>(), widened.Data<float>() + normalized.size());
            });
            run(original);
            OpTester rope("RotaryEmbedding", 2, kMSDomain);
            rope.AddAttribute<int64_t>("num_heads", heads);
            rope.AddAttribute<int64_t>("rotary_embedding_dim", rotary);
            rope.AddAttribute<int64_t>("use_fp32_arithmetic", fp32 ? 1 : 0);
            rope.AddInput<BFloat16>("X", output_shape, norm);
            rope.AddInput<int64_t>("position", position_shape, positions);
            rope.AddInput<BFloat16>("cos", {cache_rows, rotary / 2}, cosine);
            rope.AddInput<BFloat16>("sin", {cache_rows, rotary / 2}, sine);
            rope.AddInput<int32_t>("mode", {1}, {dense});
            rope.AddOutput<BFloat16>("Y", output_shape, expected);
            rope.SetCustomOutputVerifier([&](const std::vector<OrtValue>& values, const std::string&) {
              if (expected.empty()) return;
              const auto& y = values[0].Get<Tensor>();
              expected.assign(y.Data<BFloat16>(), y.Data<BFloat16>() + expected.size());
            });
            if (sequence > 0) run(rope);
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
             mode == 0 ? "slice" : mode == 1 ? "rotary" : mode == 2 ? "requires finite" :
             mode == 3 ? "invalid slice" :
             mode == 4 ? "scale" : mode == 5 ? "positions" : mode == 6 ? "cosine/sine" : "prefill_numerics",
             {}, nullptr, &providers);
  }
}

}  // namespace
}  // namespace onnxruntime::test
