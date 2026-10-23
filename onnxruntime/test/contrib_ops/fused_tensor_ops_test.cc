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

float RoundedBf16Sigmoid(float value) {
  const float e = BFloat16(std::exp(-std::abs(value))).ToFloat();
  const float reciprocal = BFloat16(1.0f / BFloat16(1.0f + e).ToFloat()).ToFloat();
  return value > 0 ? reciprocal : BFloat16(1.0f - reciprocal).ToFloat();
}

TEST(FusedTensorOpsTest, RMSNormRotaryExactUnfusedParity) {
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

TEST(FusedTensorOpsTest, RMSNormRotaryRejectsInvalidContracts) {
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

TEST(FusedTensorOpsTest, DotProductGateRoundingAndBroadcast) {
  for (int64_t rows : {0, 1, 4}) {
    for (int64_t width : {1, 17, 2560}) {
      for (bool broadcast : {false, true}) {
        auto providers = CudaProvider();
        if (providers.empty()) GTEST_SKIP();
        const int64_t value_width = broadcast ? width : 4 * width;
        std::vector<BFloat16> key(rows * 4 * width), query(key.size(), BFloat16(0.25f));
        std::vector<BFloat16> value(rows * value_width), expected(key.size());
        for (size_t i = 0; i < value.size(); ++i) value[i] = BFloat16(static_cast<float>(i % 7) * 0.25f);
        for (int64_t group = 0; group < rows * 4; ++group) {
          const float x = static_cast<float>(group % 4 - 2) * 0.125f;
          const float scaled = BFloat16(BFloat16(x * 0.25f * static_cast<float>(width)).ToFloat() / 2.0f).ToFloat();
          const float root = BFloat16(std::sqrt(std::max(std::abs(scaled), BFloat16(1e-6f).ToFloat()))).ToFloat();
          const float gate = RoundedBf16Sigmoid(scaled == 0 ? 0 : std::copysign(root, scaled));
          for (int64_t k = 0; k < width; ++k) {
            key[group * width + k] = BFloat16(x);
            const int64_t index = (broadcast ? group / 4 : group) * width + k;
            expected[group * width + k] = BFloat16(value[index].ToFloat() * gate);
          }
        }
        OpTester test("DotProductGate", 1, kMSDomain);
        test.AddAttribute<int64_t>("group_width", width);
        test.AddAttribute<float>("divisor", 2.0f);
        test.AddInput<BFloat16>("key", {1, rows, 4 * width}, key);
        test.AddInput<BFloat16>("query", {1, rows, 4 * width}, query);
        test.AddInput<BFloat16>("value", {1, rows, value_width}, value);
        test.AddOutput<BFloat16>("output", {1, rows, 4 * width}, expected);
        test.SetOutputTolerance(0.0f, 0.0f);
        test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
      }
    }
  }
}

TEST(FusedTensorOpsTest, DotProductGateRejectsInvalidShapeAndAttributes) {
  for (int mode = 0; mode < 6; ++mode) {
    auto providers = CudaProvider();
    if (providers.empty()) GTEST_SKIP();
    OpTester test("DotProductGate", 1, kMSDomain);
    test.AddAttribute<int64_t>("group_width", mode == 0 ? 0 : 2);
    if (mode == 1) test.AddAttribute<float>("divisor", 0.0f);
    if (mode == 2) test.AddAttribute<float>("epsilon", -1.0f);
    test.AddInput<BFloat16>("key", {1, 4}, std::vector<BFloat16>(4));
    test.AddInput<BFloat16>("query", {1, mode == 3 ? 2 : 4}, std::vector<BFloat16>(mode == 3 ? 2 : 4));
    test.AddInput<BFloat16>("value", {mode == 4 ? 2 : 1, mode == 5 ? 3 : 2},
                            std::vector<BFloat16>(mode == 4 ? 4 : mode == 5 ? 3 : 2));
    test.AddOutput<BFloat16>("output", {1, 4}, std::vector<BFloat16>(4));
    test.Run(OpTester::ExpectResult::kExpectFailure,
             mode < 3 ? "requires positive" : mode == 3 ? "key/query must" :
             mode == 4 ? "leading dimensions" : "value width",
             {}, nullptr, &providers);
  }
}

TEST(FusedTensorOpsTest, HyperConnectionPreMixSigmoidModes) {
  for (int64_t mode : {0, 1, 2}) {
    for (bool feature_gates : {false, true}) {
      auto providers = CudaProvider();
      if (providers.empty()) GTEST_SKIP();
      const int64_t width = feature_gates ? 4 : 1;
      std::vector<BFloat16> gates = ToBFloat16({-100.0f, -5.5f, -0.0f, 0.3f, 5.5f, 100.0f, -1.0f, 1.0f});
      gates.resize(2 * width);
      std::vector<BFloat16> streams = ToBFloat16({1, 2, 3, 4, 5, 6, 7, 8}), expected(4);
      for (int64_t k = 0; k < 4; ++k) {
        float sum = 0;
        for (int64_t branch = 0; branch < 2; ++branch) {
          float g = gates[branch * width + (feature_gates ? k : 0)].ToFloat();
          if (mode == 1) g = BFloat16(1.0f / (1.0f + std::exp(-g))).ToFloat();
          if (mode == 2) g = RoundedBf16Sigmoid(g);
          sum += streams[branch * 4 + k].ToFloat() * g;
        }
        expected[k] = BFloat16(sum * 0.5f);
      }
      OpTester test("HyperConnectionPreMix", 1, kMSDomain);
      test.AddAttribute<int64_t>("sigmoid_mode", mode);
      test.AddAttribute<float>("reduction_scale", 0.5f);
      test.AddInput<BFloat16>("streams", {1, 2, 4}, streams);
      test.AddInput<BFloat16>("pre_mix", {1, 2, width}, gates);
      test.AddOutput<BFloat16>("output", {1, 4}, expected);
      test.SetOutputTolerance(0.0f, 0.0f);
      test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
    }
  }
}

TEST(FusedTensorOpsTest, MtpInputProjectionOriginalBf16Weights) {
  for (int64_t rows : {0, 1, 3}) {
    auto providers = CudaProvider();
    if (providers.empty()) GTEST_SKIP();
    std::vector<BFloat16> hidden(rows * 8), embedding(rows * 4), expected(rows * 8);
    std::vector<BFloat16> weight(16, BFloat16(0.0f));
    for (int i = 0; i < 4; ++i) weight[i * 4 + i] = BFloat16(1.0f);
    for (size_t i = 0; i < hidden.size(); ++i) hidden[i] = BFloat16(static_cast<float>(i) * 0.125f);
    for (size_t i = 0; i < embedding.size(); ++i) embedding[i] = BFloat16(-static_cast<float>(i) * 0.25f);
    for (size_t i = 0; i < expected.size(); ++i) {
      expected[i] = BFloat16(hidden[i].ToFloat() + embedding[i / 8 * 4 + i % 4].ToFloat());
    }
    OpTester test("MtpInputProjection", 1, kMSDomain);
    test.AddInput<BFloat16>("hidden", {1, rows, 2, 4}, hidden);
    test.AddInput<BFloat16>("embedding", {1, rows, 4}, embedding);
    test.AddInput<BFloat16>("hidden_weight", {4, 4}, weight);
    test.AddInput<BFloat16>("embedding_weight", {4, 4}, weight);
    test.AddOutput<BFloat16>("output", {1, rows, 8}, expected);
    test.SetOutputTolerance(0.0f, 0.0f);
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
  }
}

TEST(FusedTensorOpsTest, HyperConnectionMixerOriginalBf16Weights) {
  for (int64_t rows : {0, 1, 3}) {
    auto providers = CudaProvider();
    if (providers.empty()) GTEST_SKIP();
    std::vector<BFloat16> streams(rows * 8), down(16, BFloat16(0.0f)), up(16, BFloat16(0.0f));
    std::vector<BFloat16> expected(rows * 4);
    down[0] = down[3] = BFloat16(1.0f);
    for (int i = 0; i < 8; ++i) up[(i % 2) * 8 + i] = BFloat16(1.0f);
    for (size_t i = 0; i < streams.size(); ++i) streams[i] = BFloat16(static_cast<float>(static_cast<int>(i % 9) - 4) * 0.5f);
    for (int64_t row = 0; row < rows; ++row) {
      float gates[2];
      for (int k = 0; k < 2; ++k) {
        const float z = BFloat16(streams[row * 8 + k].ToFloat() * 0.25f).ToFloat();
        const float activated = BFloat16(z * BFloat16(1.0f / (1.0f + std::exp(-z))).ToFloat()).ToFloat();
        gates[k] = RoundedBf16Sigmoid(activated);
      }
      for (int k = 0; k < 4; ++k) {
        expected[row * 4 + k] = BFloat16((streams[row * 8 + k].ToFloat() * gates[k % 2] +
                                         streams[row * 8 + 4 + k].ToFloat() * gates[k % 2]) * 0.25f);
      }
    }
    OpTester test("HyperConnectionMixer", 1, kMSDomain);
    test.AddAttribute<int64_t>("num_branches", 2);
    test.AddInput<BFloat16>("streams", {1, rows, 8}, streams);
    test.AddInput<BFloat16>("down_weight", {8, 2}, down);
    test.AddInput<BFloat16>("up_weight", {2, 8}, up);
    test.AddOutput<BFloat16>("output", {1, rows, 4}, expected);
    test.SetOutputTolerance(0.0f, 0.0f);
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
  }
}

TEST(FusedTensorOpsTest, MtpAttentionProjectionNormAndRotary) {
  for (int64_t rows : {0, 1, 3}) {
    auto providers = CudaProvider();
    if (providers.empty()) GTEST_SKIP();
    std::vector<BFloat16> x(rows * 8), iw(16, BFloat16(0.0f)), kw(64, BFloat16(0.0f)), vw(32, BFloat16(0.0f));
    for (size_t i = 0; i < x.size(); ++i) x[i] = BFloat16(static_cast<float>(static_cast<int>(i % 11) - 5) * 0.25f);
    for (int i = 0; i < 8; ++i) {
      kw[i * 8 + i] = BFloat16(1.0f);
      if (i < 2) iw[i * 2 + i] = BFloat16(1.0f);
      if (i < 4) vw[i * 4 + i] = BFloat16(1.0f);
    }
    std::vector<BFloat16> index(rows * 2), key(rows * 8), value(rows * 4);
    for (int64_t row = 0; row < rows; ++row) {
      for (int k = 0; k < 2; ++k) index[row * 2 + k] = x[row * 8 + k];
      for (int k = 0; k < 4; ++k) value[row * 4 + k] = x[row * 8 + k];
      for (int head = 0; head < 2; ++head) {
        float sum = 0;
        for (int k = 0; k < 4; ++k) {
          const float f = x[row * 8 + head * 4 + k].ToFloat();
          sum += f * f;
        }
        const float inverse = 1.0f / std::sqrt(sum / 4.0f + 1e-6f);
        BFloat16 normalized[4];
        for (int k = 0; k < 4; ++k) normalized[k] = BFloat16(x[row * 8 + head * 4 + k].ToFloat() * inverse);
        for (int k = 0; k < 4; ++k) {
          const BFloat16 a(normalized[k].ToFloat() * 0.5f);
          const BFloat16 b(normalized[(k + 2) % 4].ToFloat() * (k < 2 ? -0.25f : 0.25f));
          key[row * 8 + head * 4 + k] = BFloat16(a.ToFloat() + b.ToFloat());
        }
      }
    }
    OpTester test("MtpAttentionProjection", 1, kMSDomain);
    test.AddAttribute<int64_t>("rotary_embedding_dim", 4);
    test.AddInput<BFloat16>("input", {1, rows, 8}, x);
    test.AddInput<BFloat16>("index_weight", {8, 2}, iw);
    test.AddInput<BFloat16>("key_weight", {8, 8}, kw);
    test.AddInput<BFloat16>("value_weight", {8, 4}, vw);
    test.AddInput<float>("norm_scale", {4}, {1, 1, 1, 1});
    test.AddInput<BFloat16>("cosine", {1, rows, 2}, std::vector<BFloat16>(rows * 2, BFloat16(0.5f)));
    test.AddInput<BFloat16>("sine", {1, rows, 2}, std::vector<BFloat16>(rows * 2, BFloat16(0.25f)));
    test.AddOutput<BFloat16>("index", {1, rows, 2}, index);
    test.AddOutput<BFloat16>("key", {1, rows, 8}, key);
    test.AddOutput<BFloat16>("value", {1, rows, 4}, value);
    test.SetOutputTolerance(0.0f, 0.0f);
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
  }
}

TEST(FusedTensorOpsTest, Bf16ProjectionBlocksRejectInvalidShapes) {
  for (const std::string op : {"MtpInputProjection", "HyperConnectionMixer", "MtpAttentionProjection"}) {
    auto providers = CudaProvider();
    if (providers.empty()) GTEST_SKIP();
    OpTester test(op.c_str(), 1, kMSDomain);
    if (op == "MtpInputProjection") {
      test.AddInput<BFloat16>("hidden", {1, 1, 2, 4}, std::vector<BFloat16>(8));
      test.AddInput<BFloat16>("embedding", {1, 1, 4}, std::vector<BFloat16>(4));
      test.AddInput<BFloat16>("hidden_weight", {4, 3}, std::vector<BFloat16>(12));
      test.AddInput<BFloat16>("embedding_weight", {4, 4}, std::vector<BFloat16>(16));
      test.AddOutput<BFloat16>("output", {1, 1, 8}, std::vector<BFloat16>(8));
    } else if (op == "HyperConnectionMixer") {
      test.AddInput<BFloat16>("streams", {1, 1, 8}, std::vector<BFloat16>(8));
      test.AddInput<BFloat16>("down_weight", {8, 2}, std::vector<BFloat16>(16));
      test.AddInput<BFloat16>("up_weight", {3, 8}, std::vector<BFloat16>(24));
      test.AddOutput<BFloat16>("output", {1, 1, 2}, std::vector<BFloat16>(2));
    } else {
      test.AddAttribute<int64_t>("rotary_embedding_dim", 4);
      test.AddInput<BFloat16>("input", {1, 1, 8}, std::vector<BFloat16>(8));
      test.AddInput<BFloat16>("index_weight", {8, 2}, std::vector<BFloat16>(16));
      test.AddInput<BFloat16>("key_weight", {8, 8}, std::vector<BFloat16>(64));
      test.AddInput<BFloat16>("value_weight", {8, 4}, std::vector<BFloat16>(32));
      test.AddInput<float>("norm_scale", {3}, {1, 1, 1});
      test.AddInput<BFloat16>("cosine", {1, 1, 2}, std::vector<BFloat16>(2));
      test.AddInput<BFloat16>("sine", {1, 1, 2}, std::vector<BFloat16>(2));
      test.AddOutput<BFloat16>("index", {1, 1, 2}, std::vector<BFloat16>(2));
      test.AddOutput<BFloat16>("key", {1, 1, 8}, std::vector<BFloat16>(8));
      test.AddOutput<BFloat16>("value", {1, 1, 4}, std::vector<BFloat16>(4));
    }
    test.Run(OpTester::ExpectResult::kExpectFailure,
             op == "MtpInputProjection" ? "weights must project" :
             op == "HyperConnectionMixer" ? "requires [B,S,C*H]" : "norm/cache shape mismatch",
             {}, nullptr, &providers);
  }
}

TEST(FusedTensorOpsTest, PackedMtpMoeRejectsInvalidLayoutAndScale) {
  for (bool invalid_scale : {false, true}) {
    auto providers = CudaProvider();
    if (providers.empty()) GTEST_SKIP();
    OpTester test("PackedMtpMoe", 1, kMSDomain);
    if (invalid_scale) test.AddAttribute<float>("down_global_scale", -1.0f);
    test.AddInput<BFloat16>("input", {1, 0, 2560}, {});
    test.AddInput<BFloat16>("router", {1, 0, 512}, {});
    test.AddInput<uint8_t>("gate_up", {1}, {0});
    test.AddInput<uint8_t>("down", {1}, {0});
    test.AddOutput<BFloat16>("output", {1, 0, 2560}, {});
    test.Run(OpTester::ExpectResult::kExpectFailure,
             invalid_scale ? "positive finite global scales" : "matching router, NVFP4 weights",
             {}, nullptr, &providers);
  }
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

TEST(FusedTensorOpsTest, RowSelectMultiDraft) {
  for (const int64_t rows : {2, 3, 4}) {
    for (int64_t selection = -1; selection <= rows; ++selection) {
      auto providers = CudaProvider();
      if (providers.empty()) GTEST_SKIP();
      OpTester test("RowSelect", 1, kMSDomain);
      test.AddInput<int64_t>("selection", {}, {selection});
      const int64_t selected = rows == 2 ? (selection != 0 ? 1 : 0)
                                        : std::max<int64_t>(0, std::min(selection, rows - 1));
      for (int i = 0; i < 3; ++i) {
        const int64_t width = i + 1;
        std::vector<float> input(static_cast<size_t>(rows * width));
        for (size_t j = 0; j < input.size(); ++j) input[j] = static_cast<float>(j + 10 * i);
        const std::vector<float> expected(input.begin() + selected * width, input.begin() + (selected + 1) * width);
        const std::string input_name = "x" + std::to_string(i);
        const std::string output_name = "y" + std::to_string(i);
        test.AddInput<BFloat16>(input_name.c_str(), {1, rows, width}, ToTensorType<BFloat16>(input));
        test.AddOutput<BFloat16>(output_name.c_str(), {1, 1, width}, ToTensorType<BFloat16>(expected));
      }
      test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
    }
  }
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

TEST(FusedTensorOpsTest, SigmoidGateMulProjectedGate) {
  for (int64_t rows : {0, 1, 4}) {
    for (int64_t input_width : {1, 17, 2560}) {
      for (bool residual : {false, true}) {
        for (int mode : {0, 1}) {
          auto providers = CudaProvider();
          if (providers.empty()) GTEST_SKIP();
          constexpr int64_t output_width = 257;
          std::vector<BFloat16> activation(rows * input_width), weight(input_width, BFloat16(0.015625f));
          std::vector<BFloat16> values(rows * output_width), residuals(rows * output_width),
              expected(rows * output_width);
          for (int64_t row = 0; row < rows; ++row) {
            const float x = row % 2 == 0 ? 0.125f : -0.125f;
            std::fill_n(activation.begin() + row * input_width, input_width, BFloat16(x));
            const float dot = BFloat16(x * 0.015625f * static_cast<float>(input_width)).ToFloat();
            const float e = std::exp(-std::abs(dot));
            const BFloat16 denominator(1.0f + BFloat16(e).ToFloat());
            const BFloat16 reciprocal(1.0f / denominator.ToFloat());
            const BFloat16 gate = mode == 0
                                      ? BFloat16(dot >= 0 ? 1.0f / (1.0f + e) : e / (1.0f + e))
                                      : (dot > 0 ? reciprocal : BFloat16(1.0f - reciprocal.ToFloat()));
            for (int64_t col = 0; col < output_width; ++col) {
              const int64_t index = row * output_width + col;
              values[index] = BFloat16(static_cast<float>(col % 13 - 6) * 0.125f);
              residuals[index] = BFloat16(0.375f);
              const BFloat16 product(values[index].ToFloat() * gate.ToFloat());
              expected[index] = residual ? BFloat16(product.ToFloat() + residuals[index].ToFloat()) : product;
            }
          }
          OpTester test("SigmoidGateMul", 1, kMSDomain);
          test.AddAttribute<int64_t>("broadcast_gate", 1);
          test.AddInput<BFloat16>("attention", {1, rows, output_width}, values);
          test.AddInput<BFloat16>("projection", {1, rows, input_width}, activation);
          if (residual) {
            test.AddInput<BFloat16>("residual", {1, rows, output_width}, residuals);
          } else {
            test.AddOptionalInputEdge<BFloat16>();
          }
          test.AddInput<int32_t>("prefill_numerics", {1}, {mode});
          test.AddInput<BFloat16>("gate_weight", {input_width, 1}, weight);
          test.AddOutput<BFloat16>("output", {1, rows, output_width}, expected);
          test.SetOutputTolerance(0.0f, 0.0f);
          test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
        }
      }
    }
  }
}

TEST(FusedTensorOpsTest, SigmoidGateMulProjectedGateRejectsInvalidShapes) {
  for (bool invalid_weight : {false, true}) {
    auto providers = CudaProvider();
    if (providers.empty()) GTEST_SKIP();
    OpTester test("SigmoidGateMul", 1, kMSDomain);
    test.AddAttribute<int64_t>("broadcast_gate", 1);
    test.AddInput<BFloat16>("attention", {1, 2, 3}, std::vector<BFloat16>(6));
    test.AddInput<BFloat16>("projection", {1, invalid_weight ? 2 : 3, 4},
                            std::vector<BFloat16>(invalid_weight ? 8 : 12));
    test.AddOptionalInputEdge<BFloat16>();
    test.AddOptionalInputEdge<int32_t>();
    test.AddInput<BFloat16>("gate_weight", {invalid_weight ? 3 : 4, 1},
                            std::vector<BFloat16>(invalid_weight ? 3 : 4));
    test.AddOutput<BFloat16>("output", {1, 2, 3}, std::vector<BFloat16>(6));
    test.Run(OpTester::ExpectResult::kExpectFailure,
             invalid_weight ? "gate_weight must have shape" : "leading dimensions must match",
             {}, nullptr, &providers);
  }
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
