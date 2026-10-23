// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/inference_session_wrapper.h"
#include "test/util/include/test/test_environment.h"

#ifdef USE_CUDA
namespace onnxruntime::test {
namespace {

enum class Int3SessionMode { None,
                             DynamicMetadata,
                             PrivatePrepack };

struct Int3CaseBoundary {
  int64_t rows = 5;
  int64_t scratch_bytes = -1;
  bool malformed_scale_shape = false;
};

class Int3CudaSessionTester : public OpTester {
 public:
  Int3CudaSessionTester() : OpTester("QMoE", 1, kMSDomain) {}

  std::string ExportSessionModel(NameMLValMap& feeds, std::vector<std::string>& output_names) {
    SetTestFunctionCalled();
    auto& model = BuildModel();
    ORT_THROW_IF_ERROR(model.MainGraph().Resolve());
    FillFeedsAndOutputNames(feeds, output_names);
    return model.ToProto().SerializeAsString();
  }
};

template <typename T>
void RunInt3RepeatedSession(std::unique_ptr<Int3CudaSessionTester> tester, SessionOptions options,
                            const std::vector<T>& expected, int dynamic_index, Int3SessionMode mode) {
  NameMLValMap feeds;
  std::vector<std::string> output_names;
  const std::string model_bytes = tester->ExportSessionModel(feeds, output_names);
  tester.reset();
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));

  for (int generation = 0; generation < 2; ++generation) {
    SCOPED_TRACE(::testing::Message() << "session generation " << generation);
    InferenceSessionWrapper session(options, GetEnvironment());
    ASSERT_STATUS_OK(session.RegisterExecutionProvider(DefaultCudaExecutionProvider()));
    ASSERT_STATUS_OK(session.Load(model_bytes.data(), static_cast<int>(model_bytes.size())));
    ASSERT_STATUS_OK(session.Initialize());
    EXPECT_EQ(session.GetSessionState().GetNumberOfPrepacksCounter(),
              mode == Int3SessionMode::PrivatePrepack ? size_t{6} : size_t{5});
    EXPECT_EQ(session.GetSessionState().GetUsedSharedPrePackedWeightCounter(), size_t{0});
    for (auto initializer : {std::pair{2, "fc1_experts_weights"}, std::pair{3, "fc1_scales"},
                             std::pair{5, "fc2_experts_weights"}, std::pair{6, "fc2_scales"},
                             std::pair{11, "fc1_zero_points"}, std::pair{12, "fc2_zero_points"}}) {
      if (initializer.first == dynamic_index) continue;
      int initializer_index = -1;
      ASSERT_STATUS_OK(session.GetSessionState().GetOrtValueNameIdxMap().GetIdx(initializer.second, initializer_index));
      EXPECT_EQ(session.GetSessionState().GetInitializedTensors().count(initializer_index), size_t{0})
          << initializer.second;
    }

    auto check_valid = [&]() {
      std::vector<OrtValue> fetches;
      ASSERT_STATUS_OK(session.Run(feeds, output_names, &fetches));
      ASSERT_EQ(fetches.size(), size_t{1});
      const auto& output = fetches[0].Get<Tensor>();
      ASSERT_EQ(output.Shape(), TensorShape({1, 5, static_cast<int64_t>(expected.size() / 5)}));
      const auto* actual = output.Data<T>();
      for (size_t index = 0; index < expected.size(); ++index) {
        ASSERT_TRUE(std::isfinite(actual[index].ToFloat())) << index;
        ASSERT_NEAR(actual[index].ToFloat(), expected[index].ToFloat(),
                    0.001f + 0.02f * std::abs(expected[index].ToFloat()))
            << index;
      }
    };
    check_valid();
    ASSERT_FALSE(::testing::Test::HasFatalFailure());

    if (mode == Int3SessionMode::DynamicMetadata) {
      const char* name = dynamic_index == 3    ? "fc1_scales"
                         : dynamic_index == 6  ? "fc2_scales"
                         : dynamic_index == 11 ? "fc1_zero_points"
                                               : "fc2_zero_points";
      auto& tensor = *feeds.at(name).GetMutable<Tensor>();
      for (int invalid = 0; invalid < 3; ++invalid) {
        SCOPED_TRACE(::testing::Message() << name << ": invalid variant " << invalid);
        const size_t index = invalid == 0 ? 0 : static_cast<size_t>(tensor.Shape().Size() - 1);
        T saved_scale{};
        uint8_t saved_zero = 0;
        if (dynamic_index == 3 || dynamic_index == 6) {
          auto* scales = tensor.MutableData<T>();
          saved_scale = scales[index];
          scales[index] = T(invalid == 0   ? -1.0f
                            : invalid == 1 ? std::numeric_limits<float>::infinity()
                                           : std::numeric_limits<float>::quiet_NaN());
        } else {
          auto* zeros = tensor.MutableData<uint8_t>();
          saved_zero = zeros[index];
          zeros[index] ^= invalid == 0 ? 1 : (invalid == 1 ? 0x80 : 0x40);
        }
        std::vector<OrtValue> fetches;
        const auto status = session.Run(feeds, output_names, &fetches);
        EXPECT_FALSE(status.IsOK());
        EXPECT_NE(status.ErrorMessage().find("finite nonnegative scales and INT3 zero points equal to 4 with zero padding"),
                  std::string::npos)
            << status.ErrorMessage();
        if (dynamic_index == 3 || dynamic_index == 6) {
          tensor.MutableData<T>()[index] = saved_scale;
        } else {
          tensor.MutableData<uint8_t>()[index] = saved_zero;
        }
        check_valid();
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
      }
    }
    for (int repeat = 0; repeat < 3; ++repeat) {
      check_valid();
      ASSERT_FALSE(::testing::Test::HasFatalFailure());
    }
  }
}

void PackInt3TestCode(std::vector<uint8_t>& packed, int row, int count, int index, int bits, int code) {
  const int bytes = (count * bits + 7) / 8;
  for (int bit = 0; bit < bits; ++bit) {
    const int position = index * bits + bit;
    packed[row * bytes + position / 8] |= ((code >> bit) & 1) << (position % 8);
  }
}

template <typename T>
void RunInt3CudaCase(int block, int fc1_bits, int fc2_bits, bool separate_routing,
                     bool initializers, int malformed = 0, int dynamic_index = -1,
                     Int3SessionMode session_mode = Int3SessionMode::None,
                     Int3CaseBoundary boundary = {}) {
  auto provider = DefaultCudaExecutionProvider();
  if (!provider) GTEST_SKIP() << "CUDA execution provider unavailable";
  const int64_t rows = boundary.rows;
  constexpr int64_t experts = 3;
  constexpr int64_t top_k = 2;
  const int64_t hidden = block * 2;
  const int64_t inter = block * 3;
  const float scale = 0.015625f;
  std::vector<T> input(rows * hidden);
  for (int64_t index = 0; index < rows * hidden; ++index) input[index] = T((index % 11 - 5) * 0.0625f);
  std::vector<T> router(rows * experts, T(0.0f));
  std::vector<T> external_routing(rows * experts);
  for (int64_t index = 0; index < rows * experts; ++index) external_routing[index] = T(0.125f * (1 + index % experts));
  std::vector<uint8_t> first(experts * inter * 2 * (hidden * fc1_bits / 8));
  std::vector<uint8_t> second(experts * hidden * (inter * fc2_bits / 8));
  std::vector<T> first_scales(experts * inter * 2 * (hidden / block), T(scale));
  std::vector<T> second_scales(experts * hidden * (inter / block), T(scale));
  std::vector<T> first_bias(experts * inter * 2, T(0.015625f));
  std::vector<T> second_bias(experts * hidden, T(0.0078125f));
  for (int64_t row = 0; row < experts * inter * 2; ++row) {
    for (int64_t feature = 0; feature < hidden; ++feature) {
      PackInt3TestCode(first, row, hidden, feature, fc1_bits, (row * 5 + feature * 3) % (1 << fc1_bits));
    }
  }
  for (int64_t row = 0; row < experts * hidden; ++row) {
    for (int64_t feature = 0; feature < inter; ++feature) {
      PackInt3TestCode(second, row, inter, feature, fc2_bits, (row * 3 + feature * 5) % (1 << fc2_bits));
    }
  }
  std::vector<T> expected(rows * hidden);
  for (int64_t token = 0; token < rows; ++token) {
    std::vector<float> accumulated(hidden, 0);
    for (int64_t expert = 0; expert < top_k; ++expert) {
      std::vector<T> activation(inter);
      for (int64_t column = 0; column < inter; ++column) {
        const int64_t weight_row = expert * inter * 2 + column * 2;
        float gate = first_bias[weight_row].ToFloat();
        float up = first_bias[weight_row + 1].ToFloat();
        for (int64_t feature = 0; feature < hidden; ++feature) {
          const float value = input[token * hidden + feature].ToFloat();
          gate += value * (((weight_row * 5 + feature * 3) % (1 << fc1_bits)) - (1 << (fc1_bits - 1))) * scale;
          up += value * ((((weight_row + 1) * 5 + feature * 3) % (1 << fc1_bits)) - (1 << (fc1_bits - 1))) * scale;
        }
        activation[column] = T(gate / (1 + std::exp(-gate)) * up);
      }
      for (int64_t column = 0; column < hidden; ++column) {
        const int64_t weight_row = expert * hidden + column;
        float value = 0;
        for (int64_t feature = 0; feature < inter; ++feature) {
          value += activation[feature].ToFloat() *
                   (((weight_row * 3 + feature * 5) % (1 << fc2_bits)) - (1 << (fc2_bits - 1))) * scale;
        }
        value += second_bias[weight_row].ToFloat();
        const float routing_weight = separate_routing ? external_routing[token * experts + expert].ToFloat() : 0.5f;
        accumulated[column] += T(value).ToFloat() * routing_weight;
      }
    }
    for (int64_t column = 0; column < hidden; ++column) expected[token * hidden + column] = T(accumulated[column]);
  }
  const int64_t first_zero_bytes = ((hidden / block) * fc1_bits + 7) / 8;
  const int64_t second_zero_bytes = ((inter / block) * fc2_bits + 7) / 8;
  std::vector<uint8_t> first_zeros(experts * inter * 2 * first_zero_bytes);
  std::vector<uint8_t> second_zeros(experts * hidden * second_zero_bytes);
  for (int64_t row = 0; row < experts * inter * 2; ++row) {
    for (int64_t group = 0; group < hidden / block; ++group) {
      PackInt3TestCode(first_zeros, row, hidden / block, group, fc1_bits, 1 << (fc1_bits - 1));
    }
  }
  for (int64_t row = 0; row < experts * hidden; ++row) {
    for (int64_t group = 0; group < inter / block; ++group) {
      PackInt3TestCode(second_zeros, row, inter / block, group, fc2_bits, 1 << (fc2_bits - 1));
    }
  }
  if (malformed == 1) first_scales[0] = T(-1.0f);
  if (malformed == 2) second_scales[0] = T(std::numeric_limits<float>::infinity());
  if (malformed == 3) first_zeros[0] ^= 1;
  if (malformed == 4) first_zeros[0] |= 0x80;
  auto tester_owner = std::make_unique<Int3CudaSessionTester>();
  auto& tester = *tester_owner;
  tester.AddAttribute<int64_t>("k", top_k);
  tester.AddAttribute<std::string>("activation_type", "swiglu");
  tester.AddAttribute<int64_t>("swiglu_fusion", 1);
  tester.AddAttribute<int64_t>("normalize_routing_weights", 1);
  tester.AddAttribute<int64_t>("expert_weight_bits", fc1_bits);
  tester.AddAttribute<int64_t>("fc2_expert_weight_bits", fc2_bits);
  tester.AddAttribute<int64_t>("block_size", block);
  tester.AddAttribute<int64_t>("weights_prepacked", 0);
  tester.AddInput<T>("input", {1, rows, hidden}, input);
  tester.AddInput<T>("router_probs", {rows, experts}, router);
  tester.AddInput<uint8_t>("fc1_experts_weights", {experts, 2 * inter, hidden * fc1_bits / 8}, first, initializers && dynamic_index != 2);
  const int64_t first_groups = hidden / block - (boundary.malformed_scale_shape ? 1 : 0);
  if (boundary.malformed_scale_shape) first_scales.resize(experts * 2 * inter * first_groups);
  tester.AddInput<T>("fc1_scales", {experts, 2 * inter, first_groups}, first_scales, initializers && dynamic_index != 3);
  tester.AddInput<T>("fc1_experts_bias", {experts, 2 * inter}, first_bias, initializers);
  tester.AddInput<uint8_t>("fc2_experts_weights", {experts, hidden, inter * fc2_bits / 8}, second, initializers && dynamic_index != 5);
  tester.AddInput<T>("fc2_scales", {experts, hidden, inter / block}, second_scales, initializers && dynamic_index != 6);
  tester.AddInput<T>("fc2_experts_bias", {experts, hidden}, second_bias, initializers);
  tester.AddOptionalInputEdge<uint8_t>();
  tester.AddOptionalInputEdge<T>();
  tester.AddOptionalInputEdge<T>();
  tester.AddInput<uint8_t>("fc1_zero_points", {experts, 2 * inter, first_zero_bytes}, first_zeros, initializers && dynamic_index != 11);
  tester.AddInput<uint8_t>("fc2_zero_points", {experts, hidden, second_zero_bytes}, second_zeros, initializers && dynamic_index != 12);
  tester.AddOptionalInputEdge<uint8_t>();
  if (separate_routing) tester.AddInput<T>("router_weights", {rows, experts}, external_routing);
  tester.AddOutput<T>("output", {1, rows, hidden}, expected);
  tester.SetOutputAbsErr("output", 0.001f);
  tester.SetOutputRelErr("output", 0.02f);
  tester.SetNumRunCalls(malformed ? 1 : 3);
  SessionOptions options;
  options.config_options.configurations["ep.cuda.qmoe_row_tile_size"] = "2";
  if (boundary.scratch_bytes >= 0) {
    options.config_options.configurations["ep.cuda.qmoe_int_dequant_max_scratch_bytes"] =
        std::to_string(boundary.scratch_bytes);
  }
  if (session_mode != Int3SessionMode::None) {
    RunInt3RepeatedSession(std::move(tester_owner), options, expected, dynamic_index, session_mode);
    return;
  }
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(std::move(provider));
  const int64_t per_row_bytes = top_k * (inter * sizeof(T) + 3 * sizeof(int));
  const bool insufficient_scratch = rows > 0 && boundary.scratch_bytes >= 0 && boundary.scratch_bytes < per_row_bytes;
  const std::string expected_error = malformed                        ? "finite nonnegative scales and INT3 zero points equal to 4 with zero padding"
                                     : boundary.malformed_scale_shape ? "QMoE INT3 incompatible scale blocks dimension"
                                     : insufficient_scratch           ? "INT3 QMoE scratch budget cannot accommodate one token"
                                                                      : "";
  tester.Run(options, expected_error.empty() ? OpTester::ExpectResult::kExpectSuccess : OpTester::ExpectResult::kExpectFailure,
             expected_error, {}, nullptr, &providers);
}

TEST(QMoEInt3CudaTest, MixedWidthsAndTiledRank3Input) {
  for (int block : {32, 64, 128}) {
    for (auto bits : {std::pair{3, 3}, std::pair{3, 4}, std::pair{4, 3}}) {
      SCOPED_TRACE(::testing::Message() << block << ":" << bits.first << "/" << bits.second);
      for (bool initializers : {false, true}) {
        RunInt3CudaCase<MLFloat16>(block, bits.first, bits.second, false, initializers);
        RunInt3CudaCase<BFloat16>(block, bits.first, bits.second, true, initializers);
      }
    }
  }
}

TEST(QMoEInt3CudaTest, RejectsMalformedDeviceContent) {
  for (int malformed : {1, 2, 3, 4}) {
    RunInt3CudaCase<MLFloat16>(32, 3, 3, false, malformed % 2 == 0, malformed);
    RunInt3CudaCase<BFloat16>(32, 3, 3, false, malformed % 2 == 1, malformed);
  }
}

TEST(QMoEInt3CudaTest, PartialPrepackKeepsIndependentDynamicInputs) {
  for (auto bits : {std::pair{3, 3}, std::pair{3, 4}, std::pair{4, 3}}) {
    for (int dynamic_index : {2, 3, 5, 6, 11, 12}) {
      SCOPED_TRACE(::testing::Message() << bits.first << "/" << bits.second << ":" << dynamic_index);
      RunInt3CudaCase<MLFloat16>(64, bits.first, bits.second, false, true, 0, dynamic_index);
      RunInt3CudaCase<BFloat16>(64, bits.first, bits.second, true, true, 0, dynamic_index);
    }
  }
  for (auto malformed : {std::pair{1, 3}, std::pair{2, 6}, std::pair{3, 11}, std::pair{4, 11}}) {
    RunInt3CudaCase<MLFloat16>(32, 3, 3, false, true, malformed.first, malformed.second);
    RunInt3CudaCase<BFloat16>(32, 3, 3, false, true, malformed.first, malformed.second);
  }
}

TEST(QMoEInt3CudaTest, SameSessionRevalidatesDynamicScales) {
  for (auto bits : {std::pair{3, 3}, std::pair{3, 4}, std::pair{4, 3}}) {
    for (int dynamic_index : {3, 6}) {
      SCOPED_TRACE(::testing::Message() << bits.first << "/" << bits.second << ":" << dynamic_index);
      RunInt3CudaCase<MLFloat16>(32, bits.first, bits.second, false, true, 0, dynamic_index,
                                 Int3SessionMode::DynamicMetadata);
      RunInt3CudaCase<BFloat16>(32, bits.first, bits.second, true, true, 0, dynamic_index,
                                Int3SessionMode::DynamicMetadata);
    }
  }
}

TEST(QMoEInt3CudaTest, SameSessionRevalidatesDynamicZeroPoints) {
  for (auto bits : {std::pair{3, 3}, std::pair{3, 4}, std::pair{4, 3}}) {
    for (int dynamic_index : {11, 12}) {
      if ((dynamic_index == 11 ? bits.first : bits.second) != 3) continue;
      SCOPED_TRACE(::testing::Message() << bits.first << "/" << bits.second << ":" << dynamic_index);
      RunInt3CudaCase<MLFloat16>(32, bits.first, bits.second, false, true, 0, dynamic_index,
                                 Int3SessionMode::DynamicMetadata);
      RunInt3CudaCase<BFloat16>(32, bits.first, bits.second, true, true, 0, dynamic_index,
                                Int3SessionMode::DynamicMetadata);
    }
  }
}

TEST(QMoEInt3CudaTest, PrivatePrepackSurvivesInitializerReleaseAndSessionRecreation) {
  for (auto bits : {std::pair{3, 3}, std::pair{3, 4}, std::pair{4, 3}}) {
    SCOPED_TRACE(::testing::Message() << bits.first << "/" << bits.second);
    RunInt3CudaCase<MLFloat16>(64, bits.first, bits.second, false, true, 0, -1,
                               Int3SessionMode::PrivatePrepack);
    RunInt3CudaCase<BFloat16>(64, bits.first, bits.second, true, true, 0, -1,
                              Int3SessionMode::PrivatePrepack);
  }
}

TEST(QMoEInt3CudaTest, ScratchBudgetOneTokenBoundary) {
  constexpr int64_t per_row_bytes = 2 * (96 * sizeof(MLFloat16) + 3 * sizeof(int));
  for (auto bits : {std::pair{3, 3}, std::pair{3, 4}, std::pair{4, 3}}) {
    for (bool initializers : {false, true}) {
      for (int64_t budget : {per_row_bytes - 1, per_row_bytes, per_row_bytes + 1}) {
        SCOPED_TRACE(::testing::Message() << bits.first << "/" << bits.second << ":" << initializers << ":" << budget);
        RunInt3CudaCase<MLFloat16>(32, bits.first, bits.second, false, initializers, 0, -1,
                                   Int3SessionMode::None, {5, budget, false});
        RunInt3CudaCase<BFloat16>(32, bits.first, bits.second, true, initializers, 0, -1,
                                  Int3SessionMode::None, {5, budget, false});
      }
    }
  }
}

TEST(QMoEInt3CudaTest, EmptyRowsDoNotRequireOneTokenScratch) {
  for (auto bits : {std::pair{3, 3}, std::pair{3, 4}, std::pair{4, 3}}) {
    for (bool initializers : {false, true}) {
      SCOPED_TRACE(::testing::Message() << bits.first << "/" << bits.second << ":" << initializers);
      RunInt3CudaCase<MLFloat16>(32, bits.first, bits.second, false, initializers, 0, -1,
                                 Int3SessionMode::None, {0, 1, false});
      RunInt3CudaCase<BFloat16>(32, bits.first, bits.second, true, initializers, 0, -1,
                                Int3SessionMode::None, {0, 1, false});
    }
  }
}

TEST(QMoEInt3CudaTest, SchemaRejectsMalformedScaleShape) {
  for (auto bits : {std::pair{3, 3}, std::pair{3, 4}, std::pair{4, 3}}) {
    SCOPED_TRACE(::testing::Message() << bits.first << "/" << bits.second);
    RunInt3CudaCase<MLFloat16>(32, bits.first, bits.second, false, false, 0, -1,
                               Int3SessionMode::None, {5, -1, true});
    RunInt3CudaCase<BFloat16>(32, bits.first, bits.second, true, false, 0, -1,
                              Int3SessionMode::None, {5, -1, true});
  }
}

}  // namespace
}  // namespace onnxruntime::test
#endif
