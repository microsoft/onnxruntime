// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "core/session/onnxruntime_session_options_config_keys.h"
#include "default_providers.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {
namespace {

struct ConvTransposeTestAttributes {
  std::vector<int64_t> strides;
  std::vector<int64_t> dilations;
  std::vector<int64_t> pads;
  std::vector<int64_t> output_padding;
  std::vector<int64_t> output_shape;
  const char* auto_pad = "NOTSET";
  int64_t group = 1;
};

// Layout, float16, initializer weights/bias, opset.
class ConvTransposeWebGpuTestBase : public testing::TestWithParam<std::tuple<bool, bool, bool, int>> {
 protected:
  void Run(const TensorShapeVector& input_shape,
           const TensorShapeVector& weight_shape,
           const TensorShapeVector& output_shape,
           const ConvTransposeTestAttributes& attrs = {}, bool has_bias = false,
           bool disable_prepacking = false) {
    const auto [is_nhwc, is_fp16, initializer, opset] = GetParam();
    auto ep = DefaultWebGpuExecutionProvider(is_nhwc);
    if (!ep) {
      GTEST_SKIP() << "WebGPU execution provider is not available.";
    }

    std::vector<float> x(static_cast<size_t>(TensorShape(input_shape).Size()));
    std::vector<float> w(static_cast<size_t>(TensorShape(weight_shape).Size()));
    std::vector<float> bias(static_cast<size_t>(output_shape[1]));
    for (size_t i = 0; i < x.size(); ++i) {
      x[i] = static_cast<float>(static_cast<int>(i % 13) - 6) * 0.125f;
    }
    for (size_t i = 0; i < w.size(); ++i) {
      w[i] = static_cast<float>(static_cast<int>(i % 7) - 3) * 0.25f;
    }
    for (size_t i = 0; i < bias.size(); ++i) {
      bias[i] = static_cast<float>(i + 1) * 0.5f;
    }
    auto add_attributes = [&](OpTester& test) {
      test.AddAttribute("group", attrs.group);
      if (!attrs.strides.empty()) test.AddAttribute("strides", attrs.strides);
      if (!attrs.dilations.empty()) test.AddAttribute("dilations", attrs.dilations);
      if (!attrs.pads.empty()) test.AddAttribute("pads", attrs.pads);
      if (!attrs.output_padding.empty()) test.AddAttribute("output_padding", attrs.output_padding);
      if (!attrs.output_shape.empty()) test.AddAttribute("output_shape", attrs.output_shape);
      if (attrs.pads.empty()) test.AddAttribute("auto_pad", std::string(attrs.auto_pad));
    };

    // The CPU kernel supplies an independent reference. All input values are exactly
    // representable in float16, so both data types use the same reference inputs.
    std::vector<float> expected;
    OpTester cpu_test("ConvTranspose", opset);
    add_attributes(cpu_test);
    cpu_test.AddInput<float>("X", input_shape, x);
    cpu_test.AddInput<float>("W", weight_shape, w, initializer);
    if (has_bias) cpu_test.AddInput<float>("B", {output_shape[1]}, bias, initializer);
    cpu_test.AddOutput<float>("Y", output_shape,
                              std::vector<float>(static_cast<size_t>(TensorShape(output_shape).Size())));
    cpu_test.SetCustomOutputVerifier([&](const std::vector<OrtValue>& fetches, const std::string&) {
      const auto& tensor = fetches[0].Get<Tensor>();
      EXPECT_EQ(tensor.Shape(), TensorShape(output_shape));
      expected.assign(tensor.Data<float>(), tensor.Data<float>() + tensor.Shape().Size());
    });
    ASSERT_NO_FATAL_FAILURE(cpu_test.ConfigEp(DefaultCpuExecutionProvider()).RunWithConfig());
    ASSERT_EQ(expected.size(), static_cast<size_t>(TensorShape(output_shape).Size()));

    OpTester test("ConvTranspose", opset);
    add_attributes(test);
    if (is_fp16) {
      test.AddInput<MLFloat16>("X", input_shape, ToFloat16(x));
      test.AddInput<MLFloat16>("W", weight_shape, ToFloat16(w), initializer);
      if (has_bias) test.AddInput<MLFloat16>("B", {output_shape[1]}, ToFloat16(bias), initializer);
      test.AddOutput<MLFloat16>("Y", output_shape, ToFloat16(expected));
    } else {
      test.AddInput<float>("X", input_shape, x);
      test.AddInput<float>("W", weight_shape, w, initializer);
      if (has_bias) test.AddInput<float>("B", {output_shape[1]}, bias, initializer);
      test.AddOutput<float>("Y", output_shape, expected);
    }
    SessionOptions options;
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
    if (disable_prepacking) {
      ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigDisablePrepacking, "1"));
    }
    test.SetNumRunCalls(2);
    size_t prepacked_weights = 0;
    ASSERT_NO_FATAL_FAILURE(test.Config(options).ConfigEp(std::move(ep)).RunWithConfig(&prepacked_weights));
    if (testing::Test::IsSkipped()) {
      return;
    }
    EXPECT_EQ(prepacked_weights, initializer && !disable_prepacking ? 1U : 0U);
  }
};

using ConvTranspose3DWebGpuTest = ConvTransposeWebGpuTestBase;

class ConvTransposePrepackWebGpuTest : public ConvTransposeWebGpuTestBase {
 protected:
  void Run1DAnd2D(int64_t input_channels_per_group, int64_t output_channels_per_group,
                  int64_t group = 1, bool has_bias = true, bool use_strides_and_dilations = false,
                  bool disable_prepacking = false) {
    for (int spatial_rank : {1, 2}) {
      SCOPED_TRACE(spatial_rank);
      ConvTransposeTestAttributes attrs;
      attrs.group = group;
      TensorShapeVector input_shape{2, group * input_channels_per_group, 3};
      TensorShapeVector weight_shape{input_shape[1], output_channels_per_group, 3};
      TensorShapeVector output_shape{2, group * output_channels_per_group, 5};
      if (use_strides_and_dilations) {
        attrs.strides = {2};
        attrs.dilations = {2};
        attrs.pads = {1, 0};
        attrs.output_padding = {1};
        output_shape[2] = 9;
      }
      if (spatial_rank == 2) {
        input_shape.push_back(2);
        weight_shape.push_back(2);
        output_shape.push_back(use_strides_and_dilations ? 5 : 3);
        if (use_strides_and_dilations) {
          attrs.strides = {2, 3};
          attrs.dilations = {2, 1};
          attrs.pads = {1, 0, 0, 1};
          attrs.output_padding = {1, 1};
        }
      }
      Run(input_shape, weight_shape, output_shape, attrs, has_bias, disable_prepacking);
    }
  }
};

TEST_P(ConvTransposePrepackWebGpuTest, ScalarChannels) {
  Run1DAnd2D(3, 3, 1, false);
}

TEST_P(ConvTransposePrepackWebGpuTest, GroupedVec2Channels) {
  Run1DAnd2D(6, 6, 2);
}

TEST_P(ConvTransposePrepackWebGpuTest, GroupedVec4Channels) {
  Run1DAnd2D(8, 4, 2);
}

TEST_P(ConvTransposePrepackWebGpuTest, Vec2InputVec4Output) {
  Run1DAnd2D(2, 4, 2);
}

TEST_P(ConvTransposePrepackWebGpuTest, Vec4InputVec2Output) {
  Run1DAnd2D(4, 2, 2);
}

TEST_P(ConvTransposePrepackWebGpuTest, SingleOutputChannel) {
  // Cover both the vec2 dot path and the four-input-channel packing path.
  Run1DAnd2D(2, 1, 2);
  Run1DAnd2D(8, 1, 2);
}

TEST_P(ConvTransposePrepackWebGpuTest, SingleOutputChannelRemainders) {
  for (int64_t channels : {5, 6, 7}) {
    SCOPED_TRACE(channels);
    Run1DAnd2D(channels, 1, 2);
  }
}

TEST_P(ConvTransposePrepackWebGpuTest, Depthwise) {
  Run1DAnd2D(1, 1, 3);
}

TEST_P(ConvTransposePrepackWebGpuTest, StridesDilationsAndPadding) {
  Run1DAnd2D(4, 2, 2, true, true);
}

TEST_P(ConvTransposePrepackWebGpuTest, PrepackingDisabled) {
  Run1DAnd2D(4, 2, 2, true, false, true);
}

TEST_P(ConvTransposePrepackWebGpuTest, HeightOne2DKernel) {
  // This packs to the same shape as 1D weights, but must retain its 2D semantics.
  Run({2, 8, 3, 4}, {8, 4, 1, 3}, {2, 4, 3, 6});
}

INSTANTIATE_TEST_SUITE_P(WebGPU, ConvTransposePrepackWebGpuTest,
                         testing::Combine(testing::Bool(), testing::Bool(), testing::Bool(), testing::Values(10, 11)));

TEST(ConvTransposeWebGpuTest, InputWeightRankMismatch) {
  for (bool initializer : {false, true}) {
    SCOPED_TRACE(initializer);
    for (size_t input_rank : {3U, 4U, 5U}) {
      SCOPED_TRACE(input_rank);
      for (size_t weight_rank : {3U, 4U, 5U}) {
        if (input_rank == weight_rank) {
          continue;
        }
        SCOPED_TRACE(weight_rank);
        auto ep = DefaultWebGpuExecutionProvider(false);
        if (!ep) {
          GTEST_SKIP() << "WebGPU execution provider is not available.";
        }
        TensorShapeVector input_shape{1, 2};
        input_shape.insert(input_shape.end(), input_rank - 2, 1);
        TensorShapeVector weight_shape{2, 3};
        weight_shape.insert(weight_shape.end(), weight_rank - 2, 1);

        OpTester test("ConvTranspose", 11);
        // Leave ranks unknown to graph inference so the kernel validates them at runtime.
        test.AddShapeToTensorData(false);
        test.AddInput<float>("X", input_shape, {1.0f, 2.0f});
        test.AddInput<float>("W", weight_shape, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, initializer);
        test.AddOutput<float>("Y", {1}, {0.0f});
        SessionOptions options;
        ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
        size_t prepacked_weights = 0;
        ASSERT_NO_FATAL_FAILURE(test.Config(options)
                                    .Config(OpTester::ExpectResult::kExpectFailure,
                                            "X num_dims does not match W num_dims.")
                                    .ConfigEp(std::move(ep))
                                    .RunWithConfig(&prepacked_weights));
        EXPECT_EQ(prepacked_weights, initializer ? 1U : 0U);
      }
    }
  }
}

TEST(ConvTransposeWebGpuTest, UnsupportedSpatialRank) {
  auto ep = DefaultWebGpuExecutionProvider();
  if (!ep) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }
  OpTester test("ConvTranspose", 11);
  test.AddInput<float>("X", {1, 1, 1, 1, 1, 1}, {1.0f});
  test.AddInput<float>("W", {1, 1, 1, 1, 1, 1}, {1.0f});
  test.AddOutput<float>("Y", {1, 1, 1, 1, 1, 1}, {1.0f});
  test.Config(OpTester::ExpectResult::kExpectFailure,
              "Only ConvTranspose1d, ConvTranspose2d, and ConvTranspose3d are supported.")
      .ConfigEp(std::move(ep))
      .RunWithConfig();
}

TEST_P(ConvTranspose3DWebGpuTest, DefaultAttributes) {
  Run({2, 3, 2, 3, 2}, {3, 2, 2, 2, 3}, {2, 2, 3, 4, 4});
}

TEST_P(ConvTranspose3DWebGpuTest, GroupsAndBias) {
  ConvTransposeTestAttributes attrs;
  attrs.group = 2;
  Run({2, 4, 2, 2, 3}, {4, 3, 2, 3, 1}, {2, 6, 3, 4, 3}, attrs, true);
}

TEST_P(ConvTranspose3DWebGpuTest, Depthwise) {
  ConvTransposeTestAttributes attrs;
  attrs.group = 3;
  Run({1, 3, 2, 2, 2}, {3, 1, 2, 2, 2}, {1, 3, 3, 3, 3}, attrs, true);
}

TEST_P(ConvTranspose3DWebGpuTest, Vec4GroupsAndBias) {
  ConvTransposeTestAttributes attrs;
  attrs.group = 2;
  // Eight channels per group exercises multiple vec4 loads and a nonzero group offset.
  Run({2, 16, 2, 2, 2}, {16, 3, 2, 1, 3}, {2, 6, 3, 2, 4}, attrs, true);
}

TEST_P(ConvTranspose3DWebGpuTest, Vec2GroupAlignment) {
  ConvTransposeTestAttributes attrs;
  attrs.group = 2;
  // Total channels are divisible by four, but each group's six channels require vec2.
  Run({2, 12, 2, 2, 2}, {12, 3, 2, 1, 3}, {2, 6, 3, 2, 4}, attrs, true);
}

TEST_P(ConvTranspose3DWebGpuTest, StridesDilationsAndAsymmetricPadding) {
  ConvTransposeTestAttributes attrs;
  attrs.strides = {2, 3, 2};
  attrs.dilations = {2, 1, 2};
  attrs.pads = {1, 0, 3, 0, 1, 1};
  attrs.output_padding = {1, 2, 1};
  Run({1, 2, 2, 2, 2}, {2, 3, 2, 2, 3}, {1, 3, 5, 6, 4}, attrs, true);
}

TEST_P(ConvTranspose3DWebGpuTest, SameUpper) {
  ConvTransposeTestAttributes attrs;
  attrs.strides = {2, 2, 2};
  attrs.auto_pad = "SAME_UPPER";
  Run({1, 2, 2, 2, 2}, {2, 3, 3, 3, 3}, {1, 3, 4, 4, 4}, attrs);
}

TEST_P(ConvTranspose3DWebGpuTest, SameLower) {
  ConvTransposeTestAttributes attrs;
  attrs.strides = {2, 2, 2};
  attrs.auto_pad = "SAME_LOWER";
  Run({1, 2, 2, 2, 2}, {2, 3, 3, 3, 3}, {1, 3, 4, 4, 4}, attrs, true);
}

TEST_P(ConvTranspose3DWebGpuTest, ExplicitOutputShape) {
  ConvTransposeTestAttributes attrs;
  attrs.strides = {2, 2, 2};
  attrs.output_shape = {3, 3, 3};
  Run({1, 2, 2, 2, 2}, {2, 3, 3, 2, 2}, {1, 3, 3, 3, 3}, attrs);
}

TEST_P(ConvTranspose3DWebGpuTest, PaddingLargerThanKernel) {
  ConvTransposeTestAttributes attrs;
  attrs.pads = {2, 1, 0, 0, 0, 0};
  Run({1, 1, 4, 3, 2}, {1, 1, 2, 2, 2}, {1, 1, 3, 3, 3}, attrs);
}

TEST_P(ConvTranspose3DWebGpuTest, LargeSpatialCoordinates) {
  Run({1, 1, 1, 1, 2051}, {1, 1, 1, 1, 2}, {1, 1, 1, 1, 2052});
}

INSTANTIATE_TEST_SUITE_P(WebGPU, ConvTranspose3DWebGpuTest,
                         testing::Combine(testing::Bool(), testing::Bool(), testing::Bool(), testing::Values(10, 11)));

}  // namespace
}  // namespace test
}  // namespace onnxruntime
