// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/unittest_util/graph_transform_test_builder.h"

#ifndef DISABLE_CONTRIB_OPS
namespace onnxruntime {
namespace test {
namespace {

void TestResizeArgMax(const std::vector<int64_t>& shape, const std::vector<float>& data,
                      const std::vector<int64_t>& sizes, const std::string& coordinates = "half_pixel",
                      int64_t last = 0, int64_t keepdims = 0, const std::vector<float>& scales = {},
                      const std::string& guard = "") {
  auto build = [&](ModelTestBuilder& builder) {
    auto* input = guard == "int8" ? builder.MakeInput<int8_t>(shape, std::vector<int8_t>(data.begin(), data.end()))
                                  : builder.MakeInput<float>(shape, data);
    auto* roi = builder.MakeEmptyInput();
    auto* scale = scales.empty() ? builder.MakeEmptyInput() : builder.MakeInitializer<float>({4}, scales);
    auto* size = scales.empty() ? builder.MakeInitializer<int64_t>({4}, sizes) : builder.MakeEmptyInput();
    if (guard == "dynamic_sizes") size = builder.MakeInput<int64_t>({4}, sizes);
    auto* resized = guard == "graph_output" ? builder.MakeOutput() : builder.MakeIntermediate();
    auto& resize = builder.AddNode("Resize", {input, roi, scale, size}, {resized});
    resize.AddAttribute("mode", guard == "nearest" ? std::string("nearest") : std::string("linear"));
    resize.AddAttribute("coordinate_transformation_mode", coordinates);
    if (guard == "antialias") resize.AddAttribute("antialias", int64_t{1});
    if (guard == "shared") builder.AddNode("Abs", {resized}, {builder.MakeOutput()});
    auto& argmax = builder.AddNode("ArgMax", {resized}, {builder.MakeOutput()});
    argmax.AddAttribute("axis", guard == "axis" ? int64_t{3} : int64_t{-3});
    argmax.AddAttribute("keepdims", keepdims);
    argmax.AddAttribute("select_last_index", last);
  };
  auto check = [&](InferenceSessionWrapper& session) {
    const int expected = guard.empty() ? 1 : 0;
    auto counts = CountOpsInGraph(session.GetGraph());
    EXPECT_EQ(counts["com.microsoft.ResizeArgMax"], expected);
    EXPECT_EQ(counts["Resize"], 1 - expected);
    EXPECT_EQ(counts["ArgMax"], 1 - expected);
  };
  TransformerTester(build, check, TransformerLevel::Level1, TransformerLevel::Level2,
                    guard == "antialias" ? 18 : 13);
}

}  // namespace

TEST(ResizeArgMaxFusionTest, BatchTileTailAndCoordinateModes) {
  const std::vector<float> values{2, 3, 2, 3, 2, 3, 0, 1, 0, 1, 0, 1, 4, -2, 4, -2, 4, -2,
                                  -3, -2, -3, -2, -3, -2, -5, -4, -5, -4, -5, -4, 0, -6, 0, -6, 0, -6};
  for (const auto* mode : {"half_pixel", "align_corners", "asymmetric", "pytorch_half_pixel"}) {
    SCOPED_TRACE(mode);
    TestResizeArgMax({2, 3, 2, 3}, values, {2, 3, 5, 257}, mode);
  }
}

TEST(ResizeArgMaxFusionTest, RoundedTies) {
  // A strict source difference can become a tie after interpolation.
  const std::vector<float> values{std::nextafter(1.f, 0.f), 1, 1, 1, 1, 1, 1, 1, .1f, .1f, .1f, .1f};
  for (int64_t last : {0, 1}) {
    SCOPED_TRACE(last);
    TestResizeArgMax({1, 3, 2, 2}, values, {1, 3, 5, 257}, "half_pixel", last, last);
  }
}

TEST(ResizeArgMaxFusionTest, NonFiniteValues) {
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  for (int64_t last : {0, 1}) {
    SCOPED_TRACE(last);
    // Inf cannot prove dominance: a zero coefficient produces NaN.
    TestResizeArgMax({1, 3, 1, 2}, {0, 0, inf, inf, 1, 1}, {1, 3, 1, 16}, "half_pixel", last);
    // Resize copies unchanged shapes. Channel zero initializes ArgMax, even for NaN.
    TestResizeArgMax({1, 3, 1, 2}, {nan, -inf, 2, inf, 3, inf}, {1, 3, 1, 2}, "half_pixel", last);
  }
}

TEST(ResizeArgMaxFusionTest, FractionalScales) {
  // 5 * 3.3 rounds down to 16. Using 16 / 5 as the scale changes the labels.
  TestResizeArgMax({1, 2, 1, 5}, {0, 1, 2, 3, 4, 3.95f, 3.95f, 3.95f, 3.95f, 3.95f}, {},
                   "half_pixel", 0, 0, {1, 1, 1, 3.3f});
}

TEST(ResizeArgMaxFusionTest, SingleChannelAndUnitOutput) {
  TestResizeArgMax({1, 1, 2, 2}, {0, 1, 2, 3}, {1, 1, 1, 1}, "pytorch_half_pixel", 1, 1);
}

TEST(ResizeArgMaxFusionTest, UnsupportedGraphs) {
  const std::vector<float> values{0, 1, 2, 3, 4, 5, 6, 7};
  for (const auto* guard : {"graph_output", "shared", "axis", "nearest", "dynamic_sizes", "antialias", "int8"}) {
    SCOPED_TRACE(guard);
    TestResizeArgMax({1, 2, 2, 2}, values, {1, 2, 4, 4}, "half_pixel", 0, 0, {}, guard);
  }
  // This is a valid NHWC Resize. It must not become an NCHW fusion.
  TestResizeArgMax({1, 2, 2, 2}, values, {1, 4, 4, 2}, "half_pixel", 0, 0, {}, "channels");
}

}  // namespace test
}  // namespace onnxruntime
#endif  // DISABLE_CONTRIB_OPS
