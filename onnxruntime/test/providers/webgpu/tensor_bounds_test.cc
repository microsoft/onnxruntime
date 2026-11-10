// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <algorithm>
#include <limits>
#include <cmath>
#include <memory>
#include <vector>
#include "gtest/gtest.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
namespace onnxruntime::test {
#ifdef USE_WEBGPU
namespace {
std::unique_ptr<IExecutionProvider> CreateWebGpuProvider(bool nchw = false) {
  ConfigOptions options;
  ORT_THROW_IF_ERROR(options.AddConfigEntry("ep.webgpuexecutionprovider.validationMode", "full"));
  ORT_THROW_IF_ERROR(options.AddConfigEntry("ep.webgpuexecutionprovider.storageBufferCacheMode", "disabled"));
  ORT_THROW_IF_ERROR(options.AddConfigEntry("ep.webgpuexecutionprovider.enableInt64", "1"));
  if (nchw) {
    ORT_THROW_IF_ERROR(options.AddConfigEntry("ep.webgpuexecutionprovider.preferredLayout", "NCHW"));
  }
  return WebGpuExecutionProviderWithOptions(options);
}
void RunWebGpu(OpTester& test, std::unique_ptr<IExecutionProvider> provider) {
  SessionOptions options;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  options.graph_optimization_level = TransformerLevel::Default;
  test.Config(options).ConfigEp(std::move(provider)).RunWithConfig();
}
}  // namespace

TEST(WebGpuTensorBoundsTest, TopKPaddingFiniteMinimum) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();

  OpTester test("TopK", 11);
  const float v = -std::numeric_limits<float>::max();
  test.AddInput<float>("X", {3}, {v, v, v});
  test.AddInput<int64_t>("K", {1}, {3});
  test.AddOutput<float>("Y", {3}, {v, v, v});
  test.AddOutput<int64_t>("Indices", {3}, {0, 1, 2});
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuTensorBoundsTest, TopKLargePaddingFiniteMaximum) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();

  OpTester test("TopK", 11);
  test.AddAttribute<int64_t>("largest", 0);
  const float v = std::numeric_limits<float>::max();
  test.AddInput<float>("X", {2049}, std::vector<float>(2049, v));
  test.AddInput<int64_t>("K", {1}, {1});
  test.AddOutput<float>("Y", {1}, {v});
  test.AddOutput<int64_t>("Indices", {1}, {0});
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuTensorBoundsTest, TriluLargePositiveK) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();

  OpTester test("Trilu", 14);
  test.AddInput<float>("X", {2, 2}, {1, 2, 3, 4});
  test.AddInput<int64_t>("K", {}, {2147483647});
  test.AddOutput<float>("Y", {2, 2}, {0, 0, 0, 0});
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuTensorBoundsTest, RangeInt32DifferenceOverflow) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();

  OpTester test("Range", 11);
  test.AddInput<int32_t>("start", {}, {-2147483647 - 1});
  test.AddInput<int32_t>("limit", {}, {2147483647});
  test.AddInput<int32_t>("delta", {}, {2147483647});
  test.AddOutput<int32_t>("Y", {3}, {-2147483647 - 1, -1, 2147483646});
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuTensorBoundsTest, ResizeNearestRankOne) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();

  OpTester test("Resize", 13);
  test.AddAttribute<std::string>("mode", "nearest");
  test.AddInput<float>("X", {2}, {1, 2});
  test.AddInput<float>("roi", {0}, {});
  test.AddInput<float>("scales", {1}, {2});
  test.AddOutput<float>("Y", {4}, {1, 1, 2, 2});
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuTensorBoundsTest, ResizeCropUnitScale) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();

  OpTester test("Resize", 13);
  test.AddAttribute<std::string>("mode", "nearest");
  test.AddAttribute<std::string>("coordinate_transformation_mode", "tf_crop_and_resize");
  test.AddInput<float>("X", {2, 2}, {1, 2, 3, 4});
  test.AddInput<float>("roi", {4}, {0, 1, 1, 0});
  test.AddInput<float>("scales", {2}, {1, 1});
  test.AddOutput<float>("Y", {2, 2}, {2, 1, 4, 3});
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuTensorBoundsTest, SliceReversePartialRange) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();

  OpTester test("Slice", 13);
  test.AddInput<float>("X", {5}, {0, 1, 2, 3, 4});
  test.AddInput<int64_t>("starts", {1}, {3});
  test.AddInput<int64_t>("ends", {1}, {0});
  test.AddInput<int64_t>("axes", {1}, {0});
  test.AddInput<int64_t>("steps", {1}, {-1});
  test.AddOutput<float>("Y", {3}, {3, 2, 1});
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuTensorBoundsTest, ConvPointwiseTrailingPadding) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();

  OpTester test("Conv", 11);
  test.AddAttribute<std::vector<int64_t>>("pads", {0, 0, 1, 1});
  test.AddInput<float>("X", {1, 1, 2, 2}, {1, 2, 3, 4});
  test.AddInput<float>("W", {1, 1, 1, 1}, {1});
  test.AddOutput<float>("Y", {1, 1, 3, 3}, {1, 2, 0, 3, 4, 0, 0, 0, 0});
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuTensorBoundsTest, ConvTrailingPaddingLayoutsAndInitializers) {
  for (const bool nchw : {false, true}) {
    for (const bool initializer : {false, true}) {
      SCOPED_TRACE(testing::Message() << "nchw=" << nchw << " initializer=" << initializer);
      auto ep = CreateWebGpuProvider(nchw);
      if (!ep) GTEST_SKIP();
      OpTester test("Conv", 13);
      test.AddAttribute<std::vector<int64_t>>("pads", {0, 0, 1, 1});
      test.AddInput<float>("X", {1, 2, 2, 2}, {1, 2, 3, 4, 10, 20, 30, 40});
      test.AddInput<float>("W", {2, 2, 1, 1}, {1, 2, 3, 4}, initializer);
      test.AddOutput<float>("Y", {1, 2, 3, 3}, {21, 42, 0, 63, 84, 0, 0, 0, 0, 43, 86, 0, 129, 172, 0, 0, 0, 0});
      RunWebGpu(test, std::move(ep));
    }
  }
}

TEST(WebGpuTensorBoundsTest, ConvWholeInputKernelWithTrailingPadding) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();
  OpTester test("Conv", 13);
  test.AddAttribute<std::vector<int64_t>>("pads", {0, 0, 1, 1});
  test.AddInput<float>("X", {1, 1, 2, 2}, {1, 2, 3, 4});
  test.AddInput<float>("W", {1, 1, 2, 2}, {1, 1, 1, 1}, true);
  test.AddOutput<float>("Y", {1, 1, 2, 2}, {10, 6, 7, 4});
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuTensorBoundsTest, SliceReverseStridesAndAxes) {
  for (const int64_t step : {int64_t{-1}, int64_t{-2}, std::numeric_limits<int64_t>::min()}) {
    auto ep = CreateWebGpuProvider();
    if (!ep) GTEST_SKIP();
    OpTester test("Slice", 13);
    test.AddInput<float>("X", {2, 5}, {0, 1, 2, 3, 4, 10, 11, 12, 13, 14});
    test.AddInput<int64_t>("starts", {1}, {3});
    test.AddInput<int64_t>("ends", {1}, {std::numeric_limits<int64_t>::min()});
    test.AddInput<int64_t>("axes", {1}, {-1});
    test.AddInput<int64_t>("steps", {1}, {step});
    const std::vector<float> expected = step == -1   ? std::vector<float>{3, 2, 1, 0, 13, 12, 11, 10}
                                        : step == -2 ? std::vector<float>{3, 1, 13, 11}
                                                     : std::vector<float>{3, 13};
    test.AddOutput<float>("Y", {2, static_cast<int64_t>(expected.size() / 2)}, expected);
    RunWebGpu(test, std::move(ep));
  }
}

TEST(WebGpuTensorBoundsTest, SliceForwardVectorAndScalarViews) {
  for (const int64_t start : {0, 1, 4}) {
    auto ep = CreateWebGpuProvider();
    if (!ep) GTEST_SKIP();
    OpTester test("Slice", 13);
    test.AddInput<float>("X", {2, 8}, {0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17});
    test.AddInput<int64_t>("starts", {1}, {start});
    test.AddInput<int64_t>("ends", {1}, {start + 4});
    test.AddInput<int64_t>("axes", {1}, {1});
    std::vector<float> expected;
    for (int row = 0; row < 2; ++row)
      for (int j = 0; j < 4; ++j) expected.push_back(static_cast<float>(row * 10 + start + j));
    test.AddOutput<float>("Y", {2, 4}, expected);
    RunWebGpu(test, std::move(ep));
  }
}

TEST(WebGpuTensorBoundsTest, TopKPaddingInfinitiesAndDirections) {
  for (const bool largest : {false, true}) {
    for (const int64_t size : {3, 2049}) {
      auto ep = CreateWebGpuProvider();
      if (!ep) GTEST_SKIP();
      OpTester test("TopK", 11);
      test.AddAttribute<int64_t>("largest", largest ? 1 : 0);
      const float value = largest ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
      test.AddInput<float>("X", {size}, std::vector<float>(static_cast<size_t>(size), value));
      test.AddInput<int64_t>("K", {1}, {2});
      test.AddOutput<float>("Y", {2}, {value, value});
      test.AddOutput<int64_t>("Indices", {2}, {0, 1});
      RunWebGpu(test, std::move(ep));
    }
  }
}

TEST(WebGpuTensorBoundsTest, TriluExtremeOffsetsBothDirections) {
  for (const int64_t upper : {0, 1}) {
    for (const int64_t k : {int64_t{INT32_MIN}, int64_t{INT32_MAX}}) {
      auto ep = CreateWebGpuProvider();
      if (!ep) GTEST_SKIP();
      OpTester test("Trilu", 14);
      test.AddAttribute<int64_t>("upper", upper);
      const std::vector<float> input{1, 2, 3, 4};
      test.AddInput<float>("X", {2, 2}, input);
      test.AddInput<int64_t>("K", {}, {k});
      test.AddOutput<float>("Y", {2, 2}, (upper ? k < 0 : k > 0) ? input : std::vector<float>(4, 0));
      RunWebGpu(test, std::move(ep));
    }
  }
}

TEST(WebGpuTensorBoundsTest, ResizeNearestHighRank) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();
  OpTester test("Resize", 13);
  test.AddAttribute<std::string>("mode", "nearest");
  test.AddInput<float>("X", {1, 1, 1, 1, 1, 2}, {1, 2});
  test.AddInput<float>("roi", {0}, {});
  test.AddInput<float>("scales", {6}, {1, 1, 1, 1, 1, 2});
  test.AddOutput<float>("Y", {1, 1, 1, 1, 1, 4}, {1, 1, 2, 2});
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuTensorBoundsTest, RangeDescendingDifferenceOverflow) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();
  OpTester test("Range", 11);
  test.AddInput<int32_t>("start", {}, {INT32_MAX});
  test.AddInput<int32_t>("limit", {}, {INT32_MIN});
  test.AddInput<int32_t>("delta", {}, {-INT32_MAX});
  test.AddOutput<int32_t>("Y", {3}, {INT32_MAX, 0, -INT32_MAX});
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuTensorBoundsTest, RangeRejectsZeroDelta) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();
  OpTester test("Range", 11);
  test.AddInput<int32_t>("start", {}, {0});
  test.AddInput<int32_t>("limit", {}, {1});
  test.AddInput<int32_t>("delta", {}, {0});
  test.AddOutput<int32_t>("Y", {0}, {});
  test.Config(OpTester::ExpectResult::kExpectFailure, "delta");
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuTensorBoundsTest, SliceEmptyAttributes) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();
  OpTester test("Slice", 9);
  test.AddAttribute<std::vector<int64_t>>("starts", {});
  test.AddAttribute<std::vector<int64_t>>("ends", {});
  test.AddAttribute<std::vector<int64_t>>("axes", {});
  test.AddInput<float>("X", {4}, {1, 2, 3, 4});
  test.AddOutput<float>("Y", {4}, {1, 2, 3, 4});
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuTensorBoundsTest, ResizeLegacyNearestRankOne) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();
  OpTester test("Resize", 10);
  test.AddAttribute<std::string>("mode", "nearest");
  test.AddInput<float>("X", {2}, {1, 2});
  test.AddInput<float>("scales", {1}, {2});
  test.AddOutput<float>("Y", {4}, {1, 1, 2, 2});
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuTensorBoundsTest, ResizeUnitScaleStillAppliesHalfPixelOffset) {
  auto ep = CreateWebGpuProvider();
  if (!ep) GTEST_SKIP();
  OpTester test("Resize", 11);
  test.AddAttribute<std::string>("mode", "nearest");
  test.AddAttribute<std::string>("coordinate_transformation_mode", "tf_half_pixel_for_nn");
  test.AddAttribute<std::string>("nearest_mode", "ceil");
  test.AddInput<float>("X", {3}, {1, 2, 3});
  test.AddInput<float>("roi", {0}, {});
  test.AddInput<float>("scales", {1}, {1});
  test.AddOutput<float>("Y", {3}, {2, 3, 3});
  RunWebGpu(test, std::move(ep));
}

TEST(WebGpuTensorBoundsTest, TriluRectangularOffsets) {
  for (const bool upper : {false, true}) {
    auto ep = CreateWebGpuProvider();
    if (!ep) GTEST_SKIP();
    OpTester test("Trilu", 14);
    test.AddAttribute<int64_t>("upper", upper ? 1 : 0);
    test.AddInput<float>("X", {2, 3}, {1, 2, 3, 4, 5, 6});
    test.AddInput<int64_t>("K", {}, {upper ? 1 : -1});
    test.AddOutput<float>("Y", {2, 3}, upper ? std::vector<float>{0, 2, 3, 0, 0, 6} : std::vector<float>{0, 0, 0, 4, 0, 0});
    RunWebGpu(test, std::move(ep));
  }
}

#endif
}  // namespace onnxruntime::test
