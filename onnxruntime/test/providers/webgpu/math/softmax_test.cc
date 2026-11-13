// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <cstdint>
#include <memory>
#include <limits>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

#ifdef USE_WEBGPU
template <typename T>
static void TestLogSoftmaxWebGpu(double margin) {
  for (int64_t classes : {1, 3, 33, 63, 64, 65, 255, 256, 257, 1024, 4097}) {
    for (int64_t rows : {1, 2}) {
      for (int opset : {12, 13}) {
        for (bool tied : {false, true}) {
          auto ep = DefaultWebGpuExecutionProvider();
          if (!ep) GTEST_SKIP() << "WebGPU is unavailable";
          const int64_t pivot = classes / 2;
          std::vector<T> input(static_cast<size_t>(rows * classes), T(static_cast<float>(-margin / 2)));
          for (int64_t row = 0; row < rows; ++row) {
            input[static_cast<size_t>(row * classes + pivot)] = T(static_cast<float>(margin / 2));
            if (tied) input[static_cast<size_t>(row * classes)] = T(static_cast<float>(margin / 2));
          }
          const int64_t max_count = tied && pivot != 0 ? 2 : 1;
          const double normalizer = std::log1p((max_count - 1) + (classes - max_count) * std::exp(-margin));
          OpTester test("LogSoftmax", opset);
          test.AddInput<T>("X", {rows, classes}, input);
          test.AddOutput<T>("Y", {rows, classes}, std::vector<T>(input.size()));
          test.SetCustomOutputVerifier([=](const std::vector<OrtValue>& fetches, const std::string&) {
            const auto output = fetches[0].Get<Tensor>().DataAsSpan<T>();
            for (size_t j = 0; j < input.size(); ++j) {
              const float expected = static_cast<float>(T(static_cast<float>(
                  static_cast<double>(input[j]) - margin / 2 - normalizer)));
              EXPECT_NEAR(static_cast<float>(output[j]), expected, std::abs(expected) * 2e-3)
                  << "classes=" << classes << " rows=" << rows << " index=" << j << " tied=" << tied;
            }
          });
          SessionOptions options;
          ASSERT_STATUS_OK(options.config_options.AddConfigEntry("session.disable_cpu_ep_fallback", "1"));
          std::vector<std::unique_ptr<IExecutionProvider>> eps;
          eps.push_back(std::move(ep));
          test.Run(options, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &eps);
        }
      }
    }
  }
}

TEST(LogSoftmaxOperator, WebGpuCollapseFloat) {
  TestLogSoftmaxWebGpu<float>(10.0);
  TestLogSoftmaxWebGpu<float>(18.0);
  TestLogSoftmaxWebGpu<float>(80.0);
}

TEST(LogSoftmaxOperator, WebGpuCollapseHalf) {
  TestLogSoftmaxWebGpu<MLFloat16>(10.0);
}

TEST(LogSoftmaxOperator, WebGpuAxis) {
  for (int opset : {12, 13}) {
    auto ep = DefaultWebGpuExecutionProvider();
    if (!ep) GTEST_SKIP() << "WebGPU is unavailable";
    OpTester test("LogSoftmax", opset);
    test.AddAttribute("axis", int64_t(1));
    const float normalizer = static_cast<float>(std::log1p(std::exp(-10.0)));
    test.AddInput<float>("X", {1, 2, 2}, {5, -5, -5, 5});
    if (opset < 13) {
      const float tied_normalizer = static_cast<float>(std::log1p(1.0 + 2.0 * std::exp(-10.0)));
      test.AddOutput<float>("Y", {1, 2, 2},
                            {-tied_normalizer, -10 - tied_normalizer, -10 - tied_normalizer, -tied_normalizer});
    } else {
      test.AddOutput<float>("Y", {1, 2, 2}, {-normalizer, -10 - normalizer, -10 - normalizer, -normalizer});
    }
    test.SetOutputAbsErr("Y", 1e-7f);
    test.SetOutputRelErr("Y", 2e-5f);
    SessionOptions options;
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry("session.disable_cpu_ep_fallback", "1"));
    std::vector<std::unique_ptr<IExecutionProvider>> eps;
    eps.push_back(std::move(ep));
    test.Run(options, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &eps);
  }
}

TEST(LogSoftmaxOperator, WebGpuNonfinite) {
  const float inf = std::numeric_limits<float>::infinity();
  for (int64_t classes : {3, 257}) {
    for (float exceptional : {inf, -inf, std::numeric_limits<float>::quiet_NaN()}) {
      auto ep = DefaultWebGpuExecutionProvider();
      if (!ep) GTEST_SKIP() << "WebGPU is unavailable";
      std::vector<float> input(static_cast<size_t>(classes), -inf);
      input[static_cast<size_t>(classes / 2)] = exceptional;
      OpTester test("LogSoftmax", 13);
      test.AddInput<float>("X", {1, classes}, input);
      test.AddOutput<float>("Y", {1, classes}, std::vector<float>(input.size()));
      test.SetCustomOutputVerifier([](const std::vector<OrtValue>& fetches, const std::string&) {
        for (float value : fetches[0].Get<Tensor>().DataAsSpan<float>()) EXPECT_TRUE(std::isnan(value));
      });
      SessionOptions options;
      ASSERT_STATUS_OK(options.config_options.AddConfigEntry("session.disable_cpu_ep_fallback", "1"));
      std::vector<std::unique_ptr<IExecutionProvider>> eps;
      eps.push_back(std::move(ep));
      test.Run(options, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &eps);
    }
  }
}

TEST(SoftmaxOperator, webgpu_nan) {
  OpTester test("Softmax", 13);  // axis default is -1

  std::vector<float> x_vals = {-INFINITY, -INFINITY, -INFINITY};
  std::vector<float> expected_result = {0.0f, 0.0f, 0.0f};
  std::vector<int64_t> dimensions = {1, 3};

  test.AddInput<float>("X", dimensions, x_vals);
  test.AddOutput<float>("Y", dimensions, expected_result);

  // explicitly disable for EPs that do not handle NaN
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kCpuExecutionProvider, kCoreMLExecutionProvider, kDmlExecutionProvider});
}
#endif

}  // namespace test
}  // namespace onnxruntime
