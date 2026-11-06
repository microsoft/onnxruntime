// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

#ifdef USE_WEBGPU
#include "core/providers/webgpu/webgpu_provider_options.h"

namespace onnxruntime::test {
namespace {

template <typename T>
void RunIntegerUnary(const char* op, int first_version, int latest_version,
                     const std::vector<T>& input, const std::vector<T>& expected) {
  for (int version : {first_version, latest_version}) {
    for (bool initializer : {false, true}) {
      for (const auto& shape : {std::vector<int64_t>{9}, std::vector<int64_t>{1, 3, 3},
                                std::vector<int64_t>{}, std::vector<int64_t>{0}}) {
        SCOPED_TRACE(version);
        SCOPED_TRACE(initializer);
        SCOPED_TRACE(shape);
        const bool scalar = shape.empty();
        const bool empty = shape == std::vector<int64_t>{0};
        const auto values = empty ? std::vector<T>{} : scalar ? std::vector<T>{input.front()}
                                                              : input;
        const auto results = empty ? std::vector<T>{} : scalar ? std::vector<T>{expected.front()}
                                                               : expected;
        OpTester test(op, version);
        test.AddInput<T>("X", shape, values, initializer);
        test.AddOutput<T>("Y", shape, results);
        SessionOptions options;
        // Initializers must execute the GPU kernel, not constant-fold on CPU.
        options.graph_optimization_level = TransformerLevel::Default;
        ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
        ConfigOptions provider_options;
        ASSERT_STATUS_OK(provider_options.AddConfigEntry(webgpu::options::kEnableInt64, "1"));
        auto provider = WebGpuExecutionProviderWithOptions(provider_options);
        ASSERT_NE(provider, nullptr);
        test.Config(options).ConfigEp(std::move(provider)).RunWithConfig();
      }
    }
  }
}

const std::vector<int8_t> kInt8Input{-128, -127, -17, -1, 0, 1, 17, 127, -64};
constexpr int64_t kLarge = int64_t{1} << 40;
constexpr int64_t kMin = std::numeric_limits<int64_t>::min();
constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
const std::vector<int64_t> kInt64Input{kMin, -kLarge, -kLarge - 17, -1, 0, 1,
                                       kLarge + 17, kMax, int64_t{1} << 32};

TEST(WebGpuIntegerUnaryTest, AbsInt8) {
  RunIntegerUnary<int8_t>("Abs", 6, 13, kInt8Input, {-128, 127, 17, 1, 0, 1, 17, 127, 64});
}

TEST(WebGpuIntegerUnaryTest, NegInt8) {
  RunIntegerUnary<int8_t>("Neg", 6, 13, kInt8Input, {-128, 127, 17, 1, 0, -1, -17, -127, 64});
}

TEST(WebGpuIntegerUnaryTest, ReluInt8) {
  RunIntegerUnary<int8_t>("Relu", 14, 21, kInt8Input, {0, 0, 0, 0, 0, 1, 17, 127, 0});
}

TEST(WebGpuIntegerUnaryTest, SignInt8) {
  RunIntegerUnary<int8_t>("Sign", 9, 13, kInt8Input, {-1, -1, -1, -1, 0, 1, 1, 1, -1});
}

TEST(WebGpuIntegerUnaryTest, AbsInt64) {
  RunIntegerUnary<int64_t>("Abs", 6, 13, kInt64Input,
                           {kMin, kLarge, kLarge + 17, 1, 0, 1, kLarge + 17, kMax, int64_t{1} << 32});
}

TEST(WebGpuIntegerUnaryTest, NegInt64) {
  RunIntegerUnary<int64_t>("Neg", 6, 13, kInt64Input,
                           {kMin, kLarge, kLarge + 17, 1, 0, -1, -kLarge - 17, -kMax, -(int64_t{1} << 32)});
}

TEST(WebGpuIntegerUnaryTest, ReluInt64) {
  RunIntegerUnary<int64_t>("Relu", 14, 21, kInt64Input,
                           {0, 0, 0, 0, 0, 1, kLarge + 17, kMax, int64_t{1} << 32});
}

TEST(WebGpuIntegerUnaryTest, SignInt64) {
  RunIntegerUnary<int64_t>("Sign", 9, 13, kInt64Input, {-1, -1, -1, -1, 0, 1, 1, 1, 1});
}

}  // namespace
}  // namespace onnxruntime::test
#endif
