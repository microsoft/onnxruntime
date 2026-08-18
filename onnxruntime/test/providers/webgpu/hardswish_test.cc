// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <type_traits>
#include <vector>

#include "gtest/gtest.h"

#include "default_providers.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

template <typename T>
void RunHardSwishTest() {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }

  const std::vector<int64_t> kDims{2, 5};
  const std::vector<float> input_values{-6.0f, -3.0f, -1.0f, 0.0f, 1.0f, 3.0f, 6.0f, 8.0f, -8.0f, 0.5f};
  std::vector<float> expected_values;
  expected_values.reserve(input_values.size());
  std::transform(input_values.cbegin(), input_values.cend(), std::back_inserter(expected_values),
                 [](float x) { return x * std::max(0.0f, std::min(1.0f, x / 6.0f + 0.5f)); });

  OpTester test("HardSwish", 14);
  if constexpr (std::is_same_v<T, float>) {
    test.AddInput<T>("X", kDims, input_values);
    test.AddOutput<T>("Y", kDims, expected_values);
  } else {
    test.AddInput<T>("X", kDims, FloatsToMLFloat16s(input_values));
    test.AddOutput<T>("Y", kDims, FloatsToMLFloat16s(expected_values));
    test.SetOutputAbsErr("Y", 0.01f);
    test.SetOutputRelErr("Y", 0.01f);
  }

  test.ConfigEp(std::move(webgpu_ep)).RunWithConfig();
}

TEST(HardSwish_WebGPU, Float32) {
  RunHardSwishTest<float>();
}

TEST(HardSwish_WebGPU, Float16) {
  RunHardSwishTest<MLFloat16>();
}

}  // namespace test
}  // namespace onnxruntime
