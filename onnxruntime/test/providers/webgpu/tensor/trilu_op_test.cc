// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

#ifdef USE_WEBGPU
namespace {
void RunWithWebGpu(OpTester& test) {
  if (DefaultWebGpuExecutionProvider().get() == nullptr) {
    GTEST_SKIP() << "WebGPU EP not available";
  }

  auto provider = DefaultWebGpuExecutionProvider();
  test.ConfigEp(std::move(provider)).RunWithConfig();
}
}  // namespace

TEST(TriluOpTest, webgpu_upper_default_k) {
  OpTester test("Trilu", 14, kOnnxDomain);
  test.AddInput<float>("X", {2, 2}, {4.f, 7.f, 2.f, 6.f});
  test.AddOutput<float>("Y", {2, 2}, {4.f, 7.f, 0.f, 6.f});
  RunWithWebGpu(test);
}

TEST(TriluOpTest, webgpu_lower_default_k) {
  OpTester test("Trilu", 14, kOnnxDomain);
  test.AddAttribute<int64_t>("upper", 0);
  test.AddInput<float>("X", {2, 2}, {4.f, 7.f, 2.f, 6.f});
  test.AddOutput<float>("Y", {2, 2}, {4.f, 0.f, 2.f, 6.f});
  RunWithWebGpu(test);
}

TEST(TriluOpTest, webgpu_upper_positive_k) {
  OpTester test("Trilu", 14, kOnnxDomain);
  test.AddInput<float>("X", {3, 4},
                       {4.f, 1.f, 5.f, 8.f,
                        4.f, 3.f, 2.f, 4.f,
                        6.f, 1.f, 2.f, 3.f});
  test.AddInput<int64_t>("k", {}, {1});
  test.AddOutput<float>("Y", {3, 4},
                        {0.f, 1.f, 5.f, 8.f,
                         0.f, 0.f, 2.f, 4.f,
                         0.f, 0.f, 0.f, 3.f});
  RunWithWebGpu(test);
}

TEST(TriluOpTest, webgpu_lower_negative_k) {
  OpTester test("Trilu", 14, kOnnxDomain);
  test.AddAttribute<int64_t>("upper", 0);
  test.AddInput<float>("X", {3, 4},
                       {4.f, 1.f, 5.f, 8.f,
                        4.f, 3.f, 2.f, 4.f,
                        6.f, 1.f, 2.f, 3.f});
  test.AddInput<int64_t>("k", {}, {-1});
  test.AddOutput<float>("Y", {3, 4},
                        {0.f, 0.f, 0.f, 0.f,
                         4.f, 0.f, 0.f, 0.f,
                         6.f, 1.f, 0.f, 0.f});
  RunWithWebGpu(test);
}

TEST(TriluOpTest, webgpu_batched_upper) {
  OpTester test("Trilu", 14, kOnnxDomain);
  test.AddInput<float>("X", {2, 3, 4},
                       {4.f, 1.f, 5.f, 8.f,
                        4.f, 3.f, 2.f, 4.f,
                        6.f, 1.f, 2.f, 3.f,
                        1.f, 6.f, 2.f, 1.f,
                        4.f, 1.f, 5.f, 8.f,
                        4.f, 3.f, 2.f, 4.f});
  test.AddInput<int64_t>("k", {}, {1});
  test.AddOutput<float>("Y", {2, 3, 4},
                        {0.f, 1.f, 5.f, 8.f,
                         0.f, 0.f, 2.f, 4.f,
                         0.f, 0.f, 0.f, 3.f,
                         0.f, 6.f, 2.f, 1.f,
                         0.f, 0.f, 5.f, 8.f,
                         0.f, 0.f, 0.f, 4.f});
  RunWithWebGpu(test);
}
#endif  // USE_WEBGPU

}  // namespace test
}  // namespace onnxruntime
