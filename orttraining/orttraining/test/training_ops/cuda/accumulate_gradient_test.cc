// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {
namespace optimizer {

// #if USE_CUDA

namespace {

TEST(AccumulateGradientTest, Simple) {
  OpTester test("AccumulateGradient", 1, onnxruntime::kMSDomain);
  // void AddInput(const char* name, std::initializer_list<int64_t> dims, std::initializer_list<T> values,
  //             bool is_initializer = false, const std::vector<std::string>* dim_params = nullptr,
  //             bool is_strided_tensor = false, std::initializer_list<int64_t> strides = {}) {
  test.AddInput<float>("buffer", {2, 3, 2}, {1.0f}, false, nullptr, true, {0, 0, 0});
  test.AddInput<float>("gradient", {2, 3, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.f, 11.0f, 12.0f});

  test.AddOutput<int64_t>("updated_flag", {}, {1});
  test.AddOutput<float>("updated_buffer", {2, 3, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.f, 11.0f, 12.0f});
  test.Run();
}

}  // namespace

// #endif  // USE_CUDA

}  // namespace optimizer
}  // namespace test
}  // namespace onnxruntime