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

using std::vector;

#ifdef USE_WEBGPU
TEST(InstanceNormalizationOpTest, InstanceNormNCHW_webgpu) {
  OpTester test("InstanceNormalization");
  test.AddAttribute("epsilon", 0.009999999776482582f);

  vector<float> input = {1.0f, 2.0f, 3.0f, 2.0f, 2.0f, 2.0f};
  vector<int64_t> input_dims = {1, 2, 1, 3};
  test.AddInput<float>("input", input_dims, input);

  vector<float> scale = {1.0f, 1.0f};
  vector<int64_t> scale_dims = {2};
  test.AddInput<float>("scale", scale_dims, scale);

  vector<float> B = {0.0f, 2.0f};
  vector<int64_t> B_dims = {2};
  test.AddInput<float>("B", B_dims, B);

  vector<float> expected_output = {-1.21566f, 0.0f, 1.21566f, 2.0f, 2.0f, 2.0f};
  test.AddOutput<float>("Y", input_dims, expected_output);

  test.ConfigEp(DefaultWebGpuExecutionProvider(false)).RunWithConfig();
}

TEST(InstanceNormalizationOpTest, InstanceNormNCHW_webgpu_2) {
  OpTester test("InstanceNormalization");
  test.AddAttribute("epsilon", 0.009999999776482582f);

  vector<float> input = {1.0f, 2.0f, 3.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f};
  vector<int64_t> input_dims = {1, 2, 2, 2};
  test.AddInput<float>("input", input_dims, input);

  vector<float> scale = {1.0f, 1.0f};
  vector<int64_t> scale_dims = {2};
  test.AddInput<float>("scale", scale_dims, scale);

  vector<float> B = {0.0f, 2.0f};
  vector<int64_t> B_dims = {2};
  test.AddInput<float>("B", B_dims, B);

  vector<float> expected_output = {-1.40028f, 0.0f, 1.40028f, 0.0f, 2.0f, 2.0f, 2.0f, 2.0f};
  test.AddOutput<float>("Y", input_dims, expected_output);

  test.ConfigEp(DefaultWebGpuExecutionProvider(false)).RunWithConfig();
}
#endif

}  // namespace test
}  // namespace onnxruntime
