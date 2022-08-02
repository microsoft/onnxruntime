// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "core/util/math.h"

namespace onnxruntime {
namespace test {

TEST(Col2ImContribOpTest, simple) {
  OpTester test("Col2Im", 1, kMSDomain);

  test.AddAttribute("strides", std::vector<int64_t>{1, 1});
  test.AddAttribute("dilations", std::vector<int64_t>{1, 1});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0, 0, 0});

  test.AddInput<float>("input", {1, 5, 5},  std::vector<float>{1.f, 6.f, 11.f, 16.f, 21.f, 2.f, 7.f, 12.f, 17.f, 22.f, 3.f, 8.f, 13.f, 18.f, 23.f, 4.f, 9.f, 14.f, 19.f, 24.f, 5.f, 0.f, 15.f, 20.f, 25.f});
  test.AddInput<int64_t>("image_shape", {2},  std::vector<int64_t>{5, 5});
  test.AddInput<int64_t>("block_shape", {2},  std::vector<int64_t>{1, 5});

  test.AddOutput<float>("output", {1, 1, 5, 5}, std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f, 16.f, 17.f, 18.f, 19.f, 20.f, 21.f, 22.f, 23.f, 24.f, 25.f});
  test.Run();
}


}  // namespace test
}  // namespace onnxruntime
