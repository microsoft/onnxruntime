// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {
TEST(ContribOpTest, ConvTransposeWithDynamicPads) {
  OpTester test("ConvTransposeWithDynamicPads", 1, onnxruntime::kMSDomain);
  test.AddAttribute("kernel_shape", std::vector<int64_t>{3, 3});
  test.AddAttribute("output_padding", std::vector<int64_t>{1, 1});
  test.AddAttribute("strides", std::vector<int64_t>{2, 2});
  test.AddAttribute("dilations", std::vector<int64_t>{1, 1});

  test.AddInput<float>("X", {1, 1, 3, 3}, std::vector<float>{0.16857791f, -0.15161794f, 0.08540368f, 0.1820628f, -0.21746576f, 0.08245695f, 0.1431433f, -0.43156421f, 0.30591947f});
  test.AddInput<float>("W", {1, 1, 3, 3}, std::vector<float>{-0.06230065f, 0.37932432f, -0.25388849f, 0.33878803f, 0.43709868f, -0.22477469f, 0.04118127f, -0.44696793f, 0.06373066f});
  test.AddInput<int64_t>("Pads", {4}, std::vector<int64_t>{1, 1, 1, 1});
  test.AddOutput<float>("Y", {1, 1, 6, 6}, std::vector<float>{0.07368518f, -0.08925839f, -0.06627201f, 0.06301362f, 0.03732984f, -0.01919658f, -0.00628807f, -0.02817563f, -0.01472169f, 0.04392925f, -0.00689478f, -0.01549204f, 0.07957941f, -0.11459791f, -0.09505399f, 0.07681622f, 0.03604182f, -0.01853423f, -0.0270785f, -0.00680824f, -0.06650258f, 0.08004665f, 0.07918708f, -0.0724144f, 0.06256775f, -0.17838378f, -0.18863615f, 0.20064656f, 0.133717f, -0.06876295f, -0.06398046f, -0.00864975f, 0.19289537f, -0.01490572f, -0.13673618f, 0.01949645f});
  test.Run();
}
}  // namespace test
}  // namespace onnxruntime
