// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

// range = [-ve, +ve]
TEST(QuantizeLinearOpTest, DynamicQuantizeLinear) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: Expected equality of these values: 26 and 25";
  }

  OpTester test("DynamicQuantizeLinear", 11);
  std::vector<int64_t> dims{6};
  test.AddInput<float>("x", dims, {0, 2, -3, -2.5f, 1.34f, 0.5f});
  test.AddOutput<uint8_t>("y", dims, {153, 255, 0, 26, 221, 179});
  test.AddOutput<float>("y_scale", {}, {0.0196078438f});
  test.AddOutput<uint8_t>("y_zero_point", {}, {153});
  test.Run();
}

// quantize with 2D data with min adjustment to include 0 in the input range.
TEST(QuantizeLinearOpTest, DynamicQuantizeLinear_Min_Adjusted) {
  OpTester test("DynamicQuantizeLinear", 11);
  std::vector<int64_t> dims{3, 4};
  test.AddInput<float>("x", dims,
                       {1, 2.1f, 1.3f, 2.5f,
                        3.34f, 4.0f, 1.5f, 2.6f,
                        3.9f, 4.0f, 3.0f, 2.345f});

  test.AddOutput<uint8_t>("y", dims,
                          {64, 134, 83, 159,
                           213, 255, 96, 166,
                           249, 255, 191, 149});
  test.AddOutput<float>("y_scale", {}, {0.01568628f});
  test.AddOutput<uint8_t>("y_zero_point", {}, {0});
  test.Run();
}

// quantize max adjustment to include 0 in the input range.
TEST(QuantizeLinearOpTest, DynamicQuantizeLinear_Max_Adjusted) {
  OpTester test("DynamicQuantizeLinear", 11);
  std::vector<int64_t> dims{6};
  test.AddInput<float>("x", dims, {-1.0f, -2.1f, -1.3f, -2.5f, -3.34f, -4.0f});
  test.AddOutput<uint8_t>("y", dims, {191, 121, 172, 96, 42, 0});
  test.AddOutput<float>("y_scale", {}, {0.01568628f});
  test.AddOutput<uint8_t>("y_zero_point", {}, {255});
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
