// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {
TEST(ConvIntegerTest, ConvIntegerTest) {
  OpTester test("ConvInteger", 10);
  std::vector<int64_t> x_dims{1, 1, 3, 3};
  test.AddInput<uint8_t>("x", x_dims,
                         {2, 3, 4,
                          5, 6, 7,
						  8, 9, 10});
  std::vector<int64_t> w_dims{1, 1, 2, 2};
  test.AddInput<uint8_t>("w", w_dims,
                         {1, 1,
					      1, 1});
  test.AddInput<uint8_t>("x_zero_point", {}, {1});
  std::vector<int64_t> y_dims{1, 1, 2, 2};
  test.AddOutput<int32_t>("y", y_dims,
                          {12, 16,
						   24, 28});
  test.Run();
}

TEST(ConvIntegerTest_with_padding, ConvIntegerTest) {
  OpTester test("ConvInteger", 10);
  std::vector<int64_t> x_dims{1, 1, 3, 3};
  test.AddInput<uint8_t>("x", x_dims,
                         {2, 3, 4,
                          5, 6, 7,
                          8, 9, 10});
  std::vector<int64_t> w_dims{1, 1, 2, 2};
  test.AddInput<uint8_t>("w", w_dims,
                         {1, 1,
                          1, 1});
  test.AddInput<uint8_t>("x_zero_point", {}, {1});
  test.AddAttribute<std::vector<int64_t>>("pads", {1, 1, 1, 1});
  std::vector<int64_t> y_dims{1, 1, 4, 4};
  test.AddOutput<int32_t>("y", y_dims,
                          {1, 3, 5, 3,
	                       5, 12, 16, 9,
                           11, 24, 28, 15,
	                       7, 15, 17, 9});
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
