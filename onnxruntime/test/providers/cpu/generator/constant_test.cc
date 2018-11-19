// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

using namespace ONNX_NAMESPACE;
namespace onnxruntime {
namespace test {

TEST(ConstantLike, ConstantLike_with_input) {
  OpTester test("ConstantLike", 9);

  std::vector<int64_t> dims{4, 3, 2};

  test.AddInput<int32_t>("X", dims,
                         {0, 1, 2, 3,
                          4, 5, 7, 8,
                          0, 1, 2, 3,
                          4, 5, 7, 8,
                          0, 1, 2, 3,
                          4, 5, 7, 8});

  test.AddAttribute("value", 1.0f);

  std::vector<int32_t> expected_output(
      {1, 1, 1, 1,
       1, 1, 1, 1,
       1, 1, 1, 1,
       1, 1, 1, 1,
       1, 1, 1, 1,
       1, 1, 1, 1});

  test.AddOutput<int32_t>("Y", dims, expected_output);
  test.Run();
}

TEST(ConstantLike, ConstantLike_without_input) {
  OpTester test("ConstantLike", 9);

  std::vector<int64_t> dims{4, 3, 2};
  test.AddAttribute("shape", dims);
  test.AddAttribute("value", 2.0f);

  std::vector<float> expected_output(
      {2.0f, 2.0f, 2.0f, 2.0f,
       2.0f, 2.0f, 2.0f, 2.0f,
       2.0f, 2.0f, 2.0f, 2.0f,
       2.0f, 2.0f, 2.0f, 2.0f,
       2.0f, 2.0f, 2.0f, 2.0f,
       2.0f, 2.0f, 2.0f, 2.0f});

  test.AddOutput<float>("Y", dims, expected_output);
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
