// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(MathOpTest, MatMul) {
  std::vector<float> vals{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};

  struct MatMulTest {
    std::string name;
    std::vector<int64_t> input0_dims;
    std::vector<int64_t> input1_dims;
    std::vector<int64_t> expected_dims;
    std::vector<float> expected_vals;
  };

  MatMulTest testcases[] = {
      {"test padding and broadcast",
       {3, 1, 1, 2},
       {2, 2, 2},
       {3, 2, 1, 2},
       {2, 3, 6, 7, 6, 11, 26, 31, 10, 19, 46, 55}},
      {"test padding and broadcast",
       {2, 3, 2},
       {3, 2, 2, 1},
       {3, 2, 3, 1},
       {1, 3, 5, 33, 43, 53, 5, 23, 41, 85, 111, 137, 9, 43, 77, 137, 179, 221}},
      {"test left 1D",
       {2},
       {3, 2, 1},
       {3, 1},
       {1, 3, 5}},
      {"test right 1D",
       {3, 1, 2},
       {2},
       {3, 1},
       {1, 3, 5}},
      {"test scalar output",
       {3},
       {3},
       {},
       {5}},
      {"test 2D",
       {3, 4},
       {4, 3},
       {3, 3},
       {42, 48, 54, 114, 136, 158, 186, 224, 262}},
      {"test 2D special",
       {2, 2, 3},
       {3, 4},
       {2, 2, 4},
       {20, 23, 26, 29, 56, 68, 80, 92, 92, 113, 134, 155, 128, 158, 188, 218}},
      {"test 2D special 2",
       {2, 2, 3},
       {1, 3, 4},
       {2, 2, 4},
       {20, 23, 26, 29, 56, 68, 80, 92, 92, 113, 134, 155, 128, 158, 188, 218}},
  };

  for (auto t : testcases) {
    OpTester test("MatMul");

    int64_t size0 = TensorShape::ReinterpretBaseType(t.input0_dims).SizeHelper(0, t.input0_dims.size());
    std::vector<float> input0_vals(vals.cbegin(), vals.cbegin() + size0);
    test.AddInput<float>("A", t.input0_dims, input0_vals);

    int64_t size1 = TensorShape::ReinterpretBaseType(t.input1_dims).SizeHelper(0, t.input1_dims.size());
    std::vector<float> input1_vals(vals.cbegin(), vals.cbegin() + size1);
    test.AddInput<float>("B", t.input1_dims, input1_vals);

    test.AddOutput<float>("Y", t.expected_dims, t.expected_vals);
    test.Run();
  }
}

}  // namespace test
}  // namespace onnxruntime
