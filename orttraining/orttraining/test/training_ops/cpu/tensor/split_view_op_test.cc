// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

template <class T>
using ShapeAndData = std::pair<const std::vector<int64_t>, const std::vector<T>>;

template <typename T>
void TestSplitViewOp(const ShapeAndData<T>& input, const int64_t num_outputs, const std::vector<int64_t> split_sizes,
                     const std::vector<ShapeAndData<T>>& outputs) {
  OpTester test("SplitView", 1, onnxruntime::kMSDomain);
  if (num_outputs != -1) {
    test.AddAttribute<int64_t>("num_outputs", num_outputs);
  }
  test.AddInput<T>("input", input.first, input.second);
  if (!split_sizes.empty()) {
    test.AddInput<int64_t>("split", {static_cast<int64_t>(split_sizes.size())}, split_sizes, true);
  }

  for (size_t i = 0; i < outputs.size(); ++i) {
    std::string name = "output" + std::to_string(i);
    test.AddOutput<T>(name.c_str(), outputs[i].first, outputs[i].second);
  }

  test.Run();
}

TEST(SplitViewOpTest, NumOutputsEqualSplitFloat) {
  ShapeAndData<float> input = {{4, 2}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f}};
  std::vector<ShapeAndData<float>> outputs;
  ShapeAndData<float> output0 = {{2, 2}, {1.f, 2.f, 3.f, 4.f}};
  ShapeAndData<float> output1 = {{2, 2}, {5.f, 6.f, 7.f, 8.f}};
  outputs.emplace_back(output0);
  outputs.emplace_back(output1);
  TestSplitViewOp<float>(input, 2, {}, outputs);
}

TEST(SplitViewOpTest, NumOutputsNonEqualSplitFloat16) {
  ShapeAndData<MLFloat16> input = {{5, 2},
                                   {MLFloat16(1.f), MLFloat16(2.f), MLFloat16(3.f), MLFloat16(4.f), MLFloat16(5.f),
                                    MLFloat16(6.f), MLFloat16(7.f), MLFloat16(8.f), MLFloat16(9.f), MLFloat16(10.f)}};
  std::vector<ShapeAndData<MLFloat16>> outputs;
  ShapeAndData<MLFloat16> output0 = {{2, 2}, {MLFloat16(1.f), MLFloat16(2.f), MLFloat16(3.f), MLFloat16(4.f)}};
  ShapeAndData<MLFloat16> output1 = {{2, 2}, {MLFloat16(5.f), MLFloat16(6.f), MLFloat16(7.f), MLFloat16(8.f)}};
  ShapeAndData<MLFloat16> output2 = {{1, 2}, {MLFloat16(9.f), MLFloat16(10.f)}};
  outputs.emplace_back(output0);
  outputs.emplace_back(output1);
  outputs.emplace_back(output2);
  TestSplitViewOp<MLFloat16>(input, 3, {}, outputs);
}

TEST(SplitViewOpTest, SplitInputFloat) {
  ShapeAndData<float> input = {{6, 2}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f}};
  std::vector<ShapeAndData<float>> outputs;
  ShapeAndData<float> output0 = {{1, 2}, {1.f, 2.f}};
  ShapeAndData<float> output1 = {{2, 2}, {3.f, 4.f, 5.f, 6.f}};
  ShapeAndData<float> output2 = {{3, 2}, {7.f, 8.f, 9.f, 10.f, 11.f, 12.f}};
  outputs.emplace_back(output0);
  outputs.emplace_back(output1);
  outputs.emplace_back(output2);
  TestSplitViewOp<float>(input, -1, {1, 2, 3}, outputs);
}

TEST(SplitViewOpTest, SplitInputFloat16) {
  ShapeAndData<MLFloat16> input = {{5, 2},
                                   {MLFloat16(1.f), MLFloat16(2.f), MLFloat16(3.f), MLFloat16(4.f), MLFloat16(5.f),
                                    MLFloat16(6.f), MLFloat16(7.f), MLFloat16(8.f), MLFloat16(9.f), MLFloat16(10.f)}};
  std::vector<ShapeAndData<MLFloat16>> outputs;
  ShapeAndData<MLFloat16> output0 = {{2, 2}, {MLFloat16(1.f), MLFloat16(2.f), MLFloat16(3.f), MLFloat16(4.f)}};
  ShapeAndData<MLFloat16> output1 = {
      {3, 2}, {MLFloat16(5.f), MLFloat16(6.f), MLFloat16(7.f), MLFloat16(8.f), MLFloat16(9.f), MLFloat16(10.f)}};
  outputs.emplace_back(output0);
  outputs.emplace_back(output1);
  TestSplitViewOp<MLFloat16>(input, -1, {2, 3}, outputs);
}

}  // namespace test
}  // namespace onnxruntime
