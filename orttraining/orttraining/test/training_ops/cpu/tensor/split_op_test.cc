// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

template <class T>
using ShapeAndData = std::pair<const std::vector<int64_t>, const std::vector<T>>;

using ShapeAndFloatData = ShapeAndData<float>;
using ShapeAndStringData = ShapeAndData<std::string>;
using ExpectResult = OpTester::ExpectResult;

template <typename T>
void SplitTrainingOpTester(int64_t axis, const std::vector<int64_t> split_sizes, const ShapeAndData<T>& input,
             const std::vector<ShapeAndData<T>>& outputs, bool is_initializer = true,
             bool expect_failure = false, const std::string& err_msg = {}) {
  OpTester test("SplitTraining", 1, onnxruntime::kMSDomain);

  test.AddAttribute("axis", axis);

  test.AddInput<T>("input", input.first, input.second);
  test.AddInput<int64_t>("split", {static_cast<int64_t>(split_sizes.size())}, split_sizes, is_initializer);

  int i = 0;
  for (auto& output : outputs) {
    auto& shape = output.first;
    auto& data = output.second;
    std::ostringstream oss;
    oss << "output" << i++;
    test.AddOutput<T>(oss.str().c_str(), shape, data);
  }

  test.Run(expect_failure ? ExpectResult::kExpectFailure : ExpectResult::kExpectSuccess, err_msg);
}

TEST(SplitTrainingOpTest, Axis0EqualSplitFloat) {
  const int64_t axis = 0;
  std::vector<ShapeAndFloatData> outputs;

  // input shape and data
  ShapeAndFloatData input = {{4, 2},  // shape
                             {1.f, 2.f,
                              3.f, 4.f,
                              5.f, 6.f,
                              7.f, 8.f}};

  outputs.push_back({{2, 2},
                     {1.f, 2.f,
                      3.f, 4.f}});

  outputs.push_back({{2, 2},
                     {5.f, 6.f,
                      7.f, 8.f}});

  SplitTrainingOpTester<float>(axis, {}, input, outputs);  
}

TEST(SplitTrainingOpTest, Axis0UnequalSplitFloat) {
  const int64_t axis = 0;
  std::vector<ShapeAndFloatData> outputs;

  // input shape and data
  ShapeAndFloatData input = {{4, 2},  // shape
                             {1.f, 2.f,
                              3.f, 4.f,
                              5.f, 6.f,
                              7.f, 8.f}};

  std::vector<int64_t> splits{1, 3};

  outputs.push_back({{1, 2}, {1.f, 2.f}});

  outputs.push_back({{3, 2},
                     {3.f, 4.f,
                      5.f, 6.f,
                      7.f, 8.f}});

  SplitTrainingOpTester<float>(axis, splits, input, outputs);  
}


TEST(SplitTrainingOpTest, Axis0EqualSplitFloat_not_initializer) {
  const int64_t axis = 0;
  std::vector<ShapeAndFloatData> outputs;

  // input shape and data
  ShapeAndFloatData input = {{4, 2},  // shape
                             {1.f, 2.f,
                              3.f, 4.f,
                              5.f, 6.f,
                              7.f, 8.f}};

  outputs.push_back({{2, 2},
                     {1.f, 2.f,
                      3.f, 4.f}});

  outputs.push_back({{2, 2},
                     {5.f, 6.f,
                      7.f, 8.f}});

  SplitTrainingOpTester<float>(axis, {}, input, outputs, false);  
}

TEST(SplitTrainingOpTest, Axis0UnequalSplitFloat_not_initializer) {
  const int64_t axis = 0;
  std::vector<ShapeAndFloatData> outputs;

  // input shape and data
  ShapeAndFloatData input = {{4, 2},  // shape
                             {1.f, 2.f,
                              3.f, 4.f,
                              5.f, 6.f,
                              7.f, 8.f}};

  std::vector<int64_t> splits{1, 3};

  outputs.push_back({{1, 2}, {1.f, 2.f}});

  outputs.push_back({{3, 2},
                     {3.f, 4.f,
                      5.f, 6.f,
                      7.f, 8.f}});

  SplitTrainingOpTester<float>(axis, splits, input, outputs, false);  
}

}  // namespace test
}  // namespace onnxruntime
