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
  constexpr int64_t axis = 0;
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

std::tuple<ShapeAndFloatData, std::vector<ShapeAndFloatData>>
Setup_Axis0EqualSplitFloat_N_inputs(const int num_outputs) {
  float counter = 1.0f;
  std::vector<float> data(4 * num_outputs);
  std::iota(data.begin(), data.end(), counter);
  ShapeAndFloatData input = {{2 * num_outputs, 2}, data};

  data.resize(4);
  std::vector<ShapeAndFloatData> outputs;
  for (int i = 0; i < num_outputs; i++) {
    std::iota(data.begin(), data.end(), counter);
    outputs.push_back({{2, 2}, data});
    counter += (float)data.size();
  }

  // due to const on ShapeAndFloatData
  return std::make_tuple(input, outputs);
}

// <=32 with same sizes passes output addresses as kernel args
TEST(SplitTrainingOpTest, Axis0EqualSplitFloat_16_outputs) {
  constexpr int64_t axis = 0;
  auto io = Setup_Axis0EqualSplitFloat_N_inputs(16);
  SplitTrainingOpTester<float>(axis, {}, std::get<0>(io), std::get<1>(io));
}

// > 32 with same sizes passes output addresses as device buffer
TEST(SplitTrainingOpTest, Axis0EqualSplitFloat_64_outputs) {
  constexpr int64_t axis = 0;
  auto io = Setup_Axis0EqualSplitFloat_N_inputs(64);
  SplitTrainingOpTester<float>(axis, {}, std::get<0>(io), std::get<1>(io));
}

TEST(SplitTrainingOpTest, Axis0UnequalSplitFloat) {
  constexpr int64_t axis = 0;
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
  constexpr int64_t axis = 0;
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
  constexpr int64_t axis = 0;
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
