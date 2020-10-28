// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

template <class T>
using ShapeAndData = std::pair<const std::vector<int64_t>, const std::vector<T>>;

using ShapeAndFloatData = ShapeAndData<float>;
using ShapeAndDoubleData = ShapeAndData<double>;
using ShapeAndHalfData = ShapeAndData<MLFloat16>;
using ShapeData = ShapeAndData<int64_t>;
using ExpectResult = OpTester::ExpectResult;

template <typename T>
void RunTest(const ShapeAndData<T>& input,
             const std::vector<ShapeData>& shapes,
             const std::vector<ShapeAndData<T>>& outputs,
             bool expect_failure = false,
             const std::string& err_msg = {}) {
  OpTester test("View", 1, onnxruntime::kMSDomain);

  test.AddInput<T>("input0", input.first, input.second);

  int i = 1;
  for (auto& s : shapes) {
    auto& shape = s.first;
    auto& data = s.second;
    std::ostringstream oss;
    oss << "input" << i++;
    test.AddInput<int64_t>(oss.str().c_str(), shape, data);
  }

  i = 0;
  for (auto& output : outputs) {
    auto& shape = output.first;
    auto& data = output.second;
    std::ostringstream oss;
    oss << "output" << i++;
    test.AddOutput<T>(oss.str().c_str(), shape, data);
  }

  std::unordered_set<std::string> excluded_providers;

  test.Run(expect_failure ? ExpectResult::kExpectFailure : ExpectResult::kExpectSuccess, err_msg, excluded_providers);
}

TEST(ViewOperatorTest, TwoViewFloat_1) {
  std::vector<ShapeData> shapes;
  std::vector<ShapeAndFloatData> outputs;

  // input shape and data
  ShapeAndFloatData input = {{4, 2},
                             {1.f, 2.f,
                              3.f, 4.f,
                              5.f, 6.f,
                              7.f, 8.f}};

  shapes.push_back({{2}, std::vector<int64_t>(2, 2)});  
  shapes.push_back({{2}, std::vector<int64_t>(2, 2)});

  outputs.push_back({{2, 2},
                     {1.f, 2.f,
                      3.f, 4.f}});
  outputs.push_back({{2, 2},
                     {5.f, 6.f,
                      7.f, 8.f}});

  RunTest<float>(input, shapes, outputs);
}

TEST(ViewOperatorTest, TwoViewFloat_2) {
  std::vector<ShapeData> shapes;
  std::vector<ShapeAndFloatData> outputs;

  // input shape and data
  ShapeAndFloatData input = {{4, 2},
                             {1.f, 2.f,
                              3.f, 4.f,
                              5.f, 6.f,
                              7.f, 8.f}};

  shapes.push_back({{2}, {1, 2}});
  shapes.push_back({{2}, {3, 2}});

  outputs.push_back({{1, 2}, {1.f, 2.f}});
  outputs.push_back({{3, 2},
                     {3.f, 4.f,
                      5.f, 6.f,
                      7.f, 8.f}});

  RunTest<float>(input, shapes, outputs);
}

TEST(ViewOperatorTest, TwoViewFloat_3) {
  std::vector<ShapeData> shapes;
  std::vector<ShapeAndFloatData> outputs;

  // input shape and data
  ShapeAndFloatData input = {{4, 2},
                             {1.f, 2.f,
                              3.f, 4.f,
                              5.f, 6.f,
                              7.f, 8.f}};

  shapes.push_back({{2}, {1, 2}});
  shapes.push_back({{3}, {1, 3, 2}});

  outputs.push_back({{1, 2}, {1.f, 2.f}});
  outputs.push_back({{1, 3, 2},
                     {3.f, 4.f,
                      5.f, 6.f,
                      7.f, 8.f}});

  RunTest<float>(input, shapes, outputs);
}

TEST(ViewOperatorTest, ThreeViewFloat) {
  std::vector<ShapeData> shapes;
  std::vector<ShapeAndFloatData> outputs;

  // input shape and data
  ShapeAndFloatData input = {{4, 3},
                             {1.f, 2.f, 3.f, 4.f, 5.f, 6.f,
                              7.f, 8.f, 9.f, 10.f, 11.f, 12.f}};
                       
  shapes.push_back({{2}, {1, 2}});
  shapes.push_back({{3}, {1, 3, 2}});
  shapes.push_back({{2}, {4, 1}});

  outputs.push_back({{1, 2}, {1.f, 2.f}});
  outputs.push_back({{1, 3, 2},
                     {3.f, 4.f, 5.f, 6.f, 7.f, 8.f}});
  outputs.push_back({{4, 1},
                     {9.f, 10.f, 11.f, 12.f}});

  RunTest<float>(input, shapes, outputs);
}

TEST(ViewOperatorTest, TwoViewDouble) {
  std::vector<ShapeData> shapes;
  std::vector<ShapeAndDoubleData> outputs;

  // input shape and data
  ShapeAndDoubleData input = {{3, 2},  
                              {1.f, 2.f,
                               3.f, 4.f,
                               5.f, 6.f}};

  shapes.push_back({{2}, {2, 1}});
  shapes.push_back({{3}, {1, 2, 2}});

  outputs.push_back({{2, 1},
                     {1.f, 2.f}});
  outputs.push_back({{1, 2, 2},
                     {3.f, 4.f, 5.f, 6.f}});

  RunTest<double>(input, shapes, outputs);  

}

TEST(ViewOperatorTest, TwoViewHalf) {
  std::vector<ShapeData> shapes;
  std::vector<ShapeAndHalfData> outputs;

  std::vector<float> data = {1.0f, 2.0f,
                             3.0f, 4.0f,
                             5.0f, 6.0f};
  std::vector<MLFloat16> data_half(6);
  ConvertFloatToMLFloat16(data.data(), data_half.data(), 6);
  // input shape and data
  ShapeAndHalfData input = {{3, 2}, data_half};

  shapes.push_back({{2}, {2, 1}});
  shapes.push_back({{3}, {1, 2, 2}});

  std::vector<float> data1 = {1.0f, 2.0f};
  std::vector<MLFloat16> data_half1(2);
  ConvertFloatToMLFloat16(data1.data(), data_half1.data(), 2);
  outputs.push_back({{2, 1}, data_half1});

  std::vector<float> data2 = {3.f, 4.f, 5.f, 6.f};
  std::vector<MLFloat16> data_half2(4);
  ConvertFloatToMLFloat16(data2.data(), data_half2.data(), 4);
  outputs.push_back({{1, 2, 2}, data_half2});

  RunTest<MLFloat16>(input, shapes, outputs);
}

}  // namespace test
}  // namespace onnxruntime
