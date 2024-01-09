// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "core/framework/to_tensor_proto_element_type.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

namespace {

template <class T>
using ShapeAndData = std::pair<const std::vector<int64_t>, const std::vector<T>>;

using ShapeAndFloatData = ShapeAndData<float>;
using ShapeAndStringData = ShapeAndData<std::string>;
using ExpectResult = OpTester::ExpectResult;

template <typename T>
void RunTest(int64_t axis, const std::vector<int64_t>& split_sizes, const ShapeAndData<T>& input,
             const std::vector<ShapeAndData<T>>& outputs,
             const std::unordered_set<std::string>& excluded_providers,
             bool expect_failure = false, bool split_as_input = false, int64_t num_outputs = -1,
             bool is_initializer = true, const std::string& err_msg = {}, bool skip_split_if_empty = true) {
  int opset_version = num_outputs > -1 ? 18 : split_as_input ? 13
                                                             : 7;

  OpTester test("Split", opset_version, onnxruntime::kOnnxDomain);

  constexpr bool is_bool_data = std::is_same_v<T, bool>;
  [[maybe_unused]] auto bool_vector_to_array = [](const std::vector<bool>& v) -> std::unique_ptr<bool[]> {
    auto a = std::make_unique<bool[]>(v.size());
    std::copy(v.begin(), v.end(), a.get());
    return a;
  };

  test.AddAttribute("axis", axis);
  if (num_outputs != -1) {
    test.AddAttribute("num_outputs", num_outputs);
  }

  if constexpr (is_bool_data) {
    auto input_array = bool_vector_to_array(input.second);
    test.AddInput<T>("input", input.first, input_array.get(), input.second.size());
  } else {
    test.AddInput<T>("input", input.first, input.second);
  }

  if (!split_sizes.empty()) {
    if (split_as_input) {
      test.AddInput<int64_t>("split", {static_cast<int64_t>(split_sizes.size())}, split_sizes, is_initializer);
    } else {
      test.AddAttribute("split", split_sizes);
    }
  } else if (!skip_split_if_empty) {
    test.AddOptionalInputEdge<int64_t>();
  }

  int i = 0;
  for (auto& output : outputs) {
    auto& shape = output.first;
    auto& data = output.second;
    const auto output_name = MakeString("output", i++);

    if constexpr (is_bool_data) {
      auto data_array = bool_vector_to_array(data);
      test.AddOutput<T>(output_name.c_str(), shape, data_array.get(), data.size());
    } else {
      test.AddOutput<T>(output_name.c_str(), shape, data);
    }
  }

  test.Run(expect_failure ? ExpectResult::kExpectFailure : ExpectResult::kExpectSuccess, err_msg, excluded_providers);
}

template <typename>
[[maybe_unused]] constexpr bool dependent_false_v = false;

template <typename T>
constexpr T ValueFromIdx(size_t idx) {
  if constexpr (std::is_same_v<T, std::string>) {
    const char c = gsl::narrow_cast<char>('a' + idx);
    return std::string(1, c);
  } else if constexpr (std::is_same_v<T, bool>) {
    return (idx & 1) == 1;
  } else if constexpr (std::is_integral_v<T> || std::is_floating_point_v<T>) {
    return gsl::narrow_cast<T>(idx);
  } else if constexpr (std::is_same_v<T, MLFloat16> || std::is_same_v<T, BFloat16>) {
    return T{static_cast<float>(idx)};
  } else {
    static_assert(dependent_false_v<T>, "unsupported type");
  }
}

template <typename T>
void SplitTestAxis0EqualSplit() {
  SCOPED_TRACE(onnxruntime::MakeString("data type: ", utils::ToTensorProtoElementType<T>()));

  constexpr int64_t axis = 0;
  std::vector<ShapeAndData<T>> outputs;

  const auto V = ValueFromIdx<T>;

  // input shape and data
  ShapeAndData<T> input = {{4, 2},  // shape
                           {V(1), V(2),
                            V(3), V(4),
                            V(5), V(6),
                            V(7), V(8)}};

  outputs.push_back({{2, 2},
                     {V(1), V(2),
                      V(3), V(4)}});

  outputs.push_back({{2, 2},
                     {V(5), V(6),
                      V(7), V(8)}});

  // BFloat16 added in opset 13
  if constexpr (!std::is_same_v<T, BFloat16>) {
    RunTest<T>(axis, {}, input, outputs,
               // TensorRT parser: Assertion failed: axis != BATCH_DIM
               {kTensorrtExecutionProvider},  // is_tensorrt_supported
               false,                         // expect_failure
               false /*split_as_input*/);
  }

  RunTest<T>(axis, {}, input, outputs,
             // TensorRT parser: Assertion failed: axis != BATCH_DIM
             {kTensorrtExecutionProvider},  // is_tensorrt_supported
             false,                         // expect_failure
             true /*split_as_input*/);
}

}  // namespace

TEST(SplitOperatorTest, Axis0EqualSplit) {
  SplitTestAxis0EqualSplit<float>();
  SplitTestAxis0EqualSplit<double>();
  SplitTestAxis0EqualSplit<MLFloat16>();
  SplitTestAxis0EqualSplit<BFloat16>();
  SplitTestAxis0EqualSplit<int8_t>();
  SplitTestAxis0EqualSplit<int16_t>();
  SplitTestAxis0EqualSplit<int32_t>();
  SplitTestAxis0EqualSplit<int64_t>();
  SplitTestAxis0EqualSplit<uint8_t>();
  SplitTestAxis0EqualSplit<uint16_t>();
  SplitTestAxis0EqualSplit<uint32_t>();
  SplitTestAxis0EqualSplit<uint64_t>();
  SplitTestAxis0EqualSplit<bool>();
  SplitTestAxis0EqualSplit<std::string>();
}

TEST(SplitOperatorTest, Axis0UnequalSplitFloat) {
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

  // TensorRT parser: Assertion failed: axis != BATCH_DIM
  RunTest<float>(axis, splits, input, outputs, {kTensorrtExecutionProvider});
  // CoreML EP, etc. requires split to be an input. Same applies to below sets of tests.
  RunTest<float>(axis, splits, input, outputs, {kTensorrtExecutionProvider}, false, true);
}

TEST(SplitOperatorTest, Axis0UnequalSplitString) {
  constexpr int64_t axis = 0;
  std::vector<ShapeAndStringData> outputs;

  // input shape and data
  ShapeAndStringData input = {{4, 2},  // shape
                              {"a", "b",
                               "c", "d",
                               "e", "f",
                               "g", "h"}};

  std::vector<int64_t> splits{1, 3};

  outputs.push_back({{1, 2}, {"a", "b"}});

  outputs.push_back({{3, 2},
                     {"c", "d",
                      "e", "f",
                      "g", "h"}});
  // TensorRT parser: Assertion failed: axis != BATCH_DIM
  RunTest<std::string>(axis, splits, input, outputs, {kTensorrtExecutionProvider}, false, true);
  RunTest<std::string>(axis, splits, input, outputs, {kTensorrtExecutionProvider});
}

TEST(SplitOperatorTest, Axis1EqualSplitFloat) {
  constexpr int64_t axis = 1;
  std::vector<ShapeAndFloatData> outputs;

  // input shape and data
  ShapeAndFloatData input = {{2, 4},
                             {1.f, 2.f, 3.f, 4.f,
                              5.f, 6.f, 7.f, 8.f}};

  outputs.push_back({{2, 2},
                     {1.f, 2.f,
                      5.f, 6.f}});

  outputs.push_back({{2, 2},
                     {3.f, 4.f,
                      7.f, 8.f}});
  RunTest<float>(axis, {}, input, outputs, {kTensorrtExecutionProvider}, false, true);
  RunTest<float>(axis, {}, input, outputs, {kTensorrtExecutionProvider});
}

TEST(SplitOperatorTest, Axis1EqualSplitString) {
  constexpr int64_t axis = 1;
  std::vector<ShapeAndStringData> outputs;

  // input shape and data
  ShapeAndStringData input = {{2, 4},
                              {"a", "b", "c", "d",
                               "e", "f", "g", "h"}};

  outputs.push_back({{2, 2},
                     {"a", "b",
                      "e", "f"}});

  outputs.push_back({{2, 2},
                     {"c", "d",
                      "g", "h"}});

  RunTest<std::string>(axis, {}, input, outputs, {kTensorrtExecutionProvider}, false, true);
  RunTest<std::string>(axis, {}, input, outputs, {kTensorrtExecutionProvider});
}

TEST(SplitOperatorTest, Axis1UnequalSplitFloat) {
  constexpr int64_t axis = 1;
  std::vector<ShapeAndFloatData> outputs;

  // input shape and data
  ShapeAndFloatData input = {{2, 4},
                             {1.f, 2.f, 3.f, 4.f,
                              5.f, 6.f, 7.f, 8.f}};

  std::vector<int64_t> splits{3, 1};

  outputs.push_back({{2, 3},
                     {1.f, 2.f, 3.f,
                      5.f, 6.f, 7.f}});

  outputs.push_back({{2, 1},
                     {4.f,
                      8.f}});

  RunTest<float>(axis, splits, input, outputs, {kTensorrtExecutionProvider}, false, true);
  RunTest<float>(axis, splits, input, outputs, {kTensorrtExecutionProvider});
}

TEST(SplitOperatorTest, Axis1UnequalSplitString) {
  constexpr int64_t axis = 1;
  std::vector<ShapeAndStringData> outputs;

  // input shape and data
  ShapeAndStringData input = {{2, 4},
                              {"a", "b", "c", "d",
                               "e", "f", "g", "h"}};

  std::vector<int64_t> splits{3, 1};

  outputs.push_back({{2, 3},
                     {"a", "b", "c",
                      "e", "f", "g"}});

  outputs.push_back({{2, 1},
                     {"d",
                      "h"}});

  RunTest<std::string>(axis, splits, input, outputs, {kTensorrtExecutionProvider}, false, true);
  RunTest<std::string>(axis, splits, input, outputs, {kTensorrtExecutionProvider});
}

template <typename T>
ShapeAndData<T> CreateInput(std::vector<int64_t> shape) {
  auto size = TensorShape(shape).Size();

  T i = static_cast<T>(0), increment = static_cast<T>(1);
  // generate the elements for the data starting at 1
  std::vector<T> data;
  std::generate_n(std::back_inserter(data), size, [&]() { return i += increment; });

  return ShapeAndData<T>{shape, data};
}

TEST(SplitOperatorTest, Axis2EqualSplit) {
  constexpr int64_t axis = 2;
  std::vector<ShapeAndFloatData> outputs;

  ShapeAndFloatData input = CreateInput<float>({2, 2, 6});

  outputs.push_back({{2, 2, 2},
                     {1.f, 2.f,
                      7.f, 8.f,

                      13.f, 14.f,
                      19.f, 20.f}});

  outputs.push_back({{2, 2, 2},
                     {3.f, 4.f,
                      9.f, 10.f,

                      15.f, 16.f,
                      21.f, 22.f}});

  outputs.push_back({{2, 2, 2},
                     {5.f, 6.f,
                      11.f, 12.f,

                      17.f, 18.f,
                      23.f, 24.f}});

  RunTest<float>(axis, {}, input, outputs, {kTensorrtExecutionProvider}, false, true);
  RunTest<float>(axis, {}, input, outputs, {kTensorrtExecutionProvider});
}

TEST(SplitOperatorTest, Axis2UnequalSplit) {
  constexpr int64_t axis = 2;
  std::vector<ShapeAndFloatData> outputs;

  ShapeAndFloatData input = CreateInput<float>({2, 2, 6});

  std::vector<int64_t> splits{1, 2, 3};

  outputs.push_back({{2, 2, 1},
                     {1.f,
                      7.f,

                      13.f,
                      19.f}});

  outputs.push_back({{2, 2, 2},
                     {2.f, 3.f,
                      8.f, 9.f,

                      14.f, 15.f,
                      20.f, 21.f}});

  outputs.push_back({{2, 2, 3},
                     {4.f, 5.f, 6.f,
                      10.f, 11.f, 12.f,

                      16.f, 17.f, 18.f,
                      22.f, 23.f, 24.f}});

  // Note: temporarily marked qnn ep as excluded when running tests with split_as_input=true.
  // TODO: Need to resolve to see if it's not supported or test case failure.
  RunTest<float>(axis, splits, input, outputs, {kTensorrtExecutionProvider, kQnnExecutionProvider}, false, true);
  RunTest<float>(axis, splits, input, outputs, {kTensorrtExecutionProvider});
}

TEST(SplitOperatorTest, ZeroSizeInput) {
  constexpr int64_t axis = -1;
  std::vector<ShapeAndFloatData> outputs{{{0, 1}, {}}, {{0, 1}, {}}};

  ShapeAndFloatData input = CreateInput<float>({0, 2});

  RunTest<float>(axis, {}, input, outputs, {kTensorrtExecutionProvider, kQnnExecutionProvider, kCoreMLExecutionProvider});
}

// test a split of a dimension that has leading and trailing dimensions
TEST(SplitOperatorTest, Axis1SplitMiddleDimensionEqually) {
  constexpr int64_t axis = 1;
  std::vector<ShapeAndFloatData> outputs;

  ShapeAndFloatData input = CreateInput<float>({2, 4, 4});

  outputs.push_back({{2, 2, 4},
                     {1.f, 2.f, 3.f, 4.f,
                      5.f, 6.f, 7.f, 8.f,

                      17.f, 18.f, 19.f, 20.f,
                      21.f, 22.f, 23.f, 24.f}});

  outputs.push_back({{2, 2, 4},
                     {9.f, 10.f, 11.f, 12.f,
                      13.f, 14.f, 15.f, 16.f,

                      25.f, 26.f, 27.f, 28.f,
                      29.f, 30.f, 31.f, 32.f}});

  RunTest<float>(axis, {}, input, outputs, {kTensorrtExecutionProvider}, false, true);
  RunTest<float>(axis, {}, input, outputs, {kTensorrtExecutionProvider});
}

// test a split of a dimension that has leading and trailing dimensions
TEST(SplitOperatorTest, Axis1SplitMiddleDimensionUnequally) {
  constexpr int64_t axis = 1;
  std::vector<ShapeAndFloatData> outputs;

  ShapeAndFloatData input = CreateInput<float>({2, 4, 4});

  std::vector<int64_t> splits{1, 3};

  outputs.push_back({{2, 1, 4},
                     {1.f, 2.f, 3.f, 4.f,

                      17.f, 18.f, 19.f, 20.f}});

  outputs.push_back({{2, 3, 4},
                     {5.f, 6.f, 7.f, 8.f,
                      9.f, 10.f, 11.f, 12.f,
                      13.f, 14.f, 15.f, 16.f,

                      21.f, 22.f, 23.f, 24.f,
                      25.f, 26.f, 27.f, 28.f,
                      29.f, 30.f, 31.f, 32.f}});

  RunTest<float>(axis, splits, input, outputs, {kTensorrtExecutionProvider}, false, true);
  RunTest<float>(axis, splits, input, outputs, {kTensorrtExecutionProvider});
}

TEST(SplitOperatorTest, NegativeAxis) {
  constexpr int64_t axis = -1;  // split last axis equally
  std::vector<ShapeAndFloatData> outputs;

  // input shape and data
  ShapeAndFloatData input = {{2, 4},
                             {1.f, 2.f, 3.f, 4.f,
                              5.f, 6.f, 7.f, 8.f}};

  outputs.push_back({{2, 2},
                     {1.f, 2.f,
                      5.f, 6.f}});

  outputs.push_back({{2, 2},
                     {3.f, 4.f,
                      7.f, 8.f}});

  RunTest<float>(axis, {}, input, outputs, {kTensorrtExecutionProvider}, false, true);
  RunTest<float>(axis, {}, input, outputs, {kTensorrtExecutionProvider});
}

TEST(SplitOperatorTest, InvalidAxis) {
  constexpr int64_t axis = 2;
  std::vector<ShapeAndFloatData> outputs;

  // input shape and data
  ShapeAndFloatData input = {{4, 2},  // shape
                             {1.f, 2.f,
                              3.f, 4.f,
                              5.f, 6.f,
                              7.f, 8.f}};

  outputs.push_back({{1}, {0.f}});

  RunTest<float>(axis, {}, input, outputs, {}, true, true, -1, true, "Invalid value of attribute 'axis'");
  RunTest<float>(axis, {}, input, outputs, {}, true, false, -1, true, "Invalid value of attribute 'axis'");
}

// sum of values in splits is too small
TEST(SplitOperatorTest, SplitAttributeSumTooSmall) {
  constexpr int64_t axis = 0;
  std::vector<ShapeAndFloatData> outputs;

  // input shape and data
  ShapeAndFloatData input = {{4, 2},  // shape
                             {1.f, 2.f,
                              3.f, 4.f,
                              5.f, 6.f,
                              7.f, 8.f}};

  std::vector<int64_t> splits{1, 2};  // should sum to 4

  outputs.push_back({{1, 2}, {1.f, 2.f}});
  outputs.push_back({{2, 2}, {3.f, 4.f, 5.f, 6.f}});

  RunTest<float>(axis, splits, input, outputs, {kTensorrtExecutionProvider}, true, true, -1, true,
                 "[ShapeInferenceError] Mismatch between the sum of 'split'");
  RunTest<float>(axis, splits, input, outputs, {kTensorrtExecutionProvider}, true, false, -1, true,
                 "[ShapeInferenceError] Mismatch between the sum of 'split'");  // TensorRT parser: Assertion failed: axis != BATCH_DIM
}

TEST(SplitOperatorTest, InvalidValueInSplitAttribute) {
  constexpr int64_t axis = -1;
  std::vector<ShapeAndFloatData> outputs;

  // input shape and data
  ShapeAndFloatData input = {{4, 2},  // shape
                             {1.f, 2.f,
                              3.f, 4.f,
                              5.f, 6.f,
                              7.f, 8.f}};

  std::vector<int64_t> splits{1, 0, 3};  // 0 is not valid
  outputs.push_back({{1, 2}, {1.f, 2.f}});
  outputs.push_back({{3, 2}, {3.f, 4.f, 5.f, 6.f, 7.f, 8.f}});

  RunTest<float>(axis, splits, input, outputs, {kTensorrtExecutionProvider}, true, true, -1, true,
                 "[ShapeInferenceError] Mismatch between number of splits");
  RunTest<float>(axis, splits, input, outputs, {kTensorrtExecutionProvider}, true, false, -1, true,
                 "[ShapeInferenceError] Mismatch between number of splits");  // TensorRT parser: Assertion failed: axis != BATCH_DIM
}

// split as input
TEST(SplitOperatorTest, Axis0UnequalSplitInputFloat) {
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

  RunTest<float>(axis, splits, input, outputs, {kTensorrtExecutionProvider}, false, true);
}

// split as input
TEST(SplitOperatorTest, Axis0UnequalSplitInputFloat_not_initializer) {
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

  RunTest<float>(axis, splits, input, outputs, {kTensorrtExecutionProvider}, false, true, -1, false);
}

/*
Python to replicate processing:

import numpy as np

def SplitAxis0():
    input = np.array([1., 2., 3., 4., 5., 6.]).astype(np.float32)

    # 3 outputs, axis 0
    expected_outputs = np.split(input, 3, 0)
    print("Split into 3, Axis=0")
    print(expected_outputs)

    # 2 outputs, split of 2, 4
    print("Split into 2 sizes 2 and 4, Axis=0")
    # numpy split is slightly different in that the values are indices rather than sizes
    # so a onnxruntime split of sizes 2, 4 for input of size 6 equates to just the index 2 (:2, 3:)
    # see https://docs.scipy.org/doc/numpy/reference/generated/numpy.split.html
    print(np.split(input, [2]))

def SplitAxis1():
    input = np.arange(1.0, 9).reshape(2, 4)

    # 2 outputs, axis 1
    x1, x2 = np.split(input, 2, 1)
    print("Split into 2, Axis=1\nx1={}\nx2={}".format(x1, x2))

    # 2 outputs, split of 3, 1
    x1, x2 = np.split(input, [3], 1)
    print("Split into 2 sizes 3 and 1, Axis=1\nx1={}\nx2={}".format(x1, x2))

def SplitAxis2():
    x = np.arange(1.0, 25.0).reshape((2,2,6))

    # equal split. 3 outputs, axis=2
    x1, x2, x3 = np.split(x, 3, 2)
    print("Equal split. Axis=2\nx1={}\nx2={}\nx3={}".format(x1, x2, x3))

    # split axis 2 into [1, 2, 3] sizes, which is index 1 (:1, 1:3, 3:)
    x1, x2, x3 = np.split(x, [1, 3], 2)
    print("Split into sizes 1, 3, 2. Axis=2\nx1={}\nx2={}\nx3={}".format(x1, x2, x3))

def SplitMiddleDimension():
    print ("SplitMiddleDimension")
    x = np.arange(1.0, 33.0).reshape((2,4,4))

    # equal split. 2 outputs, axis=1
    x1, x2 = np.split(x, 2, 1)
    print("Equal split. Axis=2\nx1={}\nx2={}".format(x1, x2))

    # split axis 2 into [1,3] sizes, which is index 1 (:1, 1:)
    x1, x2 = np.split(x, [1], 1)
    print("Split into sizes 1 and 3. Axis=2\nx1={}\nx2={}".format(x1, x2))

SplitAxis0()
SplitAxis1()
SplitAxis2()
SplitMiddleDimension()

*/

// test split for uint8_t data that has leading and trailing dimensions
TEST(SplitOperatorTest, Uint8Axis1SplitMiddleDimensionUnequally) {
  constexpr int64_t axis = 1;
  std::vector<ShapeAndData<uint8_t>> outputs;

  ShapeAndData<uint8_t> input = CreateInput<uint8_t>({2, 4, 4});

  std::vector<int64_t> splits{1, 3};

  outputs.push_back({{2, 1, 4},
                     {1, 2, 3, 4,

                      17, 18, 19, 20}});

  outputs.push_back({{2, 3, 4},
                     {5, 6, 7, 8,
                      9, 10, 11, 12,
                      13, 14, 15, 16,

                      21, 22, 23, 24,
                      25, 26, 27, 28,
                      29, 30, 31, 32}});

  RunTest<uint8_t>(axis, splits, input, outputs, {kTensorrtExecutionProvider});
}

// test split for uint8_t data on the last axis equally
TEST(SplitOperatorTest, Uint8NegativeAxis) {
  constexpr int64_t axis = -1;
  std::vector<ShapeAndData<uint8_t>> outputs;

  ShapeAndData<uint8_t> input = {{2, 4},
                                 {1, 2, 3, 4,
                                  5, 6, 7, 8}};

  outputs.push_back({{2, 2},
                     {1, 2,
                      5, 6}});

  outputs.push_back({{2, 2},
                     {3, 4,
                      7, 8}});

  RunTest<uint8_t>(axis, {}, input, outputs, {kTensorrtExecutionProvider});
}

TEST(SplitOperatorTest, MissingOptionalInputAdded) {
  constexpr int64_t axis = 1;  // split last axis equally
  std::vector<ShapeAndFloatData> outputs;

  // input shape and data
  ShapeAndFloatData input = {{2, 4},
                             {1.f, 2.f, 3.f, 4.f,
                              5.f, 6.f, 7.f, 8.f}};

  outputs.push_back({{2, 2},
                     {1.f, 2.f,
                      5.f, 6.f}});

  outputs.push_back({{2, 2},
                     {3.f, 4.f,
                      7.f, 8.f}});

  // CoreML EP does not support the case when split_is_input==true but missing providing the split as initializer.
  RunTest<float>(axis, {}, input, outputs, {kTensorrtExecutionProvider, kCoreMLExecutionProvider}, false, true, -1, false, {}, false);
}

TEST(SplitOperatorTest, Split18_NumOutputs_EvenSplit) {
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

  int64_t num_outputs = 2;

  RunTest<float>(axis, {}, input, outputs, {kTensorrtExecutionProvider}, false, true, num_outputs, true);
  RunTest<float>(axis, {}, input, outputs, {kTensorrtExecutionProvider}, false, true, num_outputs, false);
}

TEST(SplitOperatorTest, Split18_NumOutputs_UnevenSplit) {
  constexpr int64_t axis = 0;
  std::vector<ShapeAndFloatData> outputs;

  // input shape and data
  ShapeAndFloatData input = {{5, 2},  // shape
                             {1.f, 2.f,
                              3.f, 4.f,
                              5.f, 6.f,
                              7.f, 8.f,
                              9.f, 10.f}};

  outputs.push_back({{2, 2},
                     {1.f, 2.f,
                      3.f, 4.f}});

  outputs.push_back({{2, 2},
                     {5.f, 6.f,
                      7.f, 8.f}});

  outputs.push_back({{1, 2}, {9.f, 10.f}});

  int64_t num_outputs = 3;

  RunTest<float>(axis, {}, input, outputs, {kTensorrtExecutionProvider, kQnnExecutionProvider}, false, true, num_outputs, true);
  RunTest<float>(axis, {}, input, outputs, {kTensorrtExecutionProvider, kQnnExecutionProvider}, false, true, num_outputs, false);
}

TEST(SplitOperatorTest, Split18_InvalidNumOutputs) {
  constexpr int64_t axis = 0;
  std::vector<ShapeAndFloatData> outputs;

  // input shape and data
  ShapeAndFloatData input = {{2, 2},  // shape
                             {1.f, 2.f,
                              3.f, 4.f}};

  outputs.push_back({{2, 2},
                     {1.f, 2.f,
                      3.f, 4.f}});

  int64_t num_outputs = 0;
  const std::unordered_set<std::string> excluded_providers =
      {
          kTensorrtExecutionProvider,
          kQnnExecutionProvider,
          kDmlExecutionProvider,  // Error message differs from expected CPU EP error message.
      };
  RunTest<float>(axis, {}, input, outputs, excluded_providers, true, true, num_outputs, false,
                 "Attribute `num_outputs` value cannot be lower than 1");
  RunTest<float>(axis, {}, input, outputs, excluded_providers, true, true, num_outputs, true,
                 "Attribute `num_outputs` value cannot be lower than 1");

  outputs.clear();
  outputs.push_back({{1, 2},
                     {0.f, 0.f}});
  outputs.push_back({{1, 2},
                     {0.f, 0.f}});

  num_outputs = 3;

  RunTest<float>(axis, {}, input, outputs, excluded_providers, true, true, num_outputs, false,
                 "Invalid num_outputs value of 3. Size of dimension being split is 2");
  RunTest<float>(axis, {}, input, outputs, excluded_providers, true, true, num_outputs, true,
                 "Invalid num_outputs value of 3. Size of dimension being split is 2");
}

TEST(SplitOperatorTest, Split18_NumOutputsEvenSplitAxis1) {
  constexpr int64_t axis = 1;
  std::vector<ShapeAndFloatData> outputs;

  // input shape and data
  ShapeAndFloatData input = {{2, 3},  // shape
                             {1.f, 2.f, 3.f,
                              4.f, 5.f, 6.f}};

  outputs.push_back({{2, 1}, {1.f, 4.f}});
  outputs.push_back({{2, 1}, {2.f, 5.f}});
  outputs.push_back({{2, 1}, {3.f, 6.f}});

  int64_t num_outputs = 3;
  RunTest<float>(axis, {}, input, outputs, {kTensorrtExecutionProvider}, false, true, num_outputs, false);
  RunTest<float>(axis, {}, input, outputs, {kTensorrtExecutionProvider}, false, true, num_outputs);
}

TEST(SplitOperatorTest, Split18_NumOutputsUnevenSplitAxis1) {
  constexpr int64_t axis = 1;
  std::vector<ShapeAndFloatData> outputs;

  // input shape and data
  ShapeAndFloatData input = {{2, 3},  // shape
                             {1.f, 2.f, 3.f,
                              4.f, 5.f, 6.f}};

  outputs.push_back({{2, 2},
                     {1.f, 2.f,
                      4.f, 5.f}});
  outputs.push_back({{2, 1}, {3.f, 6.f}});

  int64_t num_outputs = 2;
  RunTest<float>(axis, {}, input, outputs, {kTensorrtExecutionProvider, kQnnExecutionProvider}, false, true, num_outputs);
  RunTest<float>(axis, {}, input, outputs, {kTensorrtExecutionProvider, kQnnExecutionProvider}, false, true, num_outputs, false);
}

}  // namespace test
}  // namespace onnxruntime
