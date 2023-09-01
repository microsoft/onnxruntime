// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "core/util/math.h"

namespace onnxruntime {
namespace test {

template <typename T>
struct ShrinkTestData {
  std::string name;
  float bias;
  float lambd;
  std::vector<T> input_vals;
  std::vector<int64_t> input_dimensions;
  std::vector<T> expected_vals;
  std::vector<int64_t> expected_dimensions;
};

template <typename T>
std::vector<ShrinkTestData<T>> GenerateSignedTestCases() {
  std::vector<ShrinkTestData<T>> test_cases;
  test_cases.push_back(
      {"default attributes",
       0.0f,
       0.5f,
       {-1, 0, 0, 1},
       {2, 2},
       {-1, 0, 0, 1},
       {2, 2}});

  test_cases.push_back(
      {"non-default attributes",
       10.0f,
       2.0f,
       {-3, -1, 1, 4},
       {2, 2},
       {7, 0, 0, -6},
       {2, 2}});
  return test_cases;
}

template <typename T>
std::vector<ShrinkTestData<T>> GenerateUnsignedTestCases() {
  std::vector<ShrinkTestData<T>> test_cases;

  test_cases.push_back(
      {"default attributes",
       0.0f,
       0.5f,
       {0, 0, 0, 1},
       {2, 2},
       {0, 0, 0, 1},
       {2, 2}});

  test_cases.push_back(
      {"non-default attributes",
       10.0f,
       2.0f,
       {37, 1, 1, 11},
       {2, 2},
       {27, 0, 0, 1},
       {2, 2}});

  return test_cases;
}

template <typename T>
void RunShrinkTest(const std::vector<ShrinkTestData<T>>& test_cases,
                   const std::unordered_set<std::string>& excluded_provider_types = {}) {
  for (const auto& test_data : test_cases) {
    OpTester test("Shrink", 9);

    if (test_data.bias != 0.0f) {
      test.AddAttribute("bias", test_data.bias);
    }

    if (test_data.lambd != 0.5f) {
      test.AddAttribute("lambd", test_data.lambd);
    }

    test.AddInput<T>("X", test_data.input_dimensions, test_data.input_vals);
    test.AddOutput<T>("Y", test_data.expected_dimensions, test_data.expected_vals);
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", excluded_provider_types);
  }
}

const std::vector<MLFloat16> ConvertFloatToMLFloat16(const std::vector<float>& float_data) {
  std::vector<MLFloat16> new_data;
  for (const auto& f : float_data) {
    new_data.push_back(MLFloat16(f));
  }
  return new_data;
}

TEST(MathOpTest, ShrinkInt8Type) {
  const auto& test_cases = GenerateSignedTestCases<int8_t>();
  RunShrinkTest<int8_t>(test_cases, {kTensorrtExecutionProvider});  // For TensorRT running in these in INT8 quantization scales are needed, so skip it now
}

TEST(MathOpTest, ShrinkUint8Type) {
  const auto& test_cases = GenerateUnsignedTestCases<uint8_t>();
  RunShrinkTest<uint8_t>(test_cases);
}

TEST(MathOpTest, ShrinkInt16Type) {
  const auto& test_cases = GenerateSignedTestCases<int16_t>();
  RunShrinkTest<int16_t>(test_cases);
}

TEST(MathOpTest, ShrinkUint16Type) {
  const auto& test_cases = GenerateUnsignedTestCases<uint16_t>();
  RunShrinkTest<uint16_t>(test_cases);
}

TEST(MathOpTest, ShrinkInt32Type) {
  const auto& test_cases = GenerateSignedTestCases<int32_t>();
  RunShrinkTest<int32_t>(test_cases);
}

TEST(MathOpTest, ShrinkUint32Type) {
  const auto& test_cases = GenerateUnsignedTestCases<uint32_t>();
  RunShrinkTest<uint32_t>(test_cases);
}

TEST(MathOpTest, ShrinkInt64Type) {
  const auto& test_cases = GenerateSignedTestCases<int64_t>();
  RunShrinkTest<int64_t>(test_cases);
}

TEST(MathOpTest, ShrinkUint64Type) {
  const auto& test_cases = GenerateUnsignedTestCases<uint64_t>();
  RunShrinkTest<uint64_t>(test_cases);
}

TEST(MathOpTest, ShrinkFloatType) {
  const auto& test_cases = GenerateSignedTestCases<float>();
  RunShrinkTest<float>(test_cases);
}

TEST(MathOpTest, ShrinkDoubleType) {
  const auto& test_cases = GenerateSignedTestCases<double>();
  RunShrinkTest<double>(test_cases);
}

TEST(MathOpTest, ShrinkMLFloat16Type) {
  const std::vector<MLFloat16> input_test_data_default = ConvertFloatToMLFloat16({-1, 0, 0, 1});
  const std::vector<MLFloat16> output_test_data_default = ConvertFloatToMLFloat16({-1, 0, 0, 1});

  const std::vector<MLFloat16> input_test_data_nondefault = ConvertFloatToMLFloat16({-3, -1, 1, 4});
  const std::vector<MLFloat16> output_test_data_nondefault = ConvertFloatToMLFloat16({7, 0, 0, -6});
  std::vector<ShrinkTestData<MLFloat16>> test_cases;
  test_cases.push_back(
      {"default attributes",
       0.0f,
       0.5f,
       input_test_data_default,
       {2, 2},
       output_test_data_default,
       {2, 2}});
  test_cases.push_back(
      {"non-default attributes",
       10.0f,
       2.0f,
       input_test_data_nondefault,
       {2, 2},
       output_test_data_nondefault,
       {2, 2}});
  RunShrinkTest<MLFloat16>(test_cases, {kTensorrtExecutionProvider});
}

}  // namespace test
}  // namespace onnxruntime
