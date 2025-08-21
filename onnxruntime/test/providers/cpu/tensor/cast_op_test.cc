// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <type_traits>

#include "boost/mp11.hpp"

#include <gsl/gsl>

#include "gtest/gtest.h"

#include "core/framework/data_types_internal.h"

#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

template <typename T>
int GetMinRequiredCudaComputeCapability() {
  return 0;
}

template <>
int GetMinRequiredCudaComputeCapability<MLFloat16>() {
  return 530;
}

template <>
int GetMinRequiredCudaComputeCapability<BFloat16>() {
  return 800;
}

#if !defined(DISABLE_FLOAT8_TYPES)

template <>
int GetMinRequiredCudaComputeCapability<Float8E4M3FN>() {
  return 800;
}

template <>
int GetMinRequiredCudaComputeCapability<Float8E5M2>() {
  return 800;
}

#endif

enum Saturate { True,
                False,
                None };

template <typename SrcType,
          typename DstType>
void TestCastOp(gsl::span<const SrcType> input,
                gsl::span<const DstType> output,
                const BaseTester::DimsVariant& dimensions,
                OpTester::ExpectResult expect_result = OpTester::ExpectResult::kExpectSuccess,
                const std::string& expected_failure_string = "",
                int opset = 21,
                Saturate saturate = Saturate::None,
                bool cuda_only = false) {
  OpTester test("Cast", opset);
  test.AddAttribute<int64_t>("to", utils::ToTensorProtoElementType<DstType>());
  test.AddInput<SrcType>("input", dimensions, input.data(), input.size());
  test.AddOutput<DstType>("output", dimensions, output.data(), output.size());
  if (saturate != Saturate::None) {
    test.AddAttribute<int64_t>("saturate", saturate == Saturate::True ? 1 : 0);
  }

  std::unordered_set<std::string> excluded_provider_types{kTensorrtExecutionProvider};
  const auto min_required_cuda_compute_capability =
      std::max(GetMinRequiredCudaComputeCapability<SrcType>(), GetMinRequiredCudaComputeCapability<DstType>());
  if (!HasCudaEnvironment(min_required_cuda_compute_capability)) {
    excluded_provider_types.insert(kCudaExecutionProvider);
  }

  if (cuda_only && (excluded_provider_types.count(kCudaExecutionProvider) > 0)) {
    return;
  }

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  if (cuda_only) {
    execution_providers.push_back(DefaultCudaExecutionProvider());
    test.Run(expect_result, expected_failure_string, {}, nullptr, &execution_providers);
    return;
  }

  test.Run(expect_result, expected_failure_string, excluded_provider_types);
}

template <typename T>
using RequiresCastThroughFloat =
    boost::mp11::mp_any<
        std::is_same<T, MLFloat16>,
        std::is_same<T, BFloat16>>;

template <typename... T>
using AnyRequireCastThroughFloat = boost::mp11::mp_any<RequiresCastThroughFloat<T>...>;

template <typename SrcType, typename DstType>
typename std::enable_if<AnyRequireCastThroughFloat<SrcType, DstType>::value>::type
CastSpan(gsl::span<const SrcType> src, gsl::span<DstType> dst) {
  std::transform(
      src.begin(), src.end(), dst.begin(),
      [](SrcType s) {
        return static_cast<DstType>(static_cast<float>(s));
      });
}

template <typename SrcType, typename DstType>
typename std::enable_if<!AnyRequireCastThroughFloat<SrcType, DstType>::value>::type
CastSpan(gsl::span<const SrcType> src, gsl::span<DstType> dst) {
  std::transform(
      src.begin(), src.end(), dst.begin(),
      [](SrcType s) {
        return static_cast<DstType>(s);
      });
}

template <typename SrcType, typename DstType>
std::vector<DstType> CastedValues(gsl::span<const SrcType> src) {
  std::vector<DstType> result(src.size());
  CastSpan<SrcType, DstType>(src, gsl::make_span(result));
  return result;
}

struct CastNonStringTester {
  template <typename SrcType, typename DstType>
  void operator()(const std::pair<SrcType, DstType>&) {
    SCOPED_TRACE(
        onnxruntime::MakeString(
            "Cast from type ", utils::ToTensorProtoElementType<SrcType>(),
            " to type ", utils::ToTensorProtoElementType<DstType>()));

    const std::vector<int> input_int_values{
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    const TensorShape shape{2, 3, 2, 2};
    const size_t size = gsl::narrow<size_t>(shape.Size());
    ASSERT_EQ(input_int_values.size(), size);

    auto input_buffer = std::make_unique<SrcType[]>(size);
    auto input_span = gsl::make_span<SrcType>(input_buffer.get(), size);
    CastSpan<int, SrcType>(gsl::make_span(input_int_values), input_span);

    auto output_buffer = std::make_unique<DstType[]>(size);
    auto output_span = gsl::make_span<DstType>(output_buffer.get(), size);
    CastSpan<SrcType, DstType>(input_span, output_span);

    TestCastOp<SrcType, DstType>(input_span, output_span, shape.AsShapeVector());
  }
};

using CastNonStringTypes =
    boost::mp11::mp_list<
        bool,
        float, double,
        uint8_t, uint16_t, uint32_t, uint64_t,
        int8_t, int16_t, int32_t, int64_t,
        MLFloat16, BFloat16>;

TEST(CastOpTest, NonStringTypes) {
  boost::mp11::mp_for_each<boost::mp11::mp_product<std::pair, CastNonStringTypes, CastNonStringTypes>>(
      CastNonStringTester{});
}

TEST(CastOpTest, FromString) {
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<std::string> string_data = {"-inf", "+INF", "0.9767611", "0.28280696",
                                                "-0.12019656", "5.0", "NaN", "nan"};
  const std::vector<float> float_output = {-(std::numeric_limits<float>::infinity()), std::numeric_limits<float>::infinity(),
                                           0.9767611f, 0.28280696f,
                                           -0.12019656f, 5.0f, NAN, NAN};
  TestCastOp(gsl::make_span(string_data), gsl::make_span(float_output), shape);

  const std::vector<std::string> float16_string_data = {"-inf", "+INF", "0.5", "0.25",
                                                        "0.0", "-1.0", "-1.5", "NaN"};
  const std::vector<MLFloat16> float16_output =
      CastedValues<float, MLFloat16>(
          gsl::make_span(
              std::vector<float>{
                  -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), 0.5f, 0.25f,
                  0.0f, -1.0f, -1.5f, NAN}));
  TestCastOp(gsl::make_span(float16_string_data), gsl::make_span(float16_output), shape);

  const std::vector<std::string> int_16_string_data = {"0", "1", "2", "3", "4", "5", "-32768", "32767"};
  const std::vector<int16_t> int_16_output = {0, 1, 2, 3, 4, 5, SHRT_MIN, SHRT_MAX};
  TestCastOp(gsl::make_span(int_16_string_data), gsl::make_span(int_16_output), shape);

  const std::vector<std::string> int_64_string_data = {"0", "1", "2", "3", "4", "5", "-9223372036854775808", "9223372036854775807"};
  const std::vector<int64_t> int_64_output = {0, 1, 2, 3, 4, 5, LLONG_MIN, LLONG_MAX};
  TestCastOp(gsl::make_span(int_64_string_data), gsl::make_span(int_64_output), shape);
}

TEST(CastOpTest, ToString) {
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<float> float_input = {NAN, -1.f, 0.0391877927f, 0.296140194f, -0.120196559f, 5.0f,
                                          -std::numeric_limits<float>::infinity(),
                                          std::numeric_limits<float>::infinity()};

  // float output precision is 8, so the expected output differs slightly from the input due to that
  const std::vector<std::string> string_output = {"NaN", "-1", "0.039187793", "0.29614019",
                                                  "-0.12019656", "5", "-INF", "INF"};
  TestCastOp(gsl::make_span(float_input), gsl::make_span(string_output), shape);

  const std::vector<MLFloat16> float16_input =
      CastedValues<float, MLFloat16>(
          gsl::make_span(
              std::vector<float>{
                  -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), 0.5f, 0.25f,
                  0.0f, -1.0f, -1.5f, NAN}));
  const std::vector<std::string> float16_string_output = {"-INF", "INF", "0.5", "0.25",
                                                          "0", "-1", "-1.5", "NaN"};
  TestCastOp(gsl::make_span(float16_input), gsl::make_span(float16_string_output), shape);

  const std::vector<std::string> int_string_data = {"0", "1", "2", "3", "4", "5", "6", "7"};
  const std::vector<int16_t> int_16_input = {0, 1, 2, 3, 4, 5, 6, 7};
  TestCastOp(gsl::make_span(int_16_input), gsl::make_span(int_string_data), shape);
}

TEST(CastOpTest, Int4x2ToInt8) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<Int4x2> int4x2_input = {
      Int4x2(-8, 7),  // boundary values
      Int4x2(0, -1),  // zero and negative
      Int4x2(3, -5),  // positive and negative
      Int4x2(6, 2)    // both positive
  };

  const std::vector<int8_t> expected_int8_output = {-8, 7, 0, -1, 3, -5, 6, 2};

  // WHEN, THEN
  TestCastOp(gsl::make_span(int4x2_input), gsl::make_span(expected_int8_output), shape);
}

TEST(CastOpTest, Int4x2ToUInt8) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<Int4x2> int4x2_input = {
      Int4x2(-8, 7),  // boundary values
      Int4x2(0, -1),  // zero and negative
      Int4x2(3, -5),  // positive and negative
      Int4x2(6, 2)    // both positive
  };

  // Negative values will be cast to their unsigned representation
  const std::vector<uint8_t> expected_uint8_output = {248, 7, 0, UINT8_MAX, 3, 251, 6, 2};

  // WHEN, THEN
  TestCastOp(gsl::make_span(int4x2_input), gsl::make_span(expected_uint8_output), shape);
}

TEST(CastOpTest, Int4x2ToInt16) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<Int4x2> int4x2_input = {
      Int4x2(-8, 7),  // boundary values
      Int4x2(0, -1),  // zero and negative
      Int4x2(3, -5),  // positive and negative
      Int4x2(6, 2)    // both positive
  };

  const std::vector<int16_t> expected_int16_output = {-8, 7, 0, -1, 3, -5, 6, 2};

  // WHEN, THEN
  TestCastOp(gsl::make_span(int4x2_input), gsl::make_span(expected_int16_output), shape);
}

TEST(CastOpTest, Int4x2ToUInt16) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<Int4x2> int4x2_input = {
      Int4x2(-8, 7),  // boundary values
      Int4x2(0, -1),  // zero and negative
      Int4x2(3, -5),  // positive and negative
      Int4x2(6, 2)    // both positive
  };

  // Negative values will be cast to their unsigned representation
  const std::vector<uint16_t> expected_uint16_output = {65528, 7, 0, UINT16_MAX, 3, 65531, 6, 2};

  // WHEN, THEN
  TestCastOp(gsl::make_span(int4x2_input), gsl::make_span(expected_uint16_output), shape);
}

TEST(CastOpTest, Int4x2ToInt32) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<Int4x2> int4x2_input = {
      Int4x2(-8, 7),  // boundary values
      Int4x2(0, -1),  // zero and negative
      Int4x2(3, -5),  // positive and negative
      Int4x2(6, 2)    // both positive
  };

  const std::vector<int32_t> expected_int32_output = {-8, 7, 0, -1, 3, -5, 6, 2};

  // WHEN, THEN
  TestCastOp(gsl::make_span(int4x2_input), gsl::make_span(expected_int32_output), shape);
}

TEST(CastOpTest, Int4x2ToUInt32) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<Int4x2> int4x2_input = {
      Int4x2(-8, 7),  // boundary values
      Int4x2(0, -1),  // zero and negative
      Int4x2(3, -5),  // positive and negative
      Int4x2(6, 2)    // both positive
  };

  // Negative values will be cast to their unsigned representation
  const std::vector<uint32_t> expected_uint32_output = {4294967288, 7, 0, UINT32_MAX, 3, 4294967291, 6, 2};

  // WHEN, THEN
  TestCastOp(gsl::make_span(int4x2_input), gsl::make_span(expected_uint32_output), shape);
}

TEST(CastOpTest, Int4x2ToInt32OddNumberOfElements) {
  // GIVEN
  const std::vector<int64_t> odd_shape{5};
  const std::vector<Int4x2> odd_input = {
      Int4x2(-8, 7),  // boundary values
      Int4x2(0, -1),  // zero and negative
      Int4x2(3, 0),
  };

  const std::vector<int32_t> expected_odd_output = {-8, 7, 0, -1, 3};

  // WHEN, THEN
  TestCastOp(gsl::make_span(odd_input), gsl::make_span(expected_odd_output), odd_shape);
}

TEST(CastOpTest, Int4x2ToInt64) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<Int4x2> int4x2_input = {
      Int4x2(-8, 7),  // boundary values
      Int4x2(0, -1),  // zero and negative
      Int4x2(3, -5),  // positive and negative
      Int4x2(6, 2)    // both positive
  };

  const std::vector<int64_t> expected_int64_output = {-8, 7, 0, -1, 3, -5, 6, 2};

  // WHEN, THEN
  TestCastOp(gsl::make_span(int4x2_input), gsl::make_span(expected_int64_output), shape);
}

TEST(CastOpTest, Int4x2ToUInt64) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<Int4x2> int4x2_input = {
      Int4x2(-8, 7),  // boundary values
      Int4x2(0, -1),  // zero and negative
      Int4x2(3, -5),  // positive and negative
      Int4x2(6, 2)    // both positive
  };

  // Negative values will be cast to their unsigned representation
  const std::vector<uint64_t> expected_uint64_output = {18446744073709551608ULL, 7, 0, UINT64_MAX, 3, 18446744073709551611ULL, 6, 2};

  // WHEN, THEN
  TestCastOp(gsl::make_span(int4x2_input), gsl::make_span(expected_uint64_output), shape);
}

TEST(CastOpTest, UInt4x2ToUInt8) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<UInt4x2> uint4x2_input = {
      UInt4x2(0, 15),  // boundary values
      UInt4x2(1, 14),  // small and large
      UInt4x2(7, 8),   // middle values
      UInt4x2(3, 12)   // mixed values
  };

  const std::vector<uint8_t> expected_uint8_output = {0, 15, 1, 14, 7, 8, 3, 12};

  // WHEN, THEN
  TestCastOp(gsl::make_span(uint4x2_input), gsl::make_span(expected_uint8_output), shape);
}

TEST(CastOpTest, UInt4x2ToInt8) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<UInt4x2> uint4x2_input = {
      UInt4x2(0, 15),  // boundary values
      UInt4x2(1, 14),  // small and large
      UInt4x2(7, 8),   // middle values
      UInt4x2(3, 12)   // mixed values
  };

  const std::vector<int8_t> expected_int8_output = {0, 15, 1, 14, 7, 8, 3, 12};

  // WHEN, THEN
  TestCastOp(gsl::make_span(uint4x2_input), gsl::make_span(expected_int8_output), shape);
}

TEST(CastOpTest, UInt4x2ToUInt16) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<UInt4x2> uint4x2_input = {
      UInt4x2(0, 15),  // boundary values
      UInt4x2(1, 14),  // small and large
      UInt4x2(7, 8),   // middle values
      UInt4x2(3, 12)   // mixed values
  };

  const std::vector<uint16_t> expected_uint16_output = {0, 15, 1, 14, 7, 8, 3, 12};

  // WHEN, THEN
  TestCastOp(gsl::make_span(uint4x2_input), gsl::make_span(expected_uint16_output), shape);
}

TEST(CastOpTest, UInt4x2ToInt16) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<UInt4x2> uint4x2_input = {
      UInt4x2(0, 15),  // boundary values
      UInt4x2(1, 14),  // small and large
      UInt4x2(7, 8),   // middle values
      UInt4x2(3, 12)   // mixed values
  };

  const std::vector<int16_t> expected_int16_output = {0, 15, 1, 14, 7, 8, 3, 12};

  // WHEN, THEN
  TestCastOp(gsl::make_span(uint4x2_input), gsl::make_span(expected_int16_output), shape);
}

TEST(CastOpTest, UInt4x2ToUInt32) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<UInt4x2> uint4x2_input = {
      UInt4x2(0, 15),  // boundary values
      UInt4x2(1, 14),  // small and large
      UInt4x2(7, 8),   // middle values
      UInt4x2(3, 12)   // mixed values
  };

  const std::vector<uint32_t> expected_uint32_output = {0, 15, 1, 14, 7, 8, 3, 12};

  // WHEN, THEN
  TestCastOp(gsl::make_span(uint4x2_input), gsl::make_span(expected_uint32_output), shape);
}

TEST(CastOpTest, UInt4x2ToInt32) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<UInt4x2> uint4x2_input = {
      UInt4x2(0, 15),  // boundary values
      UInt4x2(1, 14),  // small and large
      UInt4x2(7, 8),   // middle values
      UInt4x2(3, 12)   // mixed values
  };

  const std::vector<int32_t> expected_int32_output = {0, 15, 1, 14, 7, 8, 3, 12};

  // WHEN, THEN
  TestCastOp(gsl::make_span(uint4x2_input), gsl::make_span(expected_int32_output), shape);
}

TEST(CastOpTest, UInt4x2ToUInt64) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<UInt4x2> uint4x2_input = {
      UInt4x2(0, 15),  // boundary values
      UInt4x2(1, 14),  // small and large
      UInt4x2(7, 8),   // middle values
      UInt4x2(3, 12)   // mixed values
  };

  const std::vector<uint64_t> expected_uint64_output = {0, 15, 1, 14, 7, 8, 3, 12};

  // WHEN, THEN
  TestCastOp(gsl::make_span(uint4x2_input), gsl::make_span(expected_uint64_output), shape);
}

TEST(CastOpTest, UInt4x2ToInt64) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<UInt4x2> uint4x2_input = {
      UInt4x2(0, 15),  // boundary values
      UInt4x2(1, 14),  // small and large
      UInt4x2(7, 8),   // middle values
      UInt4x2(3, 12)   // mixed values
  };

  const std::vector<int64_t> expected_int64_output = {0, 15, 1, 14, 7, 8, 3, 12};

  // WHEN, THEN
  TestCastOp(gsl::make_span(uint4x2_input), gsl::make_span(expected_int64_output), shape);
}

TEST(CastOpTest, Int4x2ToBool) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<Int4x2> int4x2_input = {
      Int4x2(0, -1),  // zero and non-zero
      Int4x2(7, 0),   // non-zero and zero
      Int4x2(-8, 3),  // both non-zero
      Int4x2(0, 0)    // both zero
  };

  const bool bool_output[] = {false, true, true, false, true, true, false, false};
  const gsl::span<const bool> expected_bool_output_span(bool_output);

  // WHEN, THEN
  TestCastOp(gsl::make_span(int4x2_input), expected_bool_output_span, shape);
}

TEST(CastOpTest, UInt4x2ToBool) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<UInt4x2> uint4x2_input = {
      UInt4x2(0, 1),   // zero and non-zero
      UInt4x2(15, 0),  // non-zero and zero
      UInt4x2(8, 7),   // both non-zero
      UInt4x2(0, 0)    // both zero
  };

  const bool bool_output[] = {false, true, true, false, true, true, false, false};
  const gsl::span<const bool> expected_bool_output_span(bool_output);

  // WHEN, THEN
  TestCastOp(gsl::make_span(uint4x2_input), expected_bool_output_span, shape);
}

TEST(CastOpTest, Int4x2ToFloat) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<Int4x2> int4x2_input = {
      Int4x2(1, 2),  // two 4-bit int elements: lower = 1, upper = 2
      Int4x2(-3, -4),
      Int4x2(5, -6),
      Int4x2(-8, 7)};

  const std::vector<float> expected_float_output = {1.0f, 2.0f, -3.0f, -4.0f, 5.0f, -6.0f, -8.0f, 7.0f};

  // WHEN, THEN
  TestCastOp(gsl::make_span(int4x2_input), gsl::make_span(expected_float_output), shape);
}

TEST(CastOpTest, UInt4x2ToFloat) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<UInt4x2> uint4x2_input = {
      UInt4x2(0, 1),
      UInt4x2(2, 3),
      UInt4x2(7, 8),
      UInt4x2(14, 15)};

  const std::vector<float> expected_float_output = {0.0f, 1.0f, 2.0f, 3.0f, 7.0f, 8.0f, 14.0f, 15.0f};

  // WHEN, THEN
  TestCastOp(gsl::make_span(uint4x2_input), gsl::make_span(expected_float_output), shape);
}

TEST(CastOpTest, Int4x2ToDouble) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<Int4x2> int4x2_input = {
      Int4x2(-8, 7),  // boundary values
      Int4x2(0, -3),  // zero and negative
      Int4x2(4, -2),  // positive and negative
      Int4x2(1, 6)    // both positive
  };

  const std::vector<double> expected_double_output = {-8.0, 7.0, 0.0, -3.0, 4.0, -2.0, 1.0, 6.0};

  // WHEN, THEN
  TestCastOp(gsl::make_span(int4x2_input), gsl::make_span(expected_double_output), shape);
}

TEST(CastOpTest, UInt4x2ToDouble) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<UInt4x2> uint4x2_input = {
      UInt4x2(0, 15),  // boundary values
      UInt4x2(1, 14),  // small and large
      UInt4x2(7, 8),   // middle values
      UInt4x2(3, 12)   // mixed values
  };

  const std::vector<double> expected_double_output = {0.0, 15.0, 1.0, 14.0, 7.0, 8.0, 3.0, 12.0};

  // WHEN, THEN
  TestCastOp(gsl::make_span(uint4x2_input), gsl::make_span(expected_double_output), shape);
}

TEST(CastOpTest, Int4x2ToMLFloat16) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<Int4x2> int4x2_input = {
      Int4x2(-8, 7),
      Int4x2(0, -1),
      Int4x2(3, -5),
      Int4x2(6, 2)};

  const std::vector<MLFloat16> expected_float16_output =
      CastedValues<float, MLFloat16>(
          gsl::make_span(
              std::vector<float>{-8.0f, 7.0f, 0.0f, -1.0f, 3.0f, -5.0f, 6.0f, 2.0f}));

  // WHEN, THEN
  TestCastOp(gsl::make_span(int4x2_input), gsl::make_span(expected_float16_output), shape);
}

TEST(CastOpTest, UInt4x2ToMLFloat16) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<UInt4x2> uint4x2_input = {
      UInt4x2(0, 15),
      UInt4x2(1, 14),
      UInt4x2(7, 8),
      UInt4x2(3, 12)};

  const std::vector<MLFloat16> expected_float16_output =
      CastedValues<float, MLFloat16>(
          gsl::make_span(
              std::vector<float>{0.0f, 15.0f, 1.0f, 14.0f, 7.0f, 8.0f, 3.0f, 12.0f}));

  // WHEN, THEN
  TestCastOp(gsl::make_span(uint4x2_input), gsl::make_span(expected_float16_output), shape);
}

TEST(CastOpTest, Int4x2ToBFloat16) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<Int4x2> int4x2_input = {
      Int4x2(-8, 7),
      Int4x2(0, -1),
      Int4x2(3, -5),
      Int4x2(6, 2)};

  const std::vector<BFloat16> expected_bfloat16_output =
      CastedValues<float, BFloat16>(
          gsl::make_span(
              std::vector<float>{-8.0f, 7.0f, 0.0f, -1.0f, 3.0f, -5.0f, 6.0f, 2.0f}));

  // WHEN, THEN
  TestCastOp(gsl::make_span(int4x2_input), gsl::make_span(expected_bfloat16_output), shape);
}

TEST(CastOpTest, UInt4x2ToBFloat16) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<UInt4x2> uint4x2_input = {
      UInt4x2(0, 15),
      UInt4x2(1, 14),
      UInt4x2(7, 8),
      UInt4x2(3, 12)};

  const std::vector<BFloat16> expected_bfloat16_output =
      CastedValues<float, BFloat16>(
          gsl::make_span(
              std::vector<float>{0.0f, 15.0f, 1.0f, 14.0f, 7.0f, 8.0f, 3.0f, 12.0f}));

  // WHEN, THEN
  TestCastOp(gsl::make_span(uint4x2_input), gsl::make_span(expected_bfloat16_output), shape);
}

TEST(CastOpTest, Int4x2ToString) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<Int4x2> int4x2_input = {
      Int4x2(-8, 7),  // boundary values
      Int4x2(0, -1),  // zero and negative
      Int4x2(3, -5),  // mixed values
      Int4x2(6, 2)    // positive values
  };

  // Each Int4x2 becomes two string values
  const std::vector<std::string> expected_output = {
      "-8", "7",  // from first Int4x2
      "0", "-1",  // from second Int4x2
      "3", "-5",  // from third Int4x2
      "6", "2"    // from fourth Int4x2
  };

  // WHEN, THEN
  TestCastOp(gsl::span<const Int4x2>(int4x2_input), gsl::span<const std::string>(expected_output), shape);
}

TEST(CastOpTest, UInt4x2ToString) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<UInt4x2> uint4x2_input = {
      UInt4x2(0, 15),  // boundary values
      UInt4x2(8, 7),   // mid-range values
      UInt4x2(3, 12),  // mixed values
      UInt4x2(10, 5)   // other values
  };

  // Each UInt4x2 becomes two string values
  const std::vector<std::string> expected_output = {
      "0", "15",  // from first UInt4x2
      "8", "7",   // from second UInt4x2
      "3", "12",  // from third UInt4x2
      "10", "5"   // from fourth UInt4x2
  };

  // WHEN, THEN
  TestCastOp(gsl::span<const UInt4x2>(uint4x2_input), gsl::span<const std::string>(expected_output), shape);
}

TEST(CastOpTest, Int4x2ToUInt4x2) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<Int4x2> int4x2_input = {
      Int4x2(-8, 7),  // min and max values
      Int4x2(0, -1),  // -1 becomes max unsigned value
      Int4x2(3, -5),  // positive and negative values
      Int4x2(6, 2)    // positive values
  };

  const std::vector<UInt4x2> expected_uint4x2_output = {
      UInt4x2(8, 7),   // -8 becomes 8
      UInt4x2(0, 15),  // -1 becomes 15
      UInt4x2(3, 11),  // -5 becomes 11
      UInt4x2(6, 2)    // unchanged
  };

  // WHEN, THEN
  TestCastOp(gsl::make_span(int4x2_input), gsl::make_span(expected_uint4x2_output), shape);
}

TEST(CastOpTest, UInt4x2ToInt4x2) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<UInt4x2> uint4x2_input = {
      UInt4x2(0, 15),  // 15 is out of int4 range
      UInt4x2(1, 14),  // 14 is out of int4 range
      UInt4x2(7, 8),   // 8 is out of int4 range
      UInt4x2(3, 6)    // both within range
  };

  const std::vector<Int4x2> expected_int4x2_output = {
      Int4x2(0, -1),  // 15 becomes -1
      Int4x2(1, -2),  // 14 becomes -2
      Int4x2(7, -8),  // 8 becomes -8
      Int4x2(3, 6)    // unchanged
  };

  // WHEN, THEN
  TestCastOp(gsl::make_span(uint4x2_input), gsl::make_span(expected_int4x2_output), shape);
}

TEST(CastOpTest, Int8ToInt4x2) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<int8_t> int8_input = {-10, 15, 0, -1, 7, -8, -128, 127};

  const std::vector<Int4x2> expected_int4x2_output = {
      // 10 in binary is 00001010.
      // Invert all bits -> 11110101, add 1 -> 11110110
      // So -10 in binary is 11110110.
      // Truncate to 4 least significant bits -> 0110.
      // In 4-bit two's complement, 0110 = 0 * -8 + 1 * 4 + 1 * 2 = 6.
      Int4x2(6, -1),  // -10 truncated to 6, 15 truncated to -1
      Int4x2(0, -1),  // 0 unchanged, -1 unchanged
      Int4x2(7, -8),  // 7 unchanged, -8 unchanged
      Int4x2(0, -1)   // -128 truncated to 0, 127 truncated to -1
  };

  // WHEN, THEN
  TestCastOp(gsl::make_span(int8_input), gsl::make_span(expected_int4x2_output), shape);
}

TEST(CastOpTest, UInt8ToUInt4x2) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<uint8_t> uint8_input = {20, 255, 0, 17, 7, 240, 15, 31};

  // values get truncated to lower 4 bits
  const std::vector<UInt4x2> expected_uint4x2_output = {
      UInt4x2(4, 15),  // 20 (0x14) truncated to 4, 255 (0xFF) truncated to 15
      UInt4x2(0, 1),   // 0 (0x00) truncated to 0, 17 (0x11) truncated to 1
      UInt4x2(7, 0),   // 7 (0x07) truncated to 7, 240 (0xF0) truncated to 0
      UInt4x2(15, 15)  // 15 (0x0F) truncated to 15, 31 (0x1F) truncated to 15
  };

  // WHEN, THEN
  TestCastOp(gsl::make_span(uint8_input), gsl::make_span(expected_uint4x2_output), shape);
}

TEST(CastOpTest, Int16ToInt4x2) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<int16_t> int16_input = {-10, 32767, 0, -32768, 7, -8, 240, 31};

  // values get truncated to lower 4 bits and sign-extended
  const std::vector<Int4x2> expected_int4x2_output = {
      Int4x2(6, -1),  // -10 (0xFFF6) truncated to 6, 32767 (0x7FFF) truncated to -1
      Int4x2(0, 0),   // 0 (0x0000) truncated to 0, -32768 (0x8000) truncated to 0
      Int4x2(7, -8),  // 7 (0x0007) truncated to 7, -8 (0xFFF8) truncated to -8
      Int4x2(0, -1)   // 240 (0x00F0) truncated to 0, 31 (0x001F) truncated to -1
  };

  // WHEN, THEN
  TestCastOp(gsl::make_span(int16_input), gsl::make_span(expected_int4x2_output), shape);
}

TEST(CastOpTest, UInt16ToUInt4x2) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<uint16_t> uint16_input = {20, 65535, 0, 256, 7, 240, 15, 4095};

  // values get truncated to lower 4 bits
  const std::vector<UInt4x2> expected_uint4x2_output = {
      UInt4x2(4, 15),  // 20 (0x0014) truncated to 4, 65535 (0xFFFF) truncated to 15
      UInt4x2(0, 0),   // 0 (0x0000) truncated to 0, 256 (0x0100) truncated to 0
      UInt4x2(7, 0),   // 7 (0x0007) truncated to 7, 240 (0x00F0) truncated to 0
      UInt4x2(15, 15)  // 15 (0x000F) truncated to 15, 4095 (0x0FFF) truncated to 15
  };

  // WHEN, THEN
  TestCastOp(gsl::make_span(uint16_input), gsl::make_span(expected_uint4x2_output), shape);
}

TEST(CastOpTest, Int32ToInt4x2) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<int32_t> int32_input = {-10, INT32_MAX, 0, INT32_MIN, 3, -5, 4080, 287};

  // values get truncated to lower 4 bits and sign-extended
  const std::vector<Int4x2> expected_int4x2_output = {
      Int4x2(6, -1),  // -10 (0xFFFFFFF6) truncated to 6, 2147483647 (0x7FFFFFFF) truncated to -1
      Int4x2(0, 0),   // 0 (0x00000000) truncated to 0, -2147483648 (0x80000000) truncated to 0
      Int4x2(3, -5),  // 3 (0x00000003) truncated to 3, -5 (0xFFFFFFFB) truncated to -5
      Int4x2(0, -1)   // 4080 (0x00000FF0) truncated to 0, 287 (0x0000011F) truncated to -1
  };

  // WHEN, THEN
  TestCastOp(gsl::make_span(int32_input), gsl::make_span(expected_int4x2_output), shape);
}

TEST(CastOpTest, Int32ToInt4x2OddNumberOfElements) {
  // GIVEN
  const std::vector<int64_t> odd_shape{5};
  const std::vector<int32_t> odd_input = {-10, INT32_MAX, 0, INT32_MIN, 4095};

  const std::vector<Int4x2> expected_odd_output = {
      Int4x2(6, -1),  // -10 truncated to 6, 2147483647 truncated to -1
      Int4x2(0, 0),   // 0 truncated to 0, -2147483648 truncated to 0
      Int4x2(-1, 0)   // 4095 truncated to -1, paired with 0
  };

  // WHEN, THEN
  TestCastOp(gsl::make_span(odd_input), gsl::make_span(expected_odd_output), odd_shape);
}

TEST(CastOpTest, Int32ToInt4x2EmptyTensor) {
  // GIVEN
  const std::vector<int64_t> empty_shape{0};
  const std::vector<int32_t> empty_input = {};
  const std::vector<Int4x2> empty_output = {};

  // WHEN, THEN
  TestCastOp(gsl::make_span(empty_input), gsl::make_span(empty_output), empty_shape);
}

TEST(CastOpTest, UInt32ToUInt4x2) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<uint32_t> uint32_input = {20, UINT32_MAX, 0, 256, 7, 240, 15, 4095};

  // values get truncated to lower 4 bits
  const std::vector<UInt4x2> expected_uint4x2_output = {
      UInt4x2(4, 15),  // 20 truncated to 4, 4294967295 truncated to 15
      UInt4x2(0, 0),   // 0 truncated to 0, 256 truncated to 0
      UInt4x2(7, 0),   // 7 truncated to 7, 240 truncated to 0
      UInt4x2(15, 15)  // 15 truncated to 15, 4095 truncated to 15
  };

  // WHEN, THEN
  TestCastOp(gsl::make_span(uint32_input), gsl::make_span(expected_uint4x2_output), shape);
}

TEST(CastOpTest, Int64ToInt4x2) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<int64_t> int64_input = {-10, INT64_MAX, 0, INT64_MIN, 7, -8, 65520, 4111};

  // values get truncated to lower 4 bits and sign-extended
  const std::vector<Int4x2> expected_int4x2_output = {
      Int4x2(6, -1),  // -10 truncated to 6, 9223372036854775807 truncated to -1
      Int4x2(0, 0),   // 0 truncated to 0, -9223372036854775808 truncated to 0
      Int4x2(7, -8),  // 7 truncated to 7, -8 truncated to -8
      Int4x2(0, -1)   // 65520 truncated to 0, 4111 truncated to -1
  };

  // WHEN, THEN
  TestCastOp(gsl::make_span(int64_input), gsl::make_span(expected_int4x2_output), shape);
}

TEST(CastOpTest, UInt64ToUInt4x2) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<uint64_t> uint64_input = {20, UINT64_MAX, 0, 256, 7, 240, 15, 4095};

  // values get truncated to lower 4 bits
  const std::vector<UInt4x2> expected_uint4x2_output = {
      UInt4x2(4, 15),  // 20 truncated to 4, 18446744073709551615 truncated to 15
      UInt4x2(0, 0),   // 0 truncated to 0, 256 truncated to 0
      UInt4x2(7, 0),   // 7 truncated to 7, 240 truncated to 0
      UInt4x2(15, 15)  // 15 truncated to 15, 4095 truncated to 15
  };

  // WHEN, THEN
  TestCastOp(gsl::make_span(uint64_input), gsl::make_span(expected_uint4x2_output), shape);
}

TEST(CastOpTest, FloatToInt4x2) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<float> float_input = {-10.7f, 15.3f, 0.4f, -1.6f, 7.0f, -8.0f, 240.1f, 31.9f};

  const std::vector<Int4x2> expected_int4x2_output = {
      Int4x2(5, -1),  // -10.7 rounded to -11 (0xF5), truncated to 5, sign-extended to 5; 15.3 rounded to 15 (0x0F), sign-extended to -1
      Int4x2(0, -2),  // 0.4 rounded to 0; -1.6 rounded to -2 (0xFE), truncated to 14 (0x0E), sign-extended to -2
      Int4x2(7, -8),  // 7.0 converted to 7; -8.0 converted to -8
      Int4x2(0, 0)    // 240.1 rounded to 240 (0xF0), truncated to 0; 31.9 rounded to 32 (0x20), truncated to 0
  };

  // WHEN, THEN
  TestCastOp(gsl::make_span(float_input), gsl::make_span(expected_int4x2_output), shape);
}

TEST(CastOpTest, DoubleToUInt4x2) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<double> double_input = {20.7, 255.3, 0.4, 1.6, 7.8, 240.2, 15.1, 31.9};

  const std::vector<UInt4x2> expected_uint4x2_output = {
      UInt4x2(5, 15),  // 20.7 rounded to 21, truncated to 5; 255.3 rounded to 255, truncated to 15
      UInt4x2(0, 2),   // 0.4 rounded to 0; 1.6 rounded to 2
      UInt4x2(8, 0),   // 7.8 rounded to 8; 240.2 rounded to 240, truncated to 0
      UInt4x2(15, 0)   // 15.1 rounded to 15; 31.9 rounded to 32, truncated to 0
  };

  // WHEN, THEN
  TestCastOp(gsl::make_span(double_input), gsl::make_span(expected_uint4x2_output), shape);
}

TEST(CastOpTest, MLFloat16ToInt4x2) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const MLFloat16 mlfloat16_array[8] = {
      MLFloat16(static_cast<float>(-10.7f)),
      MLFloat16(static_cast<float>(15.3f)),
      MLFloat16(static_cast<float>(0.4f)),
      MLFloat16(static_cast<float>(-1.6f)),
      MLFloat16(static_cast<float>(3.8f)),
      MLFloat16(static_cast<float>(-5.2f)),
      MLFloat16(static_cast<float>(240.1f)),
      MLFloat16(static_cast<float>(31.9f))};

  const std::vector<Int4x2> expected_int4x2 = {
      Int4x2(5, -1),  // -10.7 rounded to -11 (0xF5), truncated to 5; 15.3 rounded to 15 (0x0F), sign-extended to -1
      Int4x2(0, -2),  // 0.4 rounded to 0; -1.6 rounded to -2 (0xFE), truncated to 14 (0x0E), sign-extended to -2
      Int4x2(4, -5),  // 3.8 rounded to 4; -5.2 rounded to -5 (0xFB), truncated to 11 (0x0B), sign-extended to -5
      Int4x2(0, 0)    // 240.1 rounded to 240 (0xF0), truncated to 0; 31.9 rounded to 32 (0x20), truncated to 0
  };

  // WHEN, THEN
  TestCastOp(
      gsl::span<const MLFloat16>(mlfloat16_array, 8),
      gsl::span<const Int4x2>(expected_int4x2),
      shape);
}

TEST(CastOpTest, MLFloat16ToUInt4x2) {
  // GIVEN
  // 8 MLFloat16 values will compress to 4 UInt4x2 values
  const std::vector<int64_t> shape{2, 4};  // Shape that contains 8 elements

  // MLFloat16 values with edge cases and truncation scenarios
  const MLFloat16 mlfloat16_array[8] = {
      MLFloat16(static_cast<float>(20.7f)),
      MLFloat16(static_cast<float>(255.3f)),
      MLFloat16(static_cast<float>(0.4f)),
      MLFloat16(static_cast<float>(1.6f)),
      MLFloat16(static_cast<float>(7.8f)),
      MLFloat16(static_cast<float>(240.2f)),
      MLFloat16(static_cast<float>(15.1f)),
      MLFloat16(static_cast<float>(31.9f))};

  const std::vector<UInt4x2> expected_uint4x2 = {
      UInt4x2(5, 15),  // 20.7 rounded to 21, truncated to 5; 255.3 rounded to 255, truncated to 15
      UInt4x2(0, 2),   // 0.4 rounded to 0; 1.6 rounded to 2
      UInt4x2(8, 0),   // 7.8 rounded to 8; 240.2 rounded to 240, truncated to 0
      UInt4x2(15, 0)   // 15.1 rounded to 15; 31.9 rounded to 32, truncated to 0
  };

  // WHEN, THEN
  TestCastOp(
      gsl::span<const MLFloat16>(mlfloat16_array, 8),
      gsl::span<const UInt4x2>(expected_uint4x2),
      shape);
}

TEST(CastOpTest, MLFloat16ToInt4x2BoundaryValues) {
  // GIVEN
  // Test MLFloat16 values that need truncation to Int4x2 range
  const std::vector<int64_t> shape{3, 2};
  const MLFloat16 mlfloat16_array[6] = {
      MLFloat16(static_cast<float>(-10)),    // Truncated to lower 4 bits
      MLFloat16(static_cast<float>(9)),      // Truncated to lower 4 bits
      MLFloat16(static_cast<float>(-8)),     // Truncated to lower 4 bits
      MLFloat16(static_cast<float>(7)),      // Truncated to lower 4 bits
      MLFloat16(static_cast<float>(-0.6f)),  // Should round to -1
      MLFloat16(static_cast<float>(1.7f))    // Should round to 2
  };

  // Values get truncated to lower 4 bits and sign-extended
  const std::vector<Int4x2> expected_int4x2 = {
      Int4x2(6, -7),  // -10 (0xFFFFFFF6) truncated to 6, 9 (0x00000009) truncated to -7
      Int4x2(-8, 7),  // -8 (0xFFFFFFF8) truncated to -8, 7 (0x00000007) truncated to 7
      Int4x2(-1, 2)   // -0.6 rounds to -1, 1.7 rounds to 2
  };

  // WHEN, THEN
  TestCastOp(
      gsl::span<const MLFloat16>(mlfloat16_array, 6),
      gsl::span<const Int4x2>(expected_int4x2),
      shape);
}

TEST(CastOpTest, MLFloat16ToUInt4x2BoundaryValues) {
  // GIVEN
  // Test MLFloat16 values that need truncation to UInt4x2 range
  const std::vector<int64_t> shape{3, 2};  // Shape that contains 6 elements
  const MLFloat16 mlfloat16_array[6] = {
      MLFloat16(static_cast<float>(-5)),    // Negative, truncated to lower 4 bits
      MLFloat16(static_cast<float>(20)),    // Above max, truncated to lower 4 bits
      MLFloat16(static_cast<float>(0)),     // At min, should remain 0
      MLFloat16(static_cast<float>(15)),    // At max, should remain 15
      MLFloat16(static_cast<float>(3.4f)),  // Should round to 3
      MLFloat16(static_cast<float>(5.7f))   // Should round to 6
  };

  // Values get truncated to lower 4 bits (no sign extension for unsigned)
  const std::vector<UInt4x2> expected_uint4x2 = {
      UInt4x2(11, 4),  // -5 (0xFFFFFFFB) truncated to 11, 20 (0x00000014) truncated to 4
      UInt4x2(0, 15),  // 0 and 15 already within range
      UInt4x2(3, 6)    // 3.4 rounds to 3, 5.7 rounds to 6
  };

  // WHEN, THEN
  TestCastOp(
      gsl::span<const MLFloat16>(mlfloat16_array, 6),
      gsl::span<const UInt4x2>(expected_uint4x2),
      shape);
}

TEST(CastOpTest, BFloat16ToInt4x2) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const BFloat16 bfloat16_array[8] = {
      BFloat16(static_cast<float>(-10.7f)),
      BFloat16(static_cast<float>(15.3f)),
      BFloat16(static_cast<float>(0.4f)),
      BFloat16(static_cast<float>(-1.6f)),
      BFloat16(static_cast<float>(3.8f)),
      BFloat16(static_cast<float>(-5.2f)),
      BFloat16(static_cast<float>(240.1f)),
      BFloat16(static_cast<float>(31.9f))};

  const std::vector<Int4x2> expected_int4x2 = {
      Int4x2(5, -1),  // -10.7 rounded to -11 (0xF5), truncated to 5; 15.3 rounded to 15 (0x0F), sign-extended to -1
      Int4x2(0, -2),  // 0.4 rounded to 0; -1.6 rounded to -2 (0xFE), truncated to 14 (0x0E), sign-extended to -2
      Int4x2(4, -5),  // 3.8 rounded to 4; -5.2 rounded to -5 (0xFB), truncated to 11 (0x0B), sign-extended to -5
      Int4x2(0, 0)    // 240.1 rounded to 240 (0xF0), truncated to 0; 31.9 rounded to 32 (0x20), truncated to 0
  };

  // WHEN, THEN
  TestCastOp(
      gsl::span<const BFloat16>(bfloat16_array, 8),
      gsl::span<const Int4x2>(expected_int4x2),
      shape);
}

TEST(CastOpTest, BFloat16ToUInt4x2) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const BFloat16 bfloat16_array[8] = {
      BFloat16(static_cast<float>(20.7f)),
      BFloat16(static_cast<float>(255.3f)),
      BFloat16(static_cast<float>(0.4f)),
      BFloat16(static_cast<float>(1.6f)),
      BFloat16(static_cast<float>(7.8f)),
      BFloat16(static_cast<float>(240.2f)),
      BFloat16(static_cast<float>(15.1f)),
      BFloat16(static_cast<float>(31.9f))};

  const std::vector<UInt4x2> expected_uint4x2 = {
      UInt4x2(5, 15),  // 20.7 rounded to 21, truncated to 5; 255.3 rounded to 255, truncated to 15
      UInt4x2(0, 2),   // 0.4 rounded to 0; 1.6 rounded to 2
      UInt4x2(8, 0),   // 7.8 rounded to 8; 240.2 rounded to 240, truncated to 0
      UInt4x2(15, 0)   // 15.1 rounded to 15; 31.9 rounded to 32, truncated to 0
  };

  // WHEN, THEN
  TestCastOp(
      gsl::span<const BFloat16>(bfloat16_array, 8),
      gsl::span<const UInt4x2>(expected_uint4x2),
      shape);
}

TEST(CastOpTest, BFloat16ToUInt4x2BoundaryValues) {
  // GIVEN
  const std::vector<int64_t> shape{3, 2};
  const BFloat16 bfloat16_array[6] = {
      BFloat16(static_cast<float>(-5)),    // Negative, truncated to lower 4 bits
      BFloat16(static_cast<float>(20)),    // Above max, truncated to lower 4 bits
      BFloat16(static_cast<float>(0)),     // At min, should remain 0
      BFloat16(static_cast<float>(15)),    // At max, should remain 15
      BFloat16(static_cast<float>(3.4f)),  // Should round to 3
      BFloat16(static_cast<float>(5.7f))   // Should round to 6
  };

  // Values get truncated to lower 4 bits (no clamping for consistency)
  const std::vector<UInt4x2> expected_uint4x2 = {
      UInt4x2(11, 4),  // -5 (0xFFFFFFFB) truncated to 11, 20 (0x00000014) truncated to 4
      UInt4x2(0, 15),  // 0 and 15 already within range
      UInt4x2(3, 6)    // 3.4 rounds to 3, 5.7 rounds to 6
  };

  // WHEN, THEN
  TestCastOp(
      gsl::span<const BFloat16>(bfloat16_array, 6),
      gsl::span<const UInt4x2>(expected_uint4x2),
      shape);
}

TEST(CastOpTest, BoolToInt4x2) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const bool bool_input[] = {false, true, true, false, false, true, true, true};
  const gsl::span<const bool> bool_input_span(bool_input);

  const std::vector<Int4x2> expected_int4x2_output = {
      Int4x2(0, 1),
      Int4x2(1, 0),
      Int4x2(0, 1),
      Int4x2(1, 1)};

  // WHEN, THEN
  TestCastOp(bool_input_span, gsl::make_span(expected_int4x2_output), shape);
}

TEST(CastOpTest, BoolToUInt4x2) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const bool bool_input[] = {false, true, true, false, false, true, true, true};
  const gsl::span<const bool> bool_input_span(bool_input);

  const std::vector<UInt4x2> expected_uint4x2_output = {
      UInt4x2(0, 1),
      UInt4x2(1, 0),
      UInt4x2(0, 1),
      UInt4x2(1, 1)};

  // WHEN, THEN
  TestCastOp(bool_input_span, gsl::make_span(expected_uint4x2_output), shape);
}

TEST(CastOpTest, StringToInt4x2) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<std::string> string_input = {
      "-8", "7",  // boundary values
      "0", "-1",  // zero and negative
      "3", "-5",  // mixed values
      "6", "2"    // positive values
  };

  const std::vector<Int4x2> expected_output{
      Int4x2(-8, 7),
      Int4x2(0, -1),
      Int4x2(3, -5),
      Int4x2(6, 2)};

  // WHEN, THEN
  TestCastOp(gsl::span<const std::string>(string_input), gsl::span<const Int4x2>(expected_output), shape);
}

TEST(CastOpTest, StringToUInt4x2) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<std::string> string_input = {
      "0", "15",  // boundary values
      "8", "7",   // mid-range values
      "3", "12",  // mixed values
      "10", "5"   // other values
  };

  const std::vector<UInt4x2> expected_output{
      UInt4x2(0, 15),
      UInt4x2(8, 7),
      UInt4x2(3, 12),
      UInt4x2(10, 5)};

  // WHEN, THEN
  TestCastOp(gsl::span<const std::string>(string_input), gsl::span<const UInt4x2>(expected_output), shape);
}

TEST(CastOpTest, StringToUInt4x2BoundaryValues) {
  // GIVEN
  // Test string values that need truncation to UInt4x2 range
  const std::vector<int64_t> shape{3, 2};
  const std::vector<std::string> string_input = {
      "-5", "20",   // out of range values that get truncated
      "16", "100",  // out of range values that get truncated
      "0", "15"     // boundary values that are in range
  };

  // Each pair of strings becomes one UInt4x2
  // Values get truncated to lower 4 bits (no sign extension for unsigned)
  const std::vector<UInt4x2> expected_output{
      UInt4x2(11, 4),  // -5 (0xFFFFFFFB) truncated to 11, 20 (0x00000014) truncated to 4
      UInt4x2(0, 4),   // 16 (0x00000010) truncated to 0, 100 (0x00000064) truncated to 4
      UInt4x2(0, 15)   // 0 and 15 already in range
  };

  // WHEN, THEN
  TestCastOp(gsl::span<const std::string>(string_input), gsl::span<const UInt4x2>(expected_output), shape);
}

TEST(CastOpTest, FloatStringToInt4x2) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<std::string> string_input = {
      "-10.7", "255.3",
      "0.4", "2",
      "6.8", "240.2",
      "15.0", "-8"};

  const std::vector<Int4x2> expected_int4x2_output = {
      Int4x2(5, -1),  // -11 -> 5, 255 -> -1
      Int4x2(0, 2),
      Int4x2(7, 0),
      Int4x2(-1, -8)};

  // WHEN, THEN
  TestCastOp(gsl::span<const std::string>(string_input), gsl::span<const Int4x2>(expected_int4x2_output), shape);
}

#if !defined(DISABLE_FLOAT8_TYPES)

template <typename F8>
void CastOpTestFloat8(Saturate saturate) {
  ASSERT_NE(saturate, Saturate::None);
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<float> float_input = {NAN, -1.f, 0.0391877927f, 0.296140194f, -0.120196559f, 5.0f,
                                          -std::numeric_limits<float>::infinity(),
                                          std::numeric_limits<float>::infinity()};

  // float output precision is 8, so the expected output differs slightly from the input due to that
  std::vector<F8> output;
  output.reserve(float_input.size());
  for (size_t i = 0; i < float_input.size(); ++i) {
    output.emplace_back(F8(float_input[i], saturate == Saturate::True));
  }
  TestCastOp<float, F8>(gsl::make_span(float_input), gsl::make_span(output), shape, OpTester::ExpectResult::kExpectSuccess, "", 19, saturate);

  const std::vector<MLFloat16> float16_input =
      CastedValues<float, MLFloat16>(gsl::make_span(float_input));

  TestCastOp<MLFloat16, F8>(gsl::make_span(float16_input), gsl::make_span(output), shape, OpTester::ExpectResult::kExpectSuccess, "", 19, saturate);
}

TEST(CastOpTest, ToFloat8E4M3FN) {
  constexpr int min_cuda_architecture = 11080;
  bool enable_cuda = (nullptr != DefaultCudaExecutionProvider().get()) && HasCudaEnvironment(min_cuda_architecture);
  bool enable_cpu = (nullptr != DefaultCpuExecutionProvider().get());

  if (enable_cpu || enable_cuda) {
    CastOpTestFloat8<Float8E4M3FN>(Saturate::True);
    CastOpTestFloat8<Float8E4M3FN>(Saturate::False);
  }
}

TEST(CastOpTest, ToFloat8E4M3FNUZ) {
  bool enable_cpu = (nullptr != DefaultCpuExecutionProvider().get());
  if (enable_cpu) {
    CastOpTestFloat8<Float8E4M3FNUZ>(Saturate::True);
    CastOpTestFloat8<Float8E4M3FNUZ>(Saturate::False);
  }
}

TEST(CastOpTest, ToFloat8E5M2) {
  constexpr int min_cuda_architecture = 11080;
  bool enable_cuda = (nullptr != DefaultCudaExecutionProvider().get()) && HasCudaEnvironment(min_cuda_architecture);
  bool enable_cpu = (nullptr != DefaultCpuExecutionProvider().get());

  if (enable_cpu || enable_cuda) {
    CastOpTestFloat8<Float8E5M2>(Saturate::True);
    CastOpTestFloat8<Float8E5M2>(Saturate::False);
  }
}

TEST(CastOpTest, ToFloat8E5M2FNUZ) {
  bool enable_cpu = (nullptr != DefaultCpuExecutionProvider().get());
  if (enable_cpu) {
    CastOpTestFloat8<Float8E5M2FNUZ>(Saturate::True);
    CastOpTestFloat8<Float8E5M2FNUZ>(Saturate::False);
  }
}

TEST(CastOpTest, Int4x2ToFloat8E4M3FN) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<Int4x2> int4x2_input = {
      Int4x2(-8, 7),
      Int4x2(0, -1),
      Int4x2(3, -5),
      Int4x2(6, 2)};

  std::vector<Float8E4M3FN> expected_float8_output;
  expected_float8_output.reserve(8);
  const std::vector<float> float_values = {-8.0f, 7.0f, 0.0f, -1.0f, 3.0f, -5.0f, 6.0f, 2.0f};
  for (float val : float_values) {
    expected_float8_output.emplace_back(Float8E4M3FN(val, true));
  }

  // WHEN, THEN
  // Test with Saturate::None, which means the 'saturate_' bool inside the 'Cast' class defaults to 1
  TestCastOp<Int4x2, Float8E4M3FN>(gsl::make_span(int4x2_input), gsl::make_span(expected_float8_output), shape);
  // Test with Saturate::False, which means the 'saturate_' bool inside the 'Cast' class will be 0
  TestCastOp<Int4x2, Float8E4M3FN>(gsl::make_span(int4x2_input), gsl::make_span(expected_float8_output), shape,
                                   OpTester::ExpectResult::kExpectSuccess, "", 21, Saturate::False);
}

TEST(CastOpTest, UInt4x2ToFloat8E4M3FN) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<UInt4x2> uint4x2_input = {
      UInt4x2(0, 15),
      UInt4x2(1, 14),
      UInt4x2(7, 8),
      UInt4x2(3, 12)};

  std::vector<Float8E4M3FN> expected_uint_float8_output;
  expected_uint_float8_output.reserve(8);
  const std::vector<float> uint_float_values = {0.0f, 15.0f, 1.0f, 14.0f, 7.0f, 8.0f, 3.0f, 12.0f};
  for (float val : uint_float_values) {
    expected_uint_float8_output.emplace_back(Float8E4M3FN(val, true));
  }

  // WHEN, THEN
  // Test with Saturate::None, which means the 'saturate_' bool inside the 'Cast' class defaults to 1
  TestCastOp<UInt4x2, Float8E4M3FN>(gsl::make_span(uint4x2_input), gsl::make_span(expected_uint_float8_output), shape);
  // Test with Saturate::False, which means the 'saturate_' bool inside the 'Cast' class will be 0
  TestCastOp<UInt4x2, Float8E4M3FN>(gsl::make_span(uint4x2_input), gsl::make_span(expected_uint_float8_output), shape,
                                    OpTester::ExpectResult::kExpectSuccess, "", 21, Saturate::False);
}

TEST(CastOpTest, Int4x2ToFloat8E5M2) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<Int4x2> int4x2_input = {
      Int4x2(-8, 7),
      Int4x2(0, -1),
      Int4x2(3, -5),
      Int4x2(6, 2)};

  std::vector<Float8E5M2> expected_float8e5m2_output;
  expected_float8e5m2_output.reserve(8);
  const std::vector<float> float_values = {-8.0f, 7.0f, 0.0f, -1.0f, 3.0f, -5.0f, 6.0f, 2.0f};
  for (float val : float_values) {
    expected_float8e5m2_output.emplace_back(Float8E5M2(val, true));
  }

  // WHEN, THEN
  // Test with Saturate::None, which means the 'saturate_' bool inside the 'Cast' class defaults to 1
  TestCastOp<Int4x2, Float8E5M2>(gsl::make_span(int4x2_input), gsl::make_span(expected_float8e5m2_output), shape);
  // Test with Saturate::False, which means the 'saturate_' bool inside the 'Cast' class will be 0
  TestCastOp<Int4x2, Float8E5M2>(gsl::make_span(int4x2_input), gsl::make_span(expected_float8e5m2_output), shape,
                                 OpTester::ExpectResult::kExpectSuccess, "", 21, Saturate::False);
}

TEST(CastOpTest, UInt4x2ToFloat8E5M2) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  const std::vector<UInt4x2> uint4x2_input = {
      UInt4x2(0, 15),
      UInt4x2(1, 14),
      UInt4x2(7, 8),
      UInt4x2(3, 12)};

  std::vector<Float8E5M2> expected_uint_float8e5m2_output;
  expected_uint_float8e5m2_output.reserve(8);
  const std::vector<float> uint_float_values = {0.0f, 15.0f, 1.0f, 14.0f, 7.0f, 8.0f, 3.0f, 12.0f};
  for (float val : uint_float_values) {
    expected_uint_float8e5m2_output.emplace_back(Float8E5M2(val, true));
  }

  // WHEN, THEN
  // Test with Saturate::None, which means the 'saturate_' bool inside the 'Cast' class defaults to 1
  TestCastOp<UInt4x2, Float8E5M2>(gsl::make_span(uint4x2_input), gsl::make_span(expected_uint_float8e5m2_output), shape);
  // Test with Saturate::False, which means the 'saturate_' bool inside the 'Cast' class will be 0
  TestCastOp<UInt4x2, Float8E5M2>(gsl::make_span(uint4x2_input), gsl::make_span(expected_uint_float8e5m2_output), shape,
                                  OpTester::ExpectResult::kExpectSuccess, "", 21, Saturate::False);
}

TEST(CastOpTest, Float8E4M3FNToInt4x2) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  std::vector<Float8E4M3FN> float8_input;
  const std::vector<float> input_values = {-8.0f, 7.0f, 0.0f, -1.0f, 3.0f, -5.0f, 6.0f, 2.0f};
  for (float val : input_values) {
    float8_input.emplace_back(Float8E4M3FN(val, true));
  }

  const std::vector<Int4x2> expected_int4x2_output = {
      Int4x2(-8, 7),
      Int4x2(0, -1),
      Int4x2(3, -5),
      Int4x2(6, 2)};

  // WHEN, THEN
  // The 'saturate_' bool inside the 'Cast' class can only be false if the conversion is to a float 8 type,
  // so it's sufficient to test with the default saturate = 1 here, since we are not converting to float 8.
  TestCastOp<Float8E4M3FN, Int4x2>(gsl::make_span(float8_input), gsl::make_span(expected_int4x2_output), shape);
}

TEST(CastOpTest, Float8E4M3FNToInt4x2_OddShape) {
  // GIVEN
  const std::vector<int64_t> shape{1, 2, 3};
  std::vector<Float8E4M3FN> float8_input;
  const std::vector<float> input_values = {-8.0f, 7.0f, 0.0f, -1.0f, 3.0f, -5.0f};
  for (float val : input_values) {
    float8_input.emplace_back(Float8E4M3FN(val, true));
  }

  const std::vector<Int4x2> expected_int4x2_output = {
      Int4x2(-8, 7),
      Int4x2(0, -1),
      Int4x2(3, -5)};

  // WHEN, THEN
  // The 'saturate_' bool inside the 'Cast' class can only be false if the conversion is to a float 8 type,
  // so it's sufficient to test with the default saturate = 1 here, since we are not converting to float 8.
  TestCastOp<Float8E4M3FN, Int4x2>(gsl::make_span(float8_input), gsl::make_span(expected_int4x2_output), shape);
}

TEST(CastOpTest, Float8E4M3FNToUInt4x2) {
  // GIVEN
  const std::vector<int64_t> shape{2, 2, 2};
  std::vector<Float8E4M3FN> uint_float8_input;
  const std::vector<float> uint_input_values = {0.0f, 15.0f, 1.0f, 14.0f, 7.0f, 8.0f, 3.0f, 12.0f};
  for (float val : uint_input_values) {
    uint_float8_input.emplace_back(Float8E4M3FN(val, true));
  }

  const std::vector<UInt4x2> expected_uint4x2_output = {
      UInt4x2(0, 15),
      UInt4x2(1, 14),
      UInt4x2(7, 8),
      UInt4x2(3, 12)};

  // WHEN, THEN
  // The 'saturate_' bool inside the 'Cast' class can only be false if the conversion is to a float 8 type,
  // so it's sufficient to test with the default saturate = 1 here, since we are not converting to float 8.
  TestCastOp<Float8E4M3FN, UInt4x2>(gsl::make_span(uint_float8_input), gsl::make_span(expected_uint4x2_output), shape);
}

#endif

#if !defined(DISABLE_FLOAT4_TYPES) && defined(USE_CUDA)

template <typename F4>
void CastOpTestFloatFloat4(std::vector<int64_t> shape,
                           std::vector<float> float_data,
                           bool is_fp4_input = false) {
  int num_pairs = static_cast<int>(float_data.size()) / 2;
  int num_fp4_elements = static_cast<int>((float_data.size() + 1) / 2);
  bool is_odd_count = (float_data.size() % 2 != 0);

  std::vector<F4> fp4_data;
  fp4_data.reserve(num_fp4_elements);

  for (size_t i = 0; i < num_pairs; ++i) {
    fp4_data.emplace_back(F4(float_data[i * 2], float_data[i * 2 + 1]));
  }

  if (is_odd_count) {
    fp4_data.emplace_back(F4(float_data[num_pairs * 2], 0));  // Padding zero
  }

  if (!is_fp4_input) {
    TestCastOp<float, F4>(gsl::make_span(float_data), gsl::make_span(fp4_data), shape,
                          OpTester::ExpectResult::kExpectSuccess, "", 23, Saturate::None, true);

  } else {
    std::vector<float> casted_back_float;
    for (size_t i = 0; i < num_pairs; ++i) {
      auto pair = fp4_data[i].ToFloat2();
      casted_back_float.push_back(pair.first);
      casted_back_float.push_back(pair.second);
    }

    if (is_odd_count) {
      casted_back_float.push_back(fp4_data[num_pairs].ToFloat2().first);
    }

    TestCastOp<F4, float>(gsl::make_span(fp4_data), gsl::make_span(casted_back_float), shape,
                          OpTester::ExpectResult::kExpectSuccess, "", 23, Saturate::None, true);
  }
}

TEST(CastOpTest, FloatToFloat4E2M1x2) {
  // Even count tests
  CastOpTestFloatFloat4<Float4E2M1x2>({2, 2, 2},
                                      {std::numeric_limits<float>::infinity(),
                                       -std::numeric_limits<float>::infinity(),
                                       7.f, -7.f,
                                       0.5f, -0.5f,
                                       std::numeric_limits<float>::quiet_NaN(),
                                       -std::numeric_limits<float>::quiet_NaN()});

  // Odd count tests
  CastOpTestFloatFloat4<Float4E2M1x2>({1, 3, 1},
                                      {0.256f,
                                       0.987f,
                                       43.8f});
}

TEST(CastOpTest, Float4E2M1x2ToFloat) {
  // Even count tests
  CastOpTestFloatFloat4<Float4E2M1x2>({2, 2, 2},
                                      {0.5f, 7.34f,
                                       1.f, 1.5f,
                                       2.f, 3.f,
                                       4.f, 6.f},
                                      true);

  // Odd count tests
  CastOpTestFloatFloat4<Float4E2M1x2>({1, 3, 1},
                                      {0.256f,
                                       0.987f,
                                       43.8f},
                                      true);
}

#endif

}  // namespace test
}  // namespace onnxruntime
