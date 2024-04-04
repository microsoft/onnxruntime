// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <type_traits>

#include "boost/mp11.hpp"

#include "core/common/gsl.h"

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
                int opset = 13,
                Saturate saturate = Saturate::None) {
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
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: Expected equality of these values: true and true";
  }

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

#endif

}  // namespace test
}  // namespace onnxruntime
