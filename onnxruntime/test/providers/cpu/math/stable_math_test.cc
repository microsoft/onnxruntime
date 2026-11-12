// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <limits>
#include <optional>
#include <type_traits>
#include <unordered_set>

#include "gtest/gtest.h"
#include "core/framework/customregistry.h"
#include "core/providers/get_execution_providers.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {
namespace {

// Session-local contracts exercise the real provider kernels before the pinned
// ONNX dependency includes these schemas. No production schema or opset is enabled.
std::shared_ptr<CustomRegistry> StableMathTestSchemas() {
  auto registry = std::make_shared<CustomRegistry>();
  std::vector<ONNX_NAMESPACE::OpSchema> schemas;
  for (const char* op : {"Expm1", "Log1p", "LogCumSumExp"}) {
    if (ONNX_NAMESPACE::OpSchemaRegistry::Schema(op, 29, kOnnxDomain)) continue;
    ONNX_NAMESPACE::OpSchema schema;
    schema.SetName(op)
        .SetDomain(kOnnxDomain)
        .SinceVersion(29)
        .Input(0, "X", "Input", "T")
        .Output(0, "Y", "Output", "T")
        .TypeConstraint("T", ONNX_NAMESPACE::OpSchema::all_float_types_ir4(), "Floating-point tensors")
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput);
    if (std::string_view(op) == "LogCumSumExp") {
      schema.Attr("axis", "Scan axis", ONNX_NAMESPACE::AttributeProto::INT, int64_t{0});
    }
    schemas.push_back(std::move(schema));
  }
  if (!schemas.empty()) {
    ORT_THROW_IF_ERROR(registry->RegisterOpSet(schemas, kOnnxDomain, 29, 29));
  }
  return registry;
}

template <typename T, bool Cuda>
struct MathCase {
  using Type = T;
  static constexpr bool use_cuda = Cuda;
};

template <typename Case>
class StableMathTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if constexpr (Case::use_cuda) {
      if (!DefaultCudaExecutionProvider()) GTEST_SKIP() << "CUDA execution provider is unavailable";
    }
  }
};

using MathCases = ::testing::Types<
    MathCase<float, false>, MathCase<double, false>, MathCase<MLFloat16, false>, MathCase<BFloat16, false>,
    MathCase<float, true>, MathCase<double, true>, MathCase<MLFloat16, true>, MathCase<BFloat16, true>>;
TYPED_TEST_SUITE(StableMathTest, MathCases);

template <typename Case>
void RunMath(const char* op, const std::vector<int64_t>& shape, const std::vector<typename Case::Type>& input,
             const std::vector<typename Case::Type>& expected, std::optional<int64_t> axis = std::nullopt,
             bool expect_failure = false) {
  using T = typename Case::Type;
  const std::string provider = Case::use_cuda ? kCudaExecutionProvider : kCpuExecutionProvider;
  std::unordered_set<std::string> excluded;
  for (const auto& ep : GetAllExecutionProviderNames()) {
    if (ep != provider) excluded.insert(ep);
  }
  OpTester test(op, 29, kOnnxDomain);
  test.SetAllowUnreleasedOnnxOpset();
  test.AddCustomOpRegistry(StableMathTestSchemas());
  if (axis) test.AddAttribute<int64_t>("axis", *axis);
  test.AddInput<T>("X", shape, input);
  test.AddOutput<T>("Y", shape, expected);
  bool verified = false;
  test.SetCustomOutputVerifier([&](const std::vector<OrtValue>& fetches, const std::string& actual_provider) {
    verified = true;
    ASSERT_EQ(actual_provider, provider);
    ASSERT_EQ(fetches.size(), 1u);
    const auto& tensor = fetches[0].Get<Tensor>();
    ASSERT_EQ(tensor.Shape(), TensorShape(shape));
    ASSERT_EQ(tensor.GetElementType(), utils::ToTensorProtoElementType<T>());
    const T* actual = tensor.Data<T>();
    constexpr double tolerance = std::is_same_v<T, double>      ? 2e-12
                                 : std::is_same_v<T, float>     ? 3e-5
                                 : std::is_same_v<T, MLFloat16> ? 2e-3
                                                                : 2e-2;
    for (size_t i = 0; i < expected.size(); ++i) {
      SCOPED_TRACE(i);
      const double want = static_cast<double>(expected[i]);
      const double got = static_cast<double>(actual[i]);
      if (std::isnan(want)) {
        EXPECT_TRUE(std::isnan(got));
      } else if (std::isinf(want)) {
        EXPECT_EQ(got, want);
      } else {
        EXPECT_NEAR(got, want, std::abs(want) * tolerance);
        if (want == 0 && std::string_view(op) != "LogCumSumExp") {
          EXPECT_EQ(std::signbit(got), std::signbit(want));
        }
      }
    }
  });
  SessionOptions options;
  options.graph_optimization_level = TransformerLevel::Default;
  options.intra_op_param.thread_pool_size = 2;
  if constexpr (Case::use_cuda) {
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  }
  test.Run(options, expect_failure ? OpTester::ExpectResult::kExpectFailure : OpTester::ExpectResult::kExpectSuccess,
           "", excluded);
  if (!expect_failure) EXPECT_TRUE(verified) << "The requested native provider kernel did not execute";
}

template <typename T>
T FromDouble(double value) {
  using Intermediate = std::conditional_t<std::is_same_v<T, double>, double, float>;
  return static_cast<T>(static_cast<Intermediate>(value));
}

template <typename T>
std::vector<T> Convert(const std::vector<double>& values) {
  std::vector<T> result;
  result.reserve(values.size());
  for (double v : values) result.push_back(FromDouble<T>(v));
  return result;
}

template <typename T>
std::vector<T> ScanReference(const std::vector<T>& input, int64_t outer, int64_t width, int64_t inner) {
  std::vector<T> result(input.size());
  for (int64_t o = 0; o < outer; ++o) {
    for (int64_t i = 0; i < inner; ++i) {
      long double sum = 0;
      long double correction = 0;
      for (int64_t j = 0; j < width; ++j) {
        const auto offset = static_cast<size_t>((o * width + j) * inner + i);
        const long double term = std::exp(static_cast<long double>(static_cast<double>(input[offset])));
        const long double next = sum + term;
        // Retain tiny corrections even when MSVC's long double is only double precision.
        if (std::isfinite(next)) {
          correction += std::abs(sum) >= std::abs(term) ? (sum - next) + term : (term - next) + sum;
        }
        sum = next;
        long double expected = std::log(sum);
        if (sum > 0 && std::isfinite(sum)) expected += std::log1p(correction / sum);
        result[offset] = FromDouble<T>(static_cast<double>(expected));
      }
    }
  }
  return result;
}

TYPED_TEST(StableMathTest, UnaryTinyValuesSpecialValuesAndVectorTails) {
  using T = typename TypeParam::Type;
  const double inf = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  auto input = Convert<T>({0, -0.0, 1e-8, -1e-8, 1e-4, -1e-4, -1, -2, inf, -inf, nan, 0.5, 90, 1000});
  // Exercise vector packets as well as a non-aligned tail.
  input.resize(513, static_cast<T>(0.125f));
  for (const char* op : {"Expm1", "Log1p"}) {
    std::vector<T> expected;
    for (T x : input) {
      const double v = static_cast<double>(x);
      expected.push_back(FromDouble<T>(std::string_view(op) == "Expm1" ? std::expm1(v) : std::log1p(v)));
    }
    RunMath<TypeParam>(op, {513}, input, expected);
    RunMath<TypeParam>(op, {}, {static_cast<T>(-0.0f)}, {static_cast<T>(-0.0f)});
    RunMath<TypeParam>(op, {2, 0, 3}, {}, {});
  }
  const double tiny = std::is_same_v<T, double>      ? std::numeric_limits<double>::denorm_min()
                      : std::is_same_v<T, float>     ? std::numeric_limits<float>::denorm_min()
                      : std::is_same_v<T, MLFloat16> ? std::ldexp(1.0, -24)
                                                     : std::ldexp(1.0, -133);
  const auto tiny_input = Convert<T>({tiny, -tiny, tiny, -tiny, tiny, -tiny, tiny, -tiny, tiny});
  RunMath<TypeParam>("Expm1", {9}, tiny_input, tiny_input);
  RunMath<TypeParam>("Log1p", {9}, tiny_input, tiny_input);
}

TYPED_TEST(StableMathTest, ScanAxesAndDynamicRange) {
  using T = typename TypeParam::Type;
  auto input = Convert<T>({-200, -100, 100, -100, 100, 200, 0, -20, -40, 4, -2, 0,
                           0.1, 0.2, 0.3, -50, -49, -48, 30, 20, 10, -1, -2, -3});
  RunMath<TypeParam>("LogCumSumExp", {2, 3, 4}, input, ScanReference(input, 1, 2, 12));
  RunMath<TypeParam>("LogCumSumExp", {2, 3, 4}, input, ScanReference(input, 1, 2, 12), -3);
  for (int64_t axis : {int64_t{1}, int64_t{-2}}) {
    RunMath<TypeParam>("LogCumSumExp", {2, 3, 4}, input, ScanReference(input, 2, 3, 4), axis);
  }
  for (int64_t axis : {int64_t{2}, int64_t{-1}}) {
    RunMath<TypeParam>("LogCumSumExp", {2, 3, 4}, input, ScanReference(input, 6, 4, 1), axis);
  }
}

TYPED_TEST(StableMathTest, ScanNonfinitePrefixesAndTinyCorrection) {
  using T = typename TypeParam::Type;
  const double inf = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const auto input = Convert<T>({-inf, -inf, -inf, -inf, -inf, 0, inf, -inf, 0, inf, inf, 1,
                                 0, nan, inf, -inf, inf, -inf, nan, 0, 0, -18, -40, -100});
  RunMath<TypeParam>("LogCumSumExp", {6, 4}, input, ScanReference(input, 6, 4, 1), 1);
}

TYPED_TEST(StableMathTest, ScanLongLowPrecisionAccumulationAndTiles) {
  using T = typename TypeParam::Type;
  for (int64_t width : {int64_t{255}, int64_t{256}, int64_t{257}, int64_t{4097}}) {
    std::vector<T> input(static_cast<size_t>(width), static_cast<T>(0.0f));
    RunMath<TypeParam>("LogCumSumExp", {width}, input, ScanReference(input, 1, width, 1));
  }
  const double inf = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  std::vector<T> input(3 * 513, FromDouble<T>(-inf));
  input[256] = static_cast<T>(0.0f);
  input[513] = FromDouble<T>(inf);
  input[513 + 255] = FromDouble<T>(nan);
  input[1026 + 256] = FromDouble<T>(inf);
  input[1026 + 512] = FromDouble<T>(nan);
  RunMath<TypeParam>("LogCumSumExp", {3, 513}, input, ScanReference(input, 3, 513, 1), 1);
  for (int64_t inner : {int64_t{2}, int64_t{257}}) {
    input.assign(static_cast<size_t>(257 * inner), static_cast<T>(0.0f));
    RunMath<TypeParam>("LogCumSumExp", {257, inner}, input, ScanReference(input, 1, 257, inner));
  }
}

TYPED_TEST(StableMathTest, ScanSingletonAxis) {
  using T = typename TypeParam::Type;
  const auto input = Convert<T>({-1000, -0.0, 0.0, 1000, std::numeric_limits<double>::infinity(),
                                 -std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()});
  RunMath<TypeParam>("LogCumSumExp", {7, 1}, input, input, 1);
  RunMath<TypeParam>("LogCumSumExp", {1, 7}, input, input);
}

TYPED_TEST(StableMathTest, ScanEmptyAndInvalidAxes) {
  using T = typename TypeParam::Type;
  RunMath<TypeParam>("LogCumSumExp", {2, 0, 3}, {}, {}, 1);
  RunMath<TypeParam>("LogCumSumExp", {0, 3}, {}, {}, -1);
  RunMath<TypeParam>("LogCumSumExp", {2, 0, 3}, {}, {}, 3, true);
  RunMath<TypeParam>("LogCumSumExp", {2, 0, 3}, {}, {}, -4, true);
  RunMath<TypeParam>("LogCumSumExp", {}, {static_cast<T>(0.0f)}, {static_cast<T>(0.0f)}, 0, true);
}

}  // namespace
}  // namespace test
}  // namespace onnxruntime
