// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include <cmath>
#include <gsl/gsl>

#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

#ifdef USE_WEBGPU
#include "core/providers/webgpu/webgpu_provider_options.h"
#endif

namespace onnxruntime {
namespace test {

namespace {
constexpr char kOpName[] = "Where";
constexpr int kOpVersion = 9;

template <typename TDest, typename TSrc>
std::vector<TDest> CastVector(const std::vector<TSrc>& source) {
  std::vector<TDest> target{};
  target.reserve(source.size());
  std::transform(source.begin(), source.end(), std::back_inserter(target),
                 [](TSrc n) { return static_cast<TDest>(n); });
  return target;
}

template <typename T>
void WherePreservesNegativeZeroTest() {
  const auto verify_negative_zero = [](const std::vector<OrtValue>& fetches,
                                       const std::string& /*provider_type*/) {
    ASSERT_EQ(fetches.size(), 1u);
    ASSERT_TRUE(fetches[0].IsTensor());

    const auto& output_tensor = fetches[0].Get<Tensor>();
    const auto* output = output_tensor.Data<T>();
    for (int64_t i = 0; i < output_tensor.Shape().Size(); ++i) {
      EXPECT_EQ(output[i], T{0});
      EXPECT_TRUE(std::signbit(output[i]));
    }
  };

  {
    OpTester test{kOpName, kOpVersion};
    test.AddInput<bool>("condition", {1}, {true});
    test.AddInput<T>("X", {1}, {-T{0}});
    test.AddInput<T>("Y", {1}, {T{0}});
    test.AddOutput<T>("output", {1}, {-T{0}});
    test.SetCustomOutputVerifier(verify_negative_zero);

    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCpuExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }

  {
    OpTester test{kOpName, kOpVersion};
    test.AddInput<bool>("condition", {1}, {false});
    test.AddInput<T>("X", {4}, {T{1}, T{2}, T{3}, T{4}});
    test.AddInput<T>("Y", {1}, {-T{0}});
    test.AddOutput<T>("output", {4}, {-T{0}, -T{0}, -T{0}, -T{0}});
    test.SetCustomOutputVerifier(verify_negative_zero);

    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCpuExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
}

template <typename TNumeric>
void WhereBasicNumericTest() {
  OpTester test{kOpName, kOpVersion};

  const std::vector<int64_t> dims{2, 2};

  test.AddInput<bool>("condition", dims,
                      {false, true, true, false});
  test.AddInput<TNumeric>("X", dims,
                          CastVector<TNumeric, int>({1, 2, 3, 4}));
  test.AddInput<TNumeric>("Y", dims,
                          CastVector<TNumeric, int>({5, 6, 7, 8}));

  test.AddOutput<TNumeric>("output", dims,
                           CastVector<TNumeric, int>({5, 2, 3, 8}));

  test.Run();
}

template <typename T>
void WhereBroadcastTest(const T& x_value, const T& y_value) {
  auto condition_values = {true, false, true};  // std::initializer_list<bool> for OpTester::AddInput<bool>()
  const std::vector<T> X_values(3, x_value);
  const std::vector<T> Y_values(3, y_value);

  {
    OpTester test{kOpName, kOpVersion};

    test.AddInput<bool>("condition", {1, 1, 3}, condition_values);
    test.AddInput<T>("X", {1, 3, 1}, X_values);
    test.AddInput<T>("Y", {3, 1, 1}, Y_values);

    std::vector<T> result{};
    result.reserve(3 * 3 * 3);
    for (int i = 0; i < 3 * 3; ++i) {
      result.insert(result.end(), {x_value, y_value, x_value});
    }
    test.AddOutput<T>("output", {3, 3, 3}, result);

#if defined(OPENVINO_CONFIG_GPU)
    test.Run(OpTester::ExpectResult::kExpectSuccess, "",
             {kOpenVINOExecutionProvider});  // OpenVINO: Disabled due to failure for GPU
#else
    test.Run();
#endif
  }

  {
    OpTester test{kOpName, kOpVersion};

    test.AddInput<bool>("condition", {3, 1, 1}, condition_values);
    test.AddInput<T>("X", {1, 1, 3}, X_values);
    test.AddInput<T>("Y", {1, 3, 1}, Y_values);

    std::vector<T> result{};
    result.reserve(3 * 3 * 3);
    for (int i = 0; i < 3; ++i) {
      result.insert(
          result.end(), 3 * 3,
          gsl::make_span(condition_values.begin(), condition_values.size())[i] ? x_value : y_value);
    }
    test.AddOutput<T>("output", {3, 3, 3}, result);

#if defined(OPENVINO_CONFIG_GPU)
    test.Run(OpTester::ExpectResult::kExpectSuccess, "",
             {kOpenVINOExecutionProvider});  // OpenVINO: Disabled due to failure for GPU
#else
    test.Run();
#endif
  }
}
}  // namespace

TEST(WhereOpTest, BasicNumeric) {
  WhereBasicNumericTest<float>();
  WhereBasicNumericTest<double>();
}

TEST(WhereOpTest, PreservesNegativeZero) {
  WherePreservesNegativeZeroTest<float>();
  WherePreservesNegativeZeroTest<double>();
}

TEST(WhereOpTest, BasicString) {
  OpTester test{kOpName, kOpVersion};

  test.AddInput<bool>("condition", {2}, {false, true});
  const std::vector<std::string> X{"small0", "small1"};
  test.AddInput<std::string>("X", {2}, X);
  const std::vector<std::string> Y{std::string(1024, 'a'), std::string(1024, 'b')};
  test.AddInput<std::string>("Y", {2}, Y);

  test.AddOutput<std::string>("output", {2}, {Y[0], X[1]});

  test.Run();
}

TEST(WhereOpTest, Broadcast) {
  WhereBroadcastTest<float>(1.0f, 0.0f);
  WhereBroadcastTest<double>(1.0f, 0.0f);
  WhereBroadcastTest<std::string>("true", "false");
}

TEST(WhereOpTest, BroadcastDimWithZero) {
  // test where broadcast is possible, and dim of 0 should be selected
  OpTester test{kOpName, kOpVersion};

  test.AddInput<bool>("condition", {3}, {true, false, true});
  test.AddInput<int64_t>("X", {1, 3}, {1, 2, 3});
  test.AddInput<int64_t>("Y", {0, 1}, {});

  test.AddOutput<int64_t>("output", {0, 3}, {});

  test.Run();
}

TEST(WhereOpTest, BroadcastWithScalar) {
  OpTester test{kOpName, kOpVersion};

  test.AddInput<bool>("condition", {3}, {true, false, true});
  test.AddInput<int64_t>("X", {1, 3}, {1, 2, 3});
  test.AddInput<int64_t>("Y", {}, {1});

  test.AddOutput<int64_t>("output", {1, 3}, {1, 1, 3});

  test.Run();
}

#ifdef USE_WEBGPU
// Non-broadcast: all inputs have the same shape. Exercises the is_int64_ non-broadcast path.
TEST(WhereOpTest, EnableWebGpuInt64) {
  OpTester test{kOpName, kOpVersion};
  test.AddInput<bool>("condition", {4}, {true, false, true, false});
  test.AddInput<int64_t>("X", {4}, {10, 20, 30, 40});
  test.AddInput<int64_t>("Y", {4}, {1, 2, 3, 4});
  test.AddOutput<int64_t>("output", {4}, {10, 2, 30, 4});
  ConfigOptions config_options{};
  ASSERT_STATUS_OK(config_options.AddConfigEntry(webgpu::options::kEnableInt64, "1"));
  auto provider = WebGpuExecutionProviderWithOptions(config_options);
  test.ConfigEp(std::move(provider))
      .RunWithConfig();
}

// Broadcast: condition [1,4] broadcasts over X/Y [2,4]. Exercises the is_int64_ broadcast path
// where BroadcastedIndicesToOffset computes a different source offset per output element.
TEST(WhereOpTest, EnableBroadcastWebGpuInt64) {
  // condition [1,4] broadcasts against X [2,4] and Y [2,4] -> output [2,4]
  OpTester test{kOpName, kOpVersion};
  test.AddInput<bool>("condition", {1, 4}, {true, false, true, false});
  test.AddInput<int64_t>("X", {2, 4}, {10, 20, 30, 40, 50, 60, 70, 80});
  test.AddInput<int64_t>("Y", {2, 4}, {1, 2, 3, 4, 5, 6, 7, 8});
  test.AddOutput<int64_t>("output", {2, 4}, {10, 2, 30, 4, 50, 6, 70, 8});
  ConfigOptions config_options{};
  ASSERT_STATUS_OK(config_options.AddConfigEntry(webgpu::options::kEnableInt64, "1"));
  auto provider = WebGpuExecutionProviderWithOptions(config_options);
  test.ConfigEp(std::move(provider))
      .RunWithConfig();
}
#endif

}  // namespace test
}  // namespace onnxruntime
