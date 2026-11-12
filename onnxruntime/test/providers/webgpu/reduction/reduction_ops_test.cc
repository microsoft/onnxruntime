// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/providers/provider_test_utils.h"
#include "test/unittest_util/conversion.h"
#include "test/util/include/default_providers.h"

#ifdef USE_WEBGPU
#include "core/providers/webgpu/webgpu_provider_options.h"
#endif

namespace onnxruntime {
namespace test {

#ifdef USE_WEBGPU
TEST(ReductionOpTest, ReduceSum_WebGpu_EnableInt64) {
  OpTester test("ReduceSum", 13);
  test.AddInput<int64_t>("data", {3}, {10, 20, 30});
  test.AddInput<int64_t>("axes", {1}, {0}, true);
  test.AddOutput<int64_t>("reduced", {1}, {60});
  ConfigOptions config_options{};
  ASSERT_STATUS_OK(config_options.AddConfigEntry(webgpu::options::kEnableInt64, "1"));
  auto provider = WebGpuExecutionProviderWithOptions(config_options);
  test.ConfigEp(std::move(provider))
      .RunWithConfig();
}

// Size divisible by 4: catches issues if the shader is ever accidentally vectorized for INT64.
TEST(ReductionOpTest, ReduceSum_WebGpu_EnableInt64_SizeDiv4) {
  OpTester test("ReduceSum", 13);
  test.AddInput<int64_t>("data", {4}, {10, 20, 30, 40});
  test.AddInput<int64_t>("axes", {1}, {0}, true);
  test.AddOutput<int64_t>("reduced", {1}, {100});
  ConfigOptions config_options{};
  ASSERT_STATUS_OK(config_options.AddConfigEntry(webgpu::options::kEnableInt64, "1"));
  auto provider = WebGpuExecutionProviderWithOptions(config_options);
  test.ConfigEp(std::move(provider))
      .RunWithConfig();
}

namespace {
constexpr float kInf = std::numeric_limits<float>::infinity();

// Runs an opset-18 reduction on WebGPU only, with CPU fallback disabled. Which WebGPU program runs
// depends on the shape: reductions with reduce_size < 32 and output_size > 1024 use the naive
// per-output program; small outputs use the workgroup shared-memory program.
template <typename T>
void RunReduceOnWebGpu(const char* op_type, const std::vector<int64_t>& input_shape, const std::vector<T>& input,
                       const std::vector<int64_t>& axes, int64_t keepdims,
                       const std::vector<int64_t>& output_shape, const std::vector<T>& expected,
                       float abs_error = 0.0f) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }
  SessionOptions options;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));

  OpTester test(op_type, 18);
  test.AddAttribute<int64_t>("keepdims", keepdims);
  test.AddInput<T>("data", input_shape, input);
  test.AddInput<int64_t>("axes", {static_cast<int64_t>(axes.size())}, axes, true);
  test.AddOutput<T>("reduced", output_shape, expected);
  if (abs_error > 0.0f) {
    test.SetOutputAbsErr("reduced", abs_error);
  }
  test.Config(options).ConfigEp(std::move(webgpu_ep)).RunWithConfig();
}

// Reference LogSumExp with the CPU kernel's semantics: an infinite max is returned as-is.
float ReferenceLogSumExp(const float* values, size_t count) {
  float max_value = -kInf;
  for (size_t i = 0; i < count; ++i) {
    max_value = std::max(max_value, values[i]);
  }
  if (std::isinf(max_value)) {
    return max_value;
  }
  double sum = 0.0;
  for (size_t i = 0; i < count; ++i) {
    sum += std::exp(static_cast<double>(values[i]) - max_value);
  }
  return static_cast<float>(max_value + std::log(sum));
}

// Builds `num_rows` rows of `row_length` values by cycling through `row_patterns`, then reduces
// the last axis and checks against ReferenceLogSumExp. Float16 values are rounded first.
template <typename T>
void TestLogSumExpRows(const std::vector<std::vector<float>>& row_patterns, int64_t num_rows, float abs_error) {
  const size_t row_length = row_patterns[0].size();
  std::vector<float> input;
  std::vector<float> expected;
  for (int64_t r = 0; r < num_rows; ++r) {
    std::vector<float> row = row_patterns[static_cast<size_t>(r) % row_patterns.size()];
    if constexpr (std::is_same_v<T, MLFloat16>) {
      for (float& v : row) {
        v = MLFloat16(v).ToFloat();
      }
    }
    input.insert(input.end(), row.begin(), row.end());
    expected.push_back(ReferenceLogSumExp(row.data(), row.size()));
  }
  const std::vector<int64_t> input_shape{num_rows, static_cast<int64_t>(row_length)};
  if constexpr (std::is_same_v<T, MLFloat16>) {
    RunReduceOnWebGpu<MLFloat16>("ReduceLogSumExp", input_shape, FloatsToMLFloat16s(input), {1}, 0, {num_rows},
                                 FloatsToMLFloat16s(expected), abs_error);
  } else {
    RunReduceOnWebGpu<float>("ReduceLogSumExp", input_shape, input, {1}, 0, {num_rows}, expected, abs_error);
  }
}

// WebNN conformance reduceL2 float16 input. The sum of squares (~74000) exceeds the f16 range
// even though the result (~272.1) does not.
const std::vector<float> kReduceL2Input{
    4.859375f, 88.25f, 54.5f, 64.75f, 6.85546875f, 91.375f,
    41.875f, 73.625f, 35.3125f, 48.34375f, 82.375f, 77.875f,
    93.3125f, 62.5f, 60.28125f, 13.234375f, 20.53125f, 53.4375f,
    11.3203125f, 64.75f, 43.65625f, 0.83740234375f, 0.68505859375f, 33.5f};
}  // namespace

TEST(ReductionOpTest, ReduceL2_Float16_SharedPath_LargeSumOfSquares_WebGpu) {
  RunReduceOnWebGpu<MLFloat16>("ReduceL2", {2, 3, 4}, FloatsToMLFloat16s(kReduceL2Input), {0, 1, 2}, 0, {},
                               FloatsToMLFloat16s({272.0f}), 0.25f);
  // Non-innermost axes take the transpose + shared-memory path.
  RunReduceOnWebGpu<MLFloat16>("ReduceL2", {2, 2, 2, 3}, FloatsToMLFloat16s(kReduceL2Input), {1, 3}, 1, {2, 1, 2, 1},
                               FloatsToMLFloat16s({138.625f, 166.625f, 149.875f, 67.625f}), 0.125f);
}

TEST(ReductionOpTest, ReduceL2_Float16_NaivePath_LargeSquares_WebGpu) {
  // 300^2 and 400^2 overflow f16, but the norm (500) does not.
  constexpr int64_t kRows = 1040;
  std::vector<float> input;
  for (int64_t i = 0; i < kRows; ++i) {
    input.push_back(300.0f);
    input.push_back(-400.0f);
  }
  RunReduceOnWebGpu<MLFloat16>("ReduceL2", {kRows, 2}, FloatsToMLFloat16s(input), {1}, 0, {kRows},
                               FloatsToMLFloat16s(std::vector<float>(kRows, 500.0f)), 0.25f);
}

TEST(ReductionOpTest, ReduceSum_Float16_SharedPath_IntermediateOverflow_WebGpu) {
  // The workgroup tree reduction pairs elements 0+2 and 1+3, whose f16 partial sums overflow.
  RunReduceOnWebGpu<MLFloat16>("ReduceSum", {4}, FloatsToMLFloat16s({60000.0f, -60000.0f, 60000.0f, -60000.0f}), {0}, 0,
                               {}, FloatsToMLFloat16s({0.0f}));
}

TEST(ReductionOpTest, ReduceMean_Float16_SharedPath_IntermediateOverflow_WebGpu) {
  RunReduceOnWebGpu<MLFloat16>("ReduceMean", {2}, FloatsToMLFloat16s({60000.0f, 60000.0f}), {0}, 0, {},
                               FloatsToMLFloat16s({60000.0f}));
  // With 257-element rows, each invocation's own sum of 4 or 5 elements overflows f16.
  RunReduceOnWebGpu<MLFloat16>("ReduceMean", {2, 257}, FloatsToMLFloat16s(std::vector<float>(514, 60000.0f)), {1}, 0,
                               {2}, FloatsToMLFloat16s({60000.0f, 60000.0f}));
}

TEST(ReductionOpTest, ReduceLogSum_Float16_SharedPath_LargeSum_WebGpu) {
  // WebNN conformance reduce-log-sum-small-values: the sum exceeds f16's range, the log does not.
  RunReduceOnWebGpu<MLFloat16>("ReduceLogSum", {4}, FloatsToMLFloat16s({65504.0f, 65504.0f, 65504.0f, 65504.0f}), {0}, 0,
                               {}, FloatsToMLFloat16s({12.4765625f}), 0.01f);
}

TEST(ReductionOpTest, ReduceLogSumExp_LargeMagnitude_WebGpu) {
  // WebNN conformance reduce_log_sum_exp: exp(100) overflows f32 unless the max is subtracted first.
  RunReduceOnWebGpu<float>("ReduceLogSumExp", {1}, {100.0f}, {0}, 0, {}, {100.0f});
  RunReduceOnWebGpu<float>("ReduceLogSumExp", {1}, {-100.0f}, {0}, 0, {}, {-100.0f});
}

TEST(ReductionOpTest, ReduceLogSumExp_SharedPath_WebGpu) {
  std::vector<float> increasing(64);
  for (size_t i = 0; i < increasing.size(); ++i) {
    increasing[i] = 0.5f * static_cast<float>(i) - 10.0f;
  }
  std::vector<float> neg_inf_and_three(64, -kInf);
  for (size_t i = 0; i < neg_inf_and_three.size(); i += 2) {
    neg_inf_and_three[i] = 3.0f;
  }
  std::vector<float> one_pos_inf(64, 0.0f);
  one_pos_inf[37] = kInf;
  TestLogSumExpRows<float>({std::vector<float>(64, 100.0f),
                            std::vector<float>(64, -100.0f),
                            std::vector<float>(64, -kInf),
                            neg_inf_and_three,
                            one_pos_inf,
                            increasing},
                           6, 1e-4f);
}

TEST(ReductionOpTest, ReduceLogSumExp_SharedPath_LongRows_WebGpu) {
  // 257-element rows make each invocation loop over several elements: 4 or 5 with several rows (64
  // invocations); with one row (256 invocations), index 256 is invocation 0's second element. If
  // the max loop misses the 1000 in a later chunk, exp(1000 - max) overflows. In sum_late every
  // element contributes and index 256 dominates.
  constexpr size_t kRowLength = 257;
  std::vector<float> max_late(kRowLength);
  std::vector<float> sum_late(kRowLength);
  for (size_t i = 0; i < kRowLength; ++i) {
    max_late[i] = 0.25f * static_cast<float>(i % 41) - 5.0f;
    sum_late[i] = -0.125f * static_cast<float>(i % 23);
  }
  std::vector<float> max_mid = max_late;
  max_late[256] = 1000.0f;
  max_mid[130] = 1000.0f;
  sum_late[256] = 3.0f;
  TestLogSumExpRows<float>({max_late, max_mid, sum_late}, 3, 1e-4f);
  TestLogSumExpRows<MLFloat16>({max_late, max_mid, sum_late}, 3, 0.01f);
  for (const auto& row : {max_late, sum_late}) {
    TestLogSumExpRows<float>({row}, 1, 1e-4f);
    TestLogSumExpRows<MLFloat16>({row}, 1, 0.01f);
  }
}

TEST(ReductionOpTest, ReduceLogSumExp_NaivePath_WebGpu) {
  TestLogSumExpRows<float>({{100.0f, 100.0f},
                            {-100.0f, -99.0f},
                            {-kInf, -kInf},
                            {-kInf, 3.0f},
                            {5.0f, -kInf},
                            {kInf, 1.0f},
                            {1.0f, kInf}},
                           1036, 1e-4f);
}

TEST(ReductionOpTest, ReduceLogSumExp_Float16_WebGpu) {
  // WebNN conformance reduce_log_sum_exp float16 4D tensor options.axes.
  const std::vector<float> input{
      0.79736328125f, 5.046875f, 8.5234375f, 1.40625f,
      0.11883544921875f, 0.285888671875f, 1.9326171875f, 3.794921875f,
      2.603515625f, 4.9375f, 4.5703125f, 0.78662109375f,
      0.210205078125f, 9.0625f, 4.1015625f, 4.59765625f,
      0.2548828125f, 1.1591796875f, 6.8046875f, 5.234375f,
      8.9140625f, 9.1640625f, 5.71875f, 0.325439453125f};
  RunReduceOnWebGpu<MLFloat16>("ReduceLogSumExp", {2, 2, 2, 3}, FloatsToMLFloat16s(input), {1, 3}, 0, {2, 2},
                               FloatsToMLFloat16s({8.5703125f, 5.5f, 9.75f, 9.203125f}), 0.01f);
}

TEST(ReductionOpTest, ReduceLogSumExp_EmptyInput_WebGpu) {
  RunReduceOnWebGpu<float>("ReduceLogSumExp", {2, 0}, {}, {1}, 0, {2}, {-kInf, -kInf});
}
#endif

}  // namespace test
}  // namespace onnxruntime
