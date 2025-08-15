// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/span_utils.h"
#include "core/framework/tensor.h"
#include "core/session/inference_session.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/framework/test_utils.h"
#include "test/optimizer/graph_transform_test_builder.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
#include "core/util/qmath.h"

#include <chrono>
#include <random>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

using namespace std;

namespace onnxruntime {
namespace test {

template <typename T>
static void CalculateDynamicQuantizeMatMul(const int64_t M, const int64_t N, const int64_t K,
                                           const std::vector<float>& A_data, const std::vector<T>& B_data,
                                           std::vector<float>& B_scale, std::vector<T>& B_zero_point,
                                           const std::vector<float>& Bias, std::vector<float>& Y_data,
                                           bool per_column, bool has_zp, bool has_bias) {
  // Dynamic Quantize Matrix A
  const uint32_t num_elements = static_cast<uint32_t>(M * K);
  std::vector<uint8_t> QuantA_data(num_elements);
  std::vector<float> A_scale;
  std::vector<uint8_t> A_zero_point;

  // Get max and min
  float min = std::numeric_limits<float>::max();
  float max = std::numeric_limits<float>::lowest();
  float qmax = static_cast<float>(std::numeric_limits<uint8_t>::max());
  float qmin = static_cast<float>(std::numeric_limits<uint8_t>::lowest());

  for (uint32_t i = 0; i < num_elements; ++i) {
    max = std::max(A_data[i], max);
    min = std::min(A_data[i], min);
  }

  // Adjust the maximum and minimum to include zero
  max = std::max(max, 0.0f);
  min = std::min(min, 0.0f);

  float scale = static_cast<float>(max - min) / (qmax - qmin);
  float initial_zero_point = qmin - min / scale;
  float clamped_zp = std::max(qmin, std::min(qmax, initial_zero_point));
  uint8_t zeroPoint = static_cast<uint8_t>(RoundHalfToEven(clamped_zp));

  A_scale.push_back(scale);
  A_zero_point.push_back(zeroPoint);

  // Matrix Multiplication
  for (uint32_t i = 0; i < num_elements; ++i) {
    float val = (A_data[i] / scale) + static_cast<float>(zeroPoint);
    float rq = std::nearbyintf(val);
    rq = std::min(qmax, std::max(qmin, rq));
    QuantA_data[i] = static_cast<uint8_t>(rq);
  }
  if (!per_column) {
    B_zero_point.resize(N, B_zero_point[0]);
    B_scale.resize(N, B_scale[0]);
  }

  for (int64_t m = 0; m < M; m++) {
    for (int64_t n = 0; n < N; n++) {
      float sum = 0.0f;
      for (int64_t k = 0; k < K; k++) {
        float A_dequantized = (static_cast<int>(QuantA_data[m * K + k]) - static_cast<int>(A_zero_point[0])) * A_scale[0];

        float B_dequantized = has_zp ? (static_cast<int>(B_data[k * N + n]) - static_cast<int>(B_zero_point[n])) * B_scale[n] : B_data[k * N + n] * B_scale[n];

        sum += A_dequantized * B_dequantized;
      }
      if (has_bias) {
        sum += Bias[n];
      }
      Y_data[m * N + n] = sum;
    }
  }
}

struct TestDynamicQuantizeMatMulOptions {
  bool is_matrix_b_constant = true;

  bool per_column = false;

  bool is_scale_constant = false;

  bool has_zp = true;
  bool is_zp_constant = false;
  bool is_zp_zero = false;

  bool has_bias = false;
  bool is_bias_constant = false;

  bool empty_input = false;
};

template <typename T>
void TestDynamicQuantizeMatMul(const TestDynamicQuantizeMatMulOptions& opts) {
  static_assert(std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>);

  SCOPED_TRACE(MakeString(
      "b data type:", (std::is_same_v<T, uint8_t> ? "uint8" : "int8"),
      ", is_matrix_b_constant:", opts.is_matrix_b_constant,
      ", per_column:", opts.per_column,
      ", is_scale_constant:", opts.is_scale_constant,
      ", has_zp:", opts.has_zp,
      ", is_zp_constant:", opts.is_zp_constant,
      ", is_zp_zero:", opts.is_zp_zero,
      ", has_bias:", opts.has_bias,
      ", is_bias_constant:", opts.is_bias_constant,
      ", empty_input:", opts.empty_input));

  // create rand inputs
  RandomValueGenerator random{1668426375};

  int64_t M = opts.empty_input ? 1 : 4;
  int64_t N = 128;
  int64_t K = 128;
  std::vector<int64_t> A_dims{opts.empty_input ? 0 : M, K};
  std::vector<int64_t> B_dims{K, N};
  std::vector<int64_t> Y_dims{opts.empty_input ? 0 : M, N};
  std::vector<float> A_data = random.Uniform<float>(A_dims, -1.0f, 1.0f);
  std::vector<T> B_data;
  std::vector<T> tmp_B_data = random.Uniform<T>(B_dims,
                                                (std::is_same_v<T, int8_t>) ? std::numeric_limits<int8_t>::lowest() / 2 : std::numeric_limits<uint8_t>::lowest(),
                                                std::numeric_limits<T>::max() / 2);
  std::transform(tmp_B_data.begin(), tmp_B_data.end(), std::back_inserter(B_data), [](int32_t v) -> T {
    return static_cast<T>(v);
  });

  int64_t b_scale_zp_size = opts.per_column ? B_dims.back() : 1;
  std::vector<float> B_scale = random.Uniform<float>(AsSpan({b_scale_zp_size}), -0.1f, 0.1f);
  std::vector<T> B_zero_point(b_scale_zp_size);
  if (!opts.is_zp_zero) {
    std::for_each(B_zero_point.begin(),
                  B_zero_point.end(),
                  [&random](T& zp) {
                    zp = static_cast<T>(random.Uniform<T>(std::array<int64_t, 1>{1},
                                                          std::numeric_limits<T>::min(),
                                                          std::numeric_limits<T>::max())[0]);
                  });
  }

  std::vector<float> Bias = random.Uniform<float>(AsSpan({B_dims.back()}), -0.1f, 0.1f);

  OpTester test("DynamicQuantizeMatMul", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("A", A_dims, A_data);
  test.AddInput<T>("B", B_dims, B_data, opts.is_matrix_b_constant);
  test.AddInput<float>("b_scale", {b_scale_zp_size}, B_scale, opts.is_scale_constant);

  if (opts.has_zp) {
    test.AddInput<T>("b_zero_point", {b_scale_zp_size}, B_zero_point, opts.is_zp_constant);
  } else {
    test.AddOptionalInputEdge<T>();
  }

  if (opts.has_bias) {
    test.AddInput<float>("bias", {B_dims.back()}, Bias, opts.is_bias_constant);
  } else {
    test.AddOptionalInputEdge<float>();
  }

  std::vector<float> Y_data(M * N);
  CalculateDynamicQuantizeMatMul<T>(M, N, K, A_data, B_data, B_scale, B_zero_point, Bias, Y_data,
                                    opts.per_column, opts.has_zp, opts.has_bias);
  test.AddOutput<float>("Y", Y_dims, Y_data);
  test.SetOutputRelErr("Y", 0.02f);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kOpenVINOExecutionProvider});
}

template <typename T>
void TestDynamicQuantizeMatMul(bool is_matrix_b_constant,
                               bool per_column = false,
                               bool has_zp = true,
                               bool has_bias = false,
                               bool empty_input = false) {
  TestDynamicQuantizeMatMulOptions opts{};
  opts.is_matrix_b_constant = is_matrix_b_constant;
  opts.per_column = per_column;
  opts.has_zp = has_zp;
  opts.has_bias = has_bias;
  opts.empty_input = empty_input;

  TestDynamicQuantizeMatMul<T>(opts);
}

template <typename T>
void RunDynamicQuantizeMatMulTest() {
  for (bool is_matrix_b_constant : {false, true}) {
    for (bool per_column : {false, true}) {
      for (bool has_zp : {false, true}) {
        for (bool has_bias : {false, true}) {
          TestDynamicQuantizeMatMul<T>(is_matrix_b_constant,
                                       per_column,
                                       has_zp,
                                       has_bias);
        }
      }
    }
  }
}

TEST(DynamicQuantizeMatMul, Int8) {
  RunDynamicQuantizeMatMulTest<int8_t>();
}

TEST(DynamicQuantizeMatMul, UInt8) {
  RunDynamicQuantizeMatMulTest<uint8_t>();
}

TEST(DynamicQuantizeMatMul, WithConstantBInputs) {
  TestDynamicQuantizeMatMulOptions base_opts{};
  base_opts.is_matrix_b_constant = true;
  base_opts.is_scale_constant = true;
  base_opts.is_zp_constant = true;

  {
    // no zp
    auto opts = base_opts;
    opts.has_zp = false;

    TestDynamicQuantizeMatMul<int8_t>(opts);
    TestDynamicQuantizeMatMul<uint8_t>(opts);
  }

  {
    // zp that is zero (symmetric quantization)
    auto opts = base_opts;
    opts.has_zp = true;
    opts.is_zp_zero = true;

    TestDynamicQuantizeMatMul<int8_t>(opts);
    TestDynamicQuantizeMatMul<uint8_t>(opts);
  }

  {
    // zp that is non-zero
    auto opts = base_opts;
    opts.has_zp = true;
    opts.is_zp_zero = false;

    TestDynamicQuantizeMatMul<int8_t>(opts);
    TestDynamicQuantizeMatMul<uint8_t>(opts);
  }
}

TEST(DynamicQuantizeMatMul, UInt8_test_with_empty_input) {
  std::vector<int64_t> A_dims{0, 2};
  std::vector<int64_t> B_dims{2, 2};
  std::vector<int64_t> Y_dims{0, 2};
  OpTester test("DynamicQuantizeMatMul", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("T1", A_dims, {});
  test.AddInput<uint8_t>("T2", B_dims, {1, 6, 0, 8});
  test.AddInput<float>("b_scale", {1}, {1.0f});
  test.AddInput<uint8_t>("b_zero_point", {1}, {0});
  test.AddOptionalInputEdge<float>();
  test.AddOutput<float>("Y", {0, 2}, {});
  test.Run();
}

TEST(DynamicQuantizeMatMul, B_PerColumn_ND) {
  auto test_case = [&](const std::vector<int64_t>& input_shape,
                       const std::vector<int64_t>& weights_shape,
                       const std::vector<int64_t>& b_scale_zp_shape) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<float>(input_shape, -1.0f, 1.0f);
      auto* output_arg = builder.MakeOutput();
      auto* weight = builder.MakeInitializer<int8_t>(weights_shape,
                                                     std::numeric_limits<int8_t>::min() / 2,
                                                     std::numeric_limits<int8_t>::max() / 2);

      // add DynamicQuantizeLinear
      auto* dql_output = builder.MakeIntermediate();
      auto* dql_scale = builder.MakeIntermediate();
      auto* dql_zp = builder.MakeIntermediate();
      builder.AddNode("DynamicQuantizeLinear", {input_arg}, {dql_output, dql_scale, dql_zp});

      // add MatMulInteger
      auto* matmul_integer_output = builder.MakeIntermediate();
      auto* B_zp_arg = builder.MakeInput<int8_t>(b_scale_zp_shape,
                                                 std::numeric_limits<int8_t>::min() / 2,
                                                 std::numeric_limits<int8_t>::max() / 2);
      builder.AddNode("MatMulInteger", {dql_output, weight, dql_zp, B_zp_arg}, {matmul_integer_output});

      // add Cast
      auto* cast_output = builder.MakeIntermediate();
      Node& cast_node = builder.AddNode("Cast", {matmul_integer_output}, {cast_output});
      cast_node.AddAttribute("to", (int64_t)1);

      // add Mul1
      auto* B_scale_arg = builder.MakeInput<float>(b_scale_zp_shape, -0.1f, 0.f);
      auto* mul1_output = builder.MakeIntermediate();
      builder.AddNode("Mul", {dql_scale, B_scale_arg}, {mul1_output});

      // add Mul2
      builder.AddNode("Mul", {mul1_output, cast_output}, {output_arg});
    };

    auto check_mp_reshape_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["com.microsoft.DynamicQuantizeMatMul"], 1);
    };

    TransformerTester(build_test_case,
                      check_mp_reshape_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      12 /*opset_version*/,
                      1e-5 /*per_sample_tolerance*/,
                      1e-5 /*relative_per_sample_tolerance*/);
  };

  // Scale Scalar
  test_case({5, 4, 3}, {3, 4}, {1});

  // 2D B per-column
  test_case({5, 4, 3}, {3, 4}, {4});
  test_case({5, 4, 3}, {3, 4}, {1, 4});

  // ND B per-column
  test_case({15, 14, 13}, {15, 13, 27}, {15, 1, 27});
}

}  // namespace test
}  // namespace onnxruntime
