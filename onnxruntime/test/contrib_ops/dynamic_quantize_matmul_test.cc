// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensor.h"
#include "core/session/inference_session.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/framework/test_utils.h"
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
void TestDynamicQuantizeMatMul(const std::vector<int64_t>& A_dims,
                               std::vector<int64_t> B_dims,
                               const std::string& reference_model,
                               bool is_matrix_b_constant,
                               bool per_column = false,
                               bool has_zp = true,
                               bool has_bias = false) {
  // create rand inputs
  RandomValueGenerator random{};

  std::vector<float> A_data = random.Uniform<float>(A_dims, -1.0f, 1.0f);

  std::vector<T> B_data;
  std::vector<int> tmp_B_data = random.Uniform<int32_t>(B_dims, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
  std::transform(tmp_B_data.begin(), tmp_B_data.end(), std::back_inserter(B_data), [](int32_t v) -> T {
    return static_cast<T>(v);
  });

  int64_t b_scale_zp_size = per_column ? B_dims.back() : 1;
  std::vector<float> B_scale = random.Uniform<float>({b_scale_zp_size}, -0.1f, 0.1f);
  std::vector<T> B_zero_point(b_scale_zp_size);
  std::for_each(B_zero_point.begin(),
                B_zero_point.end(),
                [&random](T& zp) {
                  zp = static_cast<T>(random.Uniform<int32_t>({1}, std::numeric_limits<T>::min(), std::numeric_limits<T>::max())[0]);
                });

  std::vector<float> Bias = random.Uniform<float>({B_dims.back()}, -0.1f, 0.1f);

  OpTester test("DynamicQuantizeMatMul", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("A", A_dims, A_data);
  test.AddInput<T>("B", B_dims, B_data, is_matrix_b_constant);
  test.AddInput<float>("b_scale", {b_scale_zp_size}, B_scale);

  if (has_zp) {
    test.AddInput<T>("b_zero_point", {b_scale_zp_size}, B_zero_point);
  } else {
    test.AddMissingOptionalInput<T>();
  }

  if (has_bias) {
    test.AddInput<float>("bias", {B_dims.back()}, Bias);
  } else {
    test.AddMissingOptionalInput<float>();
  }

  test.AddReferenceOutputs(reference_model);
  test.Run();
}

template <typename Scalar, bool HasZeroPoint, bool HasBias>
void RunDynamicQuantizeMatMulTest(const string& model_path) {
  std::vector<int64_t> A_dims{4, 128};
  std::vector<int64_t> B_dims{128, 128};
  std::vector<int64_t> Y_dims{4, 128};

  TestDynamicQuantizeMatMul<Scalar>(A_dims,
                                    B_dims,
                                    model_path,
                                    false,        /*is_matrix_b_constant*/
                                    false,        /*per_column*/
                                    HasZeroPoint, /*has_zp*/
                                    HasBias       /*has_bias*/
  );

  TestDynamicQuantizeMatMul<Scalar>(A_dims,
                                    B_dims,
                                    model_path,
                                    true,         /*is_matrix_b_constant*/
                                    false,        /*per_column*/
                                    HasZeroPoint, /*has_zp*/
                                    HasBias       /*has_bias*/
  );

  TestDynamicQuantizeMatMul<Scalar>(A_dims,
                                    B_dims,
                                    model_path,
                                    false,        /*is_matrix_b_constant*/
                                    true,         /*per_column*/
                                    HasZeroPoint, /*has_zp*/
                                    HasBias       /*has_bias*/
  );

  TestDynamicQuantizeMatMul<Scalar>(A_dims,
                                    B_dims,
                                    model_path,
                                    true,         /*is_matrix_b_constant*/
                                    true,         /*per_column*/
                                    HasZeroPoint, /*has_zp*/
                                    HasBias       /*has_bias*/
  );
}

TEST(DynamicQuantizeMatMul, HasZeroPoint_NoBias_test) {
  RunDynamicQuantizeMatMulTest<int8_t, true, false>("testdata/dynamic_quantize_matmul_int8.onnx");
  RunDynamicQuantizeMatMulTest<uint8_t, true, false>("testdata/dynamic_quantize_matmul_uint8.onnx");
}

TEST(DynamicQuantizeMatMul, NoZeroPoint_HasBias_test) {
  RunDynamicQuantizeMatMulTest<int8_t, false, true>("testdata/dynamic_quantize_matmul_int8_bias.onnx");
  RunDynamicQuantizeMatMulTest<uint8_t, false, true>("testdata/dynamic_quantize_matmul_uint8_bias.onnx");
}

TEST(DynamicQuantizeMatMul, UInt8_test_with_empty_input) {
  std::vector<int64_t> A_dims{0, 128};
  std::vector<int64_t> B_dims{128, 128};
  std::vector<int64_t> Y_dims{0, 128};

  TestDynamicQuantizeMatMul<uint8_t>(A_dims,
                                     B_dims,
                                     "testdata/dynamic_quantize_matmul_uint8.onnx",
                                     false /*is_matrix_b_constant*/);

  TestDynamicQuantizeMatMul<uint8_t>(A_dims,
                                     B_dims,
                                     "testdata/dynamic_quantize_matmul_uint8.onnx",
                                     true /*is_matrix_b_constant*/);
}

}  // namespace test
}  // namespace onnxruntime
