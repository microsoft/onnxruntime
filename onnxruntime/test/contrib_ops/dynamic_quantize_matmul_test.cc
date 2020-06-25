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
                               const std::string& reference_model) {
  // create rand inputs
  RandomValueGenerator random{};

  std::vector<float> A_data = random.Uniform<float>(A_dims, -1.0f, 1.0f);

  std::vector<T> B_data;
  std::vector<int> tmp_B_data = random.Uniform<int32_t>(B_dims, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
  std::transform(tmp_B_data.begin(), tmp_B_data.end(), std::back_inserter(B_data), [](int32_t v) -> T {
    return static_cast<T>(v);
  });

  std::vector<float> B_scale = random.Uniform<float>({1}, -0.1f, 0.1f);
  std::vector<T> B_zero_point = {static_cast<T>(random.Uniform<int32_t>({1}, std::numeric_limits<T>::min(), std::numeric_limits<T>::max())[0])};

  OpTester test("DynamicQuantizeMatMul", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("A", A_dims, A_data);
  test.AddInput<T>("B", B_dims, B_data);
  test.AddInput<float>("b_scale", {1}, B_scale);
  test.AddInput<T>("b_zero_point", {1}, B_zero_point);

  test.AddReferenceOutputs(reference_model);
  test.Run();
}

TEST(DynamicQuantizeMatMul, Int8_test) {
#ifdef MLAS_SUPPORTS_GEMM_U8X8
  std::vector<int64_t> A_dims{4, 128};
  std::vector<int64_t> B_dims{128, 128};
  std::vector<int64_t> Y_dims{4, 128};

  TestDynamicQuantizeMatMul<int8_t>(A_dims, B_dims, "testdata/dynamic_quantize_matmul_int8.onnx");
#endif
}

TEST(DynamicQuantizeMatMul, UInt8_test) {
  std::vector<int64_t> A_dims{4, 128};
  std::vector<int64_t> B_dims{128, 128};
  std::vector<int64_t> Y_dims{4, 128};

  TestDynamicQuantizeMatMul<uint8_t>(A_dims, B_dims, "testdata/dynamic_quantize_matmul_uint8.onnx");
}

}  // namespace test
}  // namespace onnxruntime
