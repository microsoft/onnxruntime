// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/common/cuda_op_test_utils.h"

namespace onnxruntime {
namespace test {

// These tests are only applicable for the CUDA EP for now
#ifdef USE_CUDA
TEST(LastTokenMatMulContribOpTest, MultiTokenFloat) {
  OpTester test("LastTokenMatMul", 1, kMSDomain);

  std::vector<float> A_data = {1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f};
  test.AddInput<float>("A", {2, 2, 2}, A_data);

  std::vector<float> B_data = {1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f};
  test.AddInput<float>("B", {2, 4}, B_data);

  std::vector<float> Y_data = {7.f, 14.f, 21.f, 28.f, 7.f, 14.f, 21.f, 28.f};
  test.AddOutput<float>("Y", {2, 1, 4}, Y_data);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(LastTokenMatMulContribOpTest, SingleTokenFloat) {
  OpTester test("LastTokenMatMul", 1, kMSDomain);

  std::vector<float> A_data = {1.f, 2.f, 3.f, 4.f};
  test.AddInput<float>("A", {2, 1, 2}, A_data);

  std::vector<float> B_data = {1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 5.f};
  test.AddInput<float>("B", {2, 4}, B_data);

  std::vector<float> Y_data = {3.f, 6.f, 9.f, 12.f, 7.f, 14.f, 21.f, 28.f};
  test.AddOutput<float>("Y", {2, 1, 4}, Y_data);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(LastTokenMatMulContribOpTest, MultiTokenFloat16) {
  OpTester test("LastTokenMatMul", 1, kMSDomain);

  std::vector<float> A_data = {1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f};
  test.AddInput<MLFloat16>("A", {2, 2, 2}, ToFloat16(A_data));

  std::vector<float> B_data = {1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f};
  test.AddInput<MLFloat16>("B", {2, 4}, ToFloat16(B_data));

  std::vector<float> Y_data = {7.f, 14.f, 21.f, 28.f, 7.f, 14.f, 21.f, 28.f};
  test.AddOutput<MLFloat16>("Y", {2, 1, 4}, ToFloat16(Y_data));

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(LastTokenMatMulContribOpTest, SingleTokenFloat16) {
  OpTester test("LastTokenMatMul", 1, kMSDomain);

  std::vector<float> A_data = {1.f, 2.f, 3.f, 4.f};
  test.AddInput<MLFloat16>("A", {2, 1, 2}, ToFloat16(A_data));

  std::vector<float> B_data = {1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f};
  test.AddInput<MLFloat16>("B", {2, 4}, ToFloat16(B_data));

  std::vector<float> Y_data = {3.f, 6.f, 9.f, 12.f, 7.f, 14.f, 21.f, 28.f};
  test.AddOutput<MLFloat16>("Y", {2, 1, 4}, ToFloat16(Y_data));

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}
#endif

}  // namespace test
}  // namespace onnxruntime