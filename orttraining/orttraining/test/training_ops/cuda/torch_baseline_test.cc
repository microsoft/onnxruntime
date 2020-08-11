// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef USE_TORCH

#include "gtest/gtest.h"
#include "torch/torch.h"
#include "test/providers/provider_test_utils.h"
#include "test/providers/compare_provider_test_utils.h"

namespace onnxruntime {
namespace test {

static void TorchTestMatmul(const std::vector<int64_t>& A_shape,
                            const std::vector<int64_t>& B_shape) {
  OpTester test("MatMul");

  // Create inputs.
  RandomValueGenerator random{};
  std::vector<float> A = random.Uniform<float>(A_shape, -10.0f, 10.0f);
  std::vector<float> B = random.Uniform<float>(B_shape, -10.0f, 10.0f);

  // Add inputs to test context.
  test.AddInput<float>("A", A_shape, A);
  test.AddInput<float>("B", B_shape, B);

  // Invoke libtorch to compute result.
  auto torch_tensor_options = torch::TensorOptions().dtype(at::kFloat);
  torch::Tensor torch_A = torch::from_blob(A.data(), A_shape, torch_tensor_options);
  torch::Tensor torch_B = torch::from_blob(B.data(), B_shape, torch_tensor_options);
  torch::Tensor torch_C = at::matmul(torch_A, torch_B);

  // Copy libtorch result to test context.
  std::vector<int64_t> C_shape = torch_C.sizes().vec();
  std::vector<float> C((float*)torch_C.data_ptr(), (float*)torch_C.data_ptr() + torch_C.numel());
  test.AddOutput<float>("Y", C_shape, C);

  test.Run();
}

TEST(TorchBaseline, MatMul) {
  TorchTestMatmul({7, 16}, {16, 8});
}

}  // namespace test
}  // namespace onnxruntime

#endif // USE_TORCH