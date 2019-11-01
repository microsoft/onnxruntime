// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/providers/compare_provider_test_utils.h"

using namespace std;

namespace onnxruntime {
namespace test {

static void TestLayerNorm(const std::vector<int64_t>& X_dims,
                          const std::vector<int64_t>& scale_dims,
                          const std::vector<int64_t>& B_dims,
                          const std::vector<int64_t>& Y_dims,
                          const std::vector<int64_t>& mean_dims,
                          const std::vector<int64_t>& var_dims,
                          optional<float> epsilon,
                          int64_t axis = -1,
                          int64_t keep_dims = 1) {
  CompareOpTester test("LayerNormalization");
  test.AddAttribute("axis", axis);
  test.AddAttribute("keep_dims", keep_dims);
  if (epsilon.has_value()) {
    test.AddAttribute("epsilon", epsilon.value());
  }

  // create rand inputs
  std::vector<float> X_data = UniformRandom<float>(X_dims, -10.0f, 10.0f);
  std::vector<float> scale_data = UniformRandom<float>(scale_dims, -10.0f, 10.0f);
  std::vector<float> B_data = UniformRandom<float>(B_dims, -10.0f, 10.0f);

  test.AddInput<float>("X", X_dims, X_data);
  test.AddInput<float>("scale", scale_dims, scale_data, true);
  test.AddInput<float>("B", B_dims, B_data, true);

  std::vector<float> Y_data = FillZeros<float>(Y_dims);
  std::vector<float> mean_data = FillZeros<float>(mean_dims);
  std::vector<float> var_data = FillZeros<float>(var_dims);

  test.AddOutput<float>("output", Y_dims, Y_data);
  test.AddOutput<float>("mean", mean_dims, mean_data);
  test.AddOutput<float>("var", var_dims, var_data);

  test.CompareWithCPU(kCudaExecutionProvider);
}

TEST(CudaKernelTest, LayerNorm_SmallSizeTensor) {
  float epsilon = 1e-05f;
  std::vector<int64_t> X_dims{4, 20, 128};
  std::vector<int64_t> scale_dims{128};
  std::vector<int64_t> B_dims{128};
  std::vector<int64_t> Y_dims{4, 20, 128};
  std::vector<int64_t> mean_dims{4, 20, 1};
  std::vector<int64_t> var_dims{4, 20, 1};
  TestLayerNorm(X_dims, scale_dims, B_dims, Y_dims, mean_dims, var_dims, epsilon);
}

TEST(CudaKernelTest, LayerNorm_MidSizeTensor) {
  float epsilon = 1e-05f;
  std::vector<int64_t> X_dims{8, 80, 768};
  std::vector<int64_t> scale_dims{768};
  std::vector<int64_t> B_dims{768};
  std::vector<int64_t> Y_dims{8, 80, 768};
  std::vector<int64_t> mean_dims{8, 80, 1};
  std::vector<int64_t> var_dims{8, 80, 1};
  TestLayerNorm(X_dims, scale_dims, B_dims, Y_dims, mean_dims, var_dims, epsilon);
}

TEST(CudaKernelTest, LayerNorm_LargeSizeTensor) {
  float epsilon = 1e-05f;
  std::vector<int64_t> X_dims{16, 512, 1024};
  std::vector<int64_t> scale_dims{1024};
  std::vector<int64_t> B_dims{1024};
  std::vector<int64_t> Y_dims{16, 512, 1024};
  std::vector<int64_t> mean_dims{16, 512, 1};
  std::vector<int64_t> var_dims{16, 512, 1};
  TestLayerNorm(X_dims, scale_dims, B_dims, Y_dims, mean_dims, var_dims, epsilon);
}

}  // namespace test
}  // namespace onnxruntime
