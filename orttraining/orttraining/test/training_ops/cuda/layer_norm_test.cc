// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "Eigen/Core"

#include "test/providers/compare_provider_test_utils.h"

namespace onnxruntime {
namespace test {

constexpr auto k_epsilon_default = 1e-5f;
constexpr auto k_random_data_min = -10.0f;
constexpr auto k_random_data_max = 10.0f;

// The dimensions are split at the specified axis into N (before, exclusive) and M (after, inclusive).
static Status SplitDims(
    const std::vector<int64_t>& dims, int64_t axis,
    std::vector<int64_t>& n_dims, std::vector<int64_t>& m_dims) {
  if (axis < 0) axis += dims.size();
  ORT_RETURN_IF_NOT(0 <= axis && static_cast<decltype(dims.size())>(axis) <= dims.size());
  const auto boundary = dims.begin() + axis;
  n_dims.assign(dims.begin(), boundary);
  m_dims.assign(boundary, dims.end());
  return Status::OK();
}

static void TestLayerNormGrad(
    const std::vector<int64_t>& x_dims,
    int64_t axis = -1,
    double error_tolerance = 1e-4) {
  const std::vector<int64_t>& n_x_m_dims = x_dims;
  std::vector<int64_t> n_dims, m_dims;
  ASSERT_TRUE(SplitDims(n_x_m_dims, axis, n_dims, m_dims).IsOK());

  const auto N = std::accumulate(n_dims.begin(), n_dims.end(), static_cast<int64_t>(1), std::multiplies<>{});
  const auto M = std::accumulate(m_dims.begin(), m_dims.end(), static_cast<int64_t>(1), std::multiplies<>{});

  CompareOpTester test{"LayerNormalizationGrad"};

  test.AddAttribute("axis", axis);

  RandomValueGenerator random{};
  const auto Y_grad_data = random.Uniform<float>(n_x_m_dims, k_random_data_min, k_random_data_max);
  const auto X_data = random.Uniform<float>(n_x_m_dims, k_random_data_min, k_random_data_max);
  const auto scale_data = random.Uniform<float>(m_dims, k_random_data_min, k_random_data_max);

  // these inputs are dependent on X_data
  std::vector<float> mean_data(N);         // mean(X)
  std::vector<float> inv_std_var_data(N);  // 1 / sqrt(mean(X^2) - mean(X)^2 + epsilon)
  {
    using ConstEigenArrayMap = Eigen::Map<const Eigen::ArrayXX<float>>;
    using EigenRowVectorArrayMap = Eigen::Map<Eigen::Array<float, 1, Eigen::Dynamic>>;

    ConstEigenArrayMap X{X_data.data(), M, N};
    EigenRowVectorArrayMap mean{mean_data.data(), N};
    EigenRowVectorArrayMap inv_std_var{inv_std_var_data.data(), N};

    mean = X.colwise().mean();
    inv_std_var = ((X.colwise().squaredNorm() / X.rows()) - mean.square() + k_epsilon_default).rsqrt();
  }

  test.AddInput("Y_grad", n_x_m_dims, Y_grad_data);
  test.AddInput("X", n_x_m_dims, X_data);
  test.AddInput("scale", m_dims, scale_data, true);
  test.AddInput("mean", n_dims, mean_data);
  test.AddInput("inv_std_var", n_dims, inv_std_var_data);

  const auto X_grad_data = FillZeros<float>(n_x_m_dims);
  const auto scale_grad_data = FillZeros<float>(m_dims);
  const auto bias_grad_data = FillZeros<float>(m_dims);

  test.AddOutput("X_grad", n_x_m_dims, X_grad_data);
  test.AddOutput("scale_grad_data", m_dims, scale_grad_data);
  test.AddOutput("bias_grad_data", m_dims, bias_grad_data);

  test.CompareWithCPU(kCudaExecutionProvider, error_tolerance);
}

TEST(CudaKernelTest, LayerNormGrad_SmallSizeTensor) {
  const std::vector<int64_t> X_dims{4, 20, 128};
  TestLayerNormGrad(X_dims);
}

TEST(CudaKernelTest, LayerNormGrad_SmallSizeTensor_IntermediateAxis) {
  const std::vector<int64_t> X_dims{4, 20, 16, 8};
  const int64_t axis = -2;
  TestLayerNormGrad(X_dims, axis);
}

TEST(CudaKernelTest, LayerNormGrad_MidSizeTensor) {
  const std::vector<int64_t> X_dims{8, 80, 768};
  TestLayerNormGrad(X_dims);
}

TEST(CudaKernelTest, LayerNormGrad_LargeSizeTensor) {
  const std::vector<int64_t> X_dims{16, 512, 1024};
  TestLayerNormGrad(X_dims, -1, 5e-3);
}

}  // namespace test
}  // namespace onnxruntime
