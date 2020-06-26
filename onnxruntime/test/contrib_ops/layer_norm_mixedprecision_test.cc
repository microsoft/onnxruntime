// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/providers/compare_provider_test_utils.h"

#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

#ifdef USE_CUDA
constexpr auto k_epsilon_default = 1e-5f;
constexpr auto k_random_data_min = -1.0f;
constexpr auto k_random_data_max = 1.0f;

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

static void TestLayerNormMixedPrecision(const std::vector<int64_t>& x_dims,
                          optional<float> epsilon,
                          int64_t axis = -1,
                          int64_t keep_dims = 1) {
  const std::vector<int64_t>& n_x_m_dims = x_dims;
  std::vector<int64_t> n_dims, m_dims;
  ASSERT_TRUE(SplitDims(n_x_m_dims, axis, n_dims, m_dims).IsOK());
  // n_dims padded with ones
  std::vector<int64_t> n_and_ones_dims(n_dims.begin(), n_dims.end());
  std::fill_n(std::back_inserter(n_and_ones_dims), n_x_m_dims.size() - n_dims.size(), 1);

  // TODO keep_dims is not implemented, default behavior is to keep ones for reduced dimensions
  ASSERT_NE(keep_dims, 0);

  const std::vector<int64_t>& stats_dims = keep_dims ? n_and_ones_dims : n_dims;

  CompareOpTester test("LayerNormalization");
  test.AddAttribute("axis", axis);
  test.AddAttribute("keep_dims", keep_dims);
  if (epsilon.has_value()) {
    test.AddAttribute("epsilon", epsilon.value());
  }

  // create rand inputs
  RandomValueGenerator random{};
  std::vector<float> X_data = random.Uniform<float>(n_x_m_dims, k_random_data_min, k_random_data_max);
  std::vector<float> scale_data = random.Uniform<float>(m_dims, k_random_data_min, k_random_data_max);
  std::vector<float> B_data = random.Uniform<float>(m_dims, k_random_data_min, k_random_data_max);

  
  std::vector<MLFloat16> X_data_half(X_data.size());
  std::vector<MLFloat16> scale_data_half(scale_data.size());
  std::vector<MLFloat16> B_data_half(B_data.size());
  ConvertFloatToMLFloat16(X_data.data(), X_data_half.data(), int(X_data.size()));
  ConvertFloatToMLFloat16(scale_data.data(), scale_data_half.data(), int(scale_data.size()));
  ConvertFloatToMLFloat16(B_data.data(), B_data_half.data(), int(B_data.size()));

  test.AddInput<MLFloat16>("X", n_x_m_dims, X_data_half);
  test.AddInput<MLFloat16>("scale", m_dims, scale_data_half, true);
  test.AddInput<MLFloat16>("B", m_dims, B_data_half, true);

  std::vector<MLFloat16> Y_data = FillZeros<MLFloat16>(n_x_m_dims);
  std::vector<float> mean_data = FillZeros<float>(stats_dims);
  std::vector<float> var_data = FillZeros<float>(stats_dims);

  test.AddOutput<MLFloat16>("output", n_x_m_dims, Y_data);
  test.AddOutput<float>("mean", stats_dims, mean_data);
  test.AddOutput<float>("var", stats_dims, var_data);

  double per_sample_tolerance = 2e-3;
  double relative_per_sample_tolerance = 2e-3;
  test.CompareWithCPU(kCudaExecutionProvider, per_sample_tolerance, relative_per_sample_tolerance);
}

TEST(CudaKernelTest, LayerNormMixedPrecision_SmallSizeTensor) {
  const std::vector<int64_t> X_dims{4, 20, 128};
  TestLayerNormMixedPrecision(X_dims, k_epsilon_default);
}

TEST(CudaKernelTest, LayerNormMixedPrecision_SmallSizeTensor_IntermediateAxis) {
  const std::vector<int64_t> X_dims{4, 20, 8, 16};
  const int64_t axis = -2;
  TestLayerNormMixedPrecision(X_dims, k_epsilon_default, axis);
}

TEST(CudaKernelTest, LayerNormMixedPrecision_MidSizeTensor) {
  std::vector<int64_t> X_dims{8, 80, 768};
  TestLayerNormMixedPrecision(X_dims, k_epsilon_default);
}

TEST(CudaKernelTest, LayerNormMixedPrecision_512_HiddenSizeTensor) {
  std::vector<int64_t> X_dims{4, 4, 512};
  TestLayerNormMixedPrecision(X_dims, k_epsilon_default);
}

TEST(CudaKernelTest, LayerNormMixedPrecision_1000_HiddenSizeTensor) {
  std::vector<int64_t> X_dims{4, 4, 1000};
  TestLayerNormMixedPrecision(X_dims, k_epsilon_default);
}

TEST(CudaKernelTest, LayerNormMixedPrecision_1024_HiddenSizeTensor) {
  std::vector<int64_t> X_dims{4, 4, 1024};
  TestLayerNormMixedPrecision(X_dims, k_epsilon_default);
}

TEST(CudaKernelTest, LayerNormMixedPrecision_1536_HiddenSizeTensor) {
  std::vector<int64_t> X_dims{4, 4, 1536};
  TestLayerNormMixedPrecision(X_dims, k_epsilon_default);
}

TEST(CudaKernelTest, LayerNormMixedPrecision_2048_HiddenSizeTensor) {
  std::vector<int64_t> X_dims{4, 4, 2048};
  TestLayerNormMixedPrecision(X_dims, k_epsilon_default);
}

TEST(CudaKernelTest, LayerNormMixedPrecision_2560_HiddenSizeTensor) {
  std::vector<int64_t> X_dims{4, 4, 2560};
  TestLayerNormMixedPrecision(X_dims, k_epsilon_default);
}

#endif
}  // namespace test
}  // namespace onnxruntime
