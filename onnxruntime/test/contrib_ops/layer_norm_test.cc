// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/providers/compare_provider_test_utils.h"

namespace onnxruntime {
namespace test {

#if defined(USE_CUDA) || defined(USE_ROCM) || defined(USE_DML)
constexpr auto k_epsilon_default = 1e-5f;
constexpr auto k_random_data_min = -10.0f;
constexpr auto k_random_data_max = 10.0f;
const std::string SIMPLIFIED_LAYER_NORM_OP = "SimplifiedLayerNormalization";
const std::string LAYER_NORM_OP = "LayerNormalization";

// The dimensions are split at the specified axis into N (before, exclusive) and M (after, inclusive).
static Status SplitDims(
    const std::vector<int64_t>& dims, int64_t axis,
    std::vector<int64_t>& n_dims, std::vector<int64_t>& m_dims) {
  if (axis < 0) axis += dims.size();
  ORT_RETURN_IF_NOT(0 <= axis && static_cast<decltype(dims.size())>(axis) <= dims.size(),
                    "0 <= axis && axis <= dims.size() was false");
  const auto boundary = dims.begin() + axis;
  n_dims.assign(dims.begin(), boundary);
  m_dims.assign(boundary, dims.end());
  return Status::OK();
}

static void TestLayerNorm(const std::vector<int64_t>& x_dims,
                          const std::string& op,
                          optional<float> epsilon,
                          int64_t axis = -1,
                          int64_t keep_dims = 1,
                          bool no_bias = false,
                          int opset = 9) {
  const std::vector<int64_t>& n_x_m_dims = x_dims;
  std::vector<int64_t> n_dims, m_dims;
  ASSERT_TRUE(SplitDims(n_x_m_dims, axis, n_dims, m_dims).IsOK());
  // n_dims padded with ones
  std::vector<int64_t> n_and_ones_dims(n_dims.begin(), n_dims.end());
  std::fill_n(std::back_inserter(n_and_ones_dims), n_x_m_dims.size() - n_dims.size(), 1);

  // TODO keep_dims is not implemented, default behavior is to keep ones for reduced dimensions
  ASSERT_NE(keep_dims, 0);

  CompareOpTester test(op.c_str(), opset);
  test.AddAttribute("axis", axis);
  test.AddAttribute("keep_dims", keep_dims);
  if (epsilon.has_value()) {
    test.AddAttribute("epsilon", *epsilon);
  }

  // create rand inputs
  RandomValueGenerator random{};
  std::vector<float> X_data = random.Uniform<float>(n_x_m_dims, k_random_data_min, k_random_data_max);
  std::vector<float> scale_data = random.Uniform<float>(m_dims, k_random_data_min, k_random_data_max);
  std::vector<float> B_data = random.Uniform<float>(m_dims, k_random_data_min, k_random_data_max);

  test.AddInput<float>("X", n_x_m_dims, X_data);
  test.AddInput<float>("scale", m_dims, scale_data, true);
  if (op.compare(SIMPLIFIED_LAYER_NORM_OP) != 0 && no_bias == false) {
    test.AddInput<float>("B", m_dims, B_data, true);
  }

  std::vector<float> Y_data = FillZeros<float>(n_x_m_dims);
  test.AddOutput<float>("output", n_x_m_dims, Y_data);

#ifndef USE_DML
  // DML doesn't support more than one output for these ops yet
  const std::vector<int64_t>& stats_dims = keep_dims ? n_and_ones_dims : n_dims;
  std::vector<float> mean_data = FillZeros<float>(stats_dims);
  std::vector<float> var_data = FillZeros<float>(stats_dims);

  // the Main and InvStdDev outputs are training specific
  if (op.compare(SIMPLIFIED_LAYER_NORM_OP) != 0) {
    test.AddOutput<float>("mean", stats_dims, mean_data);
  }
  test.AddOutput<float>("var", stats_dims, var_data);
#endif

#ifdef USE_CUDA
  test.CompareWithCPU(kCudaExecutionProvider);
#elif USE_ROCM
  test.CompareWithCPU(kRocmExecutionProvider);
#elif USE_DML
  test.CompareWithCPU(kDmlExecutionProvider);
#endif
}

TEST(CudaKernelTest, LayerNorm_NullInput) {
  const std::vector<int64_t> X_dims{0, 20, 128};
  TestLayerNorm(X_dims, LAYER_NORM_OP, k_epsilon_default);
}

TEST(CudaKernelTest, LayerNorm_SmallSizeTensor) {
  const std::vector<int64_t> X_dims{4, 20, 128};
  TestLayerNorm(X_dims, LAYER_NORM_OP, k_epsilon_default);
}

TEST(CudaKernelTest, LayerNorm_SmallSizeTensor_IntermediateAxis) {
  const std::vector<int64_t> X_dims{4, 20, 8, 16};
  constexpr int64_t axis = -2;
  TestLayerNorm(X_dims, LAYER_NORM_OP, k_epsilon_default, axis);
}

TEST(CudaKernelTest, LayerNorm_MidSizeTensor) {
  std::vector<int64_t> X_dims{8, 80, 768};
  TestLayerNorm(X_dims, LAYER_NORM_OP, k_epsilon_default);
}

TEST(CudaKernelTest, LayerNorm_LargeSizeTensor) {
  std::vector<int64_t> X_dims{16, 512, 1024};
  TestLayerNorm(X_dims, LAYER_NORM_OP, k_epsilon_default);
}

TEST(CudaKernelTest, LayerNorm_4KTensor) {
  std::vector<int64_t> X_dims{3, 10, 4096};
  TestLayerNorm(X_dims, LAYER_NORM_OP, k_epsilon_default);
}

TEST(CudaKernelTest, LayerNorm_8KTensor) {
  std::vector<int64_t> X_dims{3, 10, 8192};
  TestLayerNorm(X_dims, LAYER_NORM_OP, k_epsilon_default);
}

TEST(CudaKernelTest, LayerNorm_MidSizeTensor_NoBias) {
  std::vector<int64_t> X_dims{8, 80, 768};
  constexpr int64_t axis = -1;
  constexpr int64_t keep_dims = 1;
  constexpr bool no_bias = true;
  TestLayerNorm(X_dims, LAYER_NORM_OP, k_epsilon_default, axis, keep_dims, no_bias);
}

TEST(CudaKernelTest, SimplifiedLayerNorm_SmallSizeTensor) {
  const std::vector<int64_t> X_dims{4, 20, 128};
  TestLayerNorm(X_dims, SIMPLIFIED_LAYER_NORM_OP, k_epsilon_default);
}

TEST(CudaKernelTest, SimplifiedLayerNorm_SmallSizeTensor_IntermediateAxis) {
  const std::vector<int64_t> X_dims{4, 20, 8, 16};
  constexpr int64_t axis = -2;
  TestLayerNorm(X_dims, SIMPLIFIED_LAYER_NORM_OP, k_epsilon_default, axis);
}

TEST(CudaKernelTest, SimplifiedLayerNorm_MidSizeTensor) {
  std::vector<int64_t> X_dims{8, 80, 768};
  TestLayerNorm(X_dims, SIMPLIFIED_LAYER_NORM_OP, k_epsilon_default);
}

TEST(CudaKernelTest, SimplifiedLayerNorm_LargeSizeTensor) {
  std::vector<int64_t> X_dims{16, 512, 1024};
  TestLayerNorm(X_dims, SIMPLIFIED_LAYER_NORM_OP, k_epsilon_default);
}

// LayerNormalization is an ONNX operator in opset 17. It uses the same implementation so this is just a sanity check.
TEST(CudaKernelTest, LayerNorm_SmallSizeTensor_Opset17) {
  const std::vector<int64_t> X_dims{4, 20, 128};
  TestLayerNorm(X_dims, LAYER_NORM_OP, k_epsilon_default, -1, 1, false, 17);
}

#endif
}  // namespace test
}  // namespace onnxruntime
