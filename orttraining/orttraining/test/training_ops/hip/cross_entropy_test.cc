// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/providers/compare_provider_test_utils.h"

using namespace std;

namespace onnxruntime {
namespace test {

static void TestSparseSoftmaxCrossEntropy(const std::vector<int64_t>* X_dims,
                                          const std::vector<int64_t>* index_dims,
                                          const std::vector<int64_t>* weight_dims,
                                          const std::vector<int64_t>* Y_dims,
                                          const std::vector<int64_t>* prob_dims,
                                          const std::string& reduction) {
  CompareOpTester test("SparseSoftmaxCrossEntropy");
  test.AddAttribute("reduction", reduction);

  // create rand inputs
  RandomValueGenerator random{};
  std::vector<float> X_data = random.Uniform<float>(*X_dims, -10.0f, 10.0f);
  std::vector<int64_t> index_data = random.Uniform<int64_t>(*index_dims, 0.0f, static_cast<float>(X_dims->back()));

  test.AddInput<float>("X", *X_dims, X_data);
  test.AddInput<int64_t>("index", *index_dims, index_data);

  if (weight_dims) {
    std::vector<float> weight_data = random.Uniform<float>(*weight_dims, 0.0f, 1.0f);
    test.AddInput<float>("weight", *weight_dims, weight_data);
  }

  std::vector<float> Y_data = FillZeros<float>(*Y_dims);
  std::vector<float> prob_data = FillZeros<float>(*prob_dims);

  test.AddOutput<float>("output", *Y_dims, Y_data);
  test.AddOutput<float>("prob", *prob_dims, prob_data);

  test.CompareWithCPU(kHipExecutionProvider);
}

TEST(HipKernelTest, SparseSoftmaxCrossEntropy_TinySizeTensor) {
  std::vector<int64_t> X_dims{8, 2};
  std::vector<int64_t> index_dims{8};
  std::vector<int64_t> weight_dims{8};
  std::vector<int64_t> Y_dims{};
  std::vector<int64_t> prob_dims{8};
  TestSparseSoftmaxCrossEntropy(&X_dims, &index_dims, &weight_dims, &Y_dims, &prob_dims, "mean");
  TestSparseSoftmaxCrossEntropy(&X_dims, &index_dims, nullptr, &Y_dims, &prob_dims, "mean");
  TestSparseSoftmaxCrossEntropy(&X_dims, &index_dims, &weight_dims, &Y_dims, &prob_dims, "sum");
  TestSparseSoftmaxCrossEntropy(&X_dims, &index_dims, nullptr, &Y_dims, &prob_dims, "sum");
}

TEST(HipKernelTest, SparseSoftmaxCrossEntropy_SmallSizeTensor) {
  std::vector<int64_t> X_dims{8, 20, 10};
  std::vector<int64_t> index_dims{8, 20};
  std::vector<int64_t> weight_dims{8, 20};
  std::vector<int64_t> Y_dims{};
  std::vector<int64_t> prob_dims{8, 20, 10};
  TestSparseSoftmaxCrossEntropy(&X_dims, &index_dims, &weight_dims, &Y_dims, &prob_dims, "mean");
  TestSparseSoftmaxCrossEntropy(&X_dims, &index_dims, nullptr, &Y_dims, &prob_dims, "mean");
  TestSparseSoftmaxCrossEntropy(&X_dims, &index_dims, &weight_dims, &Y_dims, &prob_dims, "sum");
  TestSparseSoftmaxCrossEntropy(&X_dims, &index_dims, nullptr, &Y_dims, &prob_dims, "sum");
}

TEST(HipKernelTest, SparseSoftmaxCrossEntropy_LargeSizeTensor) {
  std::vector<int64_t> X_dims{4, 512, 30528};
  std::vector<int64_t> index_dims{4, 512};
  std::vector<int64_t> weight_dims{4, 512};
  std::vector<int64_t> Y_dims{};
  std::vector<int64_t> prob_dims{4, 512, 30528};
  TestSparseSoftmaxCrossEntropy(&X_dims, &index_dims, &weight_dims, &Y_dims, &prob_dims, "mean");
  TestSparseSoftmaxCrossEntropy(&X_dims, &index_dims, nullptr, &Y_dims, &prob_dims, "mean");
  TestSparseSoftmaxCrossEntropy(&X_dims, &index_dims, &weight_dims, &Y_dims, &prob_dims, "sum");
  TestSparseSoftmaxCrossEntropy(&X_dims, &index_dims, nullptr, &Y_dims, &prob_dims, "sum");
}

static void TestSparseSoftmaxCrossEntropyGrad(const std::vector<int64_t>& dY_dims,
                                              const std::vector<int64_t>& prob_dims,
                                              const std::vector<int64_t>& index_dims,
                                              const std::vector<int64_t>& dX_dims,
                                              const std::string& reduction) {
  CompareOpTester test("SparseSoftmaxCrossEntropyGrad");
  test.AddAttribute("reduction", reduction);

  // create rand inputs
  RandomValueGenerator random{};
  std::vector<float> dY_data = random.Uniform<float>(dY_dims, -10.0f, 10.0f);
  std::vector<float> prob_data = random.Uniform<float>(prob_dims, -10.0f, 10.0f);
  std::vector<int64_t> index_data = random.Uniform<int64_t>(index_dims, 0.0f, static_cast<float>(dX_dims.back()));

  test.AddInput<float>("dY", dY_dims, dY_data);
  test.AddInput<float>("prob", prob_dims, prob_data);
  test.AddInput<int64_t>("index", index_dims, index_data);

  std::vector<float> dX_data = FillZeros<float>(dX_dims);

  test.AddOutput<float>("dX", dX_dims, dX_data);

  test.CompareWithCPU(kHipExecutionProvider);
}

TEST(HipKernelTest, SparseSoftmaxCrossEntropyGrad_TinySizeTensor) {
  std::vector<int64_t> dY_dims{};
  std::vector<int64_t> prob_dims{8, 2};
  std::vector<int64_t> index_dims{8};
  std::vector<int64_t> dX_dims{8, 2};
  TestSparseSoftmaxCrossEntropyGrad(dY_dims, prob_dims, index_dims, dX_dims, "mean");
  TestSparseSoftmaxCrossEntropyGrad(dY_dims, prob_dims, index_dims, dX_dims, "sum");
}

TEST(HipKernelTest, SparseSoftmaxCrossEntropyGrad_SmallSizeTensor) {
  std::vector<int64_t> dY_dims{};
  std::vector<int64_t> prob_dims{8, 20, 10};
  std::vector<int64_t> index_dims{8, 20};
  std::vector<int64_t> dX_dims{8, 20, 10};
  TestSparseSoftmaxCrossEntropyGrad(dY_dims, prob_dims, index_dims, dX_dims, "mean");
  TestSparseSoftmaxCrossEntropyGrad(dY_dims, prob_dims, index_dims, dX_dims, "sum");
}

TEST(HipKernelTest, SparseSoftmaxCrossEntropyGrad_LargeSizeTensor) {
  std::vector<int64_t> dY_dims{};
  std::vector<int64_t> prob_dims{2, 512, 30528};
  std::vector<int64_t> index_dims{2, 512};
  std::vector<int64_t> dX_dims{8, 512, 30528};
  TestSparseSoftmaxCrossEntropyGrad(dY_dims, prob_dims, index_dims, dX_dims, "mean");
  TestSparseSoftmaxCrossEntropyGrad(dY_dims, prob_dims, index_dims, dX_dims, "sum");
}
}  // namespace test
}  // namespace onnxruntime
