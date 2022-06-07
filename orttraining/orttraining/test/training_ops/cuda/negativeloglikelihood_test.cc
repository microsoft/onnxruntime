// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/providers/compare_provider_test_utils.h"

using namespace std;

namespace onnxruntime {
namespace test {

#if USE_CUDA
constexpr const char* kGpuExecutionProvider = kCudaExecutionProvider;
#elif USE_ROCM
constexpr const char* kGpuExecutionProvider = kRocmExecutionProvider;
#endif

static void TestNegativeLogLikelihoodLoss(CompareOpTester& test, const std::vector<int64_t>* X_dims,
                                          const std::vector<int64_t>* index_dims,
                                          const std::vector<int64_t>* weight_dims, const std::vector<int64_t>* Y_dims,
                                          const std::string& reduction, const std::int64_t ignore_index,
                                          const bool is_internal_op) {
  test.AddAttribute("reduction", reduction);
  if (!is_internal_op) {
    test.AddAttribute("ignore_index", ignore_index);
  }

  // create rand inputs
  RandomValueGenerator random{};
  std::vector<float> X_data = random.Uniform<float>(*X_dims, -200.0f, 200.0f);
  std::vector<int64_t> index_data = random.Uniform<int64_t>(*index_dims, 0, (*X_dims)[1]);
  // Add one data point that has ignore_index.
  if (index_data.size() > 0) {
    index_data[0] = ignore_index;
  }

  test.AddInput<float>("X", *X_dims, X_data);
  test.AddInput<int64_t>("index", *index_dims, index_data);

  if (weight_dims) {
    std::vector<float> weight_data = random.Uniform<float>(*weight_dims, 0.0f, 1.0f);
    test.AddInput<float>("weight", *weight_dims, weight_data);
  }

  if (is_internal_op && ignore_index != -1) {
    test.AddInput<int64_t>("ignore_index", {}, &ignore_index, 1);
  }

  std::vector<float> Y_data = FillZeros<float>(*Y_dims);
  test.AddOutput<float>("output", *Y_dims, Y_data);

  std::unordered_map<std::string, int> extra_domain_to_version;
  if (is_internal_op) {
    extra_domain_to_version[onnxruntime::kOnnxDomain] = 12;
  }

  test.CompareWithCPU(kGpuExecutionProvider, 1e-4, 1e-4, false, extra_domain_to_version);
}

static void TestNegativeLogLikelihoodLoss(const std::vector<int64_t>* X_dims, const std::vector<int64_t>* index_dims,
                                          const std::vector<int64_t>* weight_dims, const std::vector<int64_t>* Y_dims,
                                          const std::string& reduction, const std::int64_t ignore_index = -1) {
  CompareOpTester test("NegativeLogLikelihoodLoss", 12, onnxruntime::kOnnxDomain);
  TestNegativeLogLikelihoodLoss(test, X_dims, index_dims, weight_dims, Y_dims, reduction, ignore_index, false);

  // Can we add a empty optional input before a non-empty input?
  if (weight_dims || ignore_index == -1) {
    CompareOpTester test_internal("NegativeLogLikelihoodLossInternal", 1, onnxruntime::kMSDomain);
    TestNegativeLogLikelihoodLoss(test_internal, X_dims, index_dims, weight_dims, Y_dims, reduction, ignore_index,
                                  true);
  }
}

TEST(CudaKernelTest, NegativeLogLikelihoodLoss_TinySizeTensor) {
  std::vector<int64_t> X_dims{8, 2};
  std::vector<int64_t> index_dims{8};
  std::vector<int64_t> weight_dims{2};
  std::vector<int64_t> Y_dims{};
  std::vector<int64_t> Y_dims_none{8};
  TestNegativeLogLikelihoodLoss(&X_dims, &index_dims, &weight_dims, &Y_dims, "mean");
  TestNegativeLogLikelihoodLoss(&X_dims, &index_dims, nullptr, &Y_dims, "mean");
  TestNegativeLogLikelihoodLoss(&X_dims, &index_dims, &weight_dims, &Y_dims, "sum");
  TestNegativeLogLikelihoodLoss(&X_dims, &index_dims, nullptr, &Y_dims, "sum");
  TestNegativeLogLikelihoodLoss(&X_dims, &index_dims, &weight_dims, &Y_dims_none, "none");
  TestNegativeLogLikelihoodLoss(&X_dims, &index_dims, nullptr, &Y_dims_none, "none");

  // Just test ignore_index for small tensor because it will increase test time a lot with little verification gain.
  TestNegativeLogLikelihoodLoss(&X_dims, &index_dims, &weight_dims, &Y_dims, "mean", 0);
  TestNegativeLogLikelihoodLoss(&X_dims, &index_dims, nullptr, &Y_dims, "mean", 0);
  TestNegativeLogLikelihoodLoss(&X_dims, &index_dims, &weight_dims, &Y_dims, "sum", 0);
  TestNegativeLogLikelihoodLoss(&X_dims, &index_dims, nullptr, &Y_dims, "sum", 0);
  TestNegativeLogLikelihoodLoss(&X_dims, &index_dims, &weight_dims, &Y_dims_none, "none", 0);
  TestNegativeLogLikelihoodLoss(&X_dims, &index_dims, nullptr, &Y_dims_none, "none", 0);
}

TEST(CudaKernelTest, NegativeLogLikelihoodLoss_SmallSizeTensor) {
  std::vector<int64_t> X_dims{8, 20, 10};
  std::vector<int64_t> index_dims{8, 10};
  std::vector<int64_t> weight_dims{20};
  std::vector<int64_t> Y_dims{};
  std::vector<int64_t> Y_dims_none{8, 10};
  TestNegativeLogLikelihoodLoss(&X_dims, &index_dims, &weight_dims, &Y_dims, "mean");
  TestNegativeLogLikelihoodLoss(&X_dims, &index_dims, nullptr, &Y_dims, "mean");
  TestNegativeLogLikelihoodLoss(&X_dims, &index_dims, &weight_dims, &Y_dims, "sum");
  TestNegativeLogLikelihoodLoss(&X_dims, &index_dims, nullptr, &Y_dims, "sum");
  TestNegativeLogLikelihoodLoss(&X_dims, &index_dims, &weight_dims, &Y_dims_none, "none");
  TestNegativeLogLikelihoodLoss(&X_dims, &index_dims, nullptr, &Y_dims_none, "none");
}

TEST(CudaKernelTest, NegativeLogLikelihoodLoss_MediumSizeTensor) {
  std::vector<int64_t> X_dims{8, 1024};
  std::vector<int64_t> index_dims{8};
  std::vector<int64_t> weight_dims{1024};
  std::vector<int64_t> Y_dims{};
  std::vector<int64_t> Y_dims_none{8};
  TestNegativeLogLikelihoodLoss(&X_dims, &index_dims, &weight_dims, &Y_dims, "mean");
  TestNegativeLogLikelihoodLoss(&X_dims, &index_dims, nullptr, &Y_dims, "mean");
  TestNegativeLogLikelihoodLoss(&X_dims, &index_dims, &weight_dims, &Y_dims, "sum");
  TestNegativeLogLikelihoodLoss(&X_dims, &index_dims, nullptr, &Y_dims, "sum");
  TestNegativeLogLikelihoodLoss(&X_dims, &index_dims, &weight_dims, &Y_dims_none, "none");
  TestNegativeLogLikelihoodLoss(&X_dims, &index_dims, nullptr, &Y_dims_none, "none");
}

TEST(CudaKernelTest, DISABLED_NegativeLogLikelihoodLoss_LargeSizeTensor) {
  std::vector<int64_t> X_dims{4, 512, 30528};
  std::vector<int64_t> index_dims{4, 30528};
  std::vector<int64_t> weight_dims{512};
  std::vector<int64_t> Y_dims{};
  std::vector<int64_t> Y_dims_none{4, 30528};
  TestNegativeLogLikelihoodLoss(&X_dims, &index_dims, &weight_dims, &Y_dims, "mean");
  TestNegativeLogLikelihoodLoss(&X_dims, &index_dims, nullptr, &Y_dims, "mean");
  TestNegativeLogLikelihoodLoss(&X_dims, &index_dims, &weight_dims, &Y_dims, "sum");
  TestNegativeLogLikelihoodLoss(&X_dims, &index_dims, nullptr, &Y_dims, "sum");
  TestNegativeLogLikelihoodLoss(&X_dims, &index_dims, &weight_dims, &Y_dims_none, "none");
  TestNegativeLogLikelihoodLoss(&X_dims, &index_dims, nullptr, &Y_dims_none, "none");
}

}  // namespace test
}  // namespace onnxruntime
