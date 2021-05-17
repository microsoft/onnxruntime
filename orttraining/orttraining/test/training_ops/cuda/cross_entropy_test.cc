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

static void TestSoftmaxCrossEntropy(const std::vector<int64_t>& X_dims,
                                    const std::vector<int64_t>& label_dims,
                                    const std::vector<int64_t>& Y_dims,
                                    const std::vector<int64_t>& log_prob_dims,
                                    const std::string& reduction) {
  CompareOpTester test("SoftmaxCrossEntropy", 1, kMSDomain);
  test.AddAttribute("reduction", reduction);

  // create rand inputs
  RandomValueGenerator random{};
  std::vector<float> X_data = random.Uniform<float>(X_dims, -200.0f, 200.0f);
  std::vector<float> label_data = random.OneHot<float>(label_dims, label_dims.back());

  test.AddInput<float>("X", X_dims, X_data);
  test.AddInput<float>("label", label_dims, label_data);

  std::vector<float> Y_data = FillZeros<float>(Y_dims);
  std::vector<float> log_prob_data = FillZeros<float>(log_prob_dims);

  test.AddOutput<float>("output", Y_dims, Y_data);
  test.AddOutput<float>("log_prob", log_prob_dims, log_prob_data);

  test.CompareWithCPU(kGpuExecutionProvider);
}

static void TestSoftmaxCrossEntropyGrad(const std::vector<int64_t>& dY_dims,
                                        const std::vector<int64_t>& log_prob_dims,
                                        const std::vector<int64_t>& label_dims,
                                        const std::vector<int64_t>& dX_dims,
                                        const std::string& reduction) {
  CompareOpTester test("SoftmaxCrossEntropyGrad", 1, kMSDomain);
  test.AddAttribute("reduction", reduction);

  // create rand inputs
  RandomValueGenerator random{};
  std::vector<float> dY_data = random.Uniform<float>(dY_dims, -10.0f, 10.0f);
  std::vector<float> log_prob_data = random.Uniform<float>(log_prob_dims, -10.0f, 10.0f);
  std::vector<float> label_data = random.Uniform<float>(label_dims, 0.0f, 1.0f);

  test.AddInput<float>("dY", dY_dims, dY_data);
  test.AddInput<float>("log_prob", log_prob_dims, log_prob_data);
  test.AddInput<float>("label", label_dims, label_data);

  std::vector<float> dX_data = FillZeros<float>(dX_dims);

  test.AddOutput<float>("dX", dX_dims, dX_data);

  test.CompareWithCPU(kGpuExecutionProvider);
}

TEST(CudaKernelTest, SoftmaxCrossEntropy_TinySizeTensor) {
  std::vector<int64_t> X_dims{8, 2};
  std::vector<int64_t> label_dims{8, 2};
  std::vector<int64_t> Y_dims{};
  std::vector<int64_t> log_prob_dims{8, 2};
  TestSoftmaxCrossEntropy(X_dims, label_dims, Y_dims, log_prob_dims, "mean");
  TestSoftmaxCrossEntropy(X_dims, label_dims, Y_dims, log_prob_dims, "sum");
}

TEST(CudaKernelTest, SoftmaxCrossEntropy_SmallSizeTensor) {
  std::vector<int64_t> X_dims{8, 20, 10};
  std::vector<int64_t> label_dims{8, 20, 10};
  std::vector<int64_t> Y_dims{};
  std::vector<int64_t> log_prob_dims{8, 20, 10};
  TestSoftmaxCrossEntropy(X_dims, label_dims, Y_dims, log_prob_dims, "mean");
  TestSoftmaxCrossEntropy(X_dims, label_dims, Y_dims, log_prob_dims, "sum");
}

TEST(CudaKernelTest, SoftmaxCrossEntropy_MediumSizeTensor) {
  std::vector<int64_t> X_dims{7, 1024};
  std::vector<int64_t> label_dims{7, 1024};
  std::vector<int64_t> Y_dims{};
  std::vector<int64_t> log_prob_dims{7, 1024};
  TestSoftmaxCrossEntropy(X_dims, label_dims, Y_dims, log_prob_dims, "mean");
  TestSoftmaxCrossEntropy(X_dims, label_dims, Y_dims, log_prob_dims, "sum");
}

TEST(CudaKernelTest, SoftmaxCrossEntropy_LargeSizeTensor) {
  std::vector<int64_t> X_dims{2, 512, 30528};
  std::vector<int64_t> label_dims{2, 512, 30528};
  std::vector<int64_t> Y_dims{};
  std::vector<int64_t> log_prob_dims{2, 512, 30528};
  TestSoftmaxCrossEntropy(X_dims, label_dims, Y_dims, log_prob_dims, "mean");
  TestSoftmaxCrossEntropy(X_dims, label_dims, Y_dims, log_prob_dims, "sum");
}

TEST(CudaKernelTest, SoftmaxCrossEntropyGrad_TinySizeTensor) {
  std::vector<int64_t> dY_dims{};
  std::vector<int64_t> log_prob_dims{8, 2};
  std::vector<int64_t> label_dims{8, 2};
  std::vector<int64_t> dX_dims{8, 2};
  TestSoftmaxCrossEntropyGrad(dY_dims, log_prob_dims, label_dims, dX_dims, "mean");
  TestSoftmaxCrossEntropyGrad(dY_dims, log_prob_dims, label_dims, dX_dims, "sum");
}

TEST(CudaKernelTest, SoftmaxCrossEntropyGrad_SmallSizeTensor) {
  std::vector<int64_t> dY_dims{};
  std::vector<int64_t> log_prob_dims{8, 20, 10};
  std::vector<int64_t> label_dims{8, 20, 10};
  std::vector<int64_t> dX_dims{8, 20, 10};
  TestSoftmaxCrossEntropyGrad(dY_dims, log_prob_dims, label_dims, dX_dims, "mean");
  TestSoftmaxCrossEntropyGrad(dY_dims, log_prob_dims, label_dims, dX_dims, "sum");
}

TEST(CudaKernelTest, SoftmaxCrossEntropyGrad_LargeSizeTensor) {
  std::vector<int64_t> dY_dims{};
  std::vector<int64_t> log_prob_dims{2, 512, 30528};
  std::vector<int64_t> label_dims{2, 512, 30528};
  std::vector<int64_t> dX_dims{2, 512, 30528};
  TestSoftmaxCrossEntropyGrad(dY_dims, log_prob_dims, label_dims, dX_dims, "mean");
  TestSoftmaxCrossEntropyGrad(dY_dims, log_prob_dims, label_dims, dX_dims, "sum");
}

TEST(CudaKernelTest, SparseSoftmaxCrossEntropy_Basic) {
  OpTester test("SparseSoftmaxCrossEntropy", 9);
  test.AddAttribute("reduction", "mean");

  std::vector<float> X_data{-0.9468f, 1.3250f, 1.0438f, 0.4106f, -0.2150f,
                            -0.3399f, -0.4396f, 1.1835f, 1.2089f, -1.0617f,
                            -0.5239f, -0.2767f, 0.9910f, -1.5688f, -0.2863f};
  std::vector<int64_t> index_data = {3, 4, 1};
  std::vector<float> Y_data = {2.2956f};
  std::vector<float> log_prob_data = {-3.1773f, -0.9054f, -1.1867f, -1.8199f, -2.4454f,
                                      -2.4583f, -2.5580f, -0.9349f, -0.9094f, -3.1800f,
                                      -2.1341f, -1.8869f, -0.6192f, -3.1789f, -1.8965f};

  test.AddInput<float>("X", {3, 5}, X_data);
  test.AddInput<int64_t>("index", {3}, index_data);
  test.AddOutput<float>("output", {}, Y_data);
  test.AddOutput<float>("log_prob", {3, 5}, log_prob_data);

  test.Run();
}

static void TestSparseSoftmaxCrossEntropy(const std::vector<int64_t>* X_dims,
                                          const std::vector<int64_t>* index_dims,
                                          const std::vector<int64_t>* weight_dims,
                                          const std::vector<int64_t>* Y_dims,
                                          const std::vector<int64_t>* log_prob_dims,
                                          const std::string& reduction) {
  CompareOpTester test("SparseSoftmaxCrossEntropy");
  test.AddAttribute("reduction", reduction);

  // create rand inputs
  RandomValueGenerator random{};
  std::vector<float> X_data = random.Uniform<float>(*X_dims, -200.0f, 200.0f);
  std::vector<int64_t> index_data = random.Uniform<int64_t>(*index_dims, 0, X_dims->back());

  test.AddInput<float>("X", *X_dims, X_data);
  test.AddInput<int64_t>("index", *index_dims, index_data);

  if (weight_dims) {
    std::vector<float> weight_data = random.Uniform<float>(*weight_dims, 0.0f, 1.0f);
    test.AddInput<float>("weight", *weight_dims, weight_data);
  }

  std::vector<float> Y_data = FillZeros<float>(*Y_dims);
  std::vector<float> log_prob_data = FillZeros<float>(*log_prob_dims);

  test.AddOutput<float>("output", *Y_dims, Y_data);
  test.AddOutput<float>("log_prob", *log_prob_dims, log_prob_data);

  test.CompareWithCPU(kGpuExecutionProvider);
}

TEST(CudaKernelTest, SparseSoftmaxCrossEntropy_TinySizeTensor) {
  std::vector<int64_t> X_dims{8, 2};
  std::vector<int64_t> index_dims{8};
  std::vector<int64_t> weight_dims{8};
  std::vector<int64_t> Y_dims{};
  std::vector<int64_t> log_prob_dims{8, 2};
  TestSparseSoftmaxCrossEntropy(&X_dims, &index_dims, &weight_dims, &Y_dims, &log_prob_dims, "mean");
  TestSparseSoftmaxCrossEntropy(&X_dims, &index_dims, nullptr, &Y_dims, &log_prob_dims, "mean");
  TestSparseSoftmaxCrossEntropy(&X_dims, &index_dims, &weight_dims, &Y_dims, &log_prob_dims, "sum");
  TestSparseSoftmaxCrossEntropy(&X_dims, &index_dims, nullptr, &Y_dims, &log_prob_dims, "sum");
}

TEST(CudaKernelTest, SparseSoftmaxCrossEntropy_SmallSizeTensor) {
  std::vector<int64_t> X_dims{8, 20, 10};
  std::vector<int64_t> index_dims{8, 20};
  std::vector<int64_t> weight_dims{8, 20};
  std::vector<int64_t> Y_dims{};
  std::vector<int64_t> log_prob_dims{8, 20, 10};
  TestSparseSoftmaxCrossEntropy(&X_dims, &index_dims, &weight_dims, &Y_dims, &log_prob_dims, "mean");
  TestSparseSoftmaxCrossEntropy(&X_dims, &index_dims, nullptr, &Y_dims, &log_prob_dims, "mean");
  TestSparseSoftmaxCrossEntropy(&X_dims, &index_dims, &weight_dims, &Y_dims, &log_prob_dims, "sum");
  TestSparseSoftmaxCrossEntropy(&X_dims, &index_dims, nullptr, &Y_dims, &log_prob_dims, "sum");
}

TEST(CudaKernelTest, SparseSoftmaxCrossEntropy_MediumSizeTensor) {
  std::vector<int64_t> X_dims{8, 1024};
  std::vector<int64_t> index_dims{8};
  std::vector<int64_t> weight_dims{8};
  std::vector<int64_t> Y_dims{};
  std::vector<int64_t> log_prob_dims{8, 1024};
  TestSparseSoftmaxCrossEntropy(&X_dims, &index_dims, &weight_dims, &Y_dims, &log_prob_dims, "mean");
  TestSparseSoftmaxCrossEntropy(&X_dims, &index_dims, nullptr, &Y_dims, &log_prob_dims, "mean");
  TestSparseSoftmaxCrossEntropy(&X_dims, &index_dims, &weight_dims, &Y_dims, &log_prob_dims, "sum");
  TestSparseSoftmaxCrossEntropy(&X_dims, &index_dims, nullptr, &Y_dims, &log_prob_dims, "sum");
}

TEST(CudaKernelTest, SparseSoftmaxCrossEntropy_LargeSizeTensor) {
  std::vector<int64_t> X_dims{4, 512, 30528};
  std::vector<int64_t> index_dims{4, 512};
  std::vector<int64_t> weight_dims{4, 512};
  std::vector<int64_t> Y_dims{};
  std::vector<int64_t> log_prob_dims{4, 512, 30528};
  TestSparseSoftmaxCrossEntropy(&X_dims, &index_dims, &weight_dims, &Y_dims, &log_prob_dims, "mean");
  TestSparseSoftmaxCrossEntropy(&X_dims, &index_dims, nullptr, &Y_dims, &log_prob_dims, "mean");
  TestSparseSoftmaxCrossEntropy(&X_dims, &index_dims, &weight_dims, &Y_dims, &log_prob_dims, "sum");
  TestSparseSoftmaxCrossEntropy(&X_dims, &index_dims, nullptr, &Y_dims, &log_prob_dims, "sum");
}

static void TestSparseSoftmaxCrossEntropyGrad(const std::vector<int64_t>& dY_dims,
                                              const std::vector<int64_t>& log_prob_dims,
                                              const std::vector<int64_t>& index_dims,
                                              const std::vector<int64_t>& dX_dims,
                                              const std::string& reduction) {
  CompareOpTester test("SparseSoftmaxCrossEntropyGrad");
  test.AddAttribute("reduction", reduction);

  // create rand inputs
  RandomValueGenerator random{};
  std::vector<float> dY_data = random.Uniform<float>(dY_dims, -10.0f, 10.0f);
  std::vector<float> log_prob_data = random.Uniform<float>(log_prob_dims, -10.0f, 10.0f);
  std::vector<int64_t> index_data = random.Uniform<int64_t>(index_dims, 0, dX_dims.back());

  test.AddInput<float>("dY", dY_dims, dY_data);
  test.AddInput<float>("log_prob", log_prob_dims, log_prob_data);
  test.AddInput<int64_t>("index", index_dims, index_data);

  std::vector<float> dX_data = FillZeros<float>(dX_dims);

  test.AddOutput<float>("dX", dX_dims, dX_data);

  test.CompareWithCPU(kGpuExecutionProvider);
}

TEST(CudaKernelTest, SparseSoftmaxCrossEntropyGrad_TinySizeTensor) {
  std::vector<int64_t> dY_dims{};
  std::vector<int64_t> log_prob_dims{8, 2};
  std::vector<int64_t> index_dims{8};
  std::vector<int64_t> dX_dims{8, 2};
  TestSparseSoftmaxCrossEntropyGrad(dY_dims, log_prob_dims, index_dims, dX_dims, "mean");
  TestSparseSoftmaxCrossEntropyGrad(dY_dims, log_prob_dims, index_dims, dX_dims, "sum");
}

TEST(CudaKernelTest, SparseSoftmaxCrossEntropyGrad_SmallSizeTensor) {
  std::vector<int64_t> dY_dims{};
  std::vector<int64_t> log_prob_dims{8, 20, 10};
  std::vector<int64_t> index_dims{8, 20};
  std::vector<int64_t> dX_dims{8, 20, 10};
  TestSparseSoftmaxCrossEntropyGrad(dY_dims, log_prob_dims, index_dims, dX_dims, "mean");
  TestSparseSoftmaxCrossEntropyGrad(dY_dims, log_prob_dims, index_dims, dX_dims, "sum");
}

TEST(CudaKernelTest, SparseSoftmaxCrossEntropyGrad_LargeSizeTensor) {
  std::vector<int64_t> dY_dims{};
  std::vector<int64_t> log_prob_dims{2, 512, 30528};
  std::vector<int64_t> index_dims{2, 512};
  std::vector<int64_t> dX_dims{2, 512, 30528};
  TestSparseSoftmaxCrossEntropyGrad(dY_dims, log_prob_dims, index_dims, dX_dims, "mean");
  TestSparseSoftmaxCrossEntropyGrad(dY_dims, log_prob_dims, index_dims, dX_dims, "sum");
}

static void TestSoftmaxCrossEntropyLoss(const std::vector<int64_t>* X_dims,
                                        const std::vector<int64_t>* index_dims,
                                        const std::vector<int64_t>* weight_dims,
                                        const std::vector<int64_t>* Y_dims,
                                        const std::vector<int64_t>* log_prob_dims,
                                        const std::string& reduction,
                                        const std::int64_t ignore_index = -1,
                                        const bool test_fp16 = false,
                                        const double error_tolerance = 1e-4) {
  CompareOpTester test("SoftmaxCrossEntropyLoss", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute("reduction", reduction);
  test.AddAttribute("ignore_index", ignore_index);

  // create rand inputs
  RandomValueGenerator random{};
  std::vector<float> X_data = random.Uniform<float>(*X_dims, -200.0f, 200.0f);
  std::vector<int64_t> index_data = random.Uniform<int64_t>(*index_dims, 0, (*X_dims)[1]);
  //Add one data point that has ignore_index.
  if (index_data.size() > 0) {
    index_data[0] = ignore_index;
  }
  if (test_fp16) {
    std::vector<MLFloat16> X_data_half(X_data.size());
    ConvertFloatToMLFloat16(X_data.data(), X_data_half.data(), int(X_data.size()));
    test.AddInput<MLFloat16>("X", *X_dims, X_data_half);
  } else {
    test.AddInput<float>("X", *X_dims, X_data);
  }

  test.AddInput<int64_t>("index", *index_dims, index_data);

  if (weight_dims) {
    std::vector<float> weight_data = random.Uniform<float>(*weight_dims, 0.0f, 1.0f);
    if (test_fp16) {
      std::vector<MLFloat16> weight_data_half(weight_data.size());
      ConvertFloatToMLFloat16(weight_data.data(), weight_data_half.data(), int(weight_data.size()));
      test.AddInput<MLFloat16>("weight", *weight_dims, weight_data_half);
    } else {
      test.AddInput<float>("weight", *weight_dims, weight_data);
    }
  }

  if (test_fp16) {
    std::vector<MLFloat16> Y_data = FillZeros<MLFloat16>(*Y_dims);
    test.AddOutput<MLFloat16>("output", *Y_dims, Y_data);

    if (log_prob_dims) {
      std::vector<MLFloat16> log_prob_data = FillZeros<MLFloat16>(*log_prob_dims);
      test.AddOutput<MLFloat16>("log_prob", *log_prob_dims, log_prob_data);
    }

    test.CompareWithCPU(kGpuExecutionProvider, error_tolerance, error_tolerance, true);
  } else {
    std::vector<float> Y_data = FillZeros<float>(*Y_dims);
    test.AddOutput<float>("output", *Y_dims, Y_data);

    if (log_prob_dims) {
      std::vector<float> log_prob_data = FillZeros<float>(*log_prob_dims);
      test.AddOutput<float>("log_prob", *log_prob_dims, log_prob_data);
    }
    test.CompareWithCPU(kGpuExecutionProvider);
  }
}

TEST(CudaKernelTest, SoftmaxCrossEntropyLoss_TinySizeTensor) {
  std::vector<int64_t> X_dims{8, 2};
  std::vector<int64_t> index_dims{8};
  std::vector<int64_t> weight_dims{2};
  std::vector<int64_t> Y_dims{};
  std::vector<int64_t> Y_dims_none{8};
  std::vector<int64_t> log_prob_dims{8, 2};
  TestSoftmaxCrossEntropyLoss(&X_dims, &index_dims, &weight_dims, &Y_dims, &log_prob_dims, "mean");
  TestSoftmaxCrossEntropyLoss(&X_dims, &index_dims, nullptr, &Y_dims, &log_prob_dims, "mean");
  TestSoftmaxCrossEntropyLoss(&X_dims, &index_dims, &weight_dims, &Y_dims, &log_prob_dims, "sum");
  TestSoftmaxCrossEntropyLoss(&X_dims, &index_dims, nullptr, &Y_dims, &log_prob_dims, "sum");
  TestSoftmaxCrossEntropyLoss(&X_dims, &index_dims, &weight_dims, &Y_dims_none, &log_prob_dims, "none");
  TestSoftmaxCrossEntropyLoss(&X_dims, &index_dims, nullptr, &Y_dims_none, &log_prob_dims, "none");

  // Just test ignore_index for small tensor because it will increase test time a lot with little verification gain.
  TestSoftmaxCrossEntropyLoss(&X_dims, &index_dims, &weight_dims, &Y_dims, &log_prob_dims, "mean", 0);
  TestSoftmaxCrossEntropyLoss(&X_dims, &index_dims, nullptr, &Y_dims, &log_prob_dims, "mean", 0);
  TestSoftmaxCrossEntropyLoss(&X_dims, &index_dims, &weight_dims, &Y_dims, &log_prob_dims, "sum", 0);
  TestSoftmaxCrossEntropyLoss(&X_dims, &index_dims, nullptr, &Y_dims, &log_prob_dims, "sum", 0);
  TestSoftmaxCrossEntropyLoss(&X_dims, &index_dims, &weight_dims, &Y_dims_none, &log_prob_dims, "none", 0);
  TestSoftmaxCrossEntropyLoss(&X_dims, &index_dims, nullptr, &Y_dims_none, &log_prob_dims, "none", 0);
}

TEST(CudaKernelTest, SoftmaxCrossEntropyLoss_TinySizeTensor_half) {
  std::vector<int64_t> X_dims{8, 2};
  std::vector<int64_t> index_dims{8};
  std::vector<int64_t> weight_dims{2};
  std::vector<int64_t> Y_dims{};
  std::vector<int64_t> Y_dims_none{8};
  std::vector<int64_t> log_prob_dims{8, 2};
  TestSoftmaxCrossEntropyLoss(&X_dims, &index_dims, &weight_dims, &Y_dims, &log_prob_dims, "mean", -1, true, 5e-2);
  TestSoftmaxCrossEntropyLoss(&X_dims, &index_dims, nullptr, &Y_dims, &log_prob_dims, "mean", -1, true, 5e-2);
  TestSoftmaxCrossEntropyLoss(&X_dims, &index_dims, &weight_dims, &Y_dims, &log_prob_dims, "sum", -1, true, 5e-2);
  TestSoftmaxCrossEntropyLoss(&X_dims, &index_dims, nullptr, &Y_dims, &log_prob_dims, "sum", -1, true, 5e-2);
  TestSoftmaxCrossEntropyLoss(&X_dims, &index_dims, &weight_dims, &Y_dims_none, &log_prob_dims, "none", -1, true, 5e-2);
  TestSoftmaxCrossEntropyLoss(&X_dims, &index_dims, nullptr, &Y_dims_none, &log_prob_dims, "none", -1, true, 5e-2);

  // Just test ignore_index for small tensor because it will increase test time a lot with little verification gain.
  TestSoftmaxCrossEntropyLoss(&X_dims, &index_dims, &weight_dims, &Y_dims, &log_prob_dims, "mean", 0, true, 5e-2);
  TestSoftmaxCrossEntropyLoss(&X_dims, &index_dims, nullptr, &Y_dims, &log_prob_dims, "mean", 0, true, 5e-2);
  TestSoftmaxCrossEntropyLoss(&X_dims, &index_dims, &weight_dims, &Y_dims, &log_prob_dims, "sum", 0, true, 5e-2);
  TestSoftmaxCrossEntropyLoss(&X_dims, &index_dims, nullptr, &Y_dims, &log_prob_dims, "sum", 0, true, 5e-2);
  TestSoftmaxCrossEntropyLoss(&X_dims, &index_dims, &weight_dims, &Y_dims_none, &log_prob_dims, "none", 0, true, 5e-2);
  TestSoftmaxCrossEntropyLoss(&X_dims, &index_dims, nullptr, &Y_dims_none, &log_prob_dims, "none", 0, true, 5e-2);
}

TEST(CudaKernelTest, SoftmaxCrossEntropyLoss_SmallSizeTensor) {
  std::vector<int64_t> X_dims{8, 20, 10};
  std::vector<int64_t> index_dims{8, 10};
  std::vector<int64_t> weight_dims{20};
  std::vector<int64_t> Y_dims{};
  std::vector<int64_t> Y_dims_none{8, 10};
  std::vector<int64_t> log_prob_dims{8, 20, 10};
  TestSoftmaxCrossEntropyLoss(&X_dims, &index_dims, &weight_dims, &Y_dims, &log_prob_dims, "mean");
  TestSoftmaxCrossEntropyLoss(&X_dims, &index_dims, nullptr, &Y_dims, &log_prob_dims, "mean");
  TestSoftmaxCrossEntropyLoss(&X_dims, &index_dims, &weight_dims, &Y_dims, &log_prob_dims, "sum");
  TestSoftmaxCrossEntropyLoss(&X_dims, &index_dims, nullptr, &Y_dims, &log_prob_dims, "sum");
  TestSoftmaxCrossEntropyLoss(&X_dims, &index_dims, &weight_dims, &Y_dims_none, &log_prob_dims, "none");
  TestSoftmaxCrossEntropyLoss(&X_dims, &index_dims, nullptr, &Y_dims_none, &log_prob_dims, "none");
}

TEST(CudaKernelTest, SoftmaxCrossEntropyLoss_SmallSizeTensor_half) {
  std::vector<int64_t> X_dims{8, 20, 10};
  std::vector<int64_t> index_dims{8, 10};
  std::vector<int64_t> weight_dims{20};
  std::vector<int64_t> Y_dims{};
  std::vector<int64_t> Y_dims_none{8, 10};
  std::vector<int64_t> log_prob_dims{8, 20, 10};
  TestSoftmaxCrossEntropyLoss(&X_dims, &index_dims, &weight_dims, &Y_dims, &log_prob_dims, "mean", -1, true, 5e-2);
  TestSoftmaxCrossEntropyLoss(&X_dims, &index_dims, nullptr, &Y_dims, &log_prob_dims, "mean", -1, true, 5e-2);
  TestSoftmaxCrossEntropyLoss(&X_dims, &index_dims, &weight_dims, &Y_dims, &log_prob_dims, "sum", -1, true, 5e-2);
  TestSoftmaxCrossEntropyLoss(&X_dims, &index_dims, nullptr, &Y_dims, &log_prob_dims, "sum", -1, true, 5e-2);
  TestSoftmaxCrossEntropyLoss(&X_dims, &index_dims, &weight_dims, &Y_dims_none, &log_prob_dims, "none", -1, true, 5e-2);
  TestSoftmaxCrossEntropyLoss(&X_dims, &index_dims, nullptr, &Y_dims_none, &log_prob_dims, "none", -1, true, 5e-2);
}

TEST(CudaKernelTest, SoftmaxCrossEntropyLoss_MediumSizeTensor) {
  std::vector<int64_t> X_dims{8, 1024};
  std::vector<int64_t> index_dims{8};
  std::vector<int64_t> weight_dims{1024};
  std::vector<int64_t> Y_dims{};
  std::vector<int64_t> Y_dims_none{8};
  std::vector<int64_t> log_prob_dims{8, 1024};
  TestSoftmaxCrossEntropyLoss(&X_dims, &index_dims, &weight_dims, &Y_dims, &log_prob_dims, "mean");
  TestSoftmaxCrossEntropyLoss(&X_dims, &index_dims, nullptr, &Y_dims, &log_prob_dims, "mean");
  TestSoftmaxCrossEntropyLoss(&X_dims, &index_dims, &weight_dims, &Y_dims, &log_prob_dims, "sum");
  TestSoftmaxCrossEntropyLoss(&X_dims, &index_dims, nullptr, &Y_dims, &log_prob_dims, "sum");
  TestSoftmaxCrossEntropyLoss(&X_dims, &index_dims, &weight_dims, &Y_dims_none, &log_prob_dims, "none");
  TestSoftmaxCrossEntropyLoss(&X_dims, &index_dims, nullptr, &Y_dims_none, &log_prob_dims, "none");
}

TEST(CudaKernelTest, SoftmaxCrossEntropyLoss_MediumSizeTensor_half) {
  std::vector<int64_t> X_dims{8, 1024};
  std::vector<int64_t> index_dims{8};
  std::vector<int64_t> weight_dims{1024};
  std::vector<int64_t> Y_dims{};
  std::vector<int64_t> Y_dims_none{8};
  std::vector<int64_t> log_prob_dims{8, 1024};
  TestSoftmaxCrossEntropyLoss(&X_dims, &index_dims, &weight_dims, &Y_dims, &log_prob_dims, "mean", -1, true, 5e-2);
  TestSoftmaxCrossEntropyLoss(&X_dims, &index_dims, nullptr, &Y_dims, &log_prob_dims, "mean", -1, true, 5e-2);
  TestSoftmaxCrossEntropyLoss(&X_dims, &index_dims, &weight_dims, &Y_dims, &log_prob_dims, "sum", -1, true, 5e-2);
  TestSoftmaxCrossEntropyLoss(&X_dims, &index_dims, nullptr, &Y_dims, &log_prob_dims, "sum", -1, true, 5e-2);
  TestSoftmaxCrossEntropyLoss(&X_dims, &index_dims, &weight_dims, &Y_dims_none, &log_prob_dims, "none", -1, true, 5e-2);
  TestSoftmaxCrossEntropyLoss(&X_dims, &index_dims, nullptr, &Y_dims_none, &log_prob_dims, "none", -1, true, 5e-2);
}

// TODO fix flaky test
// failing random seed: 2873512643
TEST(CudaKernelTest, DISABLED_SoftmaxCrossEntropyLoss_LargeSizeTensor) {
  std::vector<int64_t> X_dims{4, 512, 30528};
  std::vector<int64_t> index_dims{4, 30528};
  std::vector<int64_t> weight_dims{512};
  std::vector<int64_t> Y_dims{};
  std::vector<int64_t> Y_dims_none{4, 30528};
  std::vector<int64_t> log_prob_dims{4, 512, 30528};
  TestSoftmaxCrossEntropyLoss(&X_dims, &index_dims, &weight_dims, &Y_dims, &log_prob_dims, "mean");
  TestSoftmaxCrossEntropyLoss(&X_dims, &index_dims, nullptr, &Y_dims, &log_prob_dims, "mean");
  TestSoftmaxCrossEntropyLoss(&X_dims, &index_dims, &weight_dims, &Y_dims, &log_prob_dims, "sum");
  TestSoftmaxCrossEntropyLoss(&X_dims, &index_dims, nullptr, &Y_dims, &log_prob_dims, "sum");
  TestSoftmaxCrossEntropyLoss(&X_dims, &index_dims, &weight_dims, &Y_dims_none, &log_prob_dims, "none");
  TestSoftmaxCrossEntropyLoss(&X_dims, &index_dims, nullptr, &Y_dims_none, &log_prob_dims, "none");
}

static void TestSoftmaxCrossEntropyLossGrad(const std::vector<int64_t>& dY_dims,
                                            const std::vector<int64_t>& log_prob_dims,
                                            const std::vector<int64_t>& index_dims,
                                            const std::vector<int64_t>& dX_dims,
                                            const std::string& reduction,
                                            const std::int64_t ignore_index = -1,
                                            const bool test_fp16 = false,
                                            const double error_tolerance = 1e-4) {
  CompareOpTester test("SoftmaxCrossEntropyLossGrad", 1, onnxruntime::kMSDomain);
  test.AddAttribute("reduction", reduction);
  test.AddAttribute("ignore_index", ignore_index);

  // create rand inputs
  RandomValueGenerator random{};
  std::vector<float> dY_data = random.Uniform<float>(dY_dims, -10.0f, 10.0f);
  std::vector<float> log_prob_data = random.Uniform<float>(log_prob_dims, -10.0f, 10.0f);
  std::vector<int64_t> index_data = random.Uniform<int64_t>(index_dims, 0, dX_dims[1]);
  //Add one data point that has ignore_index.
  if (index_data.size() > 0) {
    index_data[0] = ignore_index;
  }
  if (test_fp16) {
    std::vector<MLFloat16> dY_data_half(dY_data.size());
    ConvertFloatToMLFloat16(dY_data.data(), dY_data_half.data(), int(dY_data.size()));
    test.AddInput<MLFloat16>("dY", dY_dims, dY_data_half);

    std::vector<MLFloat16> log_prob_data_half(log_prob_data.size());
    ConvertFloatToMLFloat16(log_prob_data.data(), log_prob_data_half.data(), int(log_prob_data.size()));
    test.AddInput<MLFloat16>("log_prob", log_prob_dims, log_prob_data_half);

    test.AddInput<int64_t>("index", index_dims, index_data);

    std::vector<MLFloat16> dX_data = FillZeros<MLFloat16>(dX_dims);

    test.AddOutput<MLFloat16>("dX", dX_dims, dX_data);
    test.CompareWithCPU(kGpuExecutionProvider, error_tolerance, error_tolerance);
  } else {
    test.AddInput<float>("dY", dY_dims, dY_data);
    test.AddInput<float>("log_prob", log_prob_dims, log_prob_data);
    test.AddInput<int64_t>("index", index_dims, index_data);

    std::vector<float> dX_data = FillZeros<float>(dX_dims);

    test.AddOutput<float>("dX", dX_dims, dX_data);
    test.CompareWithCPU(kGpuExecutionProvider);
  }
}

TEST(CudaKernelTest, SoftmaxCrossEntropyLossGrad_TinySizeTensor) {
  std::vector<int64_t> dY_dims{};
  std::vector<int64_t> log_prob_dims{8, 2};
  std::vector<int64_t> index_dims{8};
  std::vector<int64_t> dX_dims{8, 2};
  TestSoftmaxCrossEntropyLossGrad(dY_dims, log_prob_dims, index_dims, dX_dims, "mean");
  TestSoftmaxCrossEntropyLossGrad(dY_dims, log_prob_dims, index_dims, dX_dims, "sum");
  TestSoftmaxCrossEntropyLossGrad({8}, log_prob_dims, index_dims, dX_dims, "none");

  // Just test ignore_index for small tensor because it will increase test time a lot with little verification gain.
  TestSoftmaxCrossEntropyLossGrad(dY_dims, log_prob_dims, index_dims, dX_dims, "mean", 0);
  TestSoftmaxCrossEntropyLossGrad(dY_dims, log_prob_dims, index_dims, dX_dims, "sum", 0);
  TestSoftmaxCrossEntropyLossGrad({8}, log_prob_dims, index_dims, dX_dims, "none", 0);
}

TEST(CudaKernelTest, SoftmaxCrossEntropyLossGrad_SmallSizeTensor) {
  std::vector<int64_t> dY_dims{};
  std::vector<int64_t> log_prob_dims{8, 20, 10};
  std::vector<int64_t> index_dims{8, 10};
  std::vector<int64_t> dX_dims{8, 20, 10};
  TestSoftmaxCrossEntropyLossGrad(dY_dims, log_prob_dims, index_dims, dX_dims, "mean");
  TestSoftmaxCrossEntropyLossGrad(dY_dims, log_prob_dims, index_dims, dX_dims, "sum");
  TestSoftmaxCrossEntropyLossGrad({8, 10}, log_prob_dims, index_dims, dX_dims, "none");
}

TEST(CudaKernelTest, SoftmaxCrossEntropyLossGrad_LargeSizeTensor) {
  std::vector<int64_t> dY_dims{};
  std::vector<int64_t> log_prob_dims{2, 512, 30528};
  std::vector<int64_t> index_dims{2, 30528};
  std::vector<int64_t> dX_dims{2, 512, 30528};
  TestSoftmaxCrossEntropyLossGrad(dY_dims, log_prob_dims, index_dims, dX_dims, "mean");
  TestSoftmaxCrossEntropyLossGrad(dY_dims, log_prob_dims, index_dims, dX_dims, "sum");
  TestSoftmaxCrossEntropyLossGrad({2, 30528}, log_prob_dims, index_dims, dX_dims, "none");
}

TEST(CudaKernelTest, SoftmaxCrossEntropyLossGrad_TinySizeTensor_half) {
  std::vector<int64_t> dY_dims{};
  std::vector<int64_t> log_prob_dims{8, 2};
  std::vector<int64_t> index_dims{8};
  std::vector<int64_t> dX_dims{8, 2};
  TestSoftmaxCrossEntropyLossGrad(dY_dims, log_prob_dims, index_dims, dX_dims, "mean", -1, true, 5e-2);
  TestSoftmaxCrossEntropyLossGrad(dY_dims, log_prob_dims, index_dims, dX_dims, "sum", -1, true, 5e-2);
  TestSoftmaxCrossEntropyLossGrad({8}, log_prob_dims, index_dims, dX_dims, "none", -1, true, 5e-2);

  // Just test ignore_index for small tensor because it will increase test time a lot with little verification gain.
  TestSoftmaxCrossEntropyLossGrad(dY_dims, log_prob_dims, index_dims, dX_dims, "mean", 0, true, 5e-2);
  TestSoftmaxCrossEntropyLossGrad(dY_dims, log_prob_dims, index_dims, dX_dims, "sum", 0, true, 5e-2);
  TestSoftmaxCrossEntropyLossGrad({8}, log_prob_dims, index_dims, dX_dims, "none", 0, true, 5e-2);
}

TEST(CudaKernelTest, SoftmaxCrossEntropyLossGrad_SmallSizeTensor_half) {
  std::vector<int64_t> dY_dims{};
  std::vector<int64_t> log_prob_dims{8, 20, 10};
  std::vector<int64_t> index_dims{8, 10};
  std::vector<int64_t> dX_dims{8, 20, 10};
  TestSoftmaxCrossEntropyLossGrad(dY_dims, log_prob_dims, index_dims, dX_dims, "mean", -1, true, 5e-2);
  TestSoftmaxCrossEntropyLossGrad(dY_dims, log_prob_dims, index_dims, dX_dims, "sum", -1, true, 5e-2);
  TestSoftmaxCrossEntropyLossGrad({8, 10}, log_prob_dims, index_dims, dX_dims, "none", -1, true, 5e-2);
}

TEST(CudaKernelTest, SoftmaxCrossEntropyLossGrad_LargeSizeTensor_half) {
  std::vector<int64_t> dY_dims{};
  std::vector<int64_t> log_prob_dims{2, 512, 30528};
  std::vector<int64_t> index_dims{2, 30528};
  std::vector<int64_t> dX_dims{2, 512, 30528};
  TestSoftmaxCrossEntropyLossGrad(dY_dims, log_prob_dims, index_dims, dX_dims, "mean", -1, true, 5e-2);
  TestSoftmaxCrossEntropyLossGrad(dY_dims, log_prob_dims, index_dims, dX_dims, "sum", -1, true, 5e-2);
  TestSoftmaxCrossEntropyLossGrad({2, 30528}, log_prob_dims, index_dims, dX_dims, "none", -1, true, 5e-2);
}

}  // namespace test
}  // namespace onnxruntime
