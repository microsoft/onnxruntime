// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>
#include <vector>
#include <cmath>

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

static void RunCollapseTest(OpTester& test, bool cuda) {
  std::vector<std::unique_ptr<IExecutionProvider>> eps;
  eps.push_back(cuda ? DefaultCudaExecutionProvider() : DefaultCpuExecutionProvider());
  SessionOptions options;
  if (cuda) ASSERT_STATUS_OK(options.config_options.AddConfigEntry("session.disable_cpu_ep_fallback", "1"));
  test.Run(options, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &eps);
}

static void TestCrossEntropyCollapse(bool cuda) {
  if (cuda && !DefaultCudaExecutionProvider()) GTEST_SKIP() << "CUDA is unavailable";
  const double tail = 2 * std::exp(-18.0);
  const float loss = static_cast<float>(std::log1p(tail));
  const float target_grad = static_cast<float>(-tail / (1 + tail));
  const float other_grad = static_cast<float>(std::exp(-18.0) / (1 + tail));
  for (const std::string reduction : {"none", "sum", "mean"}) {
    for (bool weighted : {false, true}) {
      const float scale = weighted && reduction != "mean" ? 2.0f : 1.0f;
      OpTester forward("SoftmaxCrossEntropyLoss", 13);
      forward.AddAttribute("reduction", reduction);
      forward.AddAttribute("ignore_index", int64_t(-1));
      forward.AddInput<float>("X", {2, 3, 1}, {9, -9, -9, 9, -9, -9});
      forward.AddInput<int64_t>("label", {2, 1}, {0, -1});
      if (weighted) forward.AddInput<float>("weight", {3}, {2, 3, 4});
      forward.AddOutput<float>("loss", reduction == "none" ? std::vector<int64_t>{2, 1} : std::vector<int64_t>{},
                               reduction == "none" ? std::vector<float>{loss * scale, 0} : std::vector<float>{loss * scale});
      forward.AddOutput<float>("log_prob", {2, 3, 1}, {-loss, -18, -18, -loss, -18, -18});
      forward.SetOutputAbsErr("loss", 1e-13f);
      forward.SetOutputRelErr("loss", 2e-6f);
      forward.SetOutputAbsErr("log_prob", 1e-13f);
      forward.SetOutputRelErr("log_prob", 2e-6f);
      RunCollapseTest(forward, cuda);

      for (const char* op : {"SoftmaxCrossEntropyLossGrad", "SoftmaxCrossEntropyLossInternalGrad"}) {
        const bool internal = std::string(op) == "SoftmaxCrossEntropyLossInternalGrad";
        OpTester backward(op, 1, kMSDomain);
        backward.AddAttribute("reduction", reduction);
        if (!internal) backward.AddAttribute("ignore_index", int64_t(-1));
        backward.AddInput<float>("dY", reduction == "none" ? std::vector<int64_t>{2, 1} : std::vector<int64_t>{},
                                 reduction == "none" ? std::vector<float>{-3, 7} : std::vector<float>{-3});
        backward.AddInput<float>("log_prob", {2, 3, 1}, {-loss, -18, -18, -loss, -18, -18});
        backward.AddInput<int64_t>("label", {2, 1}, {0, -1});
        if (weighted) backward.AddInput<float>("weight", {3}, {2, 3, 4});
        if (internal) {
          if (!weighted) backward.AddOptionalInputEdge<float>();
          backward.AddInput<int64_t>("ignore_index", {}, {-1});
          backward.AddInput<float>("bias", {2, 3, 1}, std::vector<float>(6, 1e-8f));
        }
        std::vector<float> expected{-3 * scale * target_grad, -3 * scale * other_grad, -3 * scale * other_grad, 0, 0, 0};
        if (internal) {
          for (float& value : expected) value += 1e-8f;
        }
        backward.AddOutput<float>("dX", {2, 3, 1}, expected);
        backward.SetOutputAbsErr("dX", 1e-13f);
        backward.SetOutputRelErr("dX", 3e-6f);
        RunCollapseTest(backward, cuda);
      }
    }
  }
}

TEST(CrossEntropyTest, CollapseCPU) {
  TestCrossEntropyCollapse(false);
}

static void TestLegacyCrossEntropyCollapse(bool cuda) {
  if (cuda && !DefaultCudaExecutionProvider()) GTEST_SKIP() << "CUDA is unavailable";
  const double tail = 2 * std::exp(-18.0);
  const float loss = static_cast<float>(std::log1p(tail));
  for (bool sparse : {false, true}) {
    OpTester forward(sparse ? "SparseSoftmaxCrossEntropy" : "SoftmaxCrossEntropy", sparse ? 9 : 1,
                     sparse ? kOnnxDomain : kMSDomain);
    forward.AddAttribute("reduction", std::string("mean"));
    forward.AddInput<float>("X", {1, 3}, {9, -9, -9});
    if (sparse) {
      forward.AddInput<int64_t>("label", {1}, {0});
    } else {
      forward.AddInput<float>("label", {1, 3}, {1, 0, 0});
    }
    forward.AddOutput<float>("loss", {}, {loss});
    forward.AddOutput<float>("log_prob", {1, 3}, {-loss, -18, -18});
    forward.SetOutputTolerance(1e-13f, 3e-6f);
    RunCollapseTest(forward, cuda);

    OpTester backward(sparse ? "SparseSoftmaxCrossEntropyGrad" : "SoftmaxCrossEntropyGrad", sparse ? 9 : 1,
                      sparse ? kOnnxDomain : kMSDomain);
    backward.AddAttribute("reduction", std::string("mean"));
    backward.AddInput<float>("dY", {}, {-3});
    backward.AddInput<float>("log_prob", {1, 3}, {-loss, -18, -18});
    if (sparse) {
      backward.AddInput<int64_t>("label", {1}, {0});
    } else {
      backward.AddInput<float>("label", {1, 3}, {1, 0, 0});
    }
    backward.AddOutput<float>("dX", {1, 3}, {static_cast<float>(3 * tail / (1 + tail)), static_cast<float>(-3 * std::exp(-18.0) / (1 + tail)), static_cast<float>(-3 * std::exp(-18.0) / (1 + tail))});
    backward.SetOutputTolerance(1e-13f, 3e-6f);
    RunCollapseTest(backward, cuda);
  }

  const float target = std::nextafter(1.0f, 0.0f);
  const double other = std::exp(-18.0) / (1 + tail);
  OpTester backward("SoftmaxCrossEntropyGrad", 1, kMSDomain);
  backward.AddAttribute("reduction", std::string("mean"));
  backward.AddInput<float>("dY", {}, {-3});
  backward.AddInput<float>("log_prob", {1, 3}, {-loss, -18, -18});
  backward.AddInput<float>("label", {1, 3}, {target, 1.0f - target, 0.0f});
  backward.AddOutput<float>("dX", {1, 3},
                            {static_cast<float>(-3 * ((1.0 - target) - tail / (1 + tail))),
                             static_cast<float>(-3 * (other - (1.0 - target))), static_cast<float>(-3 * other)});
  backward.SetOutputTolerance(1e-13f, 3e-6f);
  RunCollapseTest(backward, cuda);
}

TEST(CrossEntropyTest, LegacyCollapseCPU) {
  TestLegacyCrossEntropyCollapse(false);
}

#ifdef USE_CUDA
TEST(CrossEntropyTest, CollapseCUDA) {
  TestCrossEntropyCollapse(true);
}

TEST(CrossEntropyTest, LegacyCollapseCUDA) {
  TestLegacyCrossEntropyCollapse(true);
}

template <typename T>
static void TestMixedPrecisionCrossEntropyCollapse(double margin) {
  if (!DefaultCudaExecutionProvider()) GTEST_SKIP() << "CUDA is unavailable";
  const double tail = 2 * std::exp(-margin);
  const T loss(static_cast<float>(std::log1p(tail)));
  OpTester forward("SoftmaxCrossEntropyLoss", 13);
  forward.AddInput<T>("X", {1, 3}, {T(static_cast<float>(margin / 2)), T(static_cast<float>(-margin / 2)), T(static_cast<float>(-margin / 2))});
  forward.AddInput<int64_t>("label", {1}, {0});
  forward.AddOutput<T>("loss", {}, {loss});
  forward.SetOutputAbsErr("loss", 0.0f);
  forward.SetOutputRelErr("loss", 0.02f);
  RunCollapseTest(forward, true);

  OpTester backward("SoftmaxCrossEntropyLossGrad", 1, kMSDomain);
  backward.AddInput<T>("dY", {}, {T(-3.0f)});
  backward.AddInput<T>("log_prob", {1, 3}, {T(-static_cast<float>(loss)), T(static_cast<float>(-margin)), T(static_cast<float>(-margin))});
  backward.AddInput<int64_t>("label", {1}, {0});
  backward.AddOutput<T>("dX", {1, 3}, {T(static_cast<float>(3 * tail / (1 + tail))), T(static_cast<float>(-3 * std::exp(-margin) / (1 + tail))), T(static_cast<float>(-3 * std::exp(-margin) / (1 + tail)))});
  backward.SetOutputAbsErr("dX", 0.0f);
  backward.SetOutputRelErr("dX", 0.02f);
  RunCollapseTest(backward, true);
}

TEST(CrossEntropyTest, MixedPrecisionCollapseCUDA) {
  TestMixedPrecisionCrossEntropyCollapse<MLFloat16>(18.0);
  TestMixedPrecisionCrossEntropyCollapse<BFloat16>(18.0);
}

TEST(CrossEntropyTest, LogSoftmaxGradientScaledUnderflowCUDA) {
  if (!DefaultCudaExecutionProvider()) GTEST_SKIP() << "CUDA is unavailable";
  for (int64_t classes : {3, 1025, 2049}) {
    const double tail = (classes - 1) * std::exp(-20.0);
    std::vector<MLFloat16> output(static_cast<size_t>(classes), MLFloat16(-20.0f));
    output[0] = MLFloat16(static_cast<float>(-std::log1p(tail)));
    std::vector<MLFloat16> upstream(output.size(), MLFloat16(0.0f));
    upstream[0] = MLFloat16(100.0f);
    std::vector<MLFloat16> expected(output.size(), MLFloat16(static_cast<float>(-100 * std::exp(-20.0) / (1 + tail))));
    expected[0] = MLFloat16(static_cast<float>(100 * tail / (1 + tail)));
    OpTester test("LogSoftmaxGrad", 1, kMSDomain);
    test.AddAttribute("axis", int64_t(1));
    test.AddInput<MLFloat16>("dY", {1, classes}, upstream);
    test.AddInput<MLFloat16>("Y", {1, classes}, output);
    test.AddOutput<MLFloat16>("dX", {1, classes}, expected);
    test.SetOutputTolerance(0.0f, 0.02f);
    RunCollapseTest(test, true);
  }
}

template <typename T>
static void TestLowPrecisionCrossEntropyRows() {
  if (!DefaultCudaExecutionProvider()) GTEST_SKIP() << "CUDA is unavailable";
  for (int64_t classes : {1, 2, 3, 7, 21, 257, 1025}) {
    const int64_t pivot = classes / 2;
    const double tail = (classes - 1) * std::exp(-20.0);
    std::vector<T> log_prob(static_cast<size_t>(3 * classes), T(-20.0f));
    log_prob[pivot] = T(static_cast<float>(-std::log1p(tail)));
    std::vector<T> bias(log_prob.size(), T(0.0f)), expected(log_prob.size(), T(0.0f));
    for (int64_t j = 0; j < classes; ++j) {
      expected[j] = T(static_cast<float>(200 * (j == pivot ? -tail : std::exp(-20.0)) / (1 + tail)));
      bias[classes + j] = expected[classes + j] = T(0.00005f);
      log_prob[2 * classes + j] = T(static_cast<float>(-std::log(static_cast<double>(classes))));
      expected[2 * classes + j] = T(static_cast<float>(-6 * (1.0 / classes - (j == pivot ? 1 : 0))));
    }
    OpTester test("SoftmaxCrossEntropyLossInternalGrad", 1, kMSDomain);
    test.AddAttribute("reduction", std::string("none"));
    test.AddInput<T>("dY", {3}, {T(100.0f), T(4.0f), T(-3.0f)});
    test.AddInput<T>("log_prob", {3, classes}, log_prob);
    test.AddInput<int64_t>("label", {3}, {pivot, -1, pivot});
    test.AddInput<T>("weight", {classes}, std::vector<T>(static_cast<size_t>(classes), T(2.0f)));
    test.AddInput<int64_t>("ignore_index", {}, {-1});
    test.AddInput<T>("bias", {3, classes}, bias);
    test.AddOutput<T>("dX", {3, classes}, expected);
    test.SetOutputTolerance(0.0f, 0.02f);
    RunCollapseTest(test, true);
  }
  for (const char* op : {"SoftmaxCrossEntropyLossGrad", "SoftmaxCrossEntropyLossInternalGrad"}) {
    OpTester test(op, 1, kMSDomain);
    test.AddAttribute("reduction", std::string("none"));
    test.AddInput<T>("dY", {0}, {});
    test.AddInput<T>("log_prob", {0, 3}, {});
    test.AddInput<int64_t>("label", {0}, {});
    test.AddOutput<T>("dX", {0, 3}, {});
    RunCollapseTest(test, true);
  }
}

TEST(CrossEntropyTest, LowPrecisionCrossEntropyRowsCUDA) {
  TestLowPrecisionCrossEntropyRows<MLFloat16>();
  TestLowPrecisionCrossEntropyRows<BFloat16>();
}
#endif

static void TestSoftmaxGradientCollapse(bool cuda) {
  if (cuda && !DefaultCudaExecutionProvider()) GTEST_SKIP() << "CUDA is unavailable";
  for (int64_t classes : {1, 3, 1024, 1025, 2048, 2049, 4097}) {
    const double tail = (classes - 1) * std::exp(-40.0);
    for (bool logarithmic : {false, true}) {
      const float non_target = static_cast<float>(std::exp(-40.0) / (1 + tail));
      const float target = static_cast<float>(tail / (1 + tail));
      for (const char* op : logarithmic ? std::vector<const char*>{"LogSoftmaxGrad", "LogSoftmaxGrad_13"}
                                        : std::vector<const char*>{"SoftmaxGrad", "SoftmaxGrad_13"}) {
        std::vector<float> output(static_cast<size_t>(classes),
                                  logarithmic ? static_cast<float>(-40.0 - std::log1p(tail)) : non_target);
        output[classes / 2] = logarithmic ? static_cast<float>(-std::log1p(tail)) : 1.0f;
        std::vector<float> upstream(output.size(), 0.0f);
        upstream[classes / 2] = -3.0f;
        std::vector<float> expected(output.size(), 3 * non_target);
        expected[classes / 2] = -3 * target;
        OpTester test(op, 1, kMSDomain);
        test.AddAttribute("axis", int64_t(1));
        test.AddInput<float>("dY", {1, classes}, upstream);
        test.AddInput<float>("Y", {1, classes}, output);
        test.AddOutput<float>("dX", {1, classes}, expected);
        test.SetOutputAbsErr("dX", 1e-24f);
        test.SetOutputRelErr("dX", 5e-5f);
        RunCollapseTest(test, cuda);
      }
    }
  }
  for (bool logarithmic : {false, true}) {
    std::vector<float> output, upstream, expected;
    const double tail = 2 * std::exp(-40.0);
    for (int row = 0; row < 16; ++row) {
      const bool dominant = (row / 2) % 2 == 0;
      const float other = dominant ? static_cast<float>(std::exp(-40.0) / (1 + tail)) : 1.0f / 3;
      if (logarithmic) {
        output.insert(output.end(), dominant ? std::initializer_list<float>{static_cast<float>(-std::log1p(tail)), -40, -40}
                                             : std::initializer_list<float>{-std::log(3.0f), -std::log(3.0f), -std::log(3.0f)});
      } else {
        output.insert(output.end(), {dominant ? 1.0f : other, other, other});
      }
      upstream.insert(upstream.end(), {-3, 0, 0});
      const float other_grad = dominant || logarithmic ? 3 * other : other;
      expected.insert(expected.end(), {dominant ? static_cast<float>(-3 * tail / (1 + tail))
                                                : (logarithmic ? -2.0f : -2.0f / 3),
                                       other_grad, other_grad});
    }
    OpTester test(logarithmic ? "LogSoftmaxGrad" : "SoftmaxGrad", 1, kMSDomain);
    test.AddAttribute("axis", int64_t(1));
    test.AddInput<float>("dY", {16, 3}, upstream);
    test.AddInput<float>("Y", {16, 3}, output);
    test.AddOutput<float>("dX", {16, 3}, expected);
    test.SetOutputTolerance(1e-24f, 5e-5f);
    RunCollapseTest(test, cuda);
  }
}

TEST(CrossEntropyTest, SoftmaxGradientCollapseCPU) {
  TestSoftmaxGradientCollapse(false);
}

#ifdef USE_CUDA
TEST(CrossEntropyTest, SoftmaxGradientCollapseCUDA) {
  TestSoftmaxGradientCollapse(true);
}
#endif

// Regression tests for OOB reads when label values are outside [0, C).

TEST(CrossEntropyTest, SoftmaxCrossEntropyLoss_LabelTooLarge) {
  OpTester test("SoftmaxCrossEntropyLoss", 12);
  test.AddAttribute("reduction", std::string("mean"));
  test.AddAttribute("ignore_index", static_cast<int64_t>(-1));

  std::vector<float> X_data(3 * 5, 1.0f);
  std::vector<int64_t> index_data = {0, 5, 2};  // 5 is out of range [0, 5)

  test.AddInput<float>("X", {3, 5}, X_data);
  test.AddInput<int64_t>("index", {3}, index_data);
  test.AddOutput<float>("output", {}, {0.0f});
  test.AddOutput<float>("log_prob", {3, 5}, std::vector<float>(15, 0.0f));

  test.Run(OpTester::ExpectResult::kExpectFailure, "out of range");
}

TEST(CrossEntropyTest, SoftmaxCrossEntropyLoss_NegativeLabel) {
  OpTester test("SoftmaxCrossEntropyLoss", 12);
  test.AddAttribute("reduction", std::string("mean"));
  test.AddAttribute("ignore_index", static_cast<int64_t>(-100));

  std::vector<float> X_data(3 * 5, 1.0f);
  std::vector<int64_t> index_data = {0, -1, 2};  // -1 is out of range (and != ignore_index)

  test.AddInput<float>("X", {3, 5}, X_data);
  test.AddInput<int64_t>("index", {3}, index_data);
  test.AddOutput<float>("output", {}, {0.0f});
  test.AddOutput<float>("log_prob", {3, 5}, std::vector<float>(15, 0.0f));

  test.Run(OpTester::ExpectResult::kExpectFailure, "out of range");
}

TEST(CrossEntropyTest, SoftmaxCrossEntropyLoss_LabelTooLargeWithWeights) {
  OpTester test("SoftmaxCrossEntropyLoss", 12);
  test.AddAttribute("reduction", std::string("mean"));
  test.AddAttribute("ignore_index", static_cast<int64_t>(-1));

  std::vector<float> X_data(3 * 5, 1.0f);
  std::vector<int64_t> index_data = {0, 100, 2};  // 100 is out of range
  std::vector<float> weight_data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

  test.AddInput<float>("X", {3, 5}, X_data);
  test.AddInput<int64_t>("index", {3}, index_data);
  test.AddInput<float>("weight", {5}, weight_data);
  test.AddOutput<float>("output", {}, {0.0f});
  test.AddOutput<float>("log_prob", {3, 5}, std::vector<float>(15, 0.0f));

  test.Run(OpTester::ExpectResult::kExpectFailure, "out of range");
}

// Covers the weighted, non-MEAN forward loop (the second per-sample loop in Compute).
TEST(CrossEntropyTest, SoftmaxCrossEntropyLoss_LabelTooLargeWithWeightsSumReduction) {
  OpTester test("SoftmaxCrossEntropyLoss", 12);
  test.AddAttribute("reduction", std::string("sum"));
  test.AddAttribute("ignore_index", static_cast<int64_t>(-1));

  std::vector<float> X_data(3 * 5, 1.0f);
  std::vector<int64_t> index_data = {0, 7, 2};  // 7 is out of range [0, 5)
  std::vector<float> weight_data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

  test.AddInput<float>("X", {3, 5}, X_data);
  test.AddInput<int64_t>("index", {3}, index_data);
  test.AddInput<float>("weight", {5}, weight_data);
  test.AddOutput<float>("output", {}, {0.0f});
  test.AddOutput<float>("log_prob", {3, 5}, std::vector<float>(15, 0.0f));

  test.Run(OpTester::ExpectResult::kExpectFailure, "out of range");
}

// int32 label type — kernel is registered for both int32_t and int64_t.
TEST(CrossEntropyTest, SoftmaxCrossEntropyLoss_LabelTooLargeInt32) {
  OpTester test("SoftmaxCrossEntropyLoss", 12);
  test.AddAttribute("reduction", std::string("mean"));
  test.AddAttribute("ignore_index", static_cast<int64_t>(-1));

  std::vector<float> X_data(3 * 5, 1.0f);
  std::vector<int32_t> index_data = {0, 5, 2};  // 5 is out of range [0, 5)

  test.AddInput<float>("X", {3, 5}, X_data);
  test.AddInput<int32_t>("index", {3}, index_data);
  test.AddOutput<float>("output", {}, {0.0f});
  test.AddOutput<float>("log_prob", {3, 5}, std::vector<float>(15, 0.0f));

  test.Run(OpTester::ExpectResult::kExpectFailure, "out of range");
}

// Higher-dimensional inputs: logit [N, C, D1, D2], label [N, D1, D2].
TEST(CrossEntropyTest, SoftmaxCrossEntropyLoss_LabelTooLargeHighDim) {
  OpTester test("SoftmaxCrossEntropyLoss", 12);
  test.AddAttribute("reduction", std::string("mean"));
  test.AddAttribute("ignore_index", static_cast<int64_t>(-1));

  // [N=2, C=4, D1=2, D2=3] -> label shape [2, 2, 3] -> 12 label entries.
  std::vector<float> X_data(2 * 4 * 2 * 3, 1.0f);
  std::vector<int64_t> index_data = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 99, 3};  // 99 is out of range
  std::vector<float> log_prob_init(2 * 4 * 2 * 3, 0.0f);

  test.AddInput<float>("X", {2, 4, 2, 3}, X_data);
  test.AddInput<int64_t>("index", {2, 2, 3}, index_data);
  test.AddOutput<float>("output", {}, {0.0f});
  test.AddOutput<float>("log_prob", {2, 4, 2, 3}, log_prob_init);

  test.Run(OpTester::ExpectResult::kExpectFailure, "out of range");
}

TEST(CrossEntropyTest, SoftmaxCrossEntropyLossGrad_LabelTooLarge) {
  OpTester test("SoftmaxCrossEntropyLossGrad", 1, onnxruntime::kMSDomain);
  test.AddAttribute("reduction", std::string("mean"));
  test.AddAttribute("ignore_index", static_cast<int64_t>(-1));

  std::vector<float> dY_data = {1.0f};
  std::vector<float> log_prob_data(3 * 5, -1.6094f);
  std::vector<int64_t> index_data = {0, 5, 2};  // 5 is out of range [0, 5)

  test.AddInput<float>("dY", {}, dY_data);
  test.AddInput<float>("log_prob", {3, 5}, log_prob_data);
  test.AddInput<int64_t>("index", {3}, index_data);
  test.AddOutput<float>("dX", {3, 5}, std::vector<float>(15, 0.0f));

  test.Run(OpTester::ExpectResult::kExpectFailure, "out of range");
}

TEST(CrossEntropyTest, SoftmaxCrossEntropyLossGrad_LabelTooLargeWithWeights) {
  OpTester test("SoftmaxCrossEntropyLossGrad", 1, onnxruntime::kMSDomain);
  test.AddAttribute("reduction", std::string("mean"));
  test.AddAttribute("ignore_index", static_cast<int64_t>(-1));

  std::vector<float> dY_data = {1.0f};
  std::vector<float> log_prob_data(3 * 5, -1.6094f);
  std::vector<int64_t> index_data = {0, 5, 2};  // 5 is out of range [0, 5)
  std::vector<float> weight_data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

  test.AddInput<float>("dY", {}, dY_data);
  test.AddInput<float>("log_prob", {3, 5}, log_prob_data);
  test.AddInput<int64_t>("index", {3}, index_data);
  test.AddInput<float>("weight", {5}, weight_data);
  test.AddOutput<float>("dX", {3, 5}, std::vector<float>(15, 0.0f));

  test.Run(OpTester::ExpectResult::kExpectFailure, "out of range");
}

TEST(CrossEntropyTest, SoftmaxCrossEntropyLossGrad_LabelTooLargeInt32) {
  OpTester test("SoftmaxCrossEntropyLossGrad", 1, onnxruntime::kMSDomain);
  test.AddAttribute("reduction", std::string("mean"));
  test.AddAttribute("ignore_index", static_cast<int64_t>(-1));

  std::vector<float> dY_data = {1.0f};
  std::vector<float> log_prob_data(3 * 5, -1.6094f);
  std::vector<int32_t> index_data = {0, 5, 2};

  test.AddInput<float>("dY", {}, dY_data);
  test.AddInput<float>("log_prob", {3, 5}, log_prob_data);
  test.AddInput<int32_t>("index", {3}, index_data);
  test.AddOutput<float>("dX", {3, 5}, std::vector<float>(15, 0.0f));

  test.Run(OpTester::ExpectResult::kExpectFailure, "out of range");
}

// SoftmaxCrossEntropyLossInternal shares the same Compute as SoftmaxCrossEntropyLoss but
// is registered separately under kMSDomain v1 with an optional runtime ignore_index input.
TEST(CrossEntropyTest, SoftmaxCrossEntropyLossInternal_LabelTooLarge) {
  OpTester test("SoftmaxCrossEntropyLossInternal", 1, onnxruntime::kMSDomain);
  test.AddAttribute("reduction", std::string("mean"));

  std::vector<float> X_data(3 * 5, 1.0f);
  std::vector<int64_t> index_data = {0, 5, 2};
  int64_t ignore_index_val = -1;

  test.AddInput<float>("X", {3, 5}, X_data);
  test.AddInput<int64_t>("index", {3}, index_data);
  // weight is optional; pass empty input to skip and still provide ignore_index input below.
  test.AddOptionalInputEdge<float>();
  test.AddInput<int64_t>("ignore_index", {}, &ignore_index_val, 1);
  test.AddOutput<float>("output", {}, {0.0f});
  test.AddOutput<float>("log_prob", {3, 5}, std::vector<float>(15, 0.0f));

  test.Run(OpTester::ExpectResult::kExpectFailure, "out of range");
}

TEST(CrossEntropyTest, SoftmaxCrossEntropyLossInternalGrad_LabelTooLarge) {
  OpTester test("SoftmaxCrossEntropyLossInternalGrad", 1, onnxruntime::kMSDomain);
  test.AddAttribute("reduction", std::string("mean"));

  std::vector<float> dY_data = {1.0f};
  std::vector<float> log_prob_data(3 * 5, -1.6094f);
  std::vector<int64_t> index_data = {0, 5, 2};
  int64_t ignore_index_val = -1;

  test.AddInput<float>("dY", {}, dY_data);
  test.AddInput<float>("log_prob", {3, 5}, log_prob_data);
  test.AddInput<int64_t>("index", {3}, index_data);
  test.AddOptionalInputEdge<float>();  // weight
  test.AddInput<int64_t>("ignore_index", {}, &ignore_index_val, 1);
  test.AddOutput<float>("dX", {3, 5}, std::vector<float>(15, 0.0f));

  test.Run(OpTester::ExpectResult::kExpectFailure, "out of range");
}

}  // namespace test
}  // namespace onnxruntime
