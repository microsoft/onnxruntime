// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

namespace {

void InplaceClipGradNormTest(std::vector<std::unique_ptr<IExecutionProvider>>* providers) {
  OpTester test("InplaceClipGradNorm", 1, onnxruntime::kMSDomain);

  SeqTensors<float> gradients_input;
  gradients_input.AddTensor({3}, {1.f, 2.f, 3.f});
  gradients_input.AddTensor({4}, {4.f, 5.f, 6.f, 7.f});
  gradients_input.AddTensor({5}, {8.f, 9.f, 10.f, 11.f, 12.f});

  test.AddSeqInput<float>("gradients", gradients_input);

  test.AddAttribute("max_norm", 12.f);

  SeqTensors<float> clipped_gradients;
  clipped_gradients.AddTensor({3}, {0.4707f, 0.9414f, 1.4120f});
  clipped_gradients.AddTensor({4}, {1.8827f, 2.3534f, 2.8241f, 3.2948f});
  clipped_gradients.AddTensor({5}, {3.7654f, 4.2361f, 4.7068f, 5.1775f, 5.6481f});
  test.AddSeqOutput<float>("clipped_gradients", clipped_gradients);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, providers);
}

void InplaceClipGradNormNoClippingTest(std::vector<std::unique_ptr<IExecutionProvider>>* providers) {
  OpTester test("InplaceClipGradNorm", 1, onnxruntime::kMSDomain);

  SeqTensors<float> gradients_input;
  gradients_input.AddTensor({3}, {1.f, 2.f, 3.f});
  gradients_input.AddTensor({4}, {4.f, 5.f, 6.f, 7.f});
  gradients_input.AddTensor({5}, {8.f, 9.f, 10.f, 11.f, 12.f});

  test.AddSeqInput<float>("gradients", gradients_input);

  test.AddAttribute("max_norm", 100.f);

  SeqTensors<float> clipped_gradients;
  clipped_gradients.AddTensor({3}, {1.f, 2.f, 3.f});
  clipped_gradients.AddTensor({4}, {4.f, 5.f, 6.f, 7.f});
  clipped_gradients.AddTensor({5}, {8.f, 9.f, 10.f, 11.f, 12.f});
  test.AddSeqOutput<float>("clipped_gradients", clipped_gradients);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, providers);
}

}  // namespace

TEST(OptimizerTest, InplaceClipGradNorm_CPU) {
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.emplace_back(DefaultCpuExecutionProvider());
  InplaceClipGradNormTest(&providers);
}

TEST(OptimizerTest, InplaceClipGradNormNoClipping_CPU) {
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.emplace_back(DefaultCpuExecutionProvider());
  InplaceClipGradNormNoClippingTest(&providers);
}

#ifdef USE_CUDA

TEST(OptimizerTest, InplaceClipGradNorm_CUDA) {
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.emplace_back(DefaultCudaExecutionProvider());
  InplaceClipGradNormTest(&providers);
}

TEST(OptimizerTest, InplaceClipGradNormNoClipping_CUDA) {
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.emplace_back(DefaultCudaExecutionProvider());
  InplaceClipGradNormNoClippingTest(&providers);
}

#endif

}  // namespace test
}  // namespace onnxruntime
