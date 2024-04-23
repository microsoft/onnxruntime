// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/providers/compare_provider_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime::test {

#if defined(USE_CUDA) 

TEST(GemmaRotaryEmbGradTest, GemmaRotaryEmbGradTest) {
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.emplace_back(DefaultCudaExecutionProvider());

  OpTester test("GemmaRotaryEmbeddingGrad", 1, onnxruntime::kMSDomain);

  std::vector<float> dY(128, 1.0f);
  std::vector<int64_t> dY_shape = {1, 2, 8, 8};

  std::vector<float> X(32, 1.0f);
  std::vector<int64_t> X_shape = {1, 2, 4, 4};

  std::vector<float> dX(32, 4.0f);
  std::vector<int64_t> dX_shape = X_shape;

  test.AddInput<float>("dY", dY_shape, dY);
  test.AddInput<float>("X", X_shape, X);

  test.AddOutput<float>("dX", dX_shape, dX);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

#endif  // defined(USE_CUDA) || defined(USE_ROCM)

}  // namespace onnxruntime::test
