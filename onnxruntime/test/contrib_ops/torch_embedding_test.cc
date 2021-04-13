// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(ENABLE_TRAINING) && defined(USE_TORCH)

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

// Some of the tests can't run on TensorrtExecutionProvider because of unsupported data types.
// Those tests will fallback to other EPs

TEST(TorchEmbeddingOpTest, TorchEmbedding_Indices1D) {
  OpTester test("TorchEmbedding", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("data", {3, 3}, {0.0f, 0.1f, 0.2f, 1.0f, 1.1f, 1.2f, 2.0f, 2.1f, 2.2f});
  test.AddInput<int64_t>("indices", {1}, {1LL});
  test.AddOutput<float>("output", {1, 3}, {1.0f, 1.1f, 1.2f});
  test.Run();
}

TEST(TorchEmbeddingOpTest, TorchEmbedding_Indices2D) {
  OpTester test("TorchEmbedding", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("data", {3, 3}, {0.0f, 0.1f, 0.2f, 1.0f, 1.1f, 1.2f, 2.0f, 2.1f, 2.2f});
  test.AddInput<int64_t>("indices", {2LL, 2LL}, {1LL, 0LL, 2LL, 1LL});
  test.AddOutput<float>("output", {2, 2, 3}, {1.0f, 1.1f, 1.2f, 0.0f, 0.1f, 0.2f, 2.0f, 2.1f, 2.2f, 1.0f, 1.1f, 1.2f});
  test.Run();
}

TEST(TorchEmbeddingOpTest, TorchEmbedding_INT32) {
  OpTester test("TorchEmbedding", 1, onnxruntime::kMSDomain);
  test.AddInput<int32_t>("data", {3, 3}, {0, 1, 2, 10, 11, 12, 20, 21, 22});
  test.AddInput<int64_t>("indices", {1}, {1LL});
  test.AddOutput<int32_t>("output", {1, 3}, {10, 11, 12});
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime

#endif
