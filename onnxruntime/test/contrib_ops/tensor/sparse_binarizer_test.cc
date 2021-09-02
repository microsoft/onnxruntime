// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(DISABLE_SPARSE_TENSORS)

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "core/framework/sparse_utils.h"

namespace onnxruntime {
namespace test {

TEST(SparseBinarizerTest, Apply) {
  const float threshold = 4.f;
  std::vector<int64_t> dense_shape{2, 4};
  std::vector<float> values{3.f, 5.f, 7.f};
  std::vector<int64_t> flat_indices{3, 5, 7};

  std::vector<int64_t> expected_shape{2, 4};
  std::vector<float> expected_values{0.f, 1.f, 1.f};

  OpTester test("Binarizer", 1, onnxruntime::kMSDomain);
  test.AddAttribute("threshold", threshold);
  test.AddSparseCooInput("X", dense_shape, values, flat_indices);
  test.AddSparseCooOutput<float>("Y", expected_shape, expected_values, flat_indices);
  test.Run(OpTester::ExpectResult::kExpectSuccess);
}


}
}  // namespace onnxruntime

#endif // DISABLE_SPARSE_TENSORS
