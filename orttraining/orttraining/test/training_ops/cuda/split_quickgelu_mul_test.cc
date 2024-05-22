// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

// #if defined(USE_CUDA) || defined(USE_ROCM)

TEST(S2SModelSplitQuickGeluTest, Int32Type2D) {
  std::cout << "Starting test" << std::endl;
  std::vector<float> input = {1, 1, 3, 2,
                                0, 3, 0, 4,
                                0, 5, 0, 6,
                                0, 0, 0, 2};
  std::vector<float> output = {0.9940, 0.9678, 1.5, 1.9880,
                                 0.0000, 2.9967, 0.0000, 3.9992};

  OpTester test("S2SModelSplitQuickGelu", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("input", {4, 4}, input);
  test.AddOutput<float>("output", {4, 2}, output);
  test.Run();
}

TEST(QuickGeluTest, Int32Type2D) {
  std::cout << "Starting QuickGelu test" << std::endl;
  std::vector<float> input = {3, 2, 0, 4,
                              0, 6, 0, 2};
  std::vector<float> output = {0.9940, 0.9678, 0.5, 0.9940,
                                 0.5000, 0.9989, 0.5000, 0.9998};

  OpTester test("QuickGelu", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("input", {4, 2}, input);
  test.AddOutput<float>("output", {4, 2}, output);
  test.Run();
}


// #endif

}  // namespace test
}  // namespace onnxruntime
