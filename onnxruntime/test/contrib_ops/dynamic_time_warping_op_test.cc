// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
#include "test/common/cuda_op_test_utils.h"

using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace test {

#ifdef USE_CUDA

TEST(DynamicTimeWarp, simple) {
  if (NeedSkipIfCudaArchLowerThan(530)) {
    return;
  }

  std::vector<float> X = {
      3.0f,
      8.0f,
      5.0f,
      1.0f,
      9.0f,
      8.0f,
      5.0f,
      7.0f,
      4.0f,
      4.0f,
      9.0f,
      6.0f,
      2.0f,
      9.0f,
      7.0f,
      2.0f,
      5.0f,
      6.0f,
      1.0f,
      8.0f,
      4.0f,
      6.0f,
      5.0f,
      8.0f,
      4.0f,
      8.0f,
      3.0f,
      6.0f,
      3.0f,
      9.0f,
      1.0f,
      1.0f,
      6.0f,
      8.0f,
      3.0f,
      5.0f,
      5.0f,
      3.0f,
      3.0f,
      8.0f,
      8.0f,
      7.0f,
      1.0f,
      2.0f,
      2.0f,
      1.0f,
      5.0f,
      4.0f,
      5.0f,
      0.0f,
      3.0f,
      6.0f,
      3.0f,
      7.0f,
      4.0f,
      5.0f,
      4.0f,
      5.0f,
      4.0f,
      0.0f,
  };

  std::vector<int32_t> Y = {
      0,
      1,
      2,
      3,
      4,
      4,
      4,
      4,
      5,
      5,
      5,
      5,
      0,
      1,
      1,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
  };

  OpTester tester("DynamicTimeWarping", 1, onnxruntime::kMSDomain);
  tester.AddInput<float>("input", {6, 10}, X);
  tester.AddOutput<int32_t>("output", {2, 12}, Y);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

#endif

}  // namespace test
}  // namespace onnxruntime
