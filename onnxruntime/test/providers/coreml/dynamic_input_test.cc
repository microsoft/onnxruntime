// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include <algorithm>
#include <vector>

#include "core/providers/coreml/coreml_execution_provider.h"
#include "test/common/random_generator.h"
#include "test/framework/test_utils.h"
#include "test/util/include/test_utils.h"

namespace onnxruntime::test {

TEST(CoreMLExecutionProviderDynamicInputTest, MatMul) {
  constexpr auto model_path = ORT_TSTR("testdata/matmul_with_dynamic_input_shape.onnx");

  auto coreml_ep = std::make_unique<CoreMLExecutionProvider>(0);

  const auto ep_verification_params = EPVerificationParams{
      ExpectedEPNodeAssignment::All,
  };

  const size_t M = 3;
  constexpr size_t K = 2;

  RandomValueGenerator gen{1234};
  const auto A_shape = std::vector<int64_t>{M, K};
  const auto A_data = gen.Uniform<float>(AsSpan(A_shape), 0.0f, 1.0f);

  OrtValue A;
  CreateMLValue<float>(std::make_shared<CPUAllocator>(), A_shape, A_data, &A);

  RunAndVerifyOutputsWithEP(model_path, "CoreMLEP.MatMulWithDynamicInputShape",
                            std::move(coreml_ep),
                            {{"A", A}},
                            ep_verification_params);
}

}  // namespace onnxruntime::test
