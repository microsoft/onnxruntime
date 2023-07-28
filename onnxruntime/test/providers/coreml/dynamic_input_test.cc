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

TEST(CoreMLExecutionProviderDynamicInputShapeTest, MatMul) {
  constexpr auto model_path = ORT_TSTR("testdata/matmul_with_dynamic_input_shape.onnx");

  auto test = [&](const size_t M) {
    SCOPED_TRACE(MakeString("M=", M));

    auto coreml_ep = std::make_unique<CoreMLExecutionProvider>(0);

    const auto ep_verification_params = EPVerificationParams{
        ExpectedEPNodeAssignment::All,
        2e-3f
    };

    constexpr size_t K = 2;

    RandomValueGenerator gen{1234};
    const auto A_shape = std::vector<int64_t>{static_cast<int64_t>(M), K};
    const auto A_data = gen.Uniform<float>(AsSpan(A_shape), 0.0f, 1.0f);

    OrtValue A;
    CreateMLValue<float>(std::make_shared<CPUAllocator>(), A_shape, A_data, &A);

    RunAndVerifyOutputsWithEP(model_path, "CoreMLEPDynamicInputShape.MatMul",
                              std::move(coreml_ep),
                              {{"A", A}},
                              ep_verification_params);
  };

  test(1);
  test(3);
  test(5);
}

TEST(CoreMLExecutionProviderDynamicInputShapeTest, MobileNet) {
  constexpr auto model_path = ORT_TSTR("testdata/mobilenet_v3_small.onnx");

  auto test = [&](const size_t batch_size) {
    SCOPED_TRACE(MakeString("batch_size=", batch_size));

    auto coreml_ep = std::make_unique<CoreMLExecutionProvider>(0);

    const auto ep_verification_params = EPVerificationParams{
        ExpectedEPNodeAssignment::Some,
        0.5f,
    };

    RandomValueGenerator gen{1234};
    const auto A_shape = std::vector<int64_t>{static_cast<int64_t>(batch_size), 3, 224, 224};
    const auto A_data = gen.Uniform<float>(AsSpan(A_shape), 0.0f, 1.0f);

    OrtValue A;
    CreateMLValue<float>(std::make_shared<CPUAllocator>(), A_shape, A_data, &A);

    RunAndVerifyOutputsWithEP(model_path, "CoreMLEPDynamicInputShape.MobileNet",
                              std::move(coreml_ep),
                              {{"input", A}},
                              ep_verification_params);
  };

  test(1);
  test(3);
  test(5);
}



}  // namespace onnxruntime::test
