// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

TEST(Dropout, Opset7) {
  OpTester test("Dropout", 7, kOnnxDomain);
  std::vector<int64_t> dims{2, 2};
  test.AddInput<float>("X", dims, {1.0f, 2.0f, 3.0f, 4.0f});
  test.AddOutput<float>("Y", dims, {1.0f, 2.0f, 3.0f, 4.0f});
  test.Run();
}

TEST(Dropout, Opset10) {
  OpTester test("Dropout", 10, kOnnxDomain);
  std::vector<int64_t> dims{2, 2};
  test.AddInput<float>("X", dims, {1.0f, 2.0f, 3.0f, 5.0f});
  test.AddOutput<float>("Y", dims, {1.0f, 2.0f, 3.0f, 5.0f});
  test.Run();
}

TEST(Dropout, WithOptionalOutputOpset10) {
  OpTester test("Dropout", 10, kOnnxDomain);
  std::vector<int64_t> dims{2, 2};
  test.AddInput<float>("X", dims, {1.0f, 2.0f, 3.0f, 5.0f});
  test.AddOutput<float>("Y", dims, {1.0f, 2.0f, 3.0f, 5.0f});
  test.AddOutput<bool>("mask", dims, {false, false, false, false});
  // The fix in onnx-tensorrt parser for dropout onnx node is not included in TRT 8.6.1 but might be included in later ORT release.
  // Simply skip this for now.
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(Dropout, WithOptionalOutputOpset7) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: MLOperatorAuthorImpl.cpp(2092): The parameter is incorrect.";
  }

  // Opset 7 differs with Opset 10 in that the type of the 'mask'
  // output is tied with the type of the input in Opset 7 whereas
  // the type of 'mask' in Opset 10 is 'bool' always
  OpTester test("Dropout", 7, kOnnxDomain);
  std::vector<int64_t> dims{2, 2};
  test.AddInput<float>("X", dims, {1.0f, 2.0f, 3.0f, 5.0f});
  test.AddOutput<float>("Y", dims, {1.0f, 2.0f, 3.0f, 5.0f});
  test.AddOutput<float>("mask", dims, {0.0f, 0.0f, 0.0f, 0.0f});
  // The TensorRT execution provider doesn't seem to support 'Dropout' with non-boolean mask output
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

}  // namespace test
}  // namespace onnxruntime
