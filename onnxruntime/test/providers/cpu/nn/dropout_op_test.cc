// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

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
  auto p_tp_model = test.BuildGraph();//slx
  std::cout << "Dropout:WithOptionalOutputOpset10: p_tp_model: " << p_tp_model->MainGraph().Name() << std::endl;//slx
  onnxruntime::Model::Save(*p_tp_model, "DropoutWithOptionalOutputOpset10.onnx");
  test.Run();
}

TEST(Dropout, WithOptionalOutputOpset7) {
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
