// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(MathOpTest, Clip_6) {
  OpTester test("Clip", 6);

  test.AddAttribute("min", -10.0f);
  test.AddAttribute("max", 10.0f);

  std::vector<int64_t> dims{3, 3};
  test.AddInput<float>("X", dims,
                       {11.0f, 4.4f, 432.3f,
                        -1.3f, 3.5f, 64.0f,
                        -5.4f, 9.3f, 82.4f});
  test.AddOutput<float>("Y", dims,
                        {10.0f, 4.4f, 10.0f,
                         -1.3f, 3.5f, 10.0f,
                         -5.4f, 9.3f, 10.0f});
  #if defined(OPENVINO_CONFIG_MYRIAD) || defined(OPENVINO_CONFIG_VAD_M)
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kOpenVINOExecutionProvider});
  #else
    test.Run();
  #endif
}

TEST(MathOpTest, Clip_Default) {
  OpTester test("Clip", -1);

  std::vector<int64_t> dims{3, 3};
  test.AddInput<float>("X", dims,
                       {11.0f, 4.4f, 432.3f,
                        -1.3f, 3.5f, 64.0f,
                        -5.4f, 9.3f, 82.4f});
  test.AddOutput<float>("Y", dims,
                        {11.0f, 4.4f, 432.3f,
                         -1.3f, 3.5f, 64.0f,
                         -5.4f, 9.3f, 82.4f});

  #if defined(OPENVINO_CONFIG_MYRIAD) || defined(OPENVINO_CONFIG_VAD_M)
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kOpenVINOExecutionProvider, kNGraphExecutionProvider});
  #else
  // nGraph does not support Clip opset 12 yet.
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kNGraphExecutionProvider});
  #endif
}

TEST(MathOpTest, Clip_Default_int8) {
  OpTester test("Clip", -1);

  std::vector<int64_t> dims{3, 3};
  test.AddInput<int8_t>("X", dims,
                         {11, 4, 127,
                          -1, 3, 64,
                          -5, 9, 82});
  test.AddOutput<int8_t>("Y", dims,
                          {11, 4, 127,
                           -1, 3, 64,
                           -5, 9, 82});

  // TensorRT, nGraph does not support Clip opset 12 yet.
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kNGraphExecutionProvider});
}

TEST(MathOpTest, Clip_Default_uint8) {
  OpTester test("Clip", -1);

  std::vector<int64_t> dims{3, 3};
  test.AddInput<uint8_t>("X", dims,
                        {11, 4, 255,
                         1, 3, 64,
                         5, 9, 82});
  test.AddOutput<uint8_t>("Y", dims,
                         {11, 4, 255,
                          1, 3, 64,
                          5, 9, 82});

  // TensorRT, nGraph does not support Clip opset 12 yet.
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kNGraphExecutionProvider});
}

TEST(MathOpTest, Clip_Default_int64) {
  OpTester test("Clip", -1);

  std::vector<int64_t> dims{3, 3};
  test.AddInput<int64_t>("X", dims,
                        {11, 4, 432,
                         -1, 3, 64,
                         -5, 9, 82});
  test.AddOutput<int64_t>("Y", dims,
                         {11, 4, 432,
                          -1, 3, 64,
                          -5, 9, 82});

  // TensorRT, nGraph does not support Clip opset 12 yet.
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kNGraphExecutionProvider});
}


TEST(MathOpTest, Clip_Default_uint64) {
  OpTester test("Clip", -1);

  std::vector<int64_t> dims{3, 3};
  test.AddInput<uint64_t>("X", dims,
                         {11, 4, 432,
                          1, 3, 64,
                          5, 9, 82});
  test.AddOutput<uint64_t>("Y", dims,
                          {11, 4, 432,
                           1, 3, 64,
                           5, 9, 82});

  // TensorRT, nGraph does not support Clip opset 12 yet.
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kNGraphExecutionProvider});
}

TEST(MathOpTest, Clip) {
  OpTester test("Clip", 11);

  std::vector<int64_t> dims{3, 3};
  test.AddInput<float>("X", dims,
                       {-1.0f, 0.0f, 1.0f,
                        -6.0f, 0.0f, 6.0f,
                        -5.4f, 2.0f, 6.0f});
  test.AddInput<float>("min", {}, {-5});
  test.AddInput<float>("max", {}, {5});
  test.AddOutput<float>("Y", dims,
                        {-1.0f, 0.0f, 1.0f,
                         -5.0f, 0.0f, 5.0f,
                         -5.0f, 2.0f, 5.0f});

  // TensorRT, nGraph and Tensorrt does not support Clip opset 11 yet.
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kNGraphExecutionProvider, kTensorrtExecutionProvider});
}

TEST(MathOpTest, ClipDimWithZero) {
  std::vector<int64_t> dims{3, 0};  // dim with value of zero should be handled

  OpTester test("Clip", 11);
  test.AddInput<float>("X", dims, {});
  test.AddInput<float>("min", {}, {-5});
  test.AddInput<float>("max", {}, {5});
  test.AddOutput<float>("Y", dims, {});

  // nGraph and Tensorrt does not support Clip opset 11 yet.
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kNGraphExecutionProvider, kTensorrtExecutionProvider});

  OpTester test1("Clip");  //
  test1.AddInput<float>("X", dims, {});
  test1.AddAttribute("min", -10.0f);
  test1.AddAttribute("max", 10.0f);
  test1.AddOutput<float>("Y", dims, {});
  // TRT doesn't handle this
  test1.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

}  // namespace test
}  // namespace onnxruntime
