// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include "core/session/environment.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

// Disable TensorRT on some of the tests because axis=0 is not supported

class FlattenOpTest : public testing::Test {
 public:
  FlattenOpTest() : test_("Flatten", 11), data0_(120, 1.0f) {}

 protected:
  OpTester test_;
  const std::vector<float> data0_;
  const std::vector<float> data1_ = {0.0f, 0.1f, 0.2f, 0.3f,
                                     1.0f, 1.1f, 1.2f, 1.3f,
                                     2.0f, 2.1f, 2.2f, 2.3f,
                                     3.0f, 3.1f, 3.2f, 3.3f};
};

TEST_F(FlattenOpTest, Flatten_axis0) {
  test_.AddAttribute<int64_t>("axis", 0L);
  test_.AddInput<float>("data", {2L, 3L, 4L, 5L}, data0_);
  test_.AddOutput<float>("output", {1L, 120L}, data0_);
  test_.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST_F(FlattenOpTest, Flatten_default_axis) {
  // default axis value = 1
  test_.AddInput<float>("data", {2L, 3L, 4L, 5L}, data0_);
  test_.AddOutput<float>("output", {2L, 60L}, data0_);
  test_.Run();
}

TEST_F(FlattenOpTest, Flatten_invalid_axis) {
  test_.AddAttribute<int64_t>("axis", 5L);
  test_.AddInput<float>("data", {1L, 2L, 4L, 2L}, data1_);
  test_.AddOutput<float>("output", {1L, 16L}, data1_);
  test_.Run(OpTester::ExpectResult::kExpectFailure, "Invalid value(5) for attribute 'axis'");
}

TEST_F(FlattenOpTest, Flatten_axis3) {
  test_.AddAttribute<int64_t>("axis", 3L);
  test_.AddInput<float>("data", {2L, 3L, 4L, 5L}, data0_);
  test_.AddOutput<float>("output", {24L, 5L}, data0_);
  test_.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST_F(FlattenOpTest, Flatten_axis4) {
  test_.AddAttribute<int64_t>("axis", 4L);
  test_.AddInput<float>("data", {1L, 2L, 4L, 2L}, data1_);
  test_.AddOutput<float>("output", {16L, 1L}, data1_);
  test_.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST_F(FlattenOpTest, Flatten_neg_axis3) {
  test_.AddAttribute<int64_t>("axis", -1L);
  test_.AddInput<float>("data", {2L, 3L, 4L, 5L}, data0_);
  test_.AddOutput<float>("output", {24L, 5L}, data0_);
  test_.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// Regression test primarily for CoreML.
// The CoreML EP implementation was not reading the axis attribute correctly causing an incorrect output shape to be
// produced for a Flatten node. That issue gets hidden as the Tensor to write the output to is created by the
// CoreML EP using the inferred output shape (which is correct) and we provide the Tensor's buffer but not the shape
// when executing the CoreML model. As the flatten isn't changing or moving any data nothing breaks when we test
// with only a Flatten node in the model.
//
// This test uses a model with a Flatten followed by a Mul which requires broadcasting. Both nodes are handled by
// CoreML, so if the axis is not correctly processed the output from Flatten will not be broadcastable and the CoreML
// model execution will fail.
TEST(FlattenOpModelTest, Flatten_broadcast) {
  auto model_uri = ORT_TSTR("testdata/flatten_broadcast.onnx");
  ModelTester tester("flatten_broadcast", model_uri);

  tester.AddInput<float>("X", {4}, {0.f, 1.f, 2.f, 3.f});
  tester.AddInput<float>("Y", {3, 4},
                         {0.f, 1.f, 2.f, 3.f,
                          4.f, 5.f, 6.f, 7.f,
                          8.f, 9.f, 10.f, 11.f});
  tester.AddOutput<float>("Z", {3, 4},
                          {0.f, 1.f, 4.f, 9.f,
                           0.f, 5.f, 12.f, 21.f,
                           0.f, 9.f, 20.f, 33.f});

  // disable TRT as it does not support axis=0 as used by the model
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}
}  // namespace test
}  // namespace onnxruntime
