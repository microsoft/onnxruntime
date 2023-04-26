// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
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

}  // namespace test
}  // namespace onnxruntime
