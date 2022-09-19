// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {
enum QLinearWhereFailCause {
  NoFail,
  FailByWrongTensorType,
  FailByWrongZeroPointType,
  FailByWrongScaleType
};

template <typename T>
void RunQLinearWhere(
    bool*  condition,
    const std::vector<T> x,
    float x_scale,
    T x_zero_point,
    const std::vector<T> y,
    float y_scale,
    T y_zero_point,
    float z_scale,
    T z_zero_point,
    const std::vector<T> z,
    bool is_x_const_input,
    bool is_y_const_input,
    const std::vector<int64_t> condition_shape,
    const std::vector<int64_t> x_shape,
    const std::vector<int64_t> y_shape,
    const std::vector<int64_t> z_shape,
   OpTester::ExpectResult expect_result = OpTester::ExpectResult::kExpectSuccess) {
  OpTester test("QLinearWhere", 1, onnxruntime::kMSDomain);
  test.AddInput<bool>("condition", condition_shape,condition,true );
  test.AddInput<T>("X", x_shape, x);
  test.AddInput<float>("x_scale" , {}, {x_scale}, is_x_const_input);
  test.AddInput<T>("x_zero_point" ,{}, {x_zero_point}, is_x_const_input);
  test.AddInput<T>("Y", y_shape, y);
  test.AddInput<float>("y_scale" , {}, {y_scale}, is_y_const_input);
  test.AddInput<T>("y_zero_point" ,{}, {y_zero_point}, is_y_const_input);
  test.AddInput<float>("z_scale", {}, {z_scale});
  test.AddInput<T>("z_zero_point", {}, {z_zero_point});
  test.AddOutput<T>("Z", z_shape, z);
  test.Run(expect_result);
}

void QLinearWhereScalarCondition() {
  bool c[] = {true};
  RunQLinearWhere<uint8_t>(
      c,
      {1},
      1.0f,
      0,
      {2},
      1.0f,
      0,
      1.0f,
      0,
      {1},
      true,
      true,
      {1},
      {1},
      {1},
      {1}, OpTester::ExpectResult::kExpectSuccess);
}
TEST(QLinearWhereTest, QLinearWhereScalarCondition) {
  QLinearWhereScalarCondition();
}
} // namespace test
} // namespace onnxruntime