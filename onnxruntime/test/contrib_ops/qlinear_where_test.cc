// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include <array>

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
    OpTester &test,
    const std::vector<T> &x,
    const std::vector<T> &y,
    const std::vector<T> &z,
    float x_scale,
    T x_zero_point,
    float y_scale,
    T y_zero_point,
    float z_scale,
    T z_zero_point,
    const std::vector<int64_t> &x_shape,
    const std::vector<int64_t> &y_shape,
    const std::vector<int64_t> &z_shape,
    OpTester::ExpectResult expect_result = OpTester::ExpectResult::kExpectSuccess) {

  test.AddInput<T>("X", x_shape, x);
  test.AddInput<float>("x_scale", {}, {x_scale}, true);
  test.AddInput<T>("x_zero_point", {}, {x_zero_point}, true);
  test.AddInput<T>("Y", y_shape, y);
  test.AddInput<float>("y_scale", {}, {y_scale}, true);
  test.AddInput<T>("y_zero_point", {}, {y_zero_point}, true);
  test.AddInput<float>("z_scale", {}, {z_scale});
  test.AddInput<T>("z_zero_point", {}, {z_zero_point});
  test.AddOutput<T>("Z", z_shape, z);
  test.Run(expect_result);
}
template <typename T>
void QLinearWhereScalarCondition() {
  OpTester test("QLinearWhere", 1, onnxruntime::kMSDomain);
  test.AddInput<bool>("condition", {2,1}, {true,false}, true  );
  RunQLinearWhere<T>(
      test,
      {1}, {2}, {2, 1},// x ,y ,z
      1.0f, 0, 1.0f, 0, 1.0f, 0, // x_scale, x_zp, y_scale, y_zp, z_scale, z_zp
      {1}, {1}, {1,2},
      OpTester::ExpectResult::kExpectSuccess);
}
template <typename T>
void populate_Z(const std::vector<T> &x, const std::vector<T> &y, std::vector<T> &z, float x_scale, T x_zero_point, float y_scale, T y_zero_point, float z_scale, T z_zero_point) {
  for (size_t i = 0; i < x.size(); i++) {
    z[i] = (x[i] - x_zero_point) * x_scale < (y[i] - y_zero_point) * y_scale ? x[i] : y[i];
  }
}
TEST(QLinearWhereTest, QLinearWhereScalarCondition) {
  QLinearWhereScalarCondition<int8_t>();
  QLinearWhereScalarCondition<uint8_t>();
}
}  // namespace test
}  // namespace onnxruntime