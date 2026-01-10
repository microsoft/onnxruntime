// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include <array>

namespace onnxruntime {
namespace test {

template <typename T>
void RunQLinearWhere(
    OpTester& test,
    const std::vector<T>& x,
    const std::vector<T>& y,
    const std::vector<T>& z,
    float x_scale,
    T x_zero_point,
    float y_scale,
    T y_zero_point,
    float z_scale,
    T z_zero_point,
    const std::vector<int64_t>& x_shape,
    const std::vector<int64_t>& y_shape,
    const std::vector<int64_t>& z_shape) {
  test.AddInput<T>("X", x_shape, x);
  test.AddInput<float>("x_scale", {}, {x_scale}, true);
  test.AddInput<T>("x_zero_point", {}, {x_zero_point}, true);
  test.AddInput<T>("Y", y_shape, y);
  test.AddInput<float>("y_scale", {}, {y_scale}, true);
  test.AddInput<T>("y_zero_point", {}, {y_zero_point}, true);
  test.AddInput<float>("z_scale", {}, {z_scale});
  test.AddInput<T>("z_zero_point", {}, {z_zero_point});
  test.AddOutput<T>("Z", z_shape, z);
  test.Run(OpTester::ExpectResult::kExpectSuccess);
}
template <typename T>
void QLinearWhereScalarAll() {
  constexpr float scale = 0.039f;
  constexpr uint8_t zp = 135;
  OpTester test("QLinearWhere", 1, onnxruntime::kMSDomain);
  test.AddInput<bool>("condition", {1}, {true}, true);
  RunQLinearWhere<T>(
      test,
      {1}, {2}, {1},                    // x ,y ,z
      scale, zp, scale, zp, scale, zp,  // x_scale, x_zp, y_scale, y_zp, z_scale, z_zp
      {1}, {1}, {1});
}

TEST(QLinearWhereTest, QLinearWhereScalarAll) {
  QLinearWhereScalarAll<int8_t>();
  QLinearWhereScalarAll<uint8_t>();
}

template <typename T>
void QLinearWhereVectorAll() {
  constexpr float scale = 0.039f;
  constexpr uint8_t zp = 135;
  OpTester test("QLinearWhere", 1, onnxruntime::kMSDomain);
  test.AddInput<bool>("condition", {4}, {true, false, false, true}, true);
  RunQLinearWhere<T>(
      test,
      {1, 1, 1, 1}, {2, 2, 2, 2}, {1, 2, 2, 1},  // x ,y ,z
      scale, zp, scale, zp, scale, zp,           // x_scale, x_zp, y_scale, y_zp, z_scale, z_zp
      {4}, {4}, {4});
}

TEST(QLinearWhereTest, QLinearWhereVectorAll) {
  QLinearWhereVectorAll<int8_t>();
  QLinearWhereVectorAll<uint8_t>();
}
template <typename T>
void QLinearWhereMatrixAll() {
  constexpr float scale = 0.039f;
  constexpr uint8_t zp = 135;
  OpTester test("QLinearWhere", 1, onnxruntime::kMSDomain);
  test.AddInput<bool>("condition", {2, 2}, {true, false, false, true}, true);
  RunQLinearWhere<T>(
      test,
      {1, 1, 1, 1}, {2, 2, 2, 2}, {1, 2, 2, 1},  // x ,y ,z
      scale, zp, scale, zp, scale, zp,           // x_scale, x_zp, y_scale, y_zp, z_scale, z_zp
      {2, 2}, {2, 2}, {2, 2});
}

TEST(QLinearWhereTest, QLinearWhereMatrixAll) {
  QLinearWhereMatrixAll<int8_t>();
  QLinearWhereMatrixAll<uint8_t>();
}
template <typename T>
void QLinearWhereScalarX_VectorY_MatrixCondition() {
  constexpr float scale = 0.039f;
  constexpr uint8_t zp = 135;
  OpTester test("QLinearWhere", 1, onnxruntime::kMSDomain);
  test.AddInput<bool>("condition", {2, 2}, {true, false, false, true}, true);
  RunQLinearWhere<T>(
      test,
      {1}, {2, 2}, {1, 2, 2, 1},        // x ,y ,z
      scale, zp, scale, zp, scale, zp,  // x_scale, x_zp, y_scale, y_zp, z_scale, z_zp
      {1}, {2}, {2, 2});
}

TEST(QLinearWhereTest, QLinearWhereScalarX_VectorY_MatrixCondition) {
  QLinearWhereScalarX_VectorY_MatrixCondition<int8_t>();
  QLinearWhereScalarX_VectorY_MatrixCondition<uint8_t>();
}

template <typename T>
void QLinearWhereVectorX_VectorY_MatrixCondition() {
  constexpr float scale = 0.039f;
  constexpr uint8_t zp = 135;
  OpTester test("QLinearWhere", 1, onnxruntime::kMSDomain);
  test.AddInput<bool>("condition", {2, 2}, {true, false, false, true}, true);
  RunQLinearWhere<T>(
      test,
      {1, 1}, {2, 2}, {1, 2, 2, 1},     // x ,y ,z
      scale, zp, scale, zp, scale, zp,  // x_scale, x_zp, y_scale, y_zp, z_scale, z_zp
      {2}, {2}, {2, 2});
}

TEST(QLinearWhereTest, QLinearWhereVectorX_VectorY_MatrixCondition) {
  QLinearWhereVectorX_VectorY_MatrixCondition<int8_t>();
  QLinearWhereVectorX_VectorY_MatrixCondition<uint8_t>();
}

}  // namespace test
}  // namespace onnxruntime