// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace test {

enum QLinearConcatFailCause {
  NoFail,
  FailByWrongTensorType,
  FailByWrongZeroPointType,
  FailByWrongScaleType
};

template <typename xint8>
void RunQLinearConcat(
    const std::vector<std::vector<int64_t>> x_shapes,
    const std::vector<std::vector<xint8>> x_vecs,
    int64_t possible_neg_concat_axis,
    const std::vector<float> x_scales,
    const std::vector<xint8> x_zero_points,
    const std::vector<int64_t> y_shape,
    const std::vector<xint8> y_vec,
    float y_scale,
    xint8 y_zero_point,
    const std::vector<bool> is_const_inputs,
    QLinearConcatFailCause fail_by = NoFail) {
  size_t input_count = x_shapes.size();

  OpTester test("QLinearConcat", 1, onnxruntime::kMSDomain);
  test.AddAttribute<int64_t>("axis", possible_neg_concat_axis);
  test.AddInput<float>("y_scale", {}, {y_scale});
  test.AddInput<xint8>("y_zero_point", {}, {y_zero_point});
  for (size_t input_index = 0; input_index < input_count; input_index++) {
    std::stringstream ss;
    ss << "X" << input_index;
    if (fail_by == FailByWrongTensorType && input_index == 0) {
      if (std::is_signed<xint8>().value) {
        const uint8_t* rdata = (const uint8_t*)(x_vecs[input_index].data());
        std::vector<uint8_t> ov(rdata, rdata + x_vecs[input_index].size());
        test.AddInput<uint8_t>(ss.str().c_str(), x_shapes[input_index], ov);
      } else {
        const int8_t* rdata = (const int8_t*)(x_vecs[input_index].data());
        std::vector<int8_t> ov(rdata, rdata + x_vecs[input_index].size());
        test.AddInput<int8_t>(ss.str().c_str(), x_shapes[input_index], ov);
      }
    } else {
      test.AddInput<xint8>(ss.str().c_str(), x_shapes[input_index], x_vecs[input_index]);
    }
    ss.str("");
    ss << "x_scale" << input_index;
    if (fail_by == FailByWrongScaleType && input_index == 0) {
      test.AddInput<int64_t>(ss.str().c_str(), {}, {1LL}, is_const_inputs[input_index]);
    } else {
      test.AddInput<float>(ss.str().c_str(), {}, {x_scales[input_index]}, is_const_inputs[input_index]);
    }
    ss.str("");
    ss << "x_zero_point" << input_index;
    if (fail_by == FailByWrongZeroPointType && input_index == 0) {
      if (std::is_signed<xint8>().value) {
        test.AddInput<uint8_t>(ss.str().c_str(), {}, {128}, is_const_inputs[input_index]);
      } else {
        test.AddInput<int8_t>(ss.str().c_str(), {}, {0}, is_const_inputs[input_index]);
      }
    } else {
      test.AddInput<xint8>(ss.str().c_str(), {}, {x_zero_points[input_index]}, is_const_inputs[input_index]);
    }
  }
  test.AddOutput<xint8>("Y", y_shape, y_vec);
  test.Run((fail_by == NoFail) ? OpTester::ExpectResult::kExpectSuccess : OpTester::ExpectResult::kExpectFailure);
}

void QLinearConcat3InputsU8(std::vector<bool> is_const_inputs, QLinearConcatFailCause fail_by = NoFail) {
  std::vector<int64_t> y_shape = {2, 6, 3};
  std::vector<std::vector<int64_t>> x_shapes = {
      {2, 1, 3},
      {2, 2, 3},
      {2, 3, 3}};
  std::vector<std::vector<uint8_t>> x_vecs = {
      {128, 126, 131, 123, 255, 0},
      {128, 127, 129, 125, 131, 124, 132, 122, 135, 120, 255, 0},
      {128, 127, 129, 126, 130, 125, 131, 124, 132, 123, 133, 122, 134, 121, 135, 120, 255, 0}};
  std::vector<uint8_t> x_zero_points = {128, 128, 128};

  std::vector<float> x_scales = {1.0, 0.5, 0.25};
  float y_scale = 0.25;
  uint8_t y_zero_point = 128;
  std::vector<uint8_t> y_vec = {
      128, 120, 140,
      128, 126, 130, 122, 134, 120,
      128, 127, 129, 126, 130, 125, 131, 124, 132,
      108, 255, 0,
      136, 116, 142, 112, 255, 0,
      123, 133, 122, 134, 121, 135, 120, 255, 0};

  RunQLinearConcat<uint8_t>(x_shapes, x_vecs, 1, x_scales, x_zero_points, y_shape, y_vec, y_scale, y_zero_point,
                            is_const_inputs, fail_by);

  RunQLinearConcat<uint8_t>(x_shapes, x_vecs, -2, x_scales, x_zero_points, y_shape, y_vec, y_scale, y_zero_point,
                            is_const_inputs, fail_by);
}

TEST(QLinearConcatU8, Input3_DynamicDynamicDynamic) {
  QLinearConcat3InputsU8({false, false, false});
}

TEST(QLinearConcatU8, Input3_ConstConstConst) {
  QLinearConcat3InputsU8({true, true, true});
}

TEST(QLinearConcatU8, Input3_MixedConstDynamic) {
  QLinearConcat3InputsU8({false, true, false});
  QLinearConcat3InputsU8({true, false, true});
  QLinearConcat3InputsU8({false, false, true});
  QLinearConcat3InputsU8({true, false, false});
}

TEST(QLinearConcatU8, ExpectFail_WrongScaleType_0) {
  QLinearConcat3InputsU8({false, false, false}, FailByWrongScaleType);
}

TEST(QLinearConcatU8, ExpectFail_WrongScaleType_1) {
  QLinearConcat3InputsU8({true, true, true}, FailByWrongScaleType);
}

TEST(QLinearConcatU8, ExpectFail_WrongTensorType_0) {
  QLinearConcat3InputsU8({false, false, false}, FailByWrongTensorType);
}

TEST(QLinearConcatU8, ExpectFail_WrongTensorType_1) {
  QLinearConcat3InputsU8({true, true, true}, FailByWrongTensorType);
}

TEST(QLinearConcatU8, ExpectFail_WrongZeroPointType_0) {
  QLinearConcat3InputsU8({false, false, false}, FailByWrongZeroPointType);
}

TEST(QLinearConcatU8, ExpectFail_WrongZeroPointType_1) {
  QLinearConcat3InputsU8({true, true, true}, FailByWrongZeroPointType);
}


void QLinearConcat3InputsS8(std::vector<bool> is_const_inputs, QLinearConcatFailCause fail_by = NoFail) {
  std::vector<int64_t> y_shape = {2, 6, 3};
  std::vector<std::vector<int64_t>> x_shapes = {
      {2, 1, 3},
      {2, 2, 3},
      {2, 3, 3}};
  std::vector<std::vector<int8_t>> x_vecs = {
      {0, -2, 3, -5, 127, -128},
      {0, -1, 2, -3, 3, -4, 4, -6, 7, -8, 127, -128},
      {0, -1, 2, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6, -7, 7, -8, 127, -128}};
  std::vector<int8_t> x_zero_points = {0, 0, 0};

  std::vector<float> x_scales = {1.0, 0.5, 0.25};
  float y_scale = 0.25;
  int8_t y_zero_point = 0;
  std::vector<int8_t> y_vec = {
      0, -8, 12,
      0, -2, 4, -6, 6, -8,
      0, -1, 2, -2, 2, -3, 3, -4, 4,
      -20, 127, -128,
      8, -12, 14, -16, 127, -128,
      -5, 5, -6, 6, -7, 7, -8, 127, -128};

  RunQLinearConcat<int8_t>(x_shapes, x_vecs, 1, x_scales, x_zero_points, y_shape, y_vec, y_scale, y_zero_point,
                           is_const_inputs, fail_by);

  RunQLinearConcat<int8_t>(x_shapes, x_vecs, -2, x_scales, x_zero_points, y_shape, y_vec, y_scale, y_zero_point,
                           is_const_inputs, fail_by);
}

TEST(QLinearConcatS8, Input3_DynamicDynamicDynamic) {
  QLinearConcat3InputsS8({false, false, false});
}

TEST(QLinearConcatS8, Input3_ConstConstConst) {
  QLinearConcat3InputsS8({true, true, true});
}

TEST(QLinearConcatS8, ExpectFail_MixedConstDynamic) {
  QLinearConcat3InputsS8({false, true, false});
  QLinearConcat3InputsS8({true, false, true});
  QLinearConcat3InputsS8({false, false, true});
  QLinearConcat3InputsS8({true, false, false});
}

TEST(QLinearConcatS8, ExpectFail_WrongScaleType_0) {
  QLinearConcat3InputsS8({false, false, false}, FailByWrongScaleType);
}

TEST(QLinearConcatS8, ExpectFail_WrongScaleType_1) {
  QLinearConcat3InputsS8({true, true, true}, FailByWrongScaleType);
}

TEST(QLinearConcatS8, ExpectFail_WrongTensorType_0) {
  QLinearConcat3InputsS8({false, false, false}, FailByWrongTensorType);
}

TEST(QLinearConcatS8, ExpectFail_WrongTensorType_1) {
  QLinearConcat3InputsS8({true, true, true}, FailByWrongTensorType);
}

TEST(QLinearConcatS8, ExpectFail_WrongZeroPointType_0) {
  QLinearConcat3InputsS8({false, false, false}, FailByWrongZeroPointType);
}

TEST(QLinearConcatS8, ExpectFail_WrongZeroPointType_1) {
  QLinearConcat3InputsS8({true, true, true}, FailByWrongZeroPointType);
}

}  // namespace test
}  // namespace onnxruntime
