// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(QuantizeLinearMatmulOpTest, QLinearMatMul3D_U8U8) {
  OpTester test("QLinearMatMul", 10);
  test.AddInput<uint8_t>("T1", {2, 2, 4},
                         {208, 236, 0, 238,
                          3, 214, 255, 29,

                          208, 236, 0, 238,
                          3, 214, 255, 29});

  test.AddInput<float>("a_scale", {}, {0.0066f});
  test.AddInput<uint8_t>("a_zero_point", {}, {113});

  test.AddInput<uint8_t>("T2", {2, 4, 3},
                         {152, 51, 244,
                          60, 26, 255,
                          0, 127, 246,
                          127, 254, 247,

                          152, 51, 244,
                          60, 26, 255,
                          0, 127, 246,
                          127, 254, 247});

  test.AddInput<float>("b_scale", {}, {0.00705f});
  test.AddInput<uint8_t>("b_zero_point", {}, {114});

  test.AddInput<float>("y_scale", {}, {0.0107f});
  test.AddInput<uint8_t>("y_zero_point", {}, {118});
  test.AddOutput<uint8_t>("T3", {2, 2, 3},
                          {168, 115, 255,
                           1, 66, 151,

                           168, 115, 255,
                           1, 66, 151});

  test.Run();
}

TEST(QuantizeLinearMatmulOpTest, QLinearMatMul3D_U8S8) {
  OpTester test("QLinearMatMul", 10);
  test.AddInput<uint8_t>("T1", {2, 2, 4},
                         {208, 126, 0, 238,
                          3, 214, 255, 29,

                          208, 236, 0, 238,
                          3, 214, 255, 29});

  test.AddInput<float>("a_scale", {}, {0.0066f});
  test.AddInput<uint8_t>("a_zero_point", {}, {113});

  test.AddInput<int8_t>("T2", {2, 4, 3},
                         {-43, 51, -34,
                          60, 26, -17,
                          0, 63, -55,
                          47, -29, -31,

                          -62, 51, -42,
                          60, 26, -22,
                          0, -8, -19,
                          37, -2, -47});

  test.AddInput<float>("b_scale", {}, {0.00802f});
  test.AddInput<int8_t>("b_zero_point", {}, {-2});

  test.AddInput<float>("y_scale", {}, {0.0123f});
  test.AddInput<uint8_t>("y_zero_point", {}, {118});
  test.AddOutput<uint8_t>("T3", {2, 2, 3},
                          {130, 95, 114,
                           148, 155, 105,

                           146, 157, 75,
                           160, 101, 134});

  test.Run();
}

static void QLinearMatMul2DTest(bool only_t1_not_initializer) {
  // Test non-empty inputs
  OpTester test_non_empty("QLinearMatMul", 10);
  test_non_empty.AddInput<uint8_t>("T1", {2, 4}, {208, 236, 0, 238, 3, 214, 255, 29});
  test_non_empty.AddInput<float>("a_scale", {1}, {0.0066f}, only_t1_not_initializer);
  test_non_empty.AddInput<uint8_t>("a_zero_point", {1}, {113}, only_t1_not_initializer);
  test_non_empty.AddInput<uint8_t>("T2", {4, 3}, {152, 51, 244, 60, 26, 255, 0, 127, 246, 127, 254, 247}, only_t1_not_initializer);
  test_non_empty.AddInput<float>("b_scale", {1}, {0.00705f}, only_t1_not_initializer);
  test_non_empty.AddInput<uint8_t>("b_zero_point", {1}, {114}, only_t1_not_initializer);
  test_non_empty.AddInput<float>("y_scale", {1}, {0.0107f}, only_t1_not_initializer);
  test_non_empty.AddInput<uint8_t>("y_zero_point", {1}, {118}, only_t1_not_initializer);
  test_non_empty.AddOutput<uint8_t>("T3", {2, 3}, {168, 115, 255, 1, 66, 151});
  test_non_empty.Run();

  // Test with an empty input
  OpTester test_empty("QLinearMatMul", 10);
  test_empty.AddInput<uint8_t>("T1", {0, 4}, {});
  test_empty.AddInput<float>("a_scale", {1}, {0.0066f}, only_t1_not_initializer);
  test_empty.AddInput<uint8_t>("a_zero_point", {1}, {113}, only_t1_not_initializer);
  test_empty.AddInput<uint8_t>("T2", {4, 3}, {152, 51, 244, 60, 26, 255, 0, 127, 246, 127, 254, 247}, only_t1_not_initializer);
  test_empty.AddInput<float>("b_scale", {1}, {0.00705f}, only_t1_not_initializer);
  test_empty.AddInput<uint8_t>("b_zero_point", {1}, {114}, only_t1_not_initializer);
  test_empty.AddInput<float>("y_scale", {1}, {0.0107f}, only_t1_not_initializer);
  test_empty.AddInput<uint8_t>("y_zero_point", {1}, {118}, only_t1_not_initializer);
  test_empty.AddOutput<uint8_t>("T3", {0, 3}, {});

  // Skip NNAPI as it doesn't support empty output for now
  test_empty.Run(OpTester::ExpectResult::kExpectSuccess, "", {kNnapiExecutionProvider});
}

TEST(QuantizeLinearMatmulOpTest, QLinearMatMul) {
  QLinearMatMul2DTest(false);
}

// NNAPI EP requires weight to be an initializer
TEST(QuantizeLinearMatmulOpTest, QLinearMatMulAllInputExceptT1AreInitializers) {
  QLinearMatMul2DTest(true);
}

}  // namespace test
}  // namespace onnxruntime
