// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"

namespace onnxruntime {
namespace test {

TEST(MatmulInteger16OpTest, MatMulInteger16_1) {
  OpTester test("MatMulInteger16", 1, onnxruntime::kMSDomain);
  test.AddInput<int16_t>("T1", {1, 1}, {15});
  test.AddInput<int16_t>("T2", {1, 1}, {16});
  test.AddOutput<int32_t>("T3", {1, 1}, {240});
  test.Run();
}

TEST(MatmulInteger16OpTest, MatMulInteger16_2) {
  OpTester test("MatMulInteger16", 1, onnxruntime::kMSDomain);
  test.AddInput<int16_t>("T1", {1, 2}, {-7, 10});
  test.AddInput<int16_t>("T2", {2, 1}, {-8, -11});
  test.AddOutput<int32_t>("T3", {1, 1}, {-54});
  test.Run();
}

TEST(MatmulInteger16OpTest, MatMulInteger16_3) {
  OpTester test("MatMulInteger16", 1, onnxruntime::kMSDomain);
  test.AddInput<int16_t>("T1", {3, 2}, {-7, 10, 10, -1113, 22, -356});
  test.AddInput<int16_t>("T2", {2, 4}, {-8, -11, 13, 14, -99, 1234, 321, -6});
  test.AddOutput<int32_t>("T3", {3, 4}, {-934, 12417, 3119, -158,
                                         110107, -1373552, -357143, 6818,
                                         35068, -439546, -113990, 2444});
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
