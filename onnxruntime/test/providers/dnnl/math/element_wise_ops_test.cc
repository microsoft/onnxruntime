// Copyright (c)Intel. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
#include "core/util/math.h"
#include <algorithm>
#include <math.h>

namespace onnxruntime {
namespace test {

#ifdef USE_DNNL
// Many of the "Pow" tests are identical to the CPU element wise ops tests with the
// exception the exponent for the "Pow" operator is setup as an initilizer value. Since
// the DNNL execution provider will only accept exponents that are initializers. This matches
// what is seen in many models that use "Pow"
TEST(MathOpTest, DNNL_Pow_Broadcast_Scalar1) {
  OpTester test("Pow");

  std::vector<int64_t> dims{3};
  test.AddInput<float>("X", dims, {1.0f, 2.0f, 3.0f});
  test.AddInput<float>("Y", {}, {2.0f}, true);
  test.AddOutput<float>("Z", dims, {1.0f, 4.0f, 9.0f});
  test.Run();
}

TEST(MathOpTest, DNNL_Pow_Broadcast_Scalar1_12) {
  OpTester test("Pow", 12);

  std::vector<int64_t> dims{3};
  test.AddInput<float>("X", dims, {1.0f, 2.0f, 3.0f});
  test.AddInput<float>("Y", {1}, {2.0f}, true);
  test.AddOutput<float>("Z", dims, {1.0f, 4.0f, 9.0f});
  test.Run();
}

TEST(MathOpTest, DNNL_Pow_Broadcast_Scalar1_float_int32_12) {
  OpTester test("Pow", 12);

  std::vector<int64_t> dims{3};
  test.AddInput<float>("X", dims, {1.0f, 2.0f, 3.0f});
  test.AddInput<int32_t>("Y", {}, {3}, true);
  test.AddOutput<float>("Z", dims, {1.0f, 8.0f, 27.0f});
  test.Run();
}

TEST(MathOpTest, DNNL_Pow_Broadcast_Scalar1_float_int8_12) {
  OpTester test("Pow", 12);

  std::vector<int64_t> dims{3};
  test.AddInput<float>("X", dims, {1.0f, 2.0f, 3.0f});
  test.AddInput<int8_t>("Y", {}, {3}, true);
  test.AddOutput<float>("Z", dims, {1.0f, 8.0f, 27.0f});
  test.Run();
}

TEST(MathOpTest, DNNL_Pow_Broadcast_Scalar1_float_uint8_12) {
  OpTester test("Pow", 12);

  std::vector<int64_t> dims{3};
  test.AddInput<float>("X", dims, {1.0f, 2.0f, 3.0f});
  test.AddInput<uint8_t>("Y", {}, {3}, true);
  test.AddOutput<float>("Z", dims, {1.0f, 8.0f, 27.0f});
  test.Run();
}
#endif  // USE_DNNL

}  // namespace test
}  // namespace onnxruntime