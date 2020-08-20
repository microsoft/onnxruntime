#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

#include <cfenv>

namespace onnxruntime {
namespace test {

TEST(QLinearLookupTableBasedOperatorTests, QLinearLeakyRelu_Int8) {
  OpTester test("QLinearLeakyRelu", 1, onnxruntime::kMSDomain);
  test.AddAttribute<float>("alpha", 0.1f);
  float X_scale = 0.25f;
  //int8_t X_zero_point = 0;
  float Y_scale = 0.1f;
  int8_t Y_zero_point = -100;

  std::vector<int64_t> dims = {16};
  test.AddInput<int8_t>("X", dims, {0, 16, 17, 18, 19, 90, 91, 127, -128, -110, -108, -100, -16, -17, -18, -1});
  test.AddInput<float>("X_scale", {}, {X_scale});
  test.AddMissingOptionalInput<int8_t>(); // optional "X_zero_point" using default value here
  test.AddInput<float>("Y_scale", {}, {Y_scale});
  test.AddInput<int8_t>("Y_zero_point", {}, {Y_zero_point});
  test.AddOutput<int8_t>("Y", dims, {-100, -60, -58, -55, -52, 125, 127, 127, -128, -128, -127, -125, -104, -104, -104, -100});
  auto origin_round_mode = std::fegetround();
  std::fesetround(FE_TONEAREST);
  test.Run();
  std::fesetround(origin_round_mode);
}


TEST(QLinearLookupTableBasedOperatorTests, QLinearLeakyRelu_UInt8) {
  OpTester test("QLinearLeakyRelu", 1, onnxruntime::kMSDomain);
  test.AddAttribute<float>("alpha", 0.1f);
  float X_scale = 0.25f;
  uint8_t X_zero_point = 128;
  float Y_scale = 0.1f;
  uint8_t Y_zero_point = 30;

  std::vector<int64_t> dims = {16};
  test.AddInput<uint8_t>("X", dims, {0, 16, 17, 18, 19, 90, 91, 127, 128, 136, 137, 138, 216, 217, 218, 255});
  test.AddInput<float>("X_scale", {}, {X_scale});
  test.AddInput<uint8_t>("X_zero_point", {}, {X_zero_point});
  test.AddInput<float>("Y_scale", {}, {Y_scale});
  test.AddInput<uint8_t>("Y_zero_point", {}, {Y_zero_point});
  test.AddOutput<uint8_t>("Y", dims, {0, 2, 2, 2, 3, 20, 21, 30, 30, 50, 52, 55, 250, 252, 255, 255});
  auto origin_round_mode = std::fegetround();
  std::fesetround(FE_TONEAREST);
  test.Run();
  std::fesetround(origin_round_mode);
}

}
}
