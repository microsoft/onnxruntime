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
  test.AddMissingOptionalInput<int8_t>();  // optional "X_zero_point" using default value here
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

TEST(QLinearLookupTableBasedOperatorTests, QLinearSigmoid_Int8) {
  OpTester test("QLinearSigmoid", 1, onnxruntime::kMSDomain);
  float X_scale = 0.025f;
  //int8_t X_zero_point = 0;
  float Y_scale = 1.0f / 256.0f;
  int8_t Y_zero_point = -120;

  std::vector<int64_t> dims = {16};
  test.AddInput<int8_t>("X", dims, {0, 16, 17, 18, 19, 90, 91, 127, -128, -110, -108, -100, -16, -17, -18, -1});
  test.AddInput<float>("X_scale", {}, {X_scale});
  test.AddMissingOptionalInput<int8_t>();  // optional "X_zero_point" using default value here
  test.AddInput<float>("Y_scale", {}, {Y_scale});
  test.AddInput<int8_t>("Y_zero_point", {}, {Y_zero_point});
  test.AddOutput<int8_t>("Y", dims, {8, 33, 35, 36, 38, 112, 112, 126, -110, -105, -104, -101, -17, -19, -20, 6});
  auto origin_round_mode = std::fegetround();
  std::fesetround(FE_TONEAREST);
  test.Run();
  std::fesetround(origin_round_mode);
}

TEST(QLinearLookupTableBasedOperatorTests, QLinearSigmoid_UInt8) {
  OpTester test("QLinearSigmoid", 1, onnxruntime::kMSDomain);
  float X_scale = 0.025f;
  uint8_t X_zero_point = 128;
  float Y_scale = 1.0f / 256.0f;
  uint8_t Y_zero_point = 8;

  std::vector<int64_t> dims = {16};
  test.AddInput<uint8_t>("X", dims, {0, 16, 17, 18, 19, 90, 91, 127, 128, 136, 137, 138, 216, 217, 218, 255});
  test.AddInput<float>("X_scale", {}, {X_scale});
  test.AddInput<uint8_t>("X_zero_point", {}, {X_zero_point});
  test.AddInput<float>("Y_scale", {}, {Y_scale});
  test.AddInput<uint8_t>("Y_zero_point", {}, {Y_zero_point});
  test.AddOutput<uint8_t>("Y", dims, {18, 23, 23, 23, 24, 79, 81, 134, 136, 149, 150, 152, 238, 239, 240, 254});
  auto origin_round_mode = std::fegetround();
  std::fesetround(FE_TONEAREST);
  test.Run();
  std::fesetround(origin_round_mode);
}

// NNAPI can only take 0 a Y_zero_point
TEST(QLinearLookupTableBasedOperatorTests, QLinearSigmoid_UInt8_0_Y_ZP) {
  auto run_test = [](bool scales_and_zp_are_initializers) {
    OpTester test("QLinearSigmoid", 1, onnxruntime::kMSDomain);
    float X_scale = 0.025f;
    uint8_t X_zero_point = 128;
    float Y_scale = 1.0f / 256.0f;
    uint8_t Y_zero_point = 0;

    std::vector<int64_t> dims = {16};
    test.AddInput<uint8_t>("X", dims, {0, 16, 17, 18, 19, 90, 91, 127, 128, 136, 137, 138, 216, 217, 218, 255});
    test.AddInput<float>("X_scale", {}, {X_scale}, scales_and_zp_are_initializers);
    test.AddInput<uint8_t>("X_zero_point", {}, {X_zero_point}, scales_and_zp_are_initializers);
    test.AddInput<float>("Y_scale", {}, {Y_scale}, scales_and_zp_are_initializers);
    test.AddInput<uint8_t>("Y_zero_point", {}, {Y_zero_point}, scales_and_zp_are_initializers);

    float abs_error = 0.0f;

    // For quantized models, NNAPI's rounding is different than CPU provider
    // Sometimes the result is within +/-1 of result of CPU provider
    // NNAPI is using std::round which is HALF_AWAY_FROM_ZERO, see
    // https://android.googlesource.com/platform/frameworks/ml/+/refs/heads/master/nn/common/operations/Quantize.cpp
    // Use 1 as abs_error which is the smallest possbile for uint8_t
    //
    // NOTE, for now the tolerance will only apply if the NNAPI is actually used,
    // if for any reason the execution falls back to CPU, we still expect an exact match
    // See, 'void Check<uint8_t>(...' in onnxruntime/test/providers/provider_test_utils.cc
#ifdef USE_NNAPI
    abs_error = 1.0f;
#endif

    test.AddOutput<uint8_t>("Y", dims, {10, 15, 15, 15, 16, 71, 73, 126, 128, 141, 142, 144, 230, 231, 232, 246},
                            false /* sort_output */, 0.0f /* rel_error */, abs_error);
    auto origin_round_mode = std::fegetround();
    std::fesetround(FE_TONEAREST);
    test.Run();
    std::fesetround(origin_round_mode);
  };

  run_test(false);
  run_test(true);
}

}  // namespace test
}  // namespace onnxruntime
