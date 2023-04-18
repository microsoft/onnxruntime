#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

#include <cfenv>

namespace onnxruntime {
namespace test {

TEST(QLinearLookupTableBasedOperatorTests, QLinearLeakyRelu_Int8) {
  OpTester test("QLinearLeakyRelu", 1, onnxruntime::kMSDomain);
  test.AddAttribute<float>("alpha", 0.1f);
  float X_scale = 0.25f;
  // int8_t X_zero_point = 0;
  float Y_scale = 0.1f;
  int8_t Y_zero_point = -100;

  std::vector<int64_t> dims = {16};
  test.AddInput<int8_t>("X", dims, {0, 16, 17, 18, 19, 90, 91, 127, -128, -110, -108, -100, -16, -17, -18, -1});
  test.AddInput<float>("X_scale", {}, {X_scale});
  test.AddOptionalInputEdge<int8_t>();  // optional "X_zero_point" using default value here
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
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: AbiCustomRegistry.cpp(507): The parameter is incorrect";
  }

  OpTester test("QLinearSigmoid", 1, onnxruntime::kMSDomain);
  float X_scale = 0.025f;
  // int8_t X_zero_point = 0;
  float Y_scale = 1.0f / 256.0f;
  int8_t Y_zero_point = -120;

  std::vector<int64_t> dims = {16};
  test.AddInput<int8_t>("X", dims, {0, 16, 17, 18, 19, 90, 91, 127, -128, -110, -108, -100, -16, -17, -18, -1});
  test.AddInput<float>("X_scale", {}, {X_scale});
  test.AddOptionalInputEdge<int8_t>();  // optional "X_zero_point" using default value here
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

// NNAPI can only take 0 as Y_zero_point
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
    test.AddOutput<uint8_t>("Y", dims, {10, 15, 15, 15, 16, 71, 73, 126, 128, 141, 142, 144, 230, 231, 232, 246});
    auto origin_round_mode = std::fegetround();
    std::fesetround(FE_TONEAREST);
    test.Run();
    std::fesetround(origin_round_mode);
  };

  run_test(false);
  run_test(true);
}

/*
\brief data is generated by pytorch script
\details model defines
```
        input(int8/uint8)
        x = self.dequant(x)
        x = self.softmax(x)
        x = self.quant2(x)
        output(int8/uint8)
```
\see then followed by the [DOC](https://pytorch.org/docs/stable/quantization.html)
*/
TEST(QLinearLookupTableBasedOperatorTests, QLinearSoftmax_UInt8_v12) {
  auto run_test = [](bool add_shape_to_input) {
    OpTester test("QLinearSoftmax", 1, onnxruntime::kMSDomain);
    test.AddAttribute<int64_t>("axis", -2);
    test.AddAttribute<int64_t>("opset", 12);
    float X_scale = 0.166099221f;
    //
    uint8_t X_zero_point = 128;
    float Y_scale = 1.0f / 256.0f;
    uint8_t Y_zero_point = 0;
    //

    std::vector<int64_t> dims = {2, 4, 5};
    auto x_in = std::vector<uint8_t>{50, 67, 58, 68, 46, 69, 77, 91, 62, 74, 67, 72, 71, 70, 83, 88, 75, 54, 74, 88};
    auto y_out = std::vector<uint8_t>{0, 2, 0, 2, 0, 2, 8, 86, 1, 5, 2, 4, 3, 3, 23, 52, 6, 0, 5, 52};
    for (int64_t i = 1; i < dims[0]; i++) {
      for (int64_t j = 0; j < dims[1] * dims[2]; j++) {
        x_in.push_back(x_in[j]);
        y_out.push_back(y_out[j]);
      }
    }

    test.AddShapeToTensorData(add_shape_to_input);
    test.AddInput<uint8_t>("X", dims, x_in);
    test.AddInput<float>("X_scale", {}, {X_scale});
    test.AddInput<uint8_t>("X_zero_point", {}, {X_zero_point});
    test.AddInput<float>("Y_scale", {}, {Y_scale});
    test.AddInput<uint8_t>("Y_zero_point", {}, {Y_zero_point});
    test.AddOutput<uint8_t>("Y", dims, y_out);
    auto origin_round_mode = std::fegetround();
    std::fesetround(FE_TONEAREST);
    test.Run();
    std::fesetround(origin_round_mode);
  };
  run_test(true);
  run_test(false);
}

TEST(QLinearLookupTableBasedOperatorTests, QLinearSoftmax_UInt8_v13) {
  auto run_test = [](bool add_shape_to_input) {
    OpTester test("QLinearSoftmax", 1, onnxruntime::kMSDomain);
    test.AddAttribute<int64_t>("axis", -2);
    test.AddAttribute<int64_t>("opset", 13);
    float X_scale = 0.0304f;
    //
    uint8_t X_zero_point = 128;
    float Y_scale = 0.0059f;
    uint8_t Y_zero_point = 0;
    //

    std::vector<int64_t> dims = {4, 4, 4};
    auto x_in = std::vector<uint8_t>{
        62, 50, 71, 37, 68, 88, 64, 51, 59, 95, 41, 54, 55, 20, 77, 32, 92,
        63, 43, 13, 76, 82, 53, 43, 60, 18, 73, 74, 22, 89, 44, 106, 17,
        95, 27, 35, 47, 57, 0, 78, 97, 66, 56, 28, 127, 33, 106, 71, 119,
        64, 16, 0, 16, 79, 27, 89, 110, 126, 88, 90, 67, 11, 4, 90};
    auto y_out = std::vector<uint8_t>{
        43, 20, 50, 33, 52, 63, 40, 51, 39, 78,
        20, 56, 35, 8, 59, 29, 80, 32, 29, 6, 49, 57, 39, 16, 30, 8, 72, 40,
        10, 71, 30, 107, 4, 90, 11, 20, 10, 28, 5, 74, 45, 37, 27, 16, 111, 14,
        125, 59, 84, 18, 14, 4, 4, 28, 20, 54, 64, 119, 126, 56, 17, 4, 10, 56};

    test.AddShapeToTensorData(add_shape_to_input);
    test.AddInput<uint8_t>("X", dims, x_in);
    test.AddInput<float>("X_scale", {}, {X_scale});
    test.AddInput<uint8_t>("X_zero_point", {}, {X_zero_point});
    test.AddInput<float>("Y_scale", {}, {Y_scale});
    test.AddInput<uint8_t>("Y_zero_point", {}, {Y_zero_point});
    test.AddOutput<uint8_t>("Y", dims, y_out);
    auto origin_round_mode = std::fegetround();
    std::fesetround(FE_TONEAREST);
    test.Run();
    std::fesetround(origin_round_mode);
  };
  run_test(true);
  run_test(false);
}

TEST(QLinearLookupTableBasedOperatorTests, QLinearSoftmax_Int8_v13) {
  auto run_test = [](bool add_shape_to_input) {
    OpTester test("QLinearSoftmax", 1, onnxruntime::kMSDomain);
    test.AddAttribute<int64_t>("axis", -2);
    test.AddAttribute<int64_t>("opset", 13);
    float X_scale = 0.0304F;
    //
    int8_t X_zero_point = 0;
    float Y_scale = 0.0059F;
    int8_t Y_zero_point = -128;
    //

    std::vector<int64_t> dims = {4, 4, 4};
    auto x_in = std::vector<int8_t>{
        -4, -16, 5, -29, 2, 22, -2, -15, -7, 29, -25, -12, -11, -46, 11, -34, 26,
        -3, -23, -53, 10, 16, -13, -23, -6, -48, 7, 8, -44, 23, -22, 40, -49, 29, -39, -31, -19, -9,
        -72, 12, 31, 0, -10, -38, 61, -33, 40, 5, 53, -2, -50, -66, -50, 13, -39, 23, 44, 60, 22, 24,
        1, -55, -62, 24};
    auto y_out = std::vector<int8_t>{
        -85, -108, -78, -95, -76, -65, -88, -77, -89, -50, -108, -72, -93,
        -120, -69, -99, -48, -96, -99, -122, -79, -71, -89, -112, -98, -120, -56, -88, -118, -57, -98,
        -21, -124, -38, -117, -108, -118, -100, -124, -54, -83, -91, -100, -112, -17, -114, -2, -69, -44,
        -110, -114, -124, -124, -100, -108, -74, -64, -9, -2, -72, -111, -124, -118, -72};

    test.AddShapeToTensorData(add_shape_to_input);
    test.AddInput<int8_t>("X", dims, x_in);
    test.AddInput<float>("X_scale", {}, {X_scale});
    test.AddInput<int8_t>("X_zero_point", {}, {X_zero_point});
    test.AddInput<float>("Y_scale", {}, {Y_scale});
    test.AddInput<int8_t>("Y_zero_point", {}, {Y_zero_point});
    test.AddOutput<int8_t>("Y", dims, y_out);
    auto origin_round_mode = std::fegetround();
    std::fesetround(FE_TONEAREST);
    test.Run();
    std::fesetround(origin_round_mode);
  };
  run_test(true);
  run_test(false);
}

TEST(QLinearLookupTableBasedOperatorTests, QLinearSoftmax_Int8_v12) {
  auto run_test = [](bool add_shape_to_input) {
    OpTester test("QLinearSoftmax", 1, onnxruntime::kMSDomain);
    test.AddAttribute<int64_t>("axis", -2);
    test.AddAttribute<int64_t>("opset", 12);
    float X_scale = 0.166099221f;
    //
    int8_t X_zero_point = 0;
    float Y_scale = 1.0f / 128.0f;
    int8_t Y_zero_point = 0;
    //

    std::vector<int64_t> dims = {2, 4, 5};
    auto x_in = std::vector<int8_t>{-28, -4, -4, -7, 3, -26, 4, -16, 23, 14, -7, 26, -8, 19, -16, -13, 7, 17, 27, 5};
    auto y_out = std::vector<int8_t>{0, 0, 0, 0, 1, 0, 1, 0, 22, 5, 0, 35, 0, 11, 0, 0, 2, 8, 42, 1};
    for (int64_t i = 1; i < dims[0]; i++) {
      for (int64_t j = 0; j < dims[1] * dims[2]; j++) {
        x_in.push_back(x_in[j]);
        y_out.push_back(y_out[j]);
      }
    }

    test.AddShapeToTensorData(add_shape_to_input);
    test.AddInput<int8_t>("X", dims, x_in);
    test.AddInput<float>("X_scale", {}, {X_scale});
    test.AddInput<int8_t>("X_zero_point", {}, {X_zero_point});
    test.AddInput<float>("Y_scale", {}, {Y_scale});
    test.AddInput<int8_t>("Y_zero_point", {}, {Y_zero_point});
    test.AddOutput<int8_t>("Y", dims, y_out);
    auto origin_round_mode = std::fegetround();
    std::fesetround(FE_TONEAREST);
    test.Run();
    std::fesetround(origin_round_mode);
  };
  run_test(true);
  run_test(false);
}

}  // namespace test
}  // namespace onnxruntime
