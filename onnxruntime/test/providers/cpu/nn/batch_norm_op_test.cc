// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensor.h"
#include "core/session/inference_session.h"
#include "test/providers/provider_test_utils.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

using namespace std;

namespace onnxruntime {
namespace test {

using InputDataMap = unordered_map<string, vector<float>>;
using InputShapesMap = unordered_map<string, vector<int64_t>>;

template <typename T>
void TestBatchNorm(const unordered_map<string, vector<T>>& input_data_map,
                   const InputShapesMap& input_shapes_map,
                   optional<float> epsilon,
                   const std::initializer_list<T>& expected_output,
                   const vector<int64_t>& expected_output_shape,
                   bool all_input_except_x_are_initializers = false,
                   int64_t spatial_mode = 1,
                   OpTester::ExpectResult expect_result = OpTester::ExpectResult::kExpectSuccess,
                   const std::string& err_str = "",
                   int opset_version = 9) {
  OpTester test("BatchNormalization", opset_version);
  if (epsilon.has_value()) {
    test.AddAttribute("epsilon", epsilon.value());
  }
  if (opset_version < 9) {  // spatial is only defined for opset-8 and below in the spec
    test.AddAttribute("spatial", spatial_mode);
  }
  test.AddInput<T>("X", input_shapes_map.at("X"), input_data_map.at("X"));
  test.AddInput<T>("scale", input_shapes_map.at("scale"), input_data_map.at("scale"), all_input_except_x_are_initializers);
  test.AddInput<T>("B", input_shapes_map.at("B"), input_data_map.at("B"), all_input_except_x_are_initializers);
  test.AddInput<T>("mean", input_shapes_map.at("mean"), input_data_map.at("mean"), all_input_except_x_are_initializers);
  test.AddInput<T>("var", input_shapes_map.at("var"), input_data_map.at("var"), all_input_except_x_are_initializers);
  test.AddOutput<T>("output", expected_output_shape, expected_output);
  // Weight as input is not supported by TensorRT and spatial == 0 is not supported by Nuphar
  std::unordered_set<std::string> excluded_eps = {kTensorrtExecutionProvider};
  if (spatial_mode == 0) {
    excluded_eps.insert(kOpenVINOExecutionProvider);
  }

  // OpenVINO: Disabled due to software limitations
  #if defined(OPENVINO_CONFIG_GPU_FP32) || defined(OPENVINO_CONFIG_GPU_FP16) || defined(OPENVINO_CONFIG_MYRIAD) || defined(OPENVINO_CONFIG_VAD_M) || defined(OPENVINO_CONFIG_CPU_FP32)
    excluded_eps.insert(kOpenVINOExecutionProvider);
  #endif
  test.Run(expect_result, err_str, excluded_eps);
}

TEST(BatchNormTest, PositiveTestCase) {
  // This input was taken from the SpatialBN_1.pb, SpatialBN_1_input.pb and SpatialBN_1_output.pb files.
  vector<float> X{0.329876f, -0.287158f, -0.411425f, 0.473621f, 0.18156f, -0.170596f, -0.329516f, -0.170733f, -0.121664f, 0.4372f,
                  -0.485668f, 0.218049f, -0.360263f, 0.107016f, 0.45358f, 0.325056f, 0.15995f, 0.098852f, -0.283453f, -0.373051f,
                  0.257542f, 0.0614853f, -0.0592363f, 0.434488f, -0.0179583f, 0.398374f, -0.451602f, -0.132009f, -0.174468f,
                  -0.0247169f, 0.418897f, -0.47159f, -0.131925f, 0.470943f, 0.118357f, 0.155664f, 0.370062f, -0.279229f, 0.240311f,
                  -0.451034f, 0.249178f, -0.294496f, 0.13683f, -0.0806475f, -0.309849f, -0.450604f, -0.28048f, -0.420197f, -0.433369f};
  vector<float> scale{0.589433f};
  vector<float> B{-0.384622f};
  vector<float> mean{-2.45673f};
  vector<float> var{1.37998f};

  InputDataMap input_data_map;
  input_data_map.insert({"X", X});
  input_data_map.insert({"scale", scale});
  input_data_map.insert({"B", B});
  input_data_map.insert({"mean", mean});
  input_data_map.insert({"var", var});

  InputShapesMap input_shapes_map;
  vector<int64_t> input_shape{1, 1, 7, 7, 1};
  input_shapes_map.insert({"X", input_shape});
  input_shapes_map.insert({"scale", {1}});
  input_shapes_map.insert({"B", {1}});
  input_shapes_map.insert({"mean", {1}});
  input_shapes_map.insert({"var", {1}});

  auto expected_output = {1.01359f, 0.703983f, 0.641631f, 1.08571f, 0.939167f, 0.762469f, 0.682729f, 0.762401f, 0.787021f,
                          1.06744f, 0.604378f, 0.957476f, 0.667302f, 0.901764f, 1.07566f, 1.01117f, 0.928324f, 0.897667f,
                          0.705842f, 0.660885f, 0.977291f, 0.878918f, 0.818345f, 1.06608f, 0.839057f, 1.04796f, 0.621471f,
                          0.781831f, 0.760527f, 0.835665f, 1.05825f, 0.611442f, 0.781873f, 1.08437f, 0.907454f, 0.926173f,
                          1.03375f, 0.707961f, 0.968646f, 0.621757f, 0.973095f, 0.700301f, 0.916723f, 0.807602f, 0.692598f,
                          0.621972f, 0.707334f, 0.63723f, 0.63062f};
  float epsilon = 1e-05f;
  TestBatchNorm(input_data_map, input_shapes_map, epsilon, expected_output, input_shape);
}

TEST(BatchNormTest, PositiveTestCaseDouble) {
  // This input was taken from the SpatialBN_1.pb, SpatialBN_1_input.pb and SpatialBN_1_output.pb files.
  vector<double> X{0.329876f, -0.287158f, -0.411425f, 0.473621f, 0.18156f, -0.170596f, -0.329516f, -0.170733f, -0.121664f, 0.4372f,
                   -0.485668f, 0.218049f, -0.360263f, 0.107016f, 0.45358f, 0.325056f, 0.15995f, 0.098852f, -0.283453f, -0.373051f,
                   0.257542f, 0.0614853f, -0.0592363f, 0.434488f, -0.0179583f, 0.398374f, -0.451602f, -0.132009f, -0.174468f,
                   -0.0247169f, 0.418897f, -0.47159f, -0.131925f, 0.470943f, 0.118357f, 0.155664f, 0.370062f, -0.279229f, 0.240311f,
                   -0.451034f, 0.249178f, -0.294496f, 0.13683f, -0.0806475f, -0.309849f, -0.450604f, -0.28048f, -0.420197f, -0.433369f};
  vector<double> scale{0.589433f};
  vector<double> B{-0.384622f};
  vector<double> mean{-2.45673f};
  vector<double> var{1.37998f};

  unordered_map<string, vector<double>> input_data_map;
  input_data_map.insert({"X", X});
  input_data_map.insert({"scale", scale});
  input_data_map.insert({"B", B});
  input_data_map.insert({"mean", mean});
  input_data_map.insert({"var", var});

  InputShapesMap input_shapes_map;
  vector<int64_t> input_shape{1, 1, 7, 7, 1};
  input_shapes_map.insert({"X", input_shape});
  input_shapes_map.insert({"scale", {1}});
  input_shapes_map.insert({"B", {1}});
  input_shapes_map.insert({"mean", {1}});
  input_shapes_map.insert({"var", {1}});

  const std::initializer_list<double> expected_output = {1.01359f, 0.703983f, 0.641631f, 1.08571f, 0.939167f, 0.762469f, 0.682729f, 0.762401f, 0.787021f,
                                                         1.06744f, 0.604378f, 0.957476f, 0.667302f, 0.901764f, 1.07566f, 1.01117f, 0.928324f, 0.897667f,
                                                         0.705842f, 0.660885f, 0.977291f, 0.878918f, 0.818345f, 1.06608f, 0.839057f, 1.04796f, 0.621471f,
                                                         0.781831f, 0.760527f, 0.835665f, 1.05825f, 0.611442f, 0.781873f, 1.08437f, 0.907454f, 0.926173f,
                                                         1.03375f, 0.707961f, 0.968646f, 0.621757f, 0.973095f, 0.700301f, 0.916723f, 0.807602f, 0.692598f,
                                                         0.621972f, 0.707334f, 0.63723f, 0.63062f};
  float epsilon = 1e-05f;
  TestBatchNorm(input_data_map, input_shapes_map, epsilon, expected_output, input_shape);
}

TEST(BatchNormTest, PositiveTestCaseDefaultEpsilon) {
  // This input was taken from the SpatialBN_1.pb, SpatialBN_1_input.pb and SpatialBN_1_output.pb files from an older version of this project
  vector<float> X{0.329876f, -0.287158f, -0.411425f, 0.473621f, 0.18156f, -0.170596f, -0.329516f, -0.170733f, -0.121664f, 0.4372f,
                  -0.485668f, 0.218049f, -0.360263f, 0.107016f, 0.45358f, 0.325056f, 0.15995f, 0.098852f, -0.283453f, -0.373051f,
                  0.257542f, 0.0614853f, -0.0592363f, 0.434488f, -0.0179583f, 0.398374f, -0.451602f, -0.132009f, -0.174468f,
                  -0.0247169f, 0.418897f, -0.47159f, -0.131925f, 0.470943f, 0.118357f, 0.155664f, 0.370062f, -0.279229f, 0.240311f,
                  -0.451034f, 0.249178f, -0.294496f, 0.13683f, -0.0806475f, -0.309849f, -0.450604f, -0.28048f, -0.420197f, -0.433369f};
  vector<float> scale{0.589433f};
  vector<float> B{-0.384622f};
  vector<float> mean{-2.45673f};
  vector<float> var{1.37998f};

  InputDataMap input_data_map;
  input_data_map.insert({"X", X});
  input_data_map.insert({"scale", scale});
  input_data_map.insert({"B", B});
  input_data_map.insert({"mean", mean});
  input_data_map.insert({"var", var});

  InputShapesMap input_shapes_map;
  vector<int64_t> input_shape{1, 1, 7, 7, 1};
  input_shapes_map.insert({"X", input_shape});
  input_shapes_map.insert({"scale", {1}});
  input_shapes_map.insert({"B", {1}});
  input_shapes_map.insert({"mean", {1}});
  input_shapes_map.insert({"var", {1}});

  auto expected_output = {1.01359f, 0.703983f, 0.641631f, 1.08571f, 0.939167f, 0.762469f, 0.682729f, 0.762401f, 0.787021f,
                          1.06744f, 0.604378f, 0.957476f, 0.667302f, 0.901764f, 1.07566f, 1.01117f, 0.928324f, 0.897667f,
                          0.705842f, 0.660885f, 0.977291f, 0.878918f, 0.818345f, 1.06608f, 0.839057f, 1.04796f, 0.621471f,
                          0.781831f, 0.760527f, 0.835665f, 1.05825f, 0.611442f, 0.781873f, 1.08437f, 0.907454f, 0.926173f,
                          1.03375f, 0.707961f, 0.968646f, 0.621757f, 0.973095f, 0.700301f, 0.916723f, 0.807602f, 0.692598f,
                          0.621972f, 0.707334f, 0.63723f, 0.63062f};
  optional<float> epsilon;
  TestBatchNorm(input_data_map, input_shapes_map, epsilon, expected_output, input_shape);
}

TEST(BatchNormTest, BatchNorm1d_3d_Pytorch) {
  vector<float> X{0.948241f, 1.23591f, -0.39321f, 1.4254f, -0.730771f, 0.439872f, 0.0265089f, 0.8748f,
                  -0.197505f, 0.962646f, 0.421469f, 1.94512f, 0.234179f, -0.931897f, -0.214905f, -0.982965f,
                  -0.495436f, 0.81949f, -0.796605f, -0.758605f, 0.665557f, 0.0909539f, 1.10448f, 1.91214f, -1.97433f,
                  -2.26429f, -0.384419f, -0.226564f, 0.230568f, 0.533968f, -1.31382f, -0.156257f, 0.532323f,
                  -0.16714f, 0.971087f, 0.600249f, 0.858778f, 0.423108f, -0.414433f, -1.17608f, 0.673753f, 0.278517f,
                  -2.19044f, -0.161453f, 1.17092f, -0.155138f, -0.094729f, 0.19479f, -1.17344f, -0.213813f, 0.118659f,
                  -2.39525f, 0.257687f, 0.784609f, 0.297942f, 1.10277f, -1.58026f, 0.197625f, 0.0432784f, 1.12924f};
  vector<float> scale{0.36102f, 0.592982f, 0.808513f, 0.0531484f, 0.0960613f};
  vector<float> B{0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  vector<float> mean{0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  vector<float> var{1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

  InputDataMap input_data_map;
  input_data_map.insert({"X", X});
  input_data_map.insert({"scale", scale});
  input_data_map.insert({"B", B});
  input_data_map.insert({"mean", mean});
  input_data_map.insert({"var", var});

  InputShapesMap input_shapes_map;
  vector<int64_t> input_shape{4, 5, 3};
  input_shapes_map.insert({"X", input_shape});
  input_shapes_map.insert({"scale", {5}});
  input_shapes_map.insert({"B", {5}});
  input_shapes_map.insert({"mean", {5}});
  input_shapes_map.insert({"var", {5}});

  auto expected_output = {0.342332f, 0.446184f, -0.141956f, 0.845231f, -0.433332f, 0.260835f, 0.0214327f, 0.707284f,
                          -0.159685f, 0.0511628f, 0.0224003f, 0.10338f, 0.0224955f, -0.0895188f, -0.020644f, -0.354868f,
                          -0.178861f, 0.295851f, -0.47237f, -0.449836f, 0.394661f, 0.073537f, 0.892985f, 1.54598f,
                          -0.104932f, -0.120343f, -0.0204312f, -0.021764f, 0.0221486f, 0.0512934f, -0.474312f,
                          -0.0564117f, 0.192178f, -0.0991106f, 0.575834f, 0.355935f, 0.69433f, 0.342087f, -0.335073f,
                          -0.0625065f, 0.0358087f, 0.0148027f, -0.210415f, -0.0155093f, 0.11248f, -0.0560077f, -0.0341989f,
                          0.0703226f, -0.695826f, -0.126787f, 0.0703623f, -1.93658f, 0.208342f, 0.634363f, 0.0158351f,
                          0.0586101f, -0.0839879f, 0.018984f, 0.00415736f, 0.108476f};
  float epsilon = 1e-05f;
  TestBatchNorm(input_data_map, input_shapes_map, epsilon, expected_output, input_shape);

  // NNAPI EP will need all inputs except X be initializers
  TestBatchNorm(input_data_map, input_shapes_map, epsilon, expected_output, input_shape, true);
}

TEST(BatchNormTest, BatchNorm2d_Pytorch) {
  vector<float> X{-0.91221f, -0.283559f, 0.937637f, 2.09818f, -0.100199f, -0.608113f, 0.444562f, -1.07505f, 0.940591f,
                  -0.922262f, 0.0931303f, 0.69611f, 1.55187f, 0.159808f, 0.914874f, -1.24856f, -1.98928f, -0.331621f,
                  2.33131f, 0.260409f, 0.0944811f, 0.442397f, 0.76461f, -0.203334f, -0.244228f, -0.387267f, -1.65039f,
                  -0.815409f, 0.931696f, -1.15328f, 0.773952f, -1.28195f, -0.437349f, 0.0644882f, -0.087637f, 1.74999f,
                  0.640154f, -0.505641f, -1.84014f, -0.00135415f, 0.782006f, -1.21172f, -0.621273f, -0.0977471f,
                  -0.941333f, -0.170302f, 0.18923f, 0.436322f, 0.870412f, -0.582312f, 0.679017f, 0.510252f, 0.0786005f,
                  0.160138f, -2.61889f, 0.402828f, 0.551144f, -1.39366f, -1.15191f, 0.160008f, -0.57735f, 0.210758f,
                  1.0541f, -2.12569f, 0.101656f, 1.10223f, 0.725811f, -1.5019f, -0.0892582f, 0.063546f, 0.822734f,
                  1.67707f, 0.478121f, -1.07438f, -0.0487855f, 0.0972885f, -1.54122f, 2.47422f, 0.596108f, 0.0026957f,
                  -0.967677f, -2.08882f, 0.469692f, 0.630784f, 0.196915f, -1.91331f, 1.26255f, 0.0491993f, -0.358415f,
                  0.720588f, 0.976776f, -0.418116f, 1.70979f, 2.49971f, 1.30942f, -1.18304f, -1.64901f, -1.11048f,
                  1.41467f, -0.275486f, -1.20602f, -0.545566f, -0.918059f, 1.48513f, 2.04224f, -0.96909f, -1.92804f,
                  0.634147f, -1.02079f, -0.000786079f, 0.72428f, 0.893569f, 1.14604f, -1.3423f, -1.05061f, -0.617524f,
                  -0.12619f, -0.203127f, -0.941956f, 2.06916f, 2.03025f, 0.37269f, -0.340471f, -1.27962f, 0.159472f,
                  0.643999f, 0.881773f, -0.50873f, 1.04599f, -0.287968f, 1.84344f, -0.728637f, 0.668021f, -2.00452f,
                  -0.585523f, -0.24982f, -0.379091f, 0.213692f, 1.21336f, -0.499157f, -1.50841f, -1.01256f, 0.745338f,
                  0.591107f, 1.33781f, -0.258927f, -1.87304f, 0.884799f, 1.63174f, 0.500887f, 1.80608f, -1.25441f,
                  -0.655316f, 1.22439f, 0.384174f, 0.401395f, 1.43172f, 1.85338f, -0.644909f, -1.46975f, -1.06138f,
                  1.09724f, 0.013438f, 0.589742f, -0.695768f, 0.758401f, 0.924533f, -0.0988563f, -0.197066f, 1.01118f,
                  0.195163f, 0.975466f, 1.7682f, 0.977977f, -0.88963f, -0.251431f, 0.115828f, -0.230065f, -1.08882f,
                  1.62318f, 0.502684f, 0.789724f, 1.13057f, -0.890021f, -0.614755f, 1.11055f, -1.21681f, 0.133085f,
                  0.564458f, 0.723117f, 1.67088f, -0.111012f, 1.39732f, -0.846095f, -0.194408f, -0.381931f, -0.735943f,
                  -0.788814f, -0.910318f, 1.16345f, -1.98542f, 0.742905f, -0.749476f, -0.110805f, 0.307949f, -1.66811f,
                  0.294031f, -0.522837f, -0.774399f, -0.264072f, -0.426894f, 0.965971f, 0.173348f, -0.991018f, 1.9406f,
                  0.0853744f};
  vector<float> scale{0.736494f, 0.580251f, 0.374834f};
  vector<float> B{0.0f, 0.0f, 0.0f};
  vector<float> mean{0.0f, 0.0f, 0.0f};
  vector<float> var{1.0f, 1.0f, 1.0f};

  InputDataMap input_data_map;
  input_data_map.insert({"X", X});
  input_data_map.insert({"scale", scale});
  input_data_map.insert({"B", B});
  input_data_map.insert({"mean", mean});
  input_data_map.insert({"var", var});

  InputShapesMap input_shapes_map;
  vector<int64_t> input_shape{2, 3, 6, 6};
  input_shapes_map.insert({"X", input_shape});
  input_shapes_map.insert({"scale", {3}});
  input_shapes_map.insert({"B", {3}});
  input_shapes_map.insert({"mean", {3}});
  input_shapes_map.insert({"var", {3}});

  auto expected_output = {-0.671834f, -0.208838f, 0.69056f, 1.54529f, -0.0737958f, -0.447869f, 0.327415f, -0.791764f,
                          0.692736f, -0.679237f, 0.0685895f, 0.512678f, 1.14294f, 0.117697f, 0.673796f, -0.91955f,
                          -1.46508f, -0.244235f, 1.71699f, 0.191789f, 0.0695843f, 0.325821f, 0.563128f, -0.149753f,
                          -0.179871f, -0.285218f, -1.2155f, -0.60054f, 0.686185f, -0.84938f, 0.570008f, -0.944146f,
                          -0.322103f, 0.0474949f, -0.0645437f, 1.28885f, 0.371448f, -0.293397f, -1.06774f, -0.00078574f,
                          0.453757f, -0.703098f, -0.360492f, -0.0567175f, -0.546206f, -0.0988174f, 0.1098f, 0.253175f,
                          0.505055f, -0.337885f, 0.393998f, 0.296073f, 0.0456077f, 0.0929198f, -1.5196f, 0.23374f, 0.3198f,
                          -0.808671f, -0.668391f, 0.0928441f, -0.335006f, 0.122292f, 0.611642f, -1.23343f, 0.0589855f,
                          0.639564f, 0.42115f, -0.871472f, -0.0517919f, 0.0368724f, 0.477389f, 0.973116f, 0.179215f,
                          -0.402711f, -0.0182864f, 0.0364668f, -0.577699f, 0.927416f, 0.22344f, 0.00101043f, -0.362716f,
                          -0.782957f, 0.176056f, 0.236438f, 0.0738101f, -0.717172f, 0.473243f, 0.0184415f, -0.134346f,
                          0.2701f, 0.366127f, -0.156723f, 0.640883f, 0.936974f, 0.490814f, -0.443443f, -0.618102f,
                          -0.416243f, 0.530263f, -0.103261f, -0.452055f, -0.204496f, -0.344118f, 0.556675f, 0.765497f,
                          -0.363246f, -0.722693f, 0.237699f, -0.751804f, -0.00057894f, 0.533425f, 0.658105f, 0.844047f,
                          -0.988591f, -0.773767f, -0.4548f, -0.0929374f, -0.149601f, -0.693741f, 1.52392f, 1.49526f,
                          0.274482f, -0.250754f, -0.942429f, 0.11745f, 0.474299f, 0.649417f, -0.374675f, 0.770361f,
                          -0.212086f, 1.35768f, -0.536634f, 0.491991f, -1.47631f, -0.431231f, -0.18399f, -0.279197f,
                          0.157382f, 0.89363f, -0.367624f, -1.11093f, -0.745738f, 0.548934f, 0.435344f, 0.776261f,
                          -0.150242f, -1.08683f, 0.513403f, 0.946813f, 0.290638f, 1.04797f, -0.72787f, -0.380246f,
                          0.710451f, 0.222916f, 0.232908f, 0.830753f, 1.07542f, -0.374207f, -0.852818f, -0.615864f,
                          0.63667f, 0.00779735f, 0.342196f, -0.403718f, 0.44006f, 0.536458f, -0.0573611f, -0.114347f,
                          0.586737f, 0.113243f, 0.566012f, 1.02599f, 0.567469f, -0.516206f, -0.145892f, 0.0672092f,
                          -0.133495f, -0.631785f, 0.941846f, 0.188422f, 0.296014f, 0.423775f, -0.333609f, -0.23043f,
                          0.416269f, -0.456101f, 0.0498845f, 0.211577f, 0.271048f, 0.626298f, -0.0416111f, 0.523762f,
                          -0.317144f, -0.0728705f, -0.14316f, -0.275855f, -0.295673f, -0.341217f, 0.436097f, -0.7442f,
                          0.278465f, -0.280928f, -0.0415335f, 0.115429f, -0.625263f, 0.110212f, -0.195976f, -0.29027f,
                          -0.0989828f, -0.160014f, 0.362077f, 0.0649763f, -0.371465f, 0.727401f, 0.0320011f};
  float epsilon = 1e-05f;
  TestBatchNorm(input_data_map, input_shapes_map, epsilon, expected_output, input_shape);

  // NNAPI EP will need all inputs except X be initializers
  TestBatchNorm(input_data_map, input_shapes_map, epsilon, expected_output, input_shape, true);
}

TEST(BatchNormTest, BatchNorm3d_Pytorch) {
  vector<float> X{2.02384f, -0.935186f, 0.488569f, -0.513934f, -1.27082f, -0.131913f, -1.806f, -0.37904f, 0.667796f,
                  -1.14826f, 1.2522f, 0.0300339f, 2.4758f, 1.55511f, 0.385341f, 1.46645f, -1.09355f, -2.56309f,
                  0.976015f, -1.47036f, 0.89486f, 0.580989f, -1.12418f, -0.339189f, 1.3314f, 0.418893f, -0.301401f,
                  -1.2983f, -0.839063f, 0.170261f, 1.15486f, -0.255735f, -0.589851f, -0.416289f, -0.952648f, -0.360487f,
                  0.253287f, 0.437195f, 0.32023f, 0.209606f, -0.279519f, -0.546527f, 0.265286f, -1.07383f, -1.65879f,
                  1.1222f, 0.946612f, 0.822549f, 0.64689f, -0.292639f, -0.73995f, -0.694949f, 1.33899f, -0.0652476f,
                  1.61791f, 1.49692f, -0.761145f, -0.201874f, -1.15431f, -1.83111f, -0.705267f, -0.143026f, -0.129819f,
                  -0.799425f, 0.168795f, 0.740422f, -0.377683f, 0.432598f, -2.07414f, -2.85251f, 0.273531f, 0.0532606f,
                  1.31052f, -0.769382f, 0.9976f, 0.850536f, -1.53812f, -0.00496016f, 0.931242f, 0.0517056f, -0.497829f,
                  0.275869f, 0.860001f, 1.23747f, 0.179686f, 1.5914f, 0.740327f, 0.798208f, 2.12478f, 1.74205f,
                  -0.322054f, -0.0112451f, 0.204525f, -0.431252f, -1.3114f, 0.186204f, 0.780569f, -1.42994f, 1.63344f,
                  -0.00839034f, -0.187035f, 1.8406f, 1.32053f, -0.636963f, 0.408944f, -1.50846f, -1.2076f, -0.129118f,
                  -0.0441307f, 1.47558f, 1.07251f, 1.05295f, -0.420297f, -1.13402f, -0.524053f, 3.20754f, -0.588935f,
                  -0.527549f, 0.591928f, -1.10529f, 0.520412f, 0.19404f, -1.21229f, -0.399594f, -0.280935f, -0.363324f,
                  -0.00804771f, 1.43102f, -0.523222f, 1.17608f, -0.53195f, 0.914993f, 2.69308f, -0.517211f, 0.472273f,
                  -0.464725f, -0.929768f, -0.631145f, 0.919709f, -0.27391f, 1.76689f, 0.894897f, 0.235798f, 1.2544f,
                  0.858985f, -0.139707f, 0.354544f, 0.200878f, 0.353255f, 0.0722632f, -1.56074f, 1.03685f, 1.73434f,
                  0.193269f, -0.864609f, 0.842739f, -0.372717f, 0.584484f, 0.16315f, 1.60674f, -0.0611289f, -1.24544f,
                  1.33361f, -0.961942f, -0.15732f, -0.348637f, 0.361842f, 0.7386f, 0.517256f, 1.20406f, -2.07277f,
                  -1.01983f, -1.9163f, 0.239934f, 0.177979f, 0.464564f, 0.988822f, 0.284607f, -1.56099f, -0.429143f,
                  0.111043f, -0.0853688f, -0.319176f, -0.279777f, 0.520971f, -1.078f, -0.670242f, 0.065652f, 0.468538f,
                  -0.825062f, 0.370068f, 1.68751f, -1.16928f, -0.411782f, 1.61624f, -0.973004f, 2.64703f, -0.220014f,
                  -1.43954f, -0.018692f, 1.34982f, -0.95197f, -1.72586f, 1.32725f, 0.280984f, 0.00847463f, 0.512869f,
                  0.0378154f, 0.13898f, 0.35758f, -0.084558f, 1.04045f, -1.79933f, 1.3002f, 0.390457f, 1.22267f, 0.959344f,
                  -0.964296f, -0.0935597f, 0.288953f, -0.158046f, 0.532672f, -0.500988f, 0.25187f, -2.14384f, -0.633315f,
                  1.24612f, -1.41525f, 0.36494f, -0.00714732f, -0.608963f, 0.508496f, 0.995365f, 1.21159f, -0.169055f,
                  -0.968783f, 1.52779f, -0.082381f, 2.2049f, 0.928655f, 0.120245f, 0.911429f, -0.885258f, -1.2072f,
                  0.770694f, 2.36621f, 1.08456f, -1.60069f, 0.0345025f, 0.359559f, -0.785411f, 0.466532f, -0.78543f,
                  0.024879f, 1.59337f, 1.13718f, -1.27073f, -0.263788f, -1.7702f, 0.203263f, 1.34631f, 1.11914f,
                  -2.04911f, -0.804137f, 0.466763f, 2.18386f, 1.4689f, 0.898297f, -0.648948f, 0.252202f, 1.12501f,
                  -0.204563f, 0.124608f, 0.377214f, 0.894327f, -0.249118f, 0.709188f, 0.999397f, -1.4079f, 0.193873f,
                  0.657753f, -0.709732f, 1.09897f, -0.145793f, 0.779199f, 0.88378f, -1.2676f, 1.15709f, 0.62295f,
                  -0.370894f, -0.103268f, -1.55949f, -0.470747f, 0.100394f, 0.422334f, -0.0685312f, -0.434488f,
                  -0.568974f, -0.256987f, 2.01276f, -0.923322f, -0.613144f, 1.50676f, 0.65756f, 1.20524f, 1.10395f,
                  -0.975241f, 2.44035f, 1.08276f, 0.330393f, -0.508918f, -1.25545f, 0.189815f, -0.156263f, -0.960866f,
                  1.0859f, -0.674478f, 2.76743f, 1.21399f, 1.71666f, -1.73198f, -1.1062f, 0.951285f, -0.713336f,
                  1.61586f, 1.96514f, 0.002603f, 0.0953297f, 0.949256f, -1.76552f, 0.372816f, -0.781229f, 1.50532f,
                  1.28462f, 1.31116f, 0.731908f, 1.54835f, 0.371081f, 0.409244f, -0.106938f, -1.79396f, -1.61198f,
                  -0.80869f, -1.10381f, 1.1872f, -0.832439f, 0.0755941f, -1.09553f, 0.960059f, 1.44252f, -0.196482f,
                  -1.07364f, 0.165547f, 0.630078f, 1.56569f, -0.669592f, 1.15974f, 0.0953399f, -0.202313f, 0.812631f,
                  -0.318567f, -0.16644f, 0.887062f, -0.0264821f, -0.740725f, 0.0797577f, -1.1037f, 0.90236f, 1.13427f,
                  0.364186f, -2.01043f, -0.415748f, 0.116046f, 0.369949f, 0.317886f, 0.530332f, 1.48341f, 0.74666f,
                  -1.64142f, 0.22569f, 1.18015f, 1.31827f, -1.33904f, -0.101125f};
  vector<float> scale{0.241661f, 0.960798f, 0.474727f};
  vector<float> B{0.0f, 0.0f, 0.0f};
  vector<float> mean{0.0f, 0.0f, 0.0f};
  vector<float> var{1.0f, 1.0f, 1.0f};

  InputDataMap input_data_map;
  input_data_map.insert({"X", X});
  input_data_map.insert({"scale", scale});
  input_data_map.insert({"B", B});
  input_data_map.insert({"mean", mean});
  input_data_map.insert({"var", var});

  InputShapesMap input_shapes_map;
  vector<int64_t> input_shape{2, 3, 4, 4, 4};
  input_shapes_map.insert({"X", input_shape});
  input_shapes_map.insert({"scale", {3}});
  input_shapes_map.insert({"B", {3}});
  input_shapes_map.insert({"mean", {3}});
  input_shapes_map.insert({"var", {3}});

  auto expected_output = {0.489082f, -0.225997f, 0.118068f, -0.124197f, -0.307105f, -0.031878f, -0.436439f, -0.0915989f,
                          0.16138f, -0.277489f, 0.302606f, 0.007258f, 0.598301f, 0.375807f, 0.0931215f, 0.354382f,
                          -0.264267f, -0.619395f, 0.235864f, -0.355328f, 0.216252f, 0.140402f, -0.271669f, -0.0819684f,
                          0.321747f, 0.10123f, -0.0728365f, -0.313746f, -0.202768f, 0.0411454f, 0.279085f, -0.0618009f,
                          -0.142543f, -0.1006f, -0.230217f, -0.0871152f, 0.0612094f, 0.105652f, 0.0773867f, 0.0506533f,
                          -0.0675486f, -0.132074f, 0.064109f, -0.259501f, -0.400863f, 0.271191f, 0.228758f, 0.198777f,
                          0.156327f, -0.0707191f, -0.178816f, -0.167941f, 0.323581f, -0.0157677f, 0.390985f, 0.361745f,
                          -0.183938f, -0.0487849f, -0.27895f, -0.442507f, -0.170435f, -0.0345637f, -0.031372f,
                          -0.193189f, 0.162177f, 0.711393f, -0.362876f, 0.415637f, -1.99282f, -2.74067f, 0.262807f,
                          0.0511725f, 1.25914f, -0.739217f, 0.958488f, 0.817189f, -1.47782f, -0.00476569f, 0.894731f,
                          0.0496784f, -0.478311f, 0.265053f, 0.826283f, 1.18895f, 0.172641f, 1.52901f, 0.711301f,
                          0.766913f, 2.04147f, 1.67375f, -0.309427f, -0.0108042f, 0.196507f, -0.414344f, -1.25999f,
                          0.178903f, 0.749965f, -1.37387f, 1.5694f, -0.00806138f, -0.179702f, 1.76844f, 1.26875f,
                          -0.61199f, 0.392911f, -1.44932f, -1.16025f, -0.124055f, -0.0424004f, 1.41773f, 1.03046f,
                          1.01167f, -0.403818f, -1.08956f, -0.503507f, 3.08178f, -0.565845f, -0.506866f, 0.56872f,
                          -1.06196f, 0.500008f, 0.186433f, -1.16476f, -0.383928f, -0.269921f, -0.349079f, -0.00773219f,
                          1.37492f, -0.248386f, 0.558316f, -0.25253f, 0.43437f, 1.27847f, -0.245533f, 0.2242f,
                          -0.220617f, -0.441384f, -0.29962f, 0.436609f, -0.130032f, 0.838785f, 0.424829f, 0.111939f,
                          0.595496f, 0.407781f, -0.0663221f, 0.168311f, 0.0953618f, 0.167699f, 0.0343051f, -0.74092f,
                          0.492219f, 0.823334f, 0.0917494f, -0.410451f, 0.400069f, -0.176938f, 0.277469f, 0.0774512f,
                          0.762761f, -0.0290194f, -0.59124f, 0.6331f, -0.456657f, -0.0746837f, -0.165507f, 0.171775f,
                          0.350631f, 0.245554f, 0.571595f, -0.983996f, -0.484139f, -0.909715f, 0.113902f, 0.0844908f,
                          0.22054f, 0.469418f, 0.13511f, -0.741041f, -0.203725f, 0.0527148f, -0.0405267f, -0.151521f,
                          -0.132817f, 0.247318f, -0.511752f, -0.31818f, 0.0311666f, 0.222426f, -0.391677f, 0.17568f,
                          0.801104f, -0.282569f, -0.0995112f, 0.39058f, -0.235136f, 0.639682f, -0.0531687f, -0.347878f,
                          -0.0045171f, 0.326198f, -0.230053f, -0.41707f, 0.320744f, 0.0679025f, 0.00204798f, 0.12394f,
                          0.00913847f, 0.0335859f, 0.0864127f, -0.0204343f, 0.251436f, -0.434827f, 0.314206f, 0.0943579f,
                          0.295471f, 0.231835f, -0.233032f, -0.0226096f, 0.0698283f, -0.0381934f, 0.128725f, -0.121069f,
                          0.060867f, -0.51808f, -0.153047f, 0.301137f, -0.342009f, 0.0881915f, -0.00172722f, -0.147162f,
                          0.122883f, 0.24054f, 0.292792f, -0.0408538f, -0.234116f, 0.369206f, -0.0199082f, 0.532835f,
                          0.224419f, 0.0290583f, 0.220256f, -0.213931f, -0.291733f, 0.186246f, 0.571817f, 0.262095f,
                          -0.386822f, 0.00833788f, 0.086891f, -0.189802f, 0.112742f, -0.189807f, 0.00601226f, 0.385054f,
                          0.274811f, -1.22091f, -0.253445f, -1.7008f, 0.195294f, 1.29353f, 1.07526f, -1.96877f, -0.772609f,
                          0.448463f, 2.09824f, 1.4113f, 0.863078f, -0.623505f, 0.242314f, 1.0809f, -0.196543f, 0.119722f,
                          0.362425f, 0.859263f, -0.239351f, 0.681383f, 0.960214f, -1.3527f, 0.186272f, 0.631964f,
                          -0.681905f, 1.05588f, -0.140077f, 0.748649f, 0.84913f, -1.2179f, 1.11172f, 0.598526f,
                          -0.356353f, -0.099219f, -1.49835f, -0.452291f, 0.0964582f, 0.405776f, -0.0658444f,
                          -0.417454f, -0.546667f, -0.246911f, 1.93385f, -0.887121f, -0.589104f, 1.44769f, 0.631779f,
                          1.15798f, 1.06067f, -0.937005f, 2.34467f, 1.04031f, 0.31744f, -0.488965f, -1.20623f, 0.182373f,
                          -0.150136f, -0.923194f, 1.04332f, -0.648034f, 2.65893f, 1.1664f, 1.64935f, -0.822216f,
                          -0.525139f, 0.451599f, -0.338638f, 0.767087f, 0.932899f, 0.00123571f, 0.0452554f, 0.450635f,
                          -0.838136f, 0.176985f, -0.370868f, 0.714614f, 0.60984f, 0.622438f, 0.347455f, 0.73504f,
                          0.176161f, 0.194278f, -0.0507662f, -0.851639f, -0.765246f, -0.383905f, -0.524005f, 0.563593f,
                          -0.395179f, 0.0358864f, -0.520076f, 0.455763f, 0.684801f, -0.093275f, -0.509682f, 0.0785892f,
                          0.299113f, 0.743272f, -0.317872f, 0.550556f, 0.0452602f, -0.0960432f, 0.385776f, -0.151232f,
                          -0.079013f, 0.42111f, -0.0125717f, -0.35164f, 0.0378629f, -0.523955f, 0.428372f, 0.538468f,
                          0.172888f, -0.954402f, -0.197366f, 0.0550898f, 0.175624f, 0.150908f, 0.251761f, 0.704209f,
                          0.354458f, -0.779221f, 0.107141f, 0.560244f, 0.625814f, -0.635675f, -0.0480064f};
  float epsilon = 1e-05f;
  TestBatchNorm(input_data_map, input_shapes_map, epsilon, expected_output, input_shape);
}

TEST(BatchNormTest, InvalidScaleDim) {
  vector<float> X{0.329876f, -0.287158f, -0.411425f, 0.473621f, 0.18156f, -0.170596f, -0.329516f, -0.170733f, -0.121664f, 0.4372f,
                  -0.485668f, 0.218049f, -0.360263f, 0.107016f, 0.45358f, 0.325056f, 0.15995f, 0.098852f, -0.283453f, -0.373051f,
                  0.257542f, 0.0614853f, -0.0592363f, 0.434488f, -0.0179583f, 0.398374f, -0.451602f, -0.132009f, -0.174468f,
                  -0.0247169f, 0.418897f, -0.47159f, -0.131925f, 0.470943f, 0.118357f, 0.155664f, 0.370062f, -0.279229f, 0.240311f,
                  -0.451034f, 0.249178f, -0.294496f, 0.13683f, -0.0806475f, -0.309849f, -0.450604f, -0.28048f, -0.420197f, -0.433369f};
  vector<float> scale{0.589433f, 0.589433f};
  vector<float> B{-0.384622f};
  vector<float> mean{-2.45673f};
  vector<float> var{1.37998f};

  InputDataMap input_data_map;
  input_data_map.insert({"X", X});
  input_data_map.insert({"scale", scale});
  input_data_map.insert({"B", B});
  input_data_map.insert({"mean", mean});
  input_data_map.insert({"var", var});

  InputShapesMap input_shapes_map;
  input_shapes_map.insert({"X", {1, 1, 7, 7}});
  input_shapes_map.insert({"scale", {1, 2}});  // invalid
  input_shapes_map.insert({"B", {1}});
  input_shapes_map.insert({"mean", {1}});
  input_shapes_map.insert({"var", {1}});

  vector<int64_t> expected_output_shape{1, 1, 7, 7};
  auto expected_output = {1.01359f, 0.703983f, 0.641631f, 1.08571f, 0.939167f, 0.762469f, 0.682729f, 0.762401f, 0.787021f,
                          1.06744f, 0.604378f, 0.957476f, 0.667302f, 0.901764f, 1.07566f, 1.01117f, 0.928324f, 0.897667f,
                          0.705842f, 0.660885f, 0.977291f, 0.878918f, 0.818345f, 1.06608f, 0.839057f, 1.04796f, 0.621471f,
                          0.781831f, 0.760527f, 0.835665f, 1.05825f, 0.611442f, 0.781873f, 1.08437f, 0.907454f, 0.926173f,
                          1.03375f, 0.707961f, 0.968646f, 0.621757f, 0.973095f, 0.700301f, 0.916723f, 0.807602f, 0.692598f,
                          0.621972f, 0.707334f, 0.63723f, 0.63062f};
  float epsilon = 1e-05f;
  TestBatchNorm(input_data_map,
                input_shapes_map,
                epsilon,
                expected_output,
                expected_output_shape, false, 1,
                OpTester::ExpectResult::kExpectFailure,
                "Invalid input scale");
}

TEST(BatchNormTest, InvalidBDim) {
  vector<float> X{0.329876f, -0.287158f, -0.411425f, 0.473621f, 0.18156f, -0.170596f, -0.329516f, -0.170733f, -0.121664f, 0.4372f,
                  -0.485668f, 0.218049f, -0.360263f, 0.107016f, 0.45358f, 0.325056f, 0.15995f, 0.098852f, -0.283453f, -0.373051f,
                  0.257542f, 0.0614853f, -0.0592363f, 0.434488f, -0.0179583f, 0.398374f, -0.451602f, -0.132009f, -0.174468f,
                  -0.0247169f, 0.418897f, -0.47159f, -0.131925f, 0.470943f, 0.118357f, 0.155664f, 0.370062f, -0.279229f, 0.240311f,
                  -0.451034f, 0.249178f, -0.294496f, 0.13683f, -0.0806475f, -0.309849f, -0.450604f, -0.28048f, -0.420197f, -0.433369f};
  vector<float> scale{0.589433f};
  vector<float> B{-0.384622f, -0.384622f};
  vector<float> mean{-2.45673f};
  vector<float> var{1.37998f};

  InputDataMap input_data_map;
  input_data_map.insert({"X", X});
  input_data_map.insert({"scale", scale});
  input_data_map.insert({"B", B});
  input_data_map.insert({"mean", mean});
  input_data_map.insert({"var", var});

  InputShapesMap input_shapes_map;
  input_shapes_map.insert({"X", {1, 1, 7, 7}});
  input_shapes_map.insert({"scale", {1}});
  input_shapes_map.insert({"B", {1, 2}});  // invalid
  input_shapes_map.insert({"mean", {1}});
  input_shapes_map.insert({"var", {1}});

  vector<int64_t> expected_output_shape{1, 1, 7, 7};
  auto expected_output = {1.01359f, 0.703983f, 0.641631f, 1.08571f, 0.939167f, 0.762469f, 0.682729f, 0.762401f, 0.787021f,
                          1.06744f, 0.604378f, 0.957476f, 0.667302f, 0.901764f, 1.07566f, 1.01117f, 0.928324f, 0.897667f,
                          0.705842f, 0.660885f, 0.977291f, 0.878918f, 0.818345f, 1.06608f, 0.839057f, 1.04796f, 0.621471f,
                          0.781831f, 0.760527f, 0.835665f, 1.05825f, 0.611442f, 0.781873f, 1.08437f, 0.907454f, 0.926173f,
                          1.03375f, 0.707961f, 0.968646f, 0.621757f, 0.973095f, 0.700301f, 0.916723f, 0.807602f, 0.692598f,
                          0.621972f, 0.707334f, 0.63723f, 0.63062f};
  float epsilon = 1e-05f;
  TestBatchNorm(input_data_map,
                input_shapes_map,
                epsilon,
                expected_output,
                expected_output_shape, false, 1,
                OpTester::ExpectResult::kExpectFailure,
                "Invalid input B");
}

TEST(BatchNormTest, InvalidMeanDim) {
  vector<float> X{0.329876f, -0.287158f, -0.411425f, 0.473621f, 0.18156f, -0.170596f, -0.329516f, -0.170733f, -0.121664f, 0.4372f,
                  -0.485668f, 0.218049f, -0.360263f, 0.107016f, 0.45358f, 0.325056f, 0.15995f, 0.098852f, -0.283453f, -0.373051f,
                  0.257542f, 0.0614853f, -0.0592363f, 0.434488f, -0.0179583f, 0.398374f, -0.451602f, -0.132009f, -0.174468f,
                  -0.0247169f, 0.418897f, -0.47159f, -0.131925f, 0.470943f, 0.118357f, 0.155664f, 0.370062f, -0.279229f, 0.240311f,
                  -0.451034f, 0.249178f, -0.294496f, 0.13683f, -0.0806475f, -0.309849f, -0.450604f, -0.28048f, -0.420197f, -0.433369f};
  vector<float> scale{0.589433f};
  vector<float> B{-0.384622f};
  vector<float> mean{-2.45673f, -2.45673f};
  vector<float> var{1.37998f};

  InputDataMap input_data_map;
  input_data_map.insert({"X", X});
  input_data_map.insert({"scale", scale});
  input_data_map.insert({"B", B});
  input_data_map.insert({"mean", mean});
  input_data_map.insert({"var", var});

  InputShapesMap input_shapes_map;
  input_shapes_map.insert({"X", {1, 1, 7, 7}});
  input_shapes_map.insert({"scale", {1}});
  input_shapes_map.insert({"B", {1}});
  input_shapes_map.insert({"mean", {1, 2}});  // invalid
  input_shapes_map.insert({"var", {1}});

  vector<int64_t> expected_output_shape{1, 1, 7, 7};
  auto expected_output = {1.01359f, 0.703983f, 0.641631f, 1.08571f, 0.939167f, 0.762469f, 0.682729f, 0.762401f, 0.787021f,
                          1.06744f, 0.604378f, 0.957476f, 0.667302f, 0.901764f, 1.07566f, 1.01117f, 0.928324f, 0.897667f,
                          0.705842f, 0.660885f, 0.977291f, 0.878918f, 0.818345f, 1.06608f, 0.839057f, 1.04796f, 0.621471f,
                          0.781831f, 0.760527f, 0.835665f, 1.05825f, 0.611442f, 0.781873f, 1.08437f, 0.907454f, 0.926173f,
                          1.03375f, 0.707961f, 0.968646f, 0.621757f, 0.973095f, 0.700301f, 0.916723f, 0.807602f, 0.692598f,
                          0.621972f, 0.707334f, 0.63723f, 0.63062f};
  float epsilon = 1e-05f;
  TestBatchNorm(input_data_map,
                input_shapes_map,
                epsilon,
                expected_output,
                expected_output_shape, false, 1,
                OpTester::ExpectResult::kExpectFailure,
                "Invalid input mean");
}

TEST(BatchNormTest, InvalidVarDim) {
  vector<float> X{0.329876f, -0.287158f, -0.411425f, 0.473621f, 0.18156f, -0.170596f, -0.329516f, -0.170733f, -0.121664f, 0.4372f,
                  -0.485668f, 0.218049f, -0.360263f, 0.107016f, 0.45358f, 0.325056f, 0.15995f, 0.098852f, -0.283453f, -0.373051f,
                  0.257542f, 0.0614853f, -0.0592363f, 0.434488f, -0.0179583f, 0.398374f, -0.451602f, -0.132009f, -0.174468f,
                  -0.0247169f, 0.418897f, -0.47159f, -0.131925f, 0.470943f, 0.118357f, 0.155664f, 0.370062f, -0.279229f, 0.240311f,
                  -0.451034f, 0.249178f, -0.294496f, 0.13683f, -0.0806475f, -0.309849f, -0.450604f, -0.28048f, -0.420197f, -0.433369f};
  vector<float> scale{0.589433f};
  vector<float> B{-0.384622f};
  vector<float> mean{-2.45673f};
  vector<float> var{1.37998f, 1.37998f};

  InputDataMap input_data_map;
  input_data_map.insert({"X", X});
  input_data_map.insert({"scale", scale});
  input_data_map.insert({"B", B});
  input_data_map.insert({"mean", mean});
  input_data_map.insert({"var", var});

  InputShapesMap input_shapes_map;
  input_shapes_map.insert({"X", {1, 1, 7, 7}});
  input_shapes_map.insert({"scale", {1}});
  input_shapes_map.insert({"B", {1}});
  input_shapes_map.insert({"mean", {1}});
  input_shapes_map.insert({"var", {1, 2}});  // invalid

  vector<int64_t> expected_output_shape{1, 1, 7, 7};
  auto expected_output = {1.01359f, 0.703983f, 0.641631f, 1.08571f, 0.939167f, 0.762469f, 0.682729f, 0.762401f, 0.787021f,
                          1.06744f, 0.604378f, 0.957476f, 0.667302f, 0.901764f, 1.07566f, 1.01117f, 0.928324f, 0.897667f,
                          0.705842f, 0.660885f, 0.977291f, 0.878918f, 0.818345f, 1.06608f, 0.839057f, 1.04796f, 0.621471f,
                          0.781831f, 0.760527f, 0.835665f, 1.05825f, 0.611442f, 0.781873f, 1.08437f, 0.907454f, 0.926173f,
                          1.03375f, 0.707961f, 0.968646f, 0.621757f, 0.973095f, 0.700301f, 0.916723f, 0.807602f, 0.692598f,
                          0.621972f, 0.707334f, 0.63723f, 0.63062f};
  float epsilon = 1e-05f;
  TestBatchNorm(input_data_map,
                input_shapes_map,
                epsilon,
                expected_output,
                expected_output_shape, false, 1,
                OpTester::ExpectResult::kExpectFailure,
                "Invalid input var");
}

TEST(BatchNormTest, NonSpatial_Simple) {
  vector<float> X{1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f};
  vector<float> scale{1.f, 1.f, 1.f, 1.f};
  vector<float> B{1.f, 0.f, 0.f, 1.f};
  vector<float> mean{0.f, 0.f, 0.f, 0.f};
  vector<float> var{1.f, 1.f, 1.f, 1.f};

  InputDataMap input_data_map;
  input_data_map.insert({"X", X});
  input_data_map.insert({"scale", scale});
  input_data_map.insert({"B", B});
  input_data_map.insert({"mean", mean});
  input_data_map.insert({"var", var});

  InputShapesMap input_shapes_map;
  input_shapes_map.insert({"X", {2, 2, 2}});
  input_shapes_map.insert({"scale", {2, 2}});
  input_shapes_map.insert({"B", {2, 2}});
  input_shapes_map.insert({"mean", {2, 2}});
  input_shapes_map.insert({"var", {2, 2}});

  vector<int64_t> expected_output_shape{2, 2, 2};
  auto expected_output = {2.f, 2.f, 3.f, 5.f, 2.f, 2.f, 3.f, 5.f};
  float epsilon = 0.f;
  TestBatchNorm(input_data_map,
                input_shapes_map,
                epsilon,
                expected_output,
                expected_output_shape,
                false,
                0,
                OpTester::ExpectResult::kExpectSuccess,
                "",
                7);  // opset-7
}

TEST(BatchNormTest, NonSpatial_Complicated) {
  vector<float> X{0.2134f, 0.32434f, 0.5644f, 0.3234f, 0.4545f, 0.3445f};
  vector<float> scale{0.5f, 0.6f};
  vector<float> B{0.2f, 0.1f};
  vector<float> mean{0.034f, 0.342f};
  vector<float> var{1.f, 1.f};

  InputDataMap input_data_map;
  input_data_map.insert({"X", X});
  input_data_map.insert({"scale", scale});
  input_data_map.insert({"B", B});
  input_data_map.insert({"mean", mean});
  input_data_map.insert({"var", var});

  InputShapesMap input_shapes_map;
  input_shapes_map.insert({"X", {3, 1, 2}});
  input_shapes_map.insert({"scale", {1, 2}});
  input_shapes_map.insert({"B", {1, 2}});
  input_shapes_map.insert({"mean", {1, 2}});
  input_shapes_map.insert({"var", {1, 2}});

  vector<int64_t> expected_output_shape{3, 1, 2};
  auto expected_output = {0.2897f, 0.089404f, 0.4652f, 0.08884f, 0.41025f, 0.1015f};
  float epsilon = 1e-05f;
  TestBatchNorm(input_data_map,
                input_shapes_map,
                epsilon,
                expected_output,
                expected_output_shape,
                false,
                0,
                OpTester::ExpectResult::kExpectSuccess,
                "",
                8);  // opset-8
}

// Only CUDA kernel has float 16 support
#ifdef USE_CUDA
TEST(BatchNormTest, BatchNorm2d_fp16) {
  vector<float> X{-0.91221f, -0.283559f, 0.937637f, 2.09818f, -0.100199f, -0.608113f, 0.444562f, -1.07505f, 0.940591f,
                  -0.922262f, 0.0931303f, 0.69611f, 1.55187f, 0.159808f, 0.914874f, -1.24856f, -1.98928f, -0.331621f,
                  2.33131f, 0.260409f, 0.0944811f, 0.442397f, 0.76461f, -0.203334f, -0.244228f, -0.387267f, -1.65039f,
                  -0.815409f, 0.931696f, -1.15328f, 0.773952f, -1.28195f, -0.437349f, 0.0644882f, -0.087637f, 1.74999f,
                  0.640154f, -0.505641f, -1.84014f, -0.00135415f, 0.782006f, -1.21172f, -0.621273f, -0.0977471f,
                  -0.941333f, -0.170302f, 0.18923f, 0.436322f, 0.870412f, -0.582312f, 0.679017f, 0.510252f, 0.0786005f,
                  0.160138f, -2.61889f, 0.402828f, 0.551144f, -1.39366f, -1.15191f, 0.160008f, -0.57735f, 0.210758f,
                  1.0541f, -2.12569f, 0.101656f, 1.10223f, 0.725811f, -1.5019f, -0.0892582f, 0.063546f, 0.822734f,
                  1.67707f, 0.478121f, -1.07438f, -0.0487855f, 0.0972885f, -1.54122f, 2.47422f, 0.596108f, 0.0026957f,
                  -0.967677f, -2.08882f, 0.469692f, 0.630784f, 0.196915f, -1.91331f, 1.26255f, 0.0491993f, -0.358415f,
                  0.720588f, 0.976776f, -0.418116f, 1.70979f, 2.49971f, 1.30942f, -1.18304f, -1.64901f, -1.11048f,
                  1.41467f, -0.275486f, -1.20602f, -0.545566f, -0.918059f, 1.48513f, 2.04224f, -0.96909f, -1.92804f,
                  0.634147f, -1.02079f, -0.000786079f, 0.72428f, 0.893569f, 1.14604f, -1.3423f, -1.05061f, -0.617524f,
                  -0.12619f, -0.203127f, -0.941956f, 2.06916f, 2.03025f, 0.37269f, -0.340471f, -1.27962f, 0.159472f,
                  0.643999f, 0.881773f, -0.50873f, 1.04599f, -0.287968f, 1.84344f, -0.728637f, 0.668021f, -2.00452f,
                  -0.585523f, -0.24982f, -0.379091f, 0.213692f, 1.21336f, -0.499157f, -1.50841f, -1.01256f, 0.745338f,
                  0.591107f, 1.33781f, -0.258927f, -1.87304f, 0.884799f, 1.63174f, 0.500887f, 1.80608f, -1.25441f,
                  -0.655316f, 1.22439f, 0.384174f, 0.401395f, 1.43172f, 1.85338f, -0.644909f, -1.46975f, -1.06138f,
                  1.09724f, 0.013438f, 0.589742f, -0.695768f, 0.758401f, 0.924533f, -0.0988563f, -0.197066f, 1.01118f,
                  0.195163f, 0.975466f, 1.7682f, 0.977977f, -0.88963f, -0.251431f, 0.115828f, -0.230065f, -1.08882f,
                  1.62318f, 0.502684f, 0.789724f, 1.13057f, -0.890021f, -0.614755f, 1.11055f, -1.21681f, 0.133085f,
                  0.564458f, 0.723117f, 1.67088f, -0.111012f, 1.39732f, -0.846095f, -0.194408f, -0.381931f, -0.735943f,
                  -0.788814f, -0.910318f, 1.16345f, -1.98542f, 0.742905f, -0.749476f, -0.110805f, 0.307949f, -1.66811f,
                  0.294031f, -0.522837f, -0.774399f, -0.264072f, -0.426894f, 0.965971f, 0.173348f, -0.991018f, 1.9406f,
                  0.0853744f};
  vector<float> scale{0.736494f, 0.580251f, 0.374834f};
  vector<float> B{0.0f, 0.0f, 0.0f};
  vector<float> mean{0.0f, 0.0f, 0.0f};
  vector<float> var{1.0f, 1.0f, 1.0f};

  vector<float> expected_output{-0.671834f, -0.208838f, 0.69056f, 1.54529f, -0.0737958f, -0.447869f, 0.327415f, -0.791764f,
                                0.692736f, -0.679237f, 0.0685895f, 0.512678f, 1.14294f, 0.117697f, 0.673796f, -0.91955f,
                                -1.46508f, -0.244235f, 1.71699f, 0.191789f, 0.0695843f, 0.325821f, 0.563128f, -0.149753f,
                                -0.179871f, -0.285218f, -1.2155f, -0.60054f, 0.686185f, -0.84938f, 0.570008f, -0.944146f,
                                -0.322103f, 0.0474949f, -0.0645437f, 1.28885f, 0.371448f, -0.293397f, -1.06774f, -0.00078574f,
                                0.453757f, -0.703098f, -0.360492f, -0.0567175f, -0.546206f, -0.0988174f, 0.1098f, 0.253175f,
                                0.505055f, -0.337885f, 0.393998f, 0.296073f, 0.0456077f, 0.0929198f, -1.5196f, 0.23374f, 0.3198f,
                                -0.808671f, -0.668391f, 0.0928441f, -0.335006f, 0.122292f, 0.611642f, -1.23343f, 0.0589855f,
                                0.639564f, 0.42115f, -0.871472f, -0.0517919f, 0.0368724f, 0.477389f, 0.973116f, 0.179215f,
                                -0.402711f, -0.0182864f, 0.0364668f, -0.577699f, 0.927416f, 0.22344f, 0.00101043f, -0.362716f,
                                -0.782957f, 0.176056f, 0.236438f, 0.0738101f, -0.717172f, 0.473243f, 0.0184415f, -0.134346f,
                                0.2701f, 0.366127f, -0.156723f, 0.640883f, 0.936974f, 0.490814f, -0.443443f, -0.618102f,
                                -0.416243f, 0.530263f, -0.103261f, -0.452055f, -0.204496f, -0.344118f, 0.556675f, 0.765497f,
                                -0.363246f, -0.722693f, 0.237699f, -0.751804f, -0.00057894f, 0.533425f, 0.658105f, 0.844047f,
                                -0.988591f, -0.773767f, -0.4548f, -0.0929374f, -0.149601f, -0.693741f, 1.52392f, 1.49526f,
                                0.274482f, -0.250754f, -0.942429f, 0.11745f, 0.474299f, 0.649417f, -0.374675f, 0.770361f,
                                -0.212086f, 1.35768f, -0.536634f, 0.491991f, -1.47631f, -0.431231f, -0.18399f, -0.279197f,
                                0.157382f, 0.89363f, -0.367624f, -1.11093f, -0.745738f, 0.548934f, 0.435344f, 0.776261f,
                                -0.150242f, -1.08683f, 0.513403f, 0.946813f, 0.290638f, 1.04797f, -0.72787f, -0.380246f,
                                0.710451f, 0.222916f, 0.232908f, 0.830753f, 1.07542f, -0.374207f, -0.852818f, -0.615864f,
                                0.63667f, 0.00779735f, 0.342196f, -0.403718f, 0.44006f, 0.536458f, -0.0573611f, -0.114347f,
                                0.586737f, 0.113243f, 0.566012f, 1.02599f, 0.567469f, -0.516206f, -0.145892f, 0.0672092f,
                                -0.133495f, -0.631785f, 0.941846f, 0.188422f, 0.296014f, 0.423775f, -0.333609f, -0.23043f,
                                0.416269f, -0.456101f, 0.0498845f, 0.211577f, 0.271048f, 0.626298f, -0.0416111f, 0.523762f,
                                -0.317144f, -0.0728705f, -0.14316f, -0.275855f, -0.295673f, -0.341217f, 0.436097f, -0.7442f,
                                0.278465f, -0.280928f, -0.0415335f, 0.115429f, -0.625263f, 0.110212f, -0.195976f, -0.29027f,
                                -0.0989828f, -0.160014f, 0.362077f, 0.0649763f, -0.371465f, 0.727401f, 0.0320011f};
  float epsilon = 1e-05f;

  OpTester test("BatchNormalization");
  test.AddAttribute("epsilon", epsilon);

  vector<int64_t> input_shape{2, 3, 6, 6};
  int input_size = 2 * 3 * 6 * 6;

  vector<MLFloat16> f_X(input_size);
  vector<MLFloat16> f_output(input_size);
  vector<MLFloat16> f_scale(3);
  vector<MLFloat16> f_B(3);
  vector<MLFloat16> f_mean(3);
  vector<MLFloat16> f_var(3);

  ConvertFloatToMLFloat16(X.data(), f_X.data(), input_size);
  ConvertFloatToMLFloat16(scale.data(), f_scale.data(), 3);
  ConvertFloatToMLFloat16(B.data(), f_B.data(), 3);
  ConvertFloatToMLFloat16(mean.data(), f_mean.data(), 3);
  ConvertFloatToMLFloat16(var.data(), f_var.data(), 3);
  ConvertFloatToMLFloat16(expected_output.data(), f_output.data(), input_size);

  test.AddInput<MLFloat16>("X", input_shape, f_X);
  test.AddInput<MLFloat16>("scale", {3}, f_scale);
  test.AddInput<MLFloat16>("B", {3}, f_B);
  test.AddInput<MLFloat16>("mean", {3}, f_mean);
  test.AddInput<MLFloat16>("var", {3}, f_var);
  test.AddOutput<MLFloat16>("output", input_shape, f_output);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}
#endif


// TODO fix flaky test for CUDA
TEST(BatchNormTest, ForwardTrainingTestWithSavedOutputsOpset9) {
  OpTester test("BatchNormalization", 9);
  float epsilon = 1e-05f;
  float momentum = 0.1f;
  test.AddAttribute("epsilon", epsilon);
  test.AddAttribute("momentum", momentum);
  std::vector<int64_t> input_output_dims{2, 2, 2, 2};
  std::vector<int64_t> channel_dims{2};
  test.AddInput<float>("X", input_output_dims, {-0.2953f, 0.1180f, 1.0973f, -0.1931f, -0.1999f, -0.0237f, 1.5181f, 0.0076f, -1.0830f, -1.5433f, 0.4327f, -0.9813f, 0.7875f, -0.4080f, -2.3144f, 1.5493f});
  test.AddInput<float>("scale", channel_dims, {1.0f, 1.0f});
  test.AddInput<float>("B", channel_dims, {0.0f, 0.0f});
  test.AddInput<float>("mean", channel_dims, {1.0f, 2.0f});
  test.AddInput<float>("var", channel_dims, {1.0f, 2.0f});

  test.AddOutput<float>("Y", input_output_dims, {0.0131f, 0.5210f, 1.7244f, 0.1387f, -0.2708f, -0.1191f, 1.2089f, -0.0922f, -0.9548f, -1.5203f, 0.9077f, -0.8298f, 0.5796f, -0.4501f, -2.0921f, 1.2358f});

  test.AddOutput<float>("running_mean", channel_dims, {-0.1754f, 0.303106f});
  test.AddOutput<float>("running_var", channel_dims, {0.696052f, 1.41316f});
  // mean and variance of X across channel dimension
  // With Opset9 we output saved_inv_std instead of saved_var to match CUDA EP
  test.AddOutput<float>("saved_mean", channel_dims, {-0.306f, 0.114562f});
  test.AddOutput<float>("saved_inv_std", channel_dims, {1.2288f, 0.861317f});

  // exclude CUDA Execution Provider due to flakiness
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kCudaExecutionProvider});
}

TEST(BatchNormTest, ForwardTrainingTestOpset14) {
  OpTester test("BatchNormalization", 14);
  float epsilon = 1e-05f;
  float momentum = 0.1f;
  int64_t training_mode = 1;
  test.AddAttribute("epsilon", epsilon);
  test.AddAttribute("momentum", momentum);
  test.AddAttribute("training_mode", training_mode);
  std::vector<int64_t> input_output_dims{2, 2, 2, 2};
  std::vector<int64_t> channel_dims{2};
  test.AddInput<float>("X", input_output_dims, {-0.2953f, 0.1180f, 1.0973f, -0.1931f, -0.1999f, -0.0237f, 1.5181f, 0.0076f, -1.0830f, -1.5433f, 0.4327f, -0.9813f, 0.7875f, -0.4080f, -2.3144f, 1.5493f});
  test.AddInput<float>("scale", channel_dims, {1.0f, 1.0f});
  test.AddInput<float>("B", channel_dims, {0.0f, 0.0f});
  test.AddInput<float>("mean", channel_dims, {1.0f, 2.0f});
  test.AddInput<float>("var", channel_dims, {1.0f, 2.0f});

  test.AddOutput<float>("Y", input_output_dims, {0.0131f, 0.5210f, 1.7244f, 0.1387f, -0.2708f, -0.1191f, 1.2089f, -0.0922f, -0.9548f, -1.5203f, 0.9077f, -0.8298f, 0.5796f, -0.4501f, -2.0921f, 1.2358f});

  test.AddOutput<float>("running_mean", channel_dims, {-0.1754f, 0.303106f});
  test.AddOutput<float>("running_var", channel_dims, {0.696052f, 1.41316f});

  // exclude CUDA Execution Provider due to flakiness
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kCudaExecutionProvider});
}

}  // namespace test
}  // namespace onnxruntime
