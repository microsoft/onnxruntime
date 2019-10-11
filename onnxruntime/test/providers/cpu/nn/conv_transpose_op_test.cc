// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
using namespace std;
namespace onnxruntime {
namespace test {

namespace {

struct ConvTransposeOpAttributes {
  vector<int64_t> kernel_shape;
  vector<int64_t> output_padding;
  vector<int64_t> output_shape;
  vector<int64_t> pads;
  vector<int64_t> strides;
  vector<int64_t> dilations;
  int64_t group;
};

void TestConvTransposeOp(const ConvTransposeOpAttributes& attributes,
                         const vector<vector<float>>& inputs,
                         const vector<vector<int64_t>>& input_shapes,
                         const std::initializer_list<float>& expected_output,
                         const vector<int64_t>& expected_output_shape,
                         OpTester::ExpectResult expect_result = OpTester::ExpectResult::kExpectSuccess,
                         const std::string& err_str = "") {
  OpTester test("ConvTranspose");
  test.AddAttribute("kernel_shape", attributes.kernel_shape);
  test.AddAttribute("pads", attributes.pads);
  test.AddAttribute("group", attributes.group);

  if (!attributes.output_padding.empty()) {
    test.AddAttribute("output_padding", attributes.output_padding);
  }
  if (!attributes.output_shape.empty()) {
    test.AddAttribute("output_shape", attributes.output_shape);
  }

  if (!attributes.strides.empty()) {
    test.AddAttribute("strides", attributes.strides);
  }

  if (!attributes.dilations.empty()) {
    test.AddAttribute("dilations", attributes.dilations);
  }

  ORT_ENFORCE(inputs.size() <= 3, "Our name array is only setup to handle 3 inputs");
  const char* szNames[] = {"X", "W", "B"};
  for (size_t i = 0; i < inputs.size(); i++) {
    test.AddInput<float>(szNames[i], input_shapes[i], inputs[i]);
  }
  test.AddOutput<float>("Y", expected_output_shape, expected_output);
  test.Run(expect_result, err_str, {kTensorrtExecutionProvider});  // Disable TensorRT because weight as input is not supported
}
}  // namespace

TEST(ConvTransposeTest, ConvTranspose_2D) {
  ConvTransposeOpAttributes attrs = {
      vector<int64_t>{3, 3},        // kernel_shape
      vector<int64_t>{1, 1},        // output_padding
      {},                           // output_shape
      vector<int64_t>{1, 1, 1, 1},  // pads
      vector<int64_t>{2, 2},        // strides
      vector<int64_t>{1, 1},        // dilations
      1                             // group
  };
  vector<float> X = {0.16857791f, -0.15161794f, 0.08540368f,
                     0.1820628f, -0.21746576f, 0.08245695f,
                     0.1431433f, -0.43156421f, 0.30591947f};
  vector<int64_t> X_shape = {1, 1, 3, 3};
  vector<float> W = {-0.06230065f, 0.37932432f, -0.25388849f,
                     0.33878803f, 0.43709868f, -0.22477469f,
                     0.04118127f, -0.44696793f, 0.06373066f};
  vector<int64_t> W_shape = {1, 1, 3, 3};
  vector<int64_t> Y_shape = {1, 1, 6, 6};
  auto expected_vals = {0.07368518f, -0.08925839f, -0.06627201f, 0.06301362f, 0.03732984f, -0.01919658f,
                        -0.00628807f, -0.02817563f, -0.01472169f, 0.04392925f, -0.00689478f, -0.01549204f,
                        0.07957941f, -0.11459791f, -0.09505399f, 0.07681622f, 0.03604182f, -0.01853423f,
                        -0.0270785f, -0.00680824f, -0.06650258f, 0.08004665f, 0.07918708f, -0.0724144f,
                        0.06256775f, -0.17838378f, -0.18863615f, 0.20064656f, 0.133717f, -0.06876295f,
                        -0.06398046f, -0.00864975f, 0.19289537f, -0.01490572f, -0.13673618f, 0.01949645f};
  TestConvTransposeOp(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape);
}

TEST(ConvTransposeTest, ConvTranspose_2D_Bias_1) {
  ConvTransposeOpAttributes attrs = {
      vector<int64_t>{3, 3},        // kernel_shape
      vector<int64_t>{0, 0},        // output_padding
      {},                           // output_shape
      vector<int64_t>{1, 1, 1, 1},  // pads
      vector<int64_t>{1, 1},        // strides
      vector<int64_t>{1, 1},        // dilations
      1                             // group
  };
  vector<float> X = {0.22572887f, -0.07105902f, -0.40399021f, -0.14461157f, 0.05367219f,
                     -0.08353302f, 0.41023391f, 0.42745841f, -0.3769345f, -0.42057109f,
                     -0.1372498f, 0.05485916f, 0.34602994f, -0.06402895f, -0.06000063f,
                     0.07891446f, -0.09410021f, 0.26251942f, -0.11043271f, 0.47966552f,
                     0.34682763f, -0.04511502f, 0.22414422f, 0.24618894f, -0.21480265f};
  vector<int64_t> X_shape = {1, 1, 5, 5};
  vector<float> W = {-0.0962126f, 0.19827795f, 0.03667754f,
                     0.36756599f, -0.01076147f, -0.11781135f,
                     -0.11574665f, -0.38404959f, 0.44403327f};
  vector<int64_t> W_shape = {1, 1, 3, 3};
  vector<float> B = {0.04676145f};
  vector<int64_t> B_shape = {1};
  vector<int64_t> Y_shape = {1, 1, 5, 5};
  auto expected_vals = {-0.03781903f, -0.09041066f, 0.14239404f, 0.09704495f, -0.03399426f,
                        0.08749044f, 0.35613984f, 0.07240347f, -0.27841991f, -0.00337578f,
                        0.07770107f, -0.09561026f, 0.13388641f, 0.30945939f, 0.14015588f,
                        0.13079405f, -0.00488365f, -0.06758944f, 0.45621645f, 0.01566098f,
                        0.00703105f, 0.12956856f, 0.0103332f, 0.04221053f, -0.21318194f};
  TestConvTransposeOp(attrs, {X, W, B}, {X_shape, W_shape, B_shape}, expected_vals, Y_shape);
}

TEST(ConvTransposeTest, ConvTranspose_2D_Bias_2) {
  ConvTransposeOpAttributes attrs = {
      vector<int64_t>{2, 2},        // kernel_shape
      vector<int64_t>{0, 0},        // output_padding
      {},                           // output_shape
      vector<int64_t>{0, 0, 0, 0},  // pads
      vector<int64_t>{1, 1},        // strides
      vector<int64_t>{1, 1},        // dilations
      1                             // group
  };
  vector<float> X = {0.01270282f, 0.09657472f, -0.36909008f, -0.08085269f,
                     0.0242992f, 0.40873009f, -0.46927932f, 0.34412372f,
                     -0.39574206f, 0.26234281f, 0.27352369f, -0.22265741f,
                     0.43270493f, -0.24710381f, -0.03418651f, -0.04413456f,
                     -0.16414353f, 0.3158558f, 0.1087395f, -0.38577938f,
                     -0.38986659f, -0.09614426f, 0.17591673f, 0.40140027f,
                     -0.0869683f, -0.47193506f, -0.05010766f, 0.29325962f,
                     0.22680271f, -0.0793834f, -0.36764491f, 0.20451134f,
                     0.46361887f, -0.12190259f, 0.03413916f, 0.12307656f,
                     0.28569579f, -0.392129f, 0.17179191f, 0.27161086f,
                     -0.12766263f, 0.1371125f, 0.28137422f, -0.39899838f,
                     0.23824286f, -0.19693244f, 0.32956779f, 0.46209556f,
                     -0.46913007f};
  vector<int64_t> X_shape = {1, 1, 7, 7};
  vector<float> W = {-0.34922412f, 0.1114341f, -0.01778314f, 0.46861196f};
  vector<int64_t> W_shape = {1, 1, 2, 2};
  vector<float> B = {0.17402864f};
  vector<int64_t> B_shape = {1};
  vector<int64_t> Y_shape = {1, 1, 8, 8};
  auto expected_vals = {0.1695925f, 0.14171794f, 0.31368554f, 0.16113512f,
                        0.15653302f, 0.033998f, 0.38345876f, 0.12173492f,
                        0.05362644f, 0.35481372f, 0.09013268f, -0.06378071f,
                        0.24394518f, 0.00222442f, 0.50842237f, -0.07341707f,
                        0.17984779f, 0.35392997f, 0.03631867f, 0.16350585f,
                        0.30338728f, 0.2088346f, 0.47435546f, 0.0147884f,
                        0.20821247f, 0.08664516f, 0.03569011f, 0.16659322f,
                        0.47522858f, 0.19675478f, -0.10781619f, 0.02401161f,
                        0.0965334f, 0.1788421f, 0.36887163f, 0.2512877f,
                        0.00254938f, 0.04799958f, 0.11982619f, 0.31525785f,
                        0.12701407f, 0.19566584f, 0.31214368f, -0.10558143f,
                        0.18591091f, 0.46830338f, 0.05418756f, 0.20530567f,
                        0.07357728f, 0.39731777f, 0.1872202f, 0.08253923f,
                        0.11266428f, 0.17892915f, 0.32709083f, 0.1860041f,
                        0.16902491f, 0.3129794f, -0.01718347f, 0.28917417f,
                        0.07588299f, 0.32025051f, 0.39891475f, -0.04581133f};
  TestConvTransposeOp(attrs, {X, W, B}, {X_shape, W_shape, B_shape}, expected_vals, Y_shape);
}

TEST(ConvTransposeTest, ConvTranspose_2D_OutputShape_1) {
  ConvTransposeOpAttributes attrs = {
      vector<int64_t>{3, 3},        // kernel_shape
      {},                           // output_padding
      vector<int64_t>{1, 3, 4, 4},  // output_shape
      vector<int64_t>{0, 0, 0, 0},  // pads
      vector<int64_t>{1, 1},        // strides
      vector<int64_t>{1, 1},        // dilations
      1                             // group
  };
  int image_size = 4 * 4;
  int input_channels = 3;
  int output_channels = 3;
  std::vector<float> X;
  for (int i = 0; i < input_channels * image_size; i++)
    X.push_back(1.0f);
  std::vector<float> W;
  int kernel_size = output_channels * input_channels * 3 * 3;
  for (int i = 0; i < kernel_size; i++)
    W.push_back(1.0f);

  vector<int64_t> X_shape = {1, 3, 4, 4};
  vector<int64_t> W_shape = {3, 3, 3, 3};

  vector<int64_t> Y_shape = {1, 3, 4, 4};
  auto expected_vals = {12.0f, 18.0f, 18.0f, 12.0f,
                        18.0f, 27.0f, 27.0f, 18.0f,
                        18.0f, 27.0f, 27.0f, 18.0f,
                        12.0f, 18.0f, 18.0f, 12.0f,
                        12.0f, 18.0f, 18.0f, 12.0f,
                        18.0f, 27.0f, 27.0f, 18.0f,
                        18.0f, 27.0f, 27.0f, 18.0f,
                        12.0f, 18.0f, 18.0f, 12.0f,
                        12.0f, 18.0f, 18.0f, 12.0f,
                        18.0f, 27.0f, 27.0f, 18.0f,
                        18.0f, 27.0f, 27.0f, 18.0f,
                        12.0f, 18.0f, 18.0f, 12.0f};
  TestConvTransposeOp(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape);
}

TEST(ConvTransposeTest, ConvTranspose_2D_OutputShape_2) {
  ConvTransposeOpAttributes attrs = {
      vector<int64_t>{1, 5},         // kernel_shape
      {},                            // output_padding
      vector<int64_t>{1, 1, 1, 14},  // output_shape
      vector<int64_t>{0, 0, 0, 0},   // pads
      vector<int64_t>{1, 1},         // strides
      vector<int64_t>{1, 1},         // dilations
      1                              // group
  };
  vector<float> X = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
  vector<int64_t> X_shape = {1, 1, 1, 10};
  vector<float> W = {1.0f, 2.0f, 3.0f, 2.0f, 1.0f};
  vector<int64_t> W_shape = {1, 1, 1, 5};
  vector<float> B = {1.0f};
  vector<int64_t> B_shape = {1};
  vector<int64_t> Y_shape = {1, 1, 1, 14};
  auto expected_vals = {1.0f, 2.0f, 5.0f, 11.0f, 19.0f, 28.0f, 37.0f, 46.0f, 55.0f, 64.0f, 63.0f, 51.0f, 27.0f, 10.0f};
  TestConvTransposeOp(attrs, {X, W, B}, {X_shape, W_shape, B_shape}, expected_vals, Y_shape);
}

TEST(ConvTransposeTest, ConvTranspose_2D_OutputShapeWithBatchSize) {
  ConvTransposeOpAttributes attrs = {
      vector<int64_t>{1, 5},         // kernel_shape
      {},                            // output_padding
      vector<int64_t>{2, 1, 1, 14},  // output_shape
      vector<int64_t>{0, 0, 0, 0},   // pads
      vector<int64_t>{1, 1},         // strides
      vector<int64_t>{1, 1},         // dilations
      1                              // group
  };
  vector<float> X = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
                     10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f};
  vector<int64_t> X_shape = {2, 1, 1, 10};
  vector<float> W = {1.0f, 2.0f, 3.0f, 2.0f, 1.0f};
  vector<int64_t> W_shape = {1, 1, 1, 5};
  vector<float> B = {1.0f};
  vector<int64_t> B_shape = {1};
  vector<int64_t> Y_shape = {2, 1, 1, 14};
  auto expected_vals = {1.0f, 2.0f, 5.0f, 11.0f, 19.0f, 28.0f, 37.0f, 46.0f, 55.0f, 64.0f, 63.0f, 51.0f, 27.0f, 10.0f,
                        11.0f, 32.0f, 65.0f, 91.0f, 109.0f, 118.0f, 127.0f, 136.0f, 145.0f, 154.0f, 143.0f, 111.0f, 57.0f, 20.0f};
  TestConvTransposeOp(attrs, {X, W, B}, {X_shape, W_shape, B_shape}, expected_vals, Y_shape);
}

#ifndef USE_NGRAPH
TEST(ConvTransposeTest, ConvTranspose_InvalidKernelShape) {
  ConvTransposeOpAttributes attrs = {
      vector<int64_t>{1, 1, 1, 5},   // invalid kernel_shape, should be [1, 5]
      {},                            // output_padding
      vector<int64_t>{2, 1, 1, 14},  // output_shape
      vector<int64_t>{0, 0, 0, 0},   // pads
      vector<int64_t>{1, 1},         // strides
      vector<int64_t>{1, 1},         // dilations
      1                              // group
  };
  vector<float> X = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
                     10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f};
  vector<int64_t> X_shape = {2, 1, 1, 10};
  vector<float> W = {1.0f, 2.0f, 3.0f, 2.0f, 1.0f};
  vector<int64_t> W_shape = {1, 1, 1, 5};
  vector<float> B = {1.0f};
  vector<int64_t> B_shape = {1};
  vector<int64_t> Y_shape = {2, 1, 1, 14};
  auto expected_vals = {1.0f, 2.0f, 5.0f, 11.0f, 19.0f, 28.0f, 37.0f, 46.0f, 55.0f, 64.0f, 63.0f, 51.0f, 27.0f, 10.0f,
                        11.0f, 32.0f, 65.0f, 91.0f, 109.0f, 118.0f, 127.0f, 136.0f, 145.0f, 154.0f, 143.0f, 111.0f, 57.0f, 20.0f};
  TestConvTransposeOp(attrs, {X, W, B}, {X_shape, W_shape, B_shape}, expected_vals, Y_shape,
                      OpTester::ExpectResult::kExpectFailure,
                      "kernel_shape num_dims is not compatible with W num_dims. kernel_shape: {1,1,1,5} W: {1,1,1,5}");
}
#endif

TEST(ConvTransposeTest, ConvTranspose_onnx) {
  ConvTransposeOpAttributes attrs = {
      vector<int64_t>{3, 3},        // kernel_shape
      {},                           // output_padding
      {},                           // output_shape
      vector<int64_t>{0, 0, 0, 0},  // pads
      vector<int64_t>{1, 1},        // strides
      vector<int64_t>{1, 1},        // dilations
      1                             // group
  };
  vector<float> X = {0., 1., 2., 3., 4., 5., 6., 7., 8.};
  vector<int64_t> X_shape = {1, 1, 3, 3};
  vector<float> W = {0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17.};
  vector<int64_t> W_shape = {1, 2, 3, 3};
  vector<int64_t> Y_shape = {1, 2, 5, 5};
  auto expected_vals = {
      0.f, 0.f, 1.f, 4.f, 4.f,
      0.f, 6.f, 20.f, 26.f, 20.f,
      9.f, 36.f, 84.f, 84.f, 57.f,
      36.f, 90.f, 164.f, 134.f, 80.f,
      36.f, 84.f, 145.f, 112.f, 64.f,
      0.f, 9.f, 28.f, 31.f, 22.f,
      27.f, 78.f, 155.f, 134.f, 83.f,
      90.f, 225.f, 408.f, 327.f, 192.f,
      117.f, 270.f, 461.f, 350.f, 197.f,
      90.f, 201.f, 334.f, 247.f, 136.f};

  TestConvTransposeOp(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape);
}

TEST(ConvTransposeTest, ConvTranspose_onnx2) {
  ConvTransposeOpAttributes attrs = {
      vector<int64_t>{2, 2},        // kernel_shape
      {},                           // output_padding
      {},                           // output_shape
      vector<int64_t>{0, 0, 0, 0},  // pads
      vector<int64_t>{1, 1},        // strides
      vector<int64_t>{1, 1},        // dilations
      1                             // group
  };
  vector<float> X = {0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17.};
  vector<int64_t> X_shape = {1, 2, 3, 3};
  vector<float> W = {0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23.};
  vector<int64_t> W_shape = {2, 3, 2, 2};  // this requires weight transpose
  vector<int64_t> Y_shape = {1, 3, 4, 4};
  auto expected_vals = {
      108.f, 237.f, 263.f, 145.f,
      270.f, 592.f, 652.f, 358.f,
      354.f, 772.f, 832.f, 454.f,
      222.f, 481.f, 515.f, 279.f,
      144.f, 317.f, 359.f, 197.f,
      366.f, 800.f, 892.f, 486.f,
      498.f, 1076.f, 1168.f, 630.f,
      306.f, 657.f, 707.f, 379.f,
      180.f, 397.f, 455.f, 249.f,
      462.f, 1008.f, 1132.f, 614.f,
      642.f, 1380.f, 1504.f, 806.f,
      390.f, 833.f, 899.f, 479.f};

  TestConvTransposeOp(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape);
}

TEST(ConvTransposeTest, ConvTranspose_onnx_group) {
  ConvTransposeOpAttributes attrs = {
      vector<int64_t>{1, 1},        // kernel_shape
      {},                           // output_padding
      {},                           // output_shape
      vector<int64_t>{0, 0, 0, 0},  // pads
      vector<int64_t>{1, 1},        // strides
      vector<int64_t>{1, 1},        // dilations
      4                             // group
  };
  vector<float> X = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f};
  vector<int64_t> X_shape = {1, 16, 1, 1};
  vector<float> W = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f, 16.f, 17.f, 18.f, 19.f, 20.f, 21.f, 22.f, 23.f, 24.f, 25.f, 26.f, 27.f, 28.f, 29.f, 30.f, 31.0f};
  vector<int64_t> W_shape = {16, 2, 1, 1};
  vector<int64_t> Y_shape = {1, 8, 1, 1};
  auto expected_vals = {28.f, 34.f, 252.f, 274.f, 732.f, 770.f, 1468.f, 1522.f};
  TestConvTransposeOp(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape);
}

TEST(ConvTransposeTest, ConvTranspose_2D_Dilation_1) {
  ConvTransposeOpAttributes attrs = {
      vector<int64_t>{2, 2},
      {},
      {},
      vector<int64_t>{0, 0, 0, 0},
      vector<int64_t>{1, 1},
      {2, 2},
      1};

  vector<float> X = {11.0f, 12.0f, 21.0f, 22.0f};
  vector<int64_t> X_shape = {1, 1, 2, 2};
  vector<float> W = {1.0f, 1.0f, 1.0f, 1.0f};
  vector<int64_t> W_shape = {1, 1, 2, 2};
  vector<int64_t> Y_shape = {1, 1, 4, 4};
  auto expected_vals = {11.0f, 12.0f, 11.0f, 12.0f,
                        21.0f, 22.0f, 21.0f, 22.0f,
                        11.0f, 12.0f, 11.0f, 12.0f,
                        21.0f, 22.0f, 21.0f, 22.0f};
  TestConvTransposeOp(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape);
}

TEST(ConvTransposeTest, ConvTranspose_2D_Dilation_2) {
  ConvTransposeOpAttributes attrs = {
      vector<int64_t>{2, 2},
      {},
      {},
      vector<int64_t>{0, 0, 0, 0},
      vector<int64_t>{1, 1},
      {3, 3},
      1};

  vector<float> X = {11.0f, 12.0f, 21.0f, 22.0f};
  vector<int64_t> X_shape = {1, 1, 2, 2};
  vector<float> W = {1.0f, 1.0f, 1.0f, 1.0f};
  vector<int64_t> W_shape = {1, 1, 2, 2};
  vector<int64_t> Y_shape = {1, 1, 5, 5};
  auto expected_vals = {11.0f, 12.0f, 0.0f, 11.0f, 12.0f,
                        21.0f, 22.0f, 0.0f, 21.0f, 22.0f,
                        0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                        11.0f, 12.0f, 0.0f, 11.0f, 12.0f,
                        21.0f, 22.0f, 0.0f, 21.0f, 22.0f};
  TestConvTransposeOp(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape);
}

TEST(ConvTransposeTest, ConvTranspose_2D_Dilation_3) {
  ConvTransposeOpAttributes attrs = {
      vector<int64_t>{2, 2},
      {},
      {},
      vector<int64_t>{0, 0, 0, 0},
      vector<int64_t>{1, 1},
      {2, 2},
      1};

  vector<float> X = {3.0f, 8.0f, 1.0f, 9.0f, 5.0f, 7.0f, 3.0f, 2.0f, 6.0f};
  vector<int64_t> X_shape = {1, 1, 3, 3};
  vector<float> W = {7.0f, 2.0f, 1.0f, 9.0f};
  vector<int64_t> W_shape = {1, 1, 2, 2};
  vector<int64_t> Y_shape = {1, 1, 5, 5};
  auto expected_vals = {21.0f, 56.0f, 13.0f, 16.0f, 2.0f,
                        63.0f, 35.0f, 67.0f, 10.0f, 14.0f,
                        24.0f, 22.0f, 76.0f, 76.0f, 21.0f,
                        9.0f, 5.0f, 88.0f, 45.0f, 63.0f,
                        3.0f, 2.0f, 33.0f, 18.0f, 54.0f};

  TestConvTransposeOp(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape);
}

TEST(ConvTransposeTest, ConvTranspose_2D_Dilation_4) {
  ConvTransposeOpAttributes attrs = {
      vector<int64_t>{2, 2},
      {},
      {},
      vector<int64_t>{0, 0, 0, 0},
      vector<int64_t>{1, 1},
      {3, 3},
      1};

  vector<float> X = {3.0f, 8.0f, 1.0f, 9.0f, 5.0f, 7.0f, 3.0f, 2.0f, 6.0f};
  vector<int64_t> X_shape = {1, 1, 3, 3};
  vector<float> W = {7.0f, 2.0f, 1.0f, 9.0f};
  vector<int64_t> W_shape = {1, 1, 2, 2};
  vector<int64_t> Y_shape = {1, 1, 6, 6};
  auto expected_vals = {21.0f, 56.0f, 7.0f, 6.0f, 16.0f, 2.0f,
                        63.0f, 35.0f, 49.0f, 18.0f, 10.0f, 14.0f,
                        21.0f, 14.0f, 42.0f, 6.0f, 4.0f, 12.0f,
                        3.0f, 8.0f, 1.0f, 27.0f, 72.0f, 9.0f,
                        9.0f, 5.0f, 7.0f, 81.0f, 45.0f, 63.0f,
                        3.0f, 2.0f, 6.0f, 27.0f, 18.0f, 54.0f};

  TestConvTransposeOp(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape);
}

TEST(ConvTransposeTest, ConvTranspose_2D_Dilation_Group_1) {
  ConvTransposeOpAttributes attrs = {
      vector<int64_t>{2, 2},
      {},
      {},
      vector<int64_t>{0, 0, 0, 0},
      vector<int64_t>{1, 1},
      {2, 2},
      2};

  vector<float> X = {3.0f, 8.0f, 1.0f, 9.0f, 5.0f, 7.0f, 3.0f, 2.0f, 3.0f, 7.0f, 9.0f, 1.0f, 5.0f, 2.0f, 3.0f, 9.0f, 0.0f, 2.0f};
  vector<int64_t> X_shape = {1, 2, 3, 3};
  vector<float> W = {9.0f, 3.0f, 1.0f, 2.0f, 3.0f, 7.0f, 0.0f, 8.0f};
  vector<int64_t> W_shape = {2, 1, 2, 2};
  vector<int64_t> Y_shape = {1, 2, 5, 5};
  auto expected_vals = {27.0f, 72.0f, 18.0f, 24.0f, 3.0f,
                        81.0f, 45.0f, 90.0f, 15.0f, 21.0f,
                        30.0f, 26.0f, 43.0f, 22.0f, 11.0f,
                        9.0f, 5.0f, 25.0f, 10.0f, 14.0f,
                        3.0f, 2.0f, 9.0f, 4.0f, 6.0f,
                        21.0f, 27.0f, 52.0f, 63.0f, 7.0f,
                        15.0f, 6.0f, 44.0f, 14.0f, 21.0f,
                        27.0f, 0.0f, 125.0f, 72.0f, 22.0f,
                        0.0f, 0.0f, 40.0f, 16.0f, 24.0f,
                        0.0f, 0.0f, 72.0f, 0.0f, 16.0f};

  TestConvTransposeOp(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape);
}

TEST(ConvTransposeTest, ConvTranspose_DefaultStridesAndDilations) {
  ConvTransposeOpAttributes attrs = {
      vector<int64_t>{2, 2},        // kernel_shape
      {},                           // output_padding
      {},                           // output_shape
      vector<int64_t>{0, 0, 0, 0},  // pads
      vector<int64_t>{},            // strides
      vector<int64_t>{},            // dilations
      1                             // group
  };
  vector<float> X = {0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17.};
  vector<int64_t> X_shape = {1, 2, 3, 3};
  vector<float> W = {0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23.};
  vector<int64_t> W_shape = {2, 3, 2, 2};  // this requires weight transpose
  vector<int64_t> Y_shape = {1, 3, 4, 4};
  auto expected_vals = {
      108.f, 237.f, 263.f, 145.f,
      270.f, 592.f, 652.f, 358.f,
      354.f, 772.f, 832.f, 454.f,
      222.f, 481.f, 515.f, 279.f,
      144.f, 317.f, 359.f, 197.f,
      366.f, 800.f, 892.f, 486.f,
      498.f, 1076.f, 1168.f, 630.f,
      306.f, 657.f, 707.f, 379.f,
      180.f, 397.f, 455.f, 249.f,
      462.f, 1008.f, 1132.f, 614.f,
      642.f, 1380.f, 1504.f, 806.f,
      390.f, 833.f, 899.f, 479.f};

  TestConvTransposeOp(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape);
}

}  // namespace test
}  // namespace onnxruntime
