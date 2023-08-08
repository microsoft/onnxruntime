// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace contrib {
namespace test {

using namespace std;
using namespace onnxruntime::test;

#if USE_CUDA || USE_ROCM
namespace {

struct ConvGradOpAttributes {
  vector<int64_t> dilations;
  int64_t group;
  vector<int64_t> kernel_shape;
  vector<int64_t> pads;
  vector<int64_t> strides;
};

void TestConvGradOp(const ConvGradOpAttributes& attributes, const vector<vector<float>>& inputs,
                    const vector<vector<int64_t>>& input_shapes, const vector<vector<float>>& outputs,
                    const vector<vector<int64_t>>& output_shapes, bool is_half = false) {
  OpTester test("ConvGrad", 1, kMSDomain);
  test.AddAttribute("group", attributes.group);
  test.AddAttribute("kernel_shape", attributes.kernel_shape);
  test.AddAttribute("pads", attributes.pads);

  if (!attributes.dilations.empty()) {
    test.AddAttribute("dilations", attributes.dilations);
  }

  if (!attributes.strides.empty()) {
    test.AddAttribute("strides", attributes.strides);
  }

  if (is_half) {
    std::vector<MLFloat16> dY_half(inputs[0].size());
    ConvertFloatToMLFloat16(inputs[0].data(), dY_half.data(), static_cast<int>(inputs[0].size()));
    test.AddInput<MLFloat16>("dY", input_shapes[0], dY_half);

    std::vector<MLFloat16> X_half(inputs[1].size());
    ConvertFloatToMLFloat16(inputs[1].data(), X_half.data(), static_cast<int>(inputs[1].size()));
    test.AddInput<MLFloat16>("X", input_shapes[1], X_half);

    std::vector<MLFloat16> W_half(inputs[2].size());
    ConvertFloatToMLFloat16(inputs[2].data(), W_half.data(), static_cast<int>(inputs[2].size()));
    test.AddInput<MLFloat16>("W", input_shapes[2], W_half);

    std::vector<MLFloat16> dX_half(outputs[0].size());
    ConvertFloatToMLFloat16(outputs[0].data(), dX_half.data(), static_cast<int>(outputs[0].size()));
    test.AddOutput<MLFloat16>("dX", output_shapes[0], dX_half);

    std::vector<MLFloat16> dW_half(outputs[1].size());
    ConvertFloatToMLFloat16(outputs[1].data(), dW_half.data(), static_cast<int>(outputs[1].size()));
    test.AddOutput<MLFloat16>("dW", output_shapes[1], dW_half);

    if (outputs.size() >= 3) {
      std::vector<MLFloat16> dB_half(outputs[2].size());
      ConvertFloatToMLFloat16(outputs[2].data(), dB_half.data(), static_cast<int>(outputs[2].size()));
      test.AddOutput<MLFloat16>("dB", output_shapes[2], dB_half);
    }
  } else {
    test.AddInput<float>("dY", input_shapes[0], inputs[0]);
    test.AddInput<float>("X", input_shapes[1], inputs[1]);
    test.AddInput<float>("W", input_shapes[2], inputs[2]);

    test.AddOutput<float>("dX", output_shapes[0], outputs[0]);
    test.AddOutput<float>("dW", output_shapes[1], outputs[1]);

    if (outputs.size() >= 3) {
      test.AddOutput<float>("dB", output_shapes[2], outputs[2]);
    }
  }

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

}  // namespace

TEST(ConvGradTest, Conv1D_1) {
  ConvGradOpAttributes attrs = {
      vector<int64_t>{1},     // dilations
      1,                      // group
      vector<int64_t>{1},     // kernel_shape
      vector<int64_t>{0, 0},  // pads
      vector<int64_t>{1},     // strides
  };

  vector<float> dY(7, 1.0f);
  vector<int64_t> dY_shape = {1, 1, 7};
  vector<float> X = {2.0349f, -1.8088f, -0.1171f, 1.1849f, -0.6590f, -2.0404f, -1.2810f};
  vector<int64_t> X_shape = {1, 1, 7};
  vector<float> W = {0.5081f};
  vector<int64_t> W_shape = {1, 1, 1};
  vector<float> dX = {0.5081f, 0.5081f, 0.5081f, 0.5081f, 0.5081f, 0.5081f, 0.5081f};
  vector<int64_t> dX_shape = {1, 1, 7};
  vector<float> dW = {-2.6865f};
  vector<int64_t> dW_shape = {1, 1, 1};

  TestConvGradOp(attrs, {dY, X, W}, {dY_shape, X_shape, W_shape}, {dX, dW}, {dX_shape, dW_shape});
  TestConvGradOp(attrs, {dY, X, W}, {dY_shape, X_shape, W_shape}, {dX, dW}, {dX_shape, dW_shape}, true);
}

TEST(ConvGradTest, Conv1D_2) {
  ConvGradOpAttributes attrs = {
      vector<int64_t>{2},     // dilations
      1,                      // group
      vector<int64_t>{2},     // kernel_shape
      vector<int64_t>{2, 2},  // pads
      vector<int64_t>{2},     // strides
  };

  vector<float> dY(30, 1.0f);
  vector<int64_t> dY_shape = {3, 2, 5};
  vector<float> X = {-0.9303f, 0.3717f, 0.4961f, 0.5068f, -0.7506f, -0.7609f, -1.8795f, 0.0536f,
                     1.5201f, -0.9580f, -1.7678f, 0.4683f, -0.3142f, 0.2097f, -1.3819f, -0.1070f,
                     -1.7558f, -0.0278f, 1.5378f, 2.6415f, 1.0004f, 1.3604f, 1.2819f, -0.4629f};
  vector<int64_t> X_shape = {3, 1, 8};
  vector<float> W = {1.6664f, -0.1582f, -0.8984f, 0.0849f};
  vector<int64_t> W_shape = {2, 1, 2};
  vector<float> dX = {0.6948f, 0.0000f, 0.6948f, 0.0000f, 0.6948f, 0.0000f, 0.6948f, 0.0000f,
                      0.6948f, 0.0000f, 0.6948f, 0.0000f, 0.6948f, 0.0000f, 0.6948f, 0.0000f,
                      0.6948f, 0.0000f, 0.6948f, 0.0000f, 0.6948f, 0.0000f, 0.6948f, 0.0000f};
  vector<int64_t> dX_shape = {3, 1, 8};
  vector<float> dW = {-2.9440f, -2.9440f, -2.9440f, -2.9440f};
  vector<int64_t> dW_shape = {2, 1, 2};

  TestConvGradOp(attrs, {dY, X, W}, {dY_shape, X_shape, W_shape}, {dX, dW}, {dX_shape, dW_shape});
  TestConvGradOp(attrs, {dY, X, W}, {dY_shape, X_shape, W_shape}, {dX, dW}, {dX_shape, dW_shape}, true);
}

TEST(ConvGradTest, Conv1D_Bias) {
  ConvGradOpAttributes attrs = {
      vector<int64_t>{2},     // dilations
      1,                      // group
      vector<int64_t>{1},     // kernel_shape
      vector<int64_t>{1, 1},  // pads
      vector<int64_t>{3},     // strides
  };

  vector<float> dY(8, 1.0f);
  vector<int64_t> dY_shape = {2, 1, 4};
  vector<float> X = {0.3305f, 2.6170f, -0.8102f, -1.1348f, -0.0850f, 0.2033f, 0.7295f, -0.2826f, -0.5977f,
                     0.5505f, 0.3895f, -1.3394f, 0.6413f, -0.4744f, -0.9943f, 0.7560f, 0.1355f, -1.3931f,
                     1.2644f, 0.0240f, 0.7571f, 0.6851f, -0.3362f, -1.1230f, 0.6475f, -0.4596f, 1.1648f,
                     0.8991f, 0.0440f, 1.5056f, 0.9504f, -0.5266f, 0.0437f, -0.3006f, 0.8489f, 0.0960f};
  vector<int64_t> X_shape = {2, 2, 9};
  vector<float> W = {0.0398f, 0.1392f};
  vector<int64_t> W_shape = {1, 2, 1};
  vector<float> dX = {0.0000f, 0.0000f, 0.0398f, 0.0000f, 0.0000f, 0.0398f, 0.0000f, 0.0000f, 0.0398f,
                      0.0000f, 0.0000f, 0.1392f, 0.0000f, 0.0000f, 0.1392f, 0.0000f, 0.0000f, 0.1392f,
                      0.0000f, 0.0000f, 0.0398f, 0.0000f, 0.0000f, 0.0398f, 0.0000f, 0.0000f, 0.0398f,
                      0.0000f, 0.0000f, 0.1392f, 0.0000f, 0.0000f, 0.1392f, 0.0000f, 0.0000f, 0.1392f};
  vector<int64_t> dX_shape = {2, 2, 9};
  vector<float> dW = {-0.4057f, -2.0815f};
  vector<int64_t> dW_shape = {1, 2, 1};
  vector<float> dB = {8.f};
  vector<int64_t> dB_shape = {1};

  TestConvGradOp(attrs, {dY, X, W}, {dY_shape, X_shape, W_shape}, {dX, dW, dB}, {dX_shape, dW_shape, dB_shape});
  TestConvGradOp(attrs, {dY, X, W}, {dY_shape, X_shape, W_shape}, {dX, dW, dB}, {dX_shape, dW_shape, dB_shape}, true);
}

TEST(ConvGradTest, Conv2D) {
  ConvGradOpAttributes attrs = {
      vector<int64_t>{1, 1},        // dilations
      1,                            // group
      vector<int64_t>{1, 1},        // kernel_shape
      vector<int64_t>{0, 0, 0, 0},  // pads
      vector<int64_t>{1, 1},        // strides
  };

  vector<float> dY(48, 1.0f);
  vector<int64_t> dY_shape = {1, 3, 4, 4};
  vector<float> X = {0.8374f, -2.0758f, 1.8918f, -1.0625f, -1.2747f, -0.1561f, 0.4573f, 2.5314f, -0.0089f, -1.0412f,
                     0.7690f, 0.2320f, 0.6535f, -0.4921f, -0.6051f, 0.5580f, 1.5682f, -1.0309f, -0.9379f, -0.1834f,
                     -1.2162f, 1.4167f, -0.2849f, -0.1625f, 0.3380f, -0.1393f, -1.1557f, 0.9718f, -0.4656f, -0.9046f,
                     1.5710f, -1.3963f, 1.2470f, 0.7327f, 0.8045f, 0.8071f, 1.1703f, -1.3566f, -0.2030f, -0.1227f,
                     -0.5881f, 2.4159f, -0.2768f, 0.5567f, -0.2805f, 0.1618f, -0.7256f, -0.1053f};
  vector<int64_t> X_shape = {1, 3, 4, 4};
  vector<float> W = {0.1094f, 1.1541f, 0.0486f, 0.5668f, 1.0372f, -0.3792f, 1.4979f, 0.1757f, 0.1733f};
  vector<int64_t> W_shape = {3, 3, 1, 1};
  vector<float> dX = {2.1741f, 2.1741f, 2.1741f, 2.1741f, 2.1741f, 2.1741f, 2.1741f, 2.1741f,
                      2.1741f, 2.1741f, 2.1741f, 2.1741f, 2.1741f, 2.1741f, 2.1741f, 2.1741f,
                      2.3670f, 2.3670f, 2.3670f, 2.3670f, 2.3670f, 2.3670f, 2.3670f, 2.3670f,
                      2.3670f, 2.3670f, 2.3670f, 2.3670f, 2.3670f, 2.3670f, 2.3670f, 2.3670f,
                      -0.1572f, -0.1572f, -0.1572f, -0.1572f, -0.1572f, -0.1572f, -0.1572f, -0.1572f,
                      -0.1572f, -0.1572f, -0.1572f, -0.1572f, -0.1572f, -0.1572f, -0.1572f, -0.1572f};
  vector<int64_t> dX_shape = {1, 3, 4, 4};
  vector<float> dW = {1.2142f, -2.0115f, 4.2372f, 1.2142f, -2.0115f, 4.2372f, 1.2142f, -2.0115f, 4.2372f};
  vector<int64_t> dW_shape = {3, 3, 1, 1};

  TestConvGradOp(attrs, {dY, X, W}, {dY_shape, X_shape, W_shape}, {dX, dW}, {dX_shape, dW_shape});
  TestConvGradOp(attrs, {dY, X, W}, {dY_shape, X_shape, W_shape}, {dX, dW}, {dX_shape, dW_shape}, true);
}

TEST(ConvGradTest, Conv2D_Bias) {
  ConvGradOpAttributes attrs = {
      vector<int64_t>{1, 1},        // dilations
      1,                            // group
      vector<int64_t>{2, 2},        // kernel_shape
      vector<int64_t>{0, 0, 0, 0},  // pads
      vector<int64_t>{1, 1},        // strides
  };

  vector<float> dY(8, 1.0f);
  vector<int64_t> dY_shape = {1, 2, 2, 2};
  vector<float> X = {-0.4406f, 0.3064f, 0.0794f, -0.2795f, 0.8228f, 0.4751f, 1.3114f, 0.9522f, 0.9082f};
  vector<int64_t> X_shape = {1, 1, 3, 3};
  vector<float> W = {-0.0820f, -0.4214f, -2.2745f, -1.5834f, 0.9746f, 0.6936f, -0.5140f, 0.9900f};
  vector<int64_t> W_shape = {2, 1, 2, 2};
  vector<float> dX = {0.8926f, 1.1648f, 0.2723f, -1.8959f, -2.2169f, -0.3211f, -2.7884f, -3.3818f, -0.5933f};
  vector<int64_t> dX_shape = {1, 1, 3, 3};
  vector<float> dW = {0.4092f, 1.6838f, 2.8069f, 3.1583f, 0.4092f, 1.6838f, 2.8069f, 3.1583f};
  vector<int64_t> dW_shape = {2, 1, 2, 2};
  vector<float> dB = {4.f, 4.f};
  vector<int64_t> dB_shape = {2};

  TestConvGradOp(attrs, {dY, X, W}, {dY_shape, X_shape, W_shape}, {dX, dW, dB}, {dX_shape, dW_shape, dB_shape});
  TestConvGradOp(attrs, {dY, X, W}, {dY_shape, X_shape, W_shape}, {dX, dW, dB}, {dX_shape, dW_shape, dB_shape}, true);
}

TEST(ConvGradTest, Conv3D) {
  ConvGradOpAttributes attrs = {
      vector<int64_t>{1, 1, 1},           // dilations
      1,                                  // group
      vector<int64_t>{2, 2, 1},           // kernel_shape
      vector<int64_t>{2, 2, 2, 2, 2, 2},  // pads
      vector<int64_t>{2, 2, 2},           // strides
  };

  vector<float> dY(81, 1.0f);
  vector<int64_t> dY_shape = {1, 3, 3, 3, 3};
  vector<float> X = {0.5598f, -1.9201f, -0.7435f, 2.0217f, 1.2615f, -1.9540f, 0.3119f, 0.0106f, -0.1752f,
                     0.1553f, 1.3088f, 0.8588f, -0.6396f, -1.1059f, -0.7768f, 0.3251f, -0.3116f, -0.1495f,
                     0.8923f, -0.1832f, -0.3995f, -0.9641f, 1.9743f, -0.3098f, -0.3029f, -0.3453f, -0.7708f,
                     -0.3267f, -0.5051f, -0.9330f, -0.2421f, -0.0874f, 0.3225f, -0.8572f, -0.2019f, -0.7069f,
                     0.4333f, 0.5562f, -1.5587f, -1.0665f, -0.6832f, -0.4320f, -0.0225f, 1.4662f, 0.4808f,
                     0.0282f, 0.6967f, 0.5708f, -1.3258f, -0.6925f, 0.1217f, 1.3211f, 0.5877f, 0.7335f};
  vector<int64_t> X_shape = {1, 3, 3, 3, 2};
  vector<float> W = {-0.1911f, 0.6604f, -1.0283f, -0.9381f, -0.3449f, 1.1152f, -1.0256f, -0.3494f, 0.4504f,
                     0.2418f, 0.2258f, -1.5920f, 1.0468f, 0.2045f, 0.8264f, -0.5797f, -0.0254f, 0.6934f,
                     -1.7728f, 0.8619f, -0.2013f, -0.1045f, -0.4713f, 1.2544f, 1.7090f, -0.7133f, -0.6160f,
                     -1.2325f, -1.2152f, 0.0935f, -0.4929f, 1.3772f, 0.3125f, -0.7773f, 1.0350f, 3.2168f};
  vector<int64_t> W_shape = {3, 3, 2, 2, 1};
  vector<float> dX = {2.5646f, 0.0000f, 0.1516f, 0.0000f, 2.5646f, 0.0000f, -0.8178f, 0.0000f, -2.7503f,
                      0.0000f, -0.8178f, 0.0000f, 2.5646f, 0.0000f, 0.1516f, 0.0000f, 2.5646f, 0.0000f,
                      -1.5855f, 0.0000f, 1.9021f, 0.0000f, -1.5855f, 0.0000f, -3.2913f, 0.0000f, 1.8898f,
                      0.0000f, -3.2913f, 0.0000f, -1.5855f, 0.0000f, 1.9021f, 0.0000f, -1.5855f, 0.0000f,
                      0.5616f, 0.0000f, -0.6399f, 0.0000f, 0.5616f, 0.0000f, 0.7895f, 0.0000f, 2.8792f,
                      0.0000f, 0.7895f, 0.0000f, 0.5616f, 0.0000f, -0.6399f, 0.0000f, 0.5616f, 0.0000f};
  vector<int64_t> dX_shape = {1, 3, 3, 3, 2};
  vector<float> dW = {0.8701f, -1.5203f, 1.6207f, -0.1752f, 2.4225f, -0.0770f, -0.8080f, -0.7708f, -0.9880f,
                      -1.4370f, 0.6742f, 0.4808f, 0.8701f, -1.5203f, 1.6207f, -0.1752f, 2.4225f, -0.0770f,
                      -0.8080f, -0.7708f, -0.9880f, -1.4370f, 0.6742f, 0.4808f, 0.8701f, -1.5203f, 1.6207f,
                      -0.1752f, 2.4225f, -0.0770f, -0.8080f, -0.7708f, -0.9880f, -1.4370f, 0.6742f, 0.4808f};
  vector<int64_t> dW_shape = {3, 3, 2, 2, 1};

  TestConvGradOp(attrs, {dY, X, W}, {dY_shape, X_shape, W_shape}, {dX, dW}, {dX_shape, dW_shape});
  TestConvGradOp(attrs, {dY, X, W}, {dY_shape, X_shape, W_shape}, {dX, dW}, {dX_shape, dW_shape}, true);
}

TEST(ConvTest, Conv3D_Bias) {
  ConvGradOpAttributes attrs = {
      vector<int64_t>{2, 2, 2},           // dilations
      1,                                  // group
      vector<int64_t>{2, 2, 2},           // kernel_shape
      vector<int64_t>{2, 2, 2, 2, 2, 2},  // pads
      vector<int64_t>{2, 2, 2},           // strides
  };

  vector<float> dY(108, 1.0f);
  vector<int64_t> dY_shape = {2, 2, 3, 3, 3};
  vector<float> X = {0.5189f, -0.8970f, 1.6637f, -0.0882f, 0.0562f, 1.5707f, 0.5424f, 0.4293f, -0.7949f, 2.1479f,
                     0.8582f, -0.5160f, -0.2406f, 0.0915f, -0.6440f, -0.3800f, 1.7989f, -1.6479f, 0.9071f, -0.4087f,
                     1.3729f, -1.5582f, 1.9312f, 1.0753f, 0.8313f, 0.5097f, -0.0664f, -1.2774f, 1.7208f, -1.7777f,
                     -2.0758f, -0.4782f, 2.0207f, 1.4053f, 0.3488f, -0.0871f, 1.4151f, 2.5243f, 0.2891f, 0.8317f,
                     1.8934f, 0.3911f, 0.2915f, -0.8505f, 1.0430f, 0.5391f, 0.8347f, 0.0633f, -0.3250f, 1.3358f,
                     -0.3121f, 0.4587f, -0.4955f, 1.8411f, 0.9877f, 1.0809f, 0.0119f, -1.2706f, 1.8457f, -0.1520f,
                     -0.4535f, -0.5325f, -0.8921f, -0.3127f, 0.5746f, -1.2514f, 0.4638f, 0.8440f, -0.6113f, 0.6936f,
                     0.0998f, 0.9767f, 0.2785f, -0.3068f, -0.4619f, 0.4801f, -2.1590f, -1.7342f, 0.7354f, 0.0234f,
                     1.8095f, 0.1252f, -0.5841f, 0.0738f, 1.4252f, 1.4222f, -0.1192f, -2.9955f, 0.8287f, 0.6252f,
                     -1.5834f, -0.1388f, 0.5532f, 0.4044f, 1.0432f, -2.3991f, 0.4339f, 0.1083f, -0.7726f, 2.0629f,
                     0.7136f, -0.0978f, -0.7905f, 0.9585f, -0.3205f, 1.3750f, 0.4137f, -0.4552f, 2.7165f, -1.6367f,
                     -0.6286f, -0.4656f, 0.6219f, -1.7275f, 1.7599f, -1.0443f, 1.3212f, 0.1621f, -0.5357f, 0.0957f,
                     0.4524f, -0.3814f, -0.0744f, 0.8301f, -0.9539f, 0.0867f, -0.7864f, 1.4918f};
  vector<int64_t> X_shape = {2, 1, 4, 4, 4};
  vector<float> W = {-0.6570f, 0.1637f, 1.7824f, 0.7986f, -0.2703f, -0.7447f, 1.2674f, 0.2019f,
                     0.1045f, -0.7279f, 0.9658f, 0.4698f, -0.6699f, -1.5259f, 2.1664f, -0.3859f};
  vector<int64_t> W_shape = {2, 1, 2, 2, 2};
  vector<float> dX = {2.9389f, 0.0000f, 2.9389f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 2.9389f, 0.0000f, 2.9389f,
                      0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f,
                      0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 2.9389f,
                      0.0000f, 2.9389f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 2.9389f, 0.0000f, 2.9389f, 0.0000f,
                      0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f,
                      0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 2.9389f, 0.0000f,
                      2.9389f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 2.9389f, 0.0000f, 2.9389f, 0.0000f, 0.0000f,
                      0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f,
                      0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 2.9389f, 0.0000f, 2.9389f,
                      0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 2.9389f, 0.0000f, 2.9389f, 0.0000f, 0.0000f, 0.0000f,
                      0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f,
                      0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f};
  vector<int64_t> dX_shape = {2, 1, 4, 4, 4};
  vector<float> dW = {7.4097f, 7.4097f, 7.4097f, 7.4097f, 7.4097f, 7.4097f, 7.4097f, 7.4097f,
                      7.4097f, 7.4097f, 7.4097f, 7.4097f, 7.4097f, 7.4097f, 7.4097f, 7.4097f};
  vector<int64_t> dW_shape = {2, 1, 2, 2, 2};
  vector<float> dB = {54.f, 54.f};
  vector<int64_t> dB_shape = {2};

  TestConvGradOp(attrs, {dY, X, W}, {dY_shape, X_shape, W_shape}, {dX, dW, dB}, {dX_shape, dW_shape, dB_shape});
  TestConvGradOp(attrs, {dY, X, W}, {dY_shape, X_shape, W_shape}, {dX, dW, dB}, {dX_shape, dW_shape, dB_shape}, true);
}
#endif  // USE_CUDA || USE_ROCM

}  // namespace test
}  // namespace contrib
}  // namespace onnxruntime
