// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime::contrib::test {

using namespace onnxruntime::test;

#if USE_CUDA
namespace {

struct ConvTransposeGradOpAttributes {
  std::vector<int64_t> dilations;
  int64_t group;
  std::vector<int64_t> kernel_shape;
  std::vector<int64_t> pads;
  std::vector<int64_t> strides;
};

void TestConvTransposeGradOp(const ConvTransposeGradOpAttributes& attributes,
                             const std::vector<std::vector<float>>& inputs,
                             const std::vector<std::vector<int64_t>>& input_shapes,
                             const std::vector<std::vector<float>>& outputs,
                             const std::vector<std::vector<int64_t>>& output_shapes,
                             bool is_half = false) {
  OpTester test("ConvTransposeGrad", 1, kMSDomain);
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

  test.Run();
}

}  // namespace

TEST(ConvTransposeGradTest, ConvTranspose1DDefaultAttributes) {
  ConvTransposeGradOpAttributes attrs = {
      std::vector<int64_t>{1},     // dilations
      1,                           // group
      std::vector<int64_t>{2},     // kernel_shape
      std::vector<int64_t>{0, 0},  // pads
      std::vector<int64_t>{1},     // strides
  };

  std::vector<float> dY(12, 1.0f);
  std::vector<int64_t> dY_shape = {1, 2, 6};
  std::vector<float> X = {0.1868f, -0.1679f, 1.2677f, 2.1288f, -0.0331f,
                          1.0454f, 0.7722f, 0.2963f, -0.8684f, -0.0547f};
  std::vector<int64_t> X_shape = {1, 2, 5};
  std::vector<float> W = {0.0847f, -0.0066f,
                          0.1212f, 0.2317f,
                          -0.4975f, 0.2762f,
                          -0.2644f, 0.3210f};
  std::vector<int64_t> W_shape = {2, 2, 2};
  std::vector<float> dX = {0.4309f, 0.4309f, 0.4309f, 0.4309f, 0.4309f,
                           -0.1647f, -0.1647f, -0.1647f, -0.1647f, -0.1647f};
  std::vector<int64_t> dX_shape = X_shape;
  std::vector<float> dW = {3.3823f, 3.3823f,
                           3.3823f, 3.3823f,
                           1.1908f, 1.1908f,
                           1.1908f, 1.1908f};
  std::vector<int64_t> dW_shape = W_shape;
  std::vector<float> dB = {6.f, 6.f};
  std::vector<int64_t> dB_shape = {2};

  for (const bool is_half : {false, true})
    TestConvTransposeGradOp(
        attrs,                           // attributes
        {dY, X, W},                      // inputs
        {dY_shape, X_shape, W_shape},    // input shapes
        {dX, dW, dB},                    // outputs
        {dX_shape, dW_shape, dB_shape},  // output shapes
        is_half);
}

TEST(ConvTransposeGradTest, ConvTranspose1DStrideAndPadding) {
  ConvTransposeGradOpAttributes attrs = {
      std::vector<int64_t>{1},     // dilations
      1,                           // group
      std::vector<int64_t>{2},     // kernel_shape
      std::vector<int64_t>{2, 2},  // pads
      std::vector<int64_t>{2},     // strides
  };

  std::vector<float> dY(12, 1.0f);
  std::vector<int64_t> dY_shape = {1, 2, 6};
  std::vector<float> X = {-0.0254f, -1.4303f, -0.1568f, 1.2318f, -0.8365f,
                          2.0836f, -1.0181f, -0.7539f, 0.4484f, -0.5799f};
  std::vector<int64_t> X_shape = {1, 2, 5};
  std::vector<float> W = {-0.1438f, 0.2386f,
                          -0.3085f, 0.1149f,
                          -0.1653f, -0.0707f,
                          -0.1479f, -0.0918f};
  std::vector<int64_t> W_shape = {2, 2, 2};
  std::vector<float> dX = {0.0000f, -0.0988f, -0.0988f, -0.0988f, 0.0000f,
                           0.0000f, -0.4757f, -0.4757f, -0.4757f, 0.0000f};
  std::vector<int64_t> dX_shape = X_shape;
  std::vector<float> dW = {-0.3553f, -0.3553f,
                           -0.3553f, -0.3553f,
                           -1.3236f, -1.3236f,
                           -1.3236f, -1.3236f};
  std::vector<int64_t> dW_shape = W_shape;
  std::vector<float> dB = {6.f, 6.f};
  std::vector<int64_t> dB_shape = {2};

  for (const bool is_half : {false, true})
    TestConvTransposeGradOp(
        attrs,                           // attributes
        {dY, X, W},                      // inputs
        {dY_shape, X_shape, W_shape},    // input shapes
        {dX, dW, dB},                    // outputs
        {dX_shape, dW_shape, dB_shape},  // output shapes
        is_half);
}

TEST(ConvTransposeGradTest, ConvTranspose1D) {
  ConvTransposeGradOpAttributes attrs = {
      std::vector<int64_t>{2},     // dilations
      2,                           // group
      std::vector<int64_t>{3},     // kernel_shape
      std::vector<int64_t>{2, 2},  // pads
      std::vector<int64_t>{2},     // strides
  };

  std::vector<float> dY(38, 1.0f);
  std::vector<int64_t> dY_shape = {1, 2, 19};
  std::vector<float> X = {0.2816f, 1.4660f, 0.1002f, -0.2460f, -0.1027f, 0.1228f, -0.8516f, -1.0246f, -0.6576f, -1.0280f,
                          0.1093f, 0.1447f, 1.1279f, 0.1085f, -0.3438f, -0.6224f, -0.0902f, 2.2791f, -2.1910f, 1.9736f};
  std::vector<int64_t> X_shape = {1, 2, 10};
  std::vector<float> W = {-0.1050f, -0.0622f, -0.3632f,
                          -0.3861f, -0.0134f, -0.0277f};
  std::vector<int64_t> W_shape = {2, 1, 3};
  std::vector<float> dX = {-0.4254f, -0.5304f, -0.5304f, -0.5304f, -0.5304f, -0.5304f, -0.5304f, -0.5304f, -0.5304f, -0.1672f,
                           -0.0411f, -0.4272f, -0.4272f, -0.4272f, -0.4272f, -0.4272f, -0.4272f, -0.4272f, -0.4272f, -0.3995f};
  std::vector<int64_t> dX_shape = X_shape;
  std::vector<float> dW = {-2.2215f, -1.9400f, -0.9120f,
                           2.3863f, 2.4956f, 0.5220f};
  std::vector<int64_t> dW_shape = W_shape;
  std::vector<float> dB = {19.f, 19.f};
  std::vector<int64_t> dB_shape = {2};

  for (const bool is_half : {false, true})
    TestConvTransposeGradOp(
        attrs,                           // attributes
        {dY, X, W},                      // inputs
        {dY_shape, X_shape, W_shape},    // input shapes
        {dX, dW, dB},                    // outputs
        {dX_shape, dW_shape, dB_shape},  // output shapes
        is_half);
}

TEST(ConvTransposeGradTest, ConvTranspose2DDefaultAttributes) {
  ConvTransposeGradOpAttributes attrs = {
      std::vector<int64_t>{1, 1},        // dilations
      1,                                 // group
      std::vector<int64_t>{3, 3},        // kernel_shape
      std::vector<int64_t>{0, 0, 0, 0},  // pads
      std::vector<int64_t>{1, 1},        // strides
  };

  std::vector<float> dY(98, 1.0f);
  std::vector<int64_t> dY_shape = {1, 2, 7, 7};
  std::vector<float> X = {1.1371f, -0.1498f, -1.7541f, -0.7585f, 1.6009f, -0.7496f, 0.1535f, -0.2533f, -1.0811f, 0.9760f,
                          -0.2528f, 0.1820f, -1.7450f, 0.1632f, -0.3469f, 1.1150f, -2.6888f, -0.1632f, -0.3269f, 0.6904f,
                          1.3036f, 0.7883f, 0.4459f, 0.1223f, 0.1576f, -0.8187f, 0.2281f, 1.5320f, 1.2643f, -0.5163f,
                          1.0677f, -0.2141f, 1.2992f, -2.1865f, -0.6346f, 0.8938f, 0.8346f, -2.7397f, 0.9223f, 0.8166f,
                          1.1736f, -1.3644f, 0.0316f, -1.2904f, 0.7062f, 0.2470f, 0.4559f, 0.8493f, 1.0519f, 0.9915f};
  std::vector<int64_t> X_shape = {1, 2, 5, 5};
  std::vector<float> W = {0.0761f, 0.0270f, -0.1677f, 0.1803f, -0.0824f, -0.0285f,
                          0.2098f, -0.0569f, -0.1514f, 0.0338f, -0.1962f, -0.2169f,
                          0.0432f, -0.1977f, -0.0814f, -0.1866f, -0.1574f, -0.0198f,
                          0.0097f, 0.0019f, -0.1204f, 0.2018f, -0.1750f, -0.0549f,
                          -0.0687f, -0.1269f, 0.1913f, 0.1331f, -0.0632f, 0.0821f,
                          0.0127f, 0.1761f, -0.0883f, -0.1370f, 0.1472f, 0.0690f};
  std::vector<int64_t> W_shape = {2, 2, 3, 3};
  std::vector<float> dX = {-0.9725f, -0.9725f, -0.9725f, -0.9725f, -0.9725f, -0.9725f, -0.9725f, -0.9725f, -0.9725f, -0.9725f,
                           -0.9725f, -0.9725f, -0.9725f, -0.9725f, -0.9725f, -0.9725f, -0.9725f, -0.9725f, -0.9725f, -0.9725f,
                           -0.9725f, -0.9725f, -0.9725f, -0.9725f, -0.9725f, 0.1905f, 0.1905f, 0.1905f, 0.1905f, 0.1905f,
                           0.1905f, 0.1905f, 0.1905f, 0.1905f, 0.1905f, 0.1905f, 0.1905f, 0.1905f, 0.1905f, 0.1905f,
                           0.1905f, 0.1905f, 0.1905f, 0.1905f, 0.1905f, 0.1905f, 0.1905f, 0.1905f, 0.1905f, 0.1905f};
  std::vector<int64_t> dX_shape = X_shape;
  std::vector<float> dW = {-1.4343f, -1.4343f, -1.4343f, -1.4343f, -1.4343f, -1.4343f,
                           -1.4343f, -1.4343f, -1.4343f, -1.4343f, -1.4343f, -1.4343f,
                           -1.4343f, -1.4343f, -1.4343f, -1.4343f, -1.4343f, -1.4343f,
                           4.6009f, 4.6009f, 4.6009f, 4.6009f, 4.6009f, 4.6009f,
                           4.6009f, 4.6009f, 4.6009f, 4.6009f, 4.6009f, 4.6009f,
                           4.6009f, 4.6009f, 4.6009f, 4.6009f, 4.6009f, 4.6009f};
  std::vector<int64_t> dW_shape = W_shape;
  std::vector<float> dB = {49.f, 49.f};
  std::vector<int64_t> dB_shape = {2};

  for (const bool is_half : {false, true})
    TestConvTransposeGradOp(
        attrs,                           // attributes
        {dY, X, W},                      // inputs
        {dY_shape, X_shape, W_shape},    // input shapes
        {dX, dW, dB},                    // outputs
        {dX_shape, dW_shape, dB_shape},  // output shapes
        is_half);
}

TEST(ConvTransposeGradTest, ConvTranspose2D) {
  ConvTransposeGradOpAttributes attrs = {
      std::vector<int64_t>{2, 2},        // dilations
      2,                                 // group
      std::vector<int64_t>{3, 3},        // kernel_shape
      std::vector<int64_t>{2, 2, 2, 2},  // pads
      std::vector<int64_t>{2, 2},        // strides
  };

  std::vector<float> dY(162U, 1.0f);
  std::vector<int64_t> dY_shape = {1, 2, 9, 9};
  std::vector<float> X = {-1.0158f, 0.1709f, -0.1660f, 0.3881f, 0.4017f, 1.5497f, 1.1205f, 0.2553f, -0.4359f, -0.0467f,
                          1.1374f, -0.0713f, 0.2248f, 0.8915f, -0.7239f, 0.1679f, -1.5604f, -0.8521f, 0.8966f, 3.3743f,
                          -0.5516f, 0.2516f, -0.4091f, -0.9868f, 0.3008f, 1.1066f, -0.7039f, -1.5273f, -0.3666f, 0.9392f,
                          0.1264f, -1.6604f, -1.4810f, 0.6654f, -0.2007f, -1.0660f, -0.5420f, -0.7030f, 0.0411f, 2.1082f,
                          -0.7995f, 0.2422f, 1.2848f, -0.1747f, 1.7935f, -0.1123f, -0.6668f, -2.2383f, 1.5419f, -2.7614f};
  std::vector<int64_t> X_shape = {1, 2, 5, 5};
  std::vector<float> W = {-0.2057f, -0.0411f, 0.0277f, 0.2221f, 0.1901f, 0.1435f,
                          -0.2249f, 0.3299f, -0.2203f, -0.1013f, -0.3326f, 0.1005f,
                          -0.0536f, 0.3067f, 0.3297f, 0.2728f, 0.1649f, -0.2548f};
  std::vector<int64_t> W_shape = {2, 1, 3, 3};
  std::vector<float> dX = {0.4431f, 0.4403f, 0.4403f, 0.4403f, 0.5171f, 0.4297f, 0.2212f, 0.2212f, 0.2212f, 0.2704f,
                           0.4297f, 0.2212f, 0.2212f, 0.2212f, 0.2704f, 0.4297f, 0.2212f, 0.2212f, 0.2212f, 0.2704f,
                           0.3202f, 0.3366f, 0.3366f, 0.3366f, 0.1654f, 0.5465f, 0.7658f, 0.7658f, 0.7658f, 0.6908f,
                           0.3144f, 0.4323f, 0.4323f, 0.4323f, 0.2569f, 0.3144f, 0.4323f, 0.4323f, 0.4323f, 0.2569f,
                           0.3144f, 0.4323f, 0.4323f, 0.4323f, 0.2569f, 0.4043f, 0.2494f, 0.2494f, 0.2494f, -0.1808f};
  std::vector<int64_t> dX_shape = X_shape;
  std::vector<float> dW = {2.2293f, 4.5327f, 1.6281f, 3.0240f, 4.3115f, 1.0052f,
                           3.8675f, 5.7067f, 2.7011f, -2.7512f, -4.6026f, -5.5423f,
                           -4.4098f, -5.1546f, -7.0335f, -0.2852f, -0.9177f, -5.5580f};
  std::vector<int64_t> dW_shape = W_shape;
  std::vector<float> dB = {81.f, 81.f};
  std::vector<int64_t> dB_shape = {2};

  for (const bool is_half : {false, true})
    TestConvTransposeGradOp(
        attrs,                           // attributes
        {dY, X, W},                      // inputs
        {dY_shape, X_shape, W_shape},    // input shapes
        {dX, dW, dB},                    // outputs
        {dX_shape, dW_shape, dB_shape},  // output shapes
        is_half);
}

TEST(ConvTransposeGradTest, ConvTranspose3D) {
  ConvTransposeGradOpAttributes attrs = {
      std::vector<int64_t>{2, 2, 2},           // dilations
      2,                                       // group
      std::vector<int64_t>{2, 2, 2},           // kernel_shape
      std::vector<int64_t>{2, 2, 2, 2, 2, 2},  // pads
      std::vector<int64_t>{2, 2, 2},           // strides
  };

  std::vector<float> dY(250U, 1.0f);
  std::vector<int64_t> dY_shape = {1, 2, 5, 5, 5};
  std::vector<float> X = {-0.2396f, 0.4280f, -1.3505f, -0.4366f, -1.3296f, 0.3531f, 0.0645f, -1.5480f,
                          -1.7464f, -0.9160f, 1.5065f, -0.0788f, 0.0487f, 2.4641f, 0.3855f, 2.0499f,
                          0.7068f, -0.8076f, -0.4442f, 0.1003f, -0.5056f, -0.1430f, -0.3744f, -0.2637f,
                          -1.1012f, 1.0213f, 0.0503f, 0.0147f, -0.3664f, 0.8834f, -1.1478f, -0.8221f,
                          -0.5649f, -0.4224f, -0.6779f, -0.9363f, 1.1972f, 0.2094f, 0.5676f, -0.2718f,
                          -0.1678f, -0.4178f, -0.4672f, 0.2777f, -0.7953f, -0.5603f, -2.8694f, 1.5743f,
                          -0.5057f, -0.2529f, 0.5894f, -0.3980f, -0.6719f, -0.3425f, 0.0821f, 0.8672f,
                          0.7218f, 1.5519f, 1.6513f, -1.1956f, 0.8471f, 0.4295f, -1.3917f, -1.2202f,
                          0.1054f, -2.2191f, -0.9546f, 1.1750f, -2.3637f, 1.6297f, -0.5796f, 0.3850f,
                          0.9287f, -0.3492f, -0.7284f, 0.2987f, -0.7534f, 0.7747f, -1.3198f, -0.3633f,
                          1.8635f, -0.3187f, 0.9032f, -0.6083f, -0.4236f, -0.1929f, -1.1715f, -0.5591f,
                          -1.8290f, -1.1503f, 0.1430f, 0.6048f, -0.3148f, 1.0638f, -0.2946f, -0.4990f,
                          -1.4443f, -0.7757f, -1.5374f, -0.4567f, -0.2998f, 0.0521f, 1.6293f, -0.6720f,
                          -0.0102f, -0.6598f, 0.5005f, 0.4203f, 1.3911f, 1.5988f, 0.3991f, 1.4931f,
                          0.9741f, 0.3557f, 0.1088f, -1.1806f, 1.1115f, -1.3283f, 1.7235f, 0.4177f,
                          0.7992f, -1.7248f, -0.5339f, -0.3153f, 0.1379f, 0.7493f, 0.3028f, -0.9473f};
  std::vector<int64_t> X_shape = {1, 2, 4, 4, 4};
  std::vector<float> W = {-0.1093f, -0.0511f, 0.1132f, 0.3369f, -0.3531f, -0.1766f, 0.0628f, 0.2118f,
                          0.3068f, 0.3217f, -0.2903f, -0.1633f, -0.3261f, -0.0990f, 0.2497f, -0.1553f};
  std::vector<int64_t> W_shape = {2, 1, 2, 2, 2};
  std::vector<float> dX = {0.2118f, 0.2746f, 0.2746f, 0.0628f, 0.0352f, -0.2550f, -0.2550f, -0.2902f,
                           0.0352f, -0.2550f, -0.2550f, -0.2902f, -0.1766f, -0.5297f, -0.5297f, -0.3531f,
                           0.5487f, 0.7247f, 0.7247f, 0.1760f, 0.3210f, 0.0346f, 0.0346f, -0.2864f,
                           0.3210f, 0.0346f, 0.0346f, -0.2864f, -0.2277f, -0.6901f, -0.6901f, -0.4624f,
                           0.5487f, 0.7247f, 0.7247f, 0.1760f, 0.3210f, 0.0346f, 0.0346f, -0.2864f,
                           0.3210f, 0.0346f, 0.0346f, -0.2864f, -0.2277f, -0.6901f, -0.6901f, -0.4624f,
                           0.3369f, 0.4501f, 0.4501f, 0.1132f, 0.2858f, 0.2897f, 0.2897f, 0.0038f,
                           0.2858f, 0.2897f, 0.2897f, 0.0038f, -0.0511f, -0.1604f, -0.1604f, -0.1093f,
                           -0.1553f, 0.0944f, 0.0944f, 0.2497f, -0.2542f, -0.3307f, -0.3307f, -0.0765f,
                           -0.2542f, -0.3307f, -0.3307f, -0.0765f, -0.0990f, -0.4251f, -0.4251f, -0.3261f,
                           -0.3185f, -0.3592f, -0.3592f, -0.0407f, -0.0958f, -0.1557f, -0.1557f, -0.0600f,
                           -0.0958f, -0.1557f, -0.1557f, -0.0600f, 0.2227f, 0.2035f, 0.2035f, -0.0193f,
                           -0.3185f, -0.3592f, -0.3592f, -0.0407f, -0.0958f, -0.1557f, -0.1557f, -0.0600f,
                           -0.0958f, -0.1557f, -0.1557f, -0.0600f, 0.2227f, 0.2035f, 0.2035f, -0.0193f,
                           -0.1633f, -0.4536f, -0.4536f, -0.2903f, 0.1584f, 0.1749f, 0.1749f, 0.0165f,
                           0.1584f, 0.1749f, 0.1749f, 0.0165f, 0.3217f, 0.6285f, 0.6285f, 0.3068f};
  std::vector<int64_t> dX_shape = X_shape;
  std::vector<float> dW = {-2.3068f, -2.1096f, -0.4322f, 0.4820f, 1.5420f, -4.1569f, -4.9628f, -5.5716f,
                           1.0492f, 1.6683f, -6.3262f, -3.2359f, 2.4532f, -2.3299f, -5.1917f, -9.2525f};
  std::vector<int64_t> dW_shape = W_shape;
  std::vector<float> dB = {125.f, 125.f};
  std::vector<int64_t> dB_shape = {2};

  for (const bool is_half : {false, true})
    TestConvTransposeGradOp(
        attrs,                           // attributes
        {dY, X, W},                      // inputs
        {dY_shape, X_shape, W_shape},    // input shapes
        {dX, dW, dB},                    // outputs
        {dX_shape, dW_shape, dB_shape},  // output shapes
        is_half);
}
#endif  // USE_CUDA

}  // namespace onnxruntime::contrib::test
