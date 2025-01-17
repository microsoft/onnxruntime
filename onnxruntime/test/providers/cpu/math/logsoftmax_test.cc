// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

static void RunTest(const std::vector<float>& x_vals,
                    const std::vector<float>& expected_vals,
                    const std::vector<int64_t>& dimensions,
                    int opset = 7,
                    int64_t axis = 1,
                    bool is_tensorrt_supported = true,
                    OpTester::ExpectResult expect_result = OpTester::ExpectResult::kExpectSuccess,
                    const std::string& error_msg = "",
                    float tolerance = 0.0f) {
  OpTester tester("LogSoftmax", opset);

  if (opset < 13) {
    if (axis != 1) {  // opset-12 and below : default axis value is 1
      tester.AddAttribute("axis", axis);
    }
  } else {
    if (axis != -1) {  // opset-13 : default axis value is -1
      tester.AddAttribute("axis", axis);
    }
  }

  tester.AddInput("X", dimensions, x_vals);
  tester.AddOutput("Y", dimensions, expected_vals);

  if (tolerance != 0.0f) {
    tester.SetOutputAbsErr("Y", tolerance);
  }

  std::unordered_set<std::string> excluded_providers;
  if (!is_tensorrt_supported) {
    excluded_providers.insert(kTensorrtExecutionProvider);
  }
  tester.Run(expect_result, error_msg, excluded_providers);
}

TEST(LogSoftmaxOperator, Simple) {
  // https://github.com/onnx/onnx/blob/main/docs/Operators.md#LogSoftmax
  // x = np.array([[-1, 0, 1]]).astype(np.float32)
  // # expected output[[-2.40760589, -1.40760589, -0.40760589]]

  std::vector<float> x_vals = {-1.0f, 0.0f, 1.0f};
  std::vector<float> expected_vals = {-2.40760589f, -1.40760589f, -0.40760589f};
  std::vector<int64_t> dimensions = {1, 3};

  RunTest(x_vals, expected_vals, dimensions);
}

TEST(LogSoftmaxOperator, LargeNumber) {
  //   x = np.array([[0, 1, 2, 3],
  //                 [10000, 10001, 10002, 10003]]).astype(np.float32)
  // expected output[[-3.4401896, -2.4401896, -1.44018972, -0.44018969],
  //                 [-3.4401896, -2.4401896, -1.44018972, -0.44018969]]

  std::vector<float> x_vals = {0.0f, 1.0f, 2.0f, 3.0f,
                               10000.0f, 10001.0f, 10002.0f, 10003.0f};
  std::vector<float> expected_vals = {-3.4401896f, -2.4401896f, -1.44018972f, -0.44018969f,
                                      -3.4401896f, -2.4401896f, -1.44018972f, -0.44018969f};
  std::vector<int64_t> dimensions = {2, 4};

  RunTest(x_vals, expected_vals, dimensions, 7, 1, true, OpTester::ExpectResult::kExpectSuccess, "", 0.0005f);
}

// np.random.seed(123)   # Use a seed so we can replicate the input and expected values here and in python
// x = np.abs(np.random.randn(3, 4, 5).astype(np.float32))
static std::vector<int64_t> three_dimensions = {3, 4, 5};
static std::vector<float> x_vals_3dims = {
    1.0856307f, 0.99734545f, 0.2829785f, 1.5062947f, 0.5786002f,
    1.6514366f, 2.4266791f, 0.42891264f, 1.2659363f, 0.8667404f,
    0.6788862f, 0.09470897f, 1.4913896f, 0.638902f, 0.44398195f,
    0.43435127f, 2.20593f, 2.1867862f, 1.004054f, 0.3861864f,

    0.7373686f, 1.4907321f, 0.9358339f, 1.175829f, 1.2538806f,
    0.6377515f, 0.9071052f, 1.4286807f, 0.14006872f, 0.8617549f,
    0.25561938f, 2.798589f, 1.7715331f, 0.69987726f, 0.92746246f,
    0.17363568f, 0.002845916f, 0.6882227f, 0.87953633f, 0.28362733f,

    0.8053665f, 1.7276695f, 0.3908998f, 0.57380587f, 0.33858904f,
    0.011830495f, 2.3923652f, 0.41291216f, 0.978736f, 2.2381434f,
    1.2940853f, 1.0387882f, 1.7437122f, 0.79806274f, 0.02968323f,
    1.0693159f, 0.8907064f, 1.7548862f, 1.4956441f, 1.0693927f};

TEST(LogSoftmaxOperator, ThreeDimsAxis0) {
  // x = <see x_vals_3dims>
  // node = onnx.helper.make_node('LogSoftmax', inputs = ['x'], outputs = ['y'], axis = 0)
  // y = logsoftmax_2d(x.reshape(1, 60)).reshape(3, 4, 5)
  // expect(node, inputs = [x], outputs = [y],
  //        name = 'test_logsoftmax_axis_0')

  std::vector<float> expected_vals = {
      -4.2514257f, -4.339711f, -5.054078f, -3.8307617f, -4.758456f,
      -3.6856198f, -2.9103773f, -4.908144f, -4.0711203f, -4.470316f,
      -4.65817f, -5.2423477f, -3.845667f, -4.6981544f, -4.8930745f,
      -4.902705f, -3.1311264f, -3.1502702f, -4.3330026f, -4.95087f,

      -4.5996876f, -3.8463244f, -4.401222f, -4.161227f, -4.0831757f,
      -4.6993046f, -4.429951f, -3.9083757f, -5.1969876f, -4.4753017f,
      -5.081437f, -2.5384674f, -3.5655231f, -4.6371794f, -4.409594f,
      -5.1634207f, -5.3342104f, -4.6488338f, -4.45752f, -5.053429f,

      -4.5316896f, -3.609387f, -4.9461565f, -4.7632504f, -4.9984674f,
      -5.325226f, -2.9446912f, -4.9241443f, -4.35832f, -3.098913f,
      -4.042971f, -4.2982683f, -3.5933442f, -4.538994f, -5.307373f,
      -4.2677402f, -4.44635f, -3.5821702f, -3.8414123f, -4.267664f};

  RunTest(x_vals_3dims, expected_vals, three_dimensions, /*opset*/ 7, /*axis*/ 0, false);  // axis=0 is not supported by TensorRT
}

TEST(LogSoftmaxOperator, ThreeDimsAxis1) {
  // x = <see x_vals_3dims>
  // node = onnx.helper.make_node('LogSoftmax', inputs = ['x'], outputs = ['y'], axis = 1)
  // y = logsoftmax_2d(x.reshape(3, 20)).reshape(3, 4, 5)
  // expect(node, inputs = [x], outputs = [y],
  //        name = 'test_logsoftmax_axis_1')

  std::vector<float> expected_vals = {
      -3.1908588f, -3.2791443f, -3.9935112f, -2.770195f, -3.6978893f,
      -2.625053f, -1.8498105f, -3.847577f, -3.0105534f, -3.409749f,
      -3.5976033f, -4.181781f, -2.7851f, -3.6375875f, -3.8325076f,
      -3.8421383f, -2.0705595f, -2.0897036f, -3.2724357f, -3.8903031f,

      -3.4205704f, -2.667207f, -3.222105f, -2.98211f, -2.9040585f,
      -3.5201874f, -3.250834f, -2.7292585f, -4.0178704f, -3.296184f,
      -3.90232f, -1.3593501f, -2.386406f, -3.4580617f, -3.2304766f,
      -3.9843035f, -4.155093f, -3.4697165f, -3.2784028f, -3.874312f,

      -3.4709241f, -2.5486212f, -3.8853908f, -3.7024848f, -3.9377017f,
      -4.26446f, -1.8839254f, -3.8633785f, -3.2975547f, -2.0381472f,
      -2.9822054f, -3.2375026f, -2.5325785f, -3.4782279f, -4.246608f,
      -3.2069747f, -3.3855844f, -2.5214045f, -2.7806466f, -3.206898f};

  RunTest(x_vals_3dims, expected_vals, three_dimensions, /*opset*/ 7, /*axis*/ 1, false);  // This test failed on TensorRT
}

TEST(LogSoftmaxOperator, ThreeDimsAxis1_opset13) {
  // For the same input, opset-13's behavior is different from an earlier opset
  // and we see different expected results for the same test input

  std::vector<float> expected_vals = {
      -1.373224f, -2.1894338f, -2.5028024f, -1.0337411f, -1.3945004f,
      -0.8074181f, -0.7601002f, -2.3568683f, -1.2740996f, -1.1063602f,
      -1.7799685f, -3.0920703f, -1.2943912f, -1.9011338f, -1.5291187f,
      -2.0245035f, -0.9808493f, -0.5989947f, -1.5359819f, -1.5869142f,

      -1.1288074f, -1.7014627f, -1.7446783f, -1.0005655f, -1.0210506f,
      -1.2284245f, -2.2850897f, -1.2518314f, -2.036326f, -1.4131763f,
      -1.6105566f, -0.3936059f, -0.90897894f, -1.4765173f, -1.3474687f,
      -1.6925404f, -3.189349f, -1.9922893f, -1.2968583f, -1.9913039f,

      -1.4780827f, -1.355593f, -2.2826173f, -1.8348985f, -2.3507955f,
      -2.2716188f, -0.69089717f, -2.2606049f, -1.4299684f, -0.45124117f,
      -0.9893639f, -2.0444741f, -0.92980486f, -1.6106415f, -2.6597016f,
      -1.2141333f, -2.192556f, -0.9186309f, -0.9130602f, -1.6199919f};

  RunTest(x_vals_3dims, expected_vals, three_dimensions, /*opset*/ 13, /*axis*/ 1, false);  // This test failed on TensorRT
}

TEST(LogSoftmaxOperator, ThreeDimsAxis2) {
  std::vector<float> expected_vals = {
      -1.5016061f, -1.5898913f, -2.3042583f, -1.080942f, -2.0086365f,
      -1.5264852f, -0.7512426f, -2.7490091f, -1.9119854f, -2.3111813f,
      -1.716058f, -2.3002353f, -0.9035546f, -1.7560422f, -1.9509623f,
      -2.7323837f, -0.96080494f, -0.97994876f, -2.162681f, -2.7805486f,

      -2.024213f, -1.2708496f, -1.8257477f, -1.5857526f, -1.507701f,
      -1.8521607f, -1.582807f, -1.0612315f, -2.3498435f, -1.6281573f,
      -3.0813656f, -0.538396f, -1.5654519f, -2.6371078f, -2.4095225f,
      -1.8958019f, -2.0665917f, -1.3812149f, -1.1899012f, -1.7858102f,

      -1.7220669f, -0.79976386f, -2.1365335f, -1.9536276f, -2.1888442f,
      -3.2268262f, -0.84629166f, -2.8257446f, -2.259921f, -1.0005134f,
      -1.4430928f, -1.6983899f, -0.9934659f, -1.9391153f, -2.7074947f,
      -1.8489327f, -2.027542f, -1.1633625f, -1.4226046f, -1.848856f};

  RunTest(x_vals_3dims, expected_vals, three_dimensions, /*opset*/ 7, /*axis*/ 2);
}

TEST(LogSoftmaxOperator, ThreeDimsAxis2_opset13) {
  std::vector<float> expected_vals = {
      -1.5016061f, -1.5898913f, -2.3042583f, -1.080942f, -2.0086365f,
      -1.5264852f, -0.7512426f, -2.7490091f, -1.9119854f, -2.3111813f,
      -1.716058f, -2.3002353f, -0.9035546f, -1.7560422f, -1.9509623f,
      -2.7323837f, -0.96080494f, -0.97994876f, -2.162681f, -2.7805486f,

      -2.024213f, -1.2708496f, -1.8257477f, -1.5857526f, -1.507701f,
      -1.8521607f, -1.582807f, -1.0612315f, -2.3498435f, -1.6281573f,
      -3.0813656f, -0.538396f, -1.5654519f, -2.6371078f, -2.4095225f,
      -1.8958019f, -2.0665917f, -1.3812149f, -1.1899012f, -1.7858102f,

      -1.7220669f, -0.79976386f, -2.1365335f, -1.9536276f, -2.1888442f,
      -3.2268262f, -0.84629166f, -2.8257446f, -2.259921f, -1.0005134f,
      -1.4430928f, -1.6983899f, -0.9934659f, -1.9391153f, -2.7074947f,
      -1.8489327f, -2.027542f, -1.1633625f, -1.4226046f, -1.848856f};

  RunTest(x_vals_3dims, expected_vals, three_dimensions, /*opset*/ 13, /*axis*/ 2);
}

TEST(LogSoftmaxOperator, ThreeDimsDefaultAxis_opset13) {
  std::vector<float> expected_vals = {
      -1.5016061f, -1.5898913f, -2.3042583f, -1.080942f, -2.0086365f,
      -1.5264852f, -0.7512426f, -2.7490091f, -1.9119854f, -2.3111813f,
      -1.716058f, -2.3002353f, -0.9035546f, -1.7560422f, -1.9509623f,
      -2.7323837f, -0.96080494f, -0.97994876f, -2.162681f, -2.7805486f,

      -2.024213f, -1.2708496f, -1.8257477f, -1.5857526f, -1.507701f,
      -1.8521607f, -1.582807f, -1.0612315f, -2.3498435f, -1.6281573f,
      -3.0813656f, -0.538396f, -1.5654519f, -2.6371078f, -2.4095225f,
      -1.8958019f, -2.0665917f, -1.3812149f, -1.1899012f, -1.7858102f,

      -1.7220669f, -0.79976386f, -2.1365335f, -1.9536276f, -2.1888442f,
      -3.2268262f, -0.84629166f, -2.8257446f, -2.259921f, -1.0005134f,
      -1.4430928f, -1.6983899f, -0.9934659f, -1.9391153f, -2.7074947f,
      -1.8489327f, -2.027542f, -1.1633625f, -1.4226046f, -1.848856f};

  RunTest(x_vals_3dims, expected_vals, three_dimensions, /*opset*/ 13, /*default axis*/ -1);
}
TEST(LogSoftmaxOperator, ThreeDimsNegativeAxis) {
  // x = <see x_vals_3dims>
  // node = onnx.helper.make_node('LogSoftmax', inputs = ['x'], outputs = ['y'], axis = 2)
  // y = logsoftmax_2d(x.reshape(12, 5)).reshape(3, 4, 5)
  // expect(node, inputs = [x], outputs = [y],
  //       name = 'test_logsoftmax_axis_2')

  std::vector<float> expected_vals = {
      -1.5016061f, -1.5898913f, -2.3042583f, -1.080942f, -2.0086365f,
      -1.5264852f, -0.7512426f, -2.7490091f, -1.9119854f, -2.3111813f,
      -1.716058f, -2.3002353f, -0.9035546f, -1.7560422f, -1.9509623f,
      -2.7323837f, -0.96080494f, -0.97994876f, -2.162681f, -2.7805486f,

      -2.024213f, -1.2708496f, -1.8257477f, -1.5857526f, -1.507701f,
      -1.8521607f, -1.582807f, -1.0612315f, -2.3498435f, -1.6281573f,
      -3.0813656f, -0.538396f, -1.5654519f, -2.6371078f, -2.4095225f,
      -1.8958019f, -2.0665917f, -1.3812149f, -1.1899012f, -1.7858102f,

      -1.7220669f, -0.79976386f, -2.1365335f, -1.9536276f, -2.1888442f,
      -3.2268262f, -0.84629166f, -2.8257446f, -2.259921f, -1.0005134f,
      -1.4430928f, -1.6983899f, -0.9934659f, -1.9391153f, -2.7074947f,
      -1.8489327f, -2.027542f, -1.1633625f, -1.4226046f, -1.848856f};

  // -1 is last axis so same as axis == 2
  RunTest(x_vals_3dims, expected_vals, three_dimensions, /*opset*/ 12, /*axis*/ -1);
}

TEST(LogSoftmaxOperator, InvalidAxis) {
  std::vector<float> x_vals = {-1.0f, 0.0f, 1.0f};
  std::vector<float> expected_vals = {0.0f, 0.0f, 0.0f};
  std::vector<int64_t> dimensions = {1, 3};

  RunTest(x_vals,
          expected_vals,
          dimensions,
          /*opset*/ 12,
          /* invalid axis */ -7,
          false,  // TensorRT parser: Assertion failed: axis >= 0 && axis < nbDims
          OpTester::ExpectResult::kExpectFailure,
          // ONNX has a bug in the error message generation so this is somewhat cryptic until it's fixed. Message should be:
          "[ShapeInferenceError] 'axis' must be in [-2 , 1]. Its actual value is: -7");
  //", 1]. Its actual value is: -7");
}

TEST(LogSoftmaxOperator, 2DInputReduceOnAxis1WithLargeDim) {
  std::vector<float> x_vals(1025, 0.0f);
  std::vector<float> expected_vals(1025, 0.0f);
  float incre_val = 0.01f;
  for (size_t i = 0; i < x_vals.size(); ++i) {
    x_vals[i] = incre_val;
    incre_val += 0.01f;
  }

  float sum = 0.0f;
  for (size_t i = 0; i < x_vals.size(); ++i) {
    expected_vals[i] = std::exp(x_vals[i]);
    sum += expected_vals[i];
  }

  for (size_t i = 0; i < x_vals.size(); ++i) {
    expected_vals[i] = std::log(expected_vals[i] / sum);
  }

  std::vector<int64_t> dimensions = {1, 1025};

  RunTest(x_vals, expected_vals, dimensions);
}

}  // namespace test
}  // namespace onnxruntime
