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
                    int64_t axis = 1,
                    const std::unordered_set<std::string>& excluded_providers = {},
                    OpTester::ExpectResult expect_result = OpTester::ExpectResult::kExpectSuccess,
                    const std::string& error_msg = "",
                    int opset = 7) {
  OpTester test("Softmax", opset);

  if (axis != 1) {
    test.AddAttribute("axis", axis);
  }

  test.AddInput<float>("X", dimensions, x_vals);
  test.AddOutput<float>("Y", dimensions, expected_vals);
  test.Run(expect_result, error_msg, excluded_providers);
}

TEST(SoftmaxOperator, Simple) {
  // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Softmax
  //    x = np.array([[-1, 0, 1]]).astype(np.float32)
  //    y = np.exp(x) / np.sum(np.exp(x), axis = 1) #expected output[[0.09003058, 0.24472848, 0.66524094]]

  std::vector<float> x_vals = {-1.0f, 0.0f, 1.0f};
  std::vector<float> expected_vals = {0.09003058f, 0.24472848f, 0.66524094f};
  std::vector<int64_t> dimensions = {1, 3};

  RunTest(x_vals, expected_vals, dimensions);
}

TEST(SoftmaxOperator, LargeNumber) {
  // x = np.array([[0, 1, 2, 3], [10000, 10001, 10002, 10003]]).astype(np.float32)
  // expected output[[0.0320586, 0.08714432, 0.23688284, 0.64391428],
  //                 [0.0320586, 0.08714432, 0.23688284, 0.64391428]]

  std::vector<float> x_vals = {0.0f, 1.0f, 2.0f, 3.0f, 10000.0f, 10001.0f, 10002.0f, 10003.0f};
  std::vector<float> expected_vals = {0.0320586f, 0.08714432f, 0.23688284f, 0.64391428f, 0.0320586f, 0.08714432f, 0.23688284f, 0.64391428f};
  std::vector<int64_t> dimensions = {2, 4};

  RunTest(x_vals, expected_vals, dimensions);
}

//np.random.seed(123)   # Use a seed so we can replicate the input and expected values here and in python
//x = np.abs(np.random.randn(3, 4, 5).astype(np.float32))
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

TEST(SoftmaxOperator, ThreeDimsAxis0) {
  // x = <see x_vals_3dims>
  // node = onnx.helper.make_node('Softmax', inputs = ['x'], outputs = ['y'], axis = 0)
  // y = softmax_2d(x.reshape(1, 60)).reshape(3, 4, 5)
  // expect(node, inputs = [x], outputs = [y],
  //        name = 'test_softmax_axis_0')

  std::vector<float> expected_vals = {
      0.01424391f, 0.013040296f, 0.0063832495f, 0.021693084f, 0.0085788425f,
      0.02508162f, 0.054455176f, 0.007386185f, 0.017058268f, 0.011443698f,
      0.009483798f, 0.0052878284f, 0.021372143f, 0.009112078f, 0.0074983323f,
      0.0074264654f, 0.04366858f, 0.04284054f, 0.01312807f, 0.007077248f,

      0.010054973f, 0.021358095f, 0.01226234f, 0.015588412f, 0.016853856f,
      0.009101601f, 0.01191507f, 0.020073075f, 0.0055332067f, 0.0113867875f,
      0.0062109763f, 0.07898735f, 0.028282179f, 0.009684978f, 0.012160114f,
      0.005722092f, 0.004823717f, 0.009572758f, 0.011591072f, 0.006387392f,

      0.010762473f, 0.027068434f, 0.007110686f, 0.00853781f, 0.0067482805f,
      0.004867251f, 0.0526183f, 0.007268943f, 0.0127998665f, 0.045098193f,
      0.017545262f, 0.0135920765f, 0.027506188f, 0.010684152f, 0.0049549243f,
      0.01401341f, 0.011721271f, 0.027815264f, 0.021463264f, 0.014014485f};

  RunTest(x_vals_3dims, expected_vals, three_dimensions, /*axis*/ 0, {kTensorrtExecutionProvider});  // Axis=0 is not supported by TensorRT
}

TEST(SoftmaxOperator, ThreeDimsAxis1) {
  // x = <see x_vals_3dims>
  // node = onnx.helper.make_node('Softmax', inputs = ['x'], outputs = ['y'], axis = 1)
  // y = softmax_2d(x.reshape(3, 20)).reshape(3, 4, 5)
  // expect(node, inputs = [x], outputs = [y],
  //        name = 'test_softmax_axis_1')

  std::vector<float> expected_vals = {
      0.04113652f, 0.037660476f, 0.018434875f, 0.0626498f, 0.024775764f,
      0.072435915f, 0.15726697f, 0.021331362f, 0.049264412f, 0.03304949f,
      0.027389284f, 0.015271291f, 0.061722923f, 0.026315752f, 0.021655245f,
      0.021447688f, 0.1261152f, 0.12372383f, 0.03791397f, 0.020439148f,

      0.032693777f, 0.069445916f, 0.039871037f, 0.05068577f, 0.054800365f,
      0.029593885f, 0.03874189f, 0.06526767f, 0.01799124f, 0.037024178f,
      0.02019501f, 0.25682762f, 0.091959596f, 0.031490736f, 0.03953865f,
      0.018605402f, 0.01568433f, 0.031125855f, 0.037688408f, 0.020768626f,

      0.031088287f, 0.0781894f, 0.020539802f, 0.024662167f, 0.019492965f,
      0.014059456f, 0.15199229f, 0.020996941f, 0.036973465f, 0.13026986f,
      0.050680935f, 0.03926183f, 0.079453886f, 0.030862054f, 0.014312706f,
      0.040478885f, 0.033857856f, 0.080346674f, 0.06199841f, 0.040481992f};

  RunTest(x_vals_3dims, expected_vals, three_dimensions, /*axis*/ 1, {kTensorrtExecutionProvider});
}

TEST(SoftmaxOperator, ThreeDimsAxis2) {
  // x = <see x_vals_3dims>
  // node = onnx.helper.make_node('Softmax', inputs = ['x'], outputs = ['y'], axis = 2)
  // y = softmax_2d(x.reshape(12, 5)).reshape(3, 4, 5)
  // expect(node, inputs = [x], outputs = [y],
  //       name = 'test_softmax_axis_2')

  std::vector<float> expected_vals = {
      0.22277209f, 0.20394778f, 0.09983283f, 0.33927578f, 0.13417149f,
      0.21729809f, 0.47177994f, 0.06399124f, 0.14778666f, 0.099144064f,
      0.1797734f, 0.10023525f, 0.40512702f, 0.17272712f, 0.14213723f,
      0.06506401f, 0.3825848f, 0.37533033f, 0.11501635f, 0.062004484f,

      0.13209775f, 0.28059313f, 0.16109712f, 0.2047936f, 0.22141843f,
      0.1568978f, 0.20539774f, 0.3460294f, 0.0953841f, 0.19629094f,
      0.045896534f, 0.5836837f, 0.20899355f, 0.07156797f, 0.08985819f,
      0.15019783f, 0.1266166f, 0.2512731f, 0.30425128f, 0.16766116f,

      0.17869644f, 0.44943509f, 0.11806339f, 0.1417589f, 0.112046175f,
      0.03968324f, 0.42900288f, 0.059264507f, 0.10435873f, 0.36769062f,
      0.23619612f, 0.1829779f, 0.37029108f, 0.14383113f, 0.0667037f,
      0.15740506f, 0.13165872f, 0.31243387f, 0.24108529f, 0.15741715f};

  RunTest(x_vals_3dims, expected_vals, three_dimensions, /*axis*/ 2);
}

TEST(SoftmaxOperator, ThreeDimsNegativeAxis) {
  // x = <see x_vals_3dims>
  // node = onnx.helper.make_node('Softmax', inputs = ['x'], outputs = ['y'], axis = 2)
  // y = softmax_2d(x.reshape(12, 5)).reshape(3, 4, 5)
  // expect(node, inputs = [x], outputs = [y],
  //       name = 'test_softmax_axis_2')

  std::vector<float> expected_vals = {
      0.22277209f, 0.20394778f, 0.09983283f, 0.33927578f, 0.13417149f,
      0.21729809f, 0.47177994f, 0.06399124f, 0.14778666f, 0.099144064f,
      0.1797734f, 0.10023525f, 0.40512702f, 0.17272712f, 0.14213723f,
      0.06506401f, 0.3825848f, 0.37533033f, 0.11501635f, 0.062004484f,

      0.13209775f, 0.28059313f, 0.16109712f, 0.2047936f, 0.22141843f,
      0.1568978f, 0.20539774f, 0.3460294f, 0.0953841f, 0.19629094f,
      0.045896534f, 0.5836837f, 0.20899355f, 0.07156797f, 0.08985819f,
      0.15019783f, 0.1266166f, 0.2512731f, 0.30425128f, 0.16766116f,

      0.17869644f, 0.44943509f, 0.11806339f, 0.1417589f, 0.112046175f,
      0.03968324f, 0.42900288f, 0.059264507f, 0.10435873f, 0.36769062f,
      0.23619612f, 0.1829779f, 0.37029108f, 0.14383113f, 0.0667037f,
      0.15740506f, 0.13165872f, 0.31243387f, 0.24108529f, 0.15741715f};

  // -1 is last axis so same as axis == 2
  RunTest(x_vals_3dims, expected_vals, three_dimensions, /*axis*/ -1);
}

TEST(SoftmaxOperator, InvalidAxis) {
  std::vector<float> x_vals = {-1.0f, 0.0f, 1.0f};
  std::vector<float> expected_vals = {0.0f, 0.0f, 0.0f};
  std::vector<int64_t> dimensions = {1, 3};

  RunTest(x_vals,
          expected_vals,
          dimensions,
          /* invalid axis */ -10, {kTensorrtExecutionProvider},
          OpTester::ExpectResult::kExpectFailure,
          // bug in ONNX error message currently. Message should be
          // "[ShapeInferenceError] 'axis' must be in [-2 , 1]. Its actual value is: -10"
          ", 1]. Its actual value is: -10",
          // latest opset so we get shape inferencing errors
          -1);
}

TEST(SoftmaxOperator, DimWithZero) {
  std::vector<float> x_vals = {};
  std::vector<float> expected_vals = {};
  std::vector<int64_t> dimensions = {1, 0};  // dim with value of 0 should be handled

  RunTest(x_vals, expected_vals, dimensions, 0,
          {kTensorrtExecutionProvider,
           kNnapiExecutionProvider},  // NNAPI softmax does not support empty input
          OpTester::ExpectResult::kExpectSuccess, "", 10);
}

}  // namespace test
}  // namespace onnxruntime
