// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "core/session/environment.h"
#include "test/providers/provider_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/common/dnnl_op_test_utils.h"

namespace onnxruntime {
namespace test {

static void RunTest(const std::vector<float>& x_vals,
                    const std::vector<float>& expected_vals,
                    const std::vector<int64_t>& dimensions,
                    int opset = 7,
                    int64_t axis = 1,
                    const std::unordered_set<std::string>& excluded_providers = {},
                    OpTester::ExpectResult expect_result = OpTester::ExpectResult::kExpectSuccess,
                    const std::string& error_msg = "") {
  OpTester test("Softmax", opset);

  if (opset < 13) {
    if (axis != 1) {  // opset-12 and below : default axis value is 1
      test.AddAttribute("axis", axis);
    }
  } else {
    if (axis != -1) {  // opset-13 : default axis value is -1
      test.AddAttribute("axis", axis);
    }
  }
  test.AddInput<float>("X", dimensions, x_vals);
  test.AddOutput<float>("Y", dimensions, expected_vals);
  test.Run(expect_result, error_msg, excluded_providers);
}

TEST(SoftmaxOperator, Simple) {
  // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Softmax
  //    x = np.array([[-1, 0, 1]]).astype(np.float32)
  //    y = np.exp(x) / np.sum(np.exp(x), axis = 1) #expected output[[0.09003058, 0.24472848, 0.66524094]]

  std::vector<float> x_vals = {-1.0f, 0.0f, 1.0f};
  std::vector<float> expected_vals = {0.09003058f, 0.24472848f, 0.66524094f};
  std::vector<int64_t> dimensions = {1, 3};

  RunTest(x_vals, expected_vals, dimensions);
}

#if defined(USE_CUDA) || defined(USE_ROCM)
TEST(SoftmaxOperator, Simple_fp16) {
#ifdef USE_CUDA
  int min_cuda_architecture = 530;
  if (!HasCudaEnvironment(min_cuda_architecture)) {
    LOGS_DEFAULT(WARNING) << "Hardware NOT support FP16";
    return;
  }
#endif
  OpTester test("Softmax", 14);

  int64_t axis = 1;
  test.AddAttribute("axis", axis);

  std::vector<float> X = {-1.0f, 0.0f, 1.0f};
  std::vector<float> Y = {0.09003058f, 0.24472848f, 0.66524094f};
  std::vector<int64_t> dimensions = {1, 3};

  std::vector<MLFloat16> f_X(3);
  std::vector<MLFloat16> f_Y(3);
  ConvertFloatToMLFloat16(X.data(), f_X.data(), 3);
  ConvertFloatToMLFloat16(Y.data(), f_Y.data(), 3);

  test.AddInput<MLFloat16>("X", dimensions, f_X);
  test.AddOutput<MLFloat16>("Y", dimensions, f_Y);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}
#endif

#if defined(USE_CUDA) || defined(USE_ROCM) || defined(USE_DNNL)
TEST(SoftmaxOperator, Simple_bfloat16) {
#ifdef USE_CUDA
  int min_cuda_architecture = 530;
  if (!HasCudaEnvironment(min_cuda_architecture)) {
    LOGS_DEFAULT(WARNING) << "Hardware NOT support BFP16";
    return;
  }
#endif
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
  OpTester test("Softmax", 14);

  int64_t axis = 1;
  test.AddAttribute("axis", axis);

  test.AddInput<BFloat16>("X", {1, 3}, MakeBFloat16({-1.0f, 0.0f, 1.0f}));
  test.AddOutput<BFloat16>("Y", {1, 3}, MakeBFloat16({0.09003058f, 0.24472848f, 0.66524094f}));
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#ifdef USE_CUDA
  execution_providers.push_back(DefaultCudaExecutionProvider());
#elif USE_ROCM
  execution_providers.push_back(DefaultRocmExecutionProvider());
#elif USE_DNNL
  execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}
#endif  //  USE_CUDA USE_ROCM USE_DNNL

TEST(SoftmaxOperator, LargeNumber) {
  // x = np.array([[0, 1, 2, 3], [10000, 10001, 10002, 10003]]).astype(np.float32)
  // expected output[[0.0320586, 0.08714432, 0.23688284, 0.64391428],
  //                 [0.0320586, 0.08714432, 0.23688284, 0.64391428]]

  std::vector<float> x_vals = {0.0f, 1.0f, 2.0f, 3.0f, 10000.0f, 10001.0f, 10002.0f, 10003.0f};
  std::vector<float> expected_vals = {0.0320586f, 0.08714432f, 0.23688284f, 0.64391428f, 0.0320586f, 0.08714432f, 0.23688284f, 0.64391428f};
  std::vector<int64_t> dimensions = {2, 4};

  RunTest(x_vals, expected_vals, dimensions);
}

// np.random.seed(123)   # Use a seed so we can replicate the input and expected values here and in python
// # create 60 input values
// x = np.abs(np.random.randn(3, 4, 5).astype(np.float32))
static const std::vector<int64_t> three_dimensions = {3, 4, 5};
static const std::vector<int64_t> four_dimensions = {1, 3, 4, 5};
static const std::vector<float> input_vals_60 = {
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

TEST(SoftmaxOperator, ThreeAndFourDimsAxis0) {
  // x = <see input_vals_60>
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

  RunTest(input_vals_60, expected_vals, three_dimensions, /*opset*/ 7, /*axis*/ 0,
          // axis=0 is not supported by TensorRT
          {kTensorrtExecutionProvider, kOpenVINOExecutionProvider, kDnnlExecutionProvider, kCoreMLExecutionProvider});

  RunTest(input_vals_60, expected_vals, four_dimensions, /*opset*/ 7, /*axis*/ 0,
          // axis=0 is not supported by TensorRT
          {kTensorrtExecutionProvider, kOpenVINOExecutionProvider, kDnnlExecutionProvider, kCoreMLExecutionProvider});
}

TEST(SoftmaxOperator, ThreeAndFourDimsSecondLastAxis) {
  // x = <see input_vals_60>
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

  RunTest(input_vals_60, expected_vals, three_dimensions, /*opset*/ 7, /*axis*/ 1,
          {kTensorrtExecutionProvider, kOpenVINOExecutionProvider, kDnnlExecutionProvider, kCoreMLExecutionProvider});

  RunTest(input_vals_60, expected_vals, four_dimensions, /*opset*/ 7, /*axis*/ 2,
          {kTensorrtExecutionProvider, kOpenVINOExecutionProvider, kDnnlExecutionProvider, kCoreMLExecutionProvider});
}

TEST(SoftmaxOperator, ThreeAndFourDimsSecondLastAxis_opset13) {
  // For the same input, opset-13's behavior is different from an earlier opset
  // and we see different expected results for the same test input

  std::vector<float> expected_vals = {
      0.253289f, 0.11198013f, 0.08185529f, 0.35567388f, 0.24795689f,
      0.44600812f, 0.46761957f, 0.09471639f, 0.2796827f, 0.3307607f,
      0.16864346f, 0.04540785f, 0.27406466f, 0.14939913f, 0.2167266f,
      0.1320594f, 0.3749925f, 0.5493636f, 0.21524426f, 0.20455585f,

      0.32341874f, 0.18241648f, 0.1747012f, 0.36767146f, 0.36021632f,
      0.29275346f, 0.10176494f, 0.28598055f, 0.13050734f, 0.24336906f,
      0.19977638f, 0.67461985f, 0.40293545f, 0.22843185f, 0.25989732f,
      0.18405138f, 0.04119869f, 0.13638285f, 0.27338937f, 0.13651732f,

      0.22807457f, 0.2577944f, 0.10201685f, 0.15962972f, 0.09529332f,
      0.10314508f, 0.5011263f, 0.10428739f, 0.23931651f, 0.63683724f,
      0.37181312f, 0.12944824f, 0.3946307f, 0.19975942f, 0.0699691f,
      0.29696727f, 0.11163106f, 0.39906505f, 0.4012943f, 0.1979003f};

  RunTest(input_vals_60, expected_vals, three_dimensions, /*opset*/ 13, /*axis*/ 1,
          {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});  // OpenVINO doesn't support opset-13 yet

  RunTest(input_vals_60, expected_vals, four_dimensions, /*opset*/ 13, /*axis*/ 2,
          {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});  // OpenVINO doesn't support opset-13 yet
}

TEST(SoftmaxOperator, ThreeAndFourDimsLastAxis) {
  // x = <see input_vals_60>
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

  RunTest(input_vals_60, expected_vals, three_dimensions, /*opset*/ 7, /*axis*/ 2);
  RunTest(input_vals_60, expected_vals, four_dimensions, /*opset*/ 7, /*axis*/ 3);
}

TEST(SoftmaxOperator, ThreeAndFourDimsLastAxis_opset13) {
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

  RunTest(input_vals_60, expected_vals, three_dimensions, /*opset*/ 13, /*axis*/ 2,
          {kOpenVINOExecutionProvider});  // OpenVINO doesn't support opset-13 yet

  RunTest(input_vals_60, expected_vals, four_dimensions, /*opset*/ 13, /*axis*/ 3,
          {kOpenVINOExecutionProvider});  // OpenVINO doesn't support opset-13 yet
}

TEST(SoftmaxOperator, ThreeAndFourDimsDefaultAxis_opset13) {
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

  RunTest(input_vals_60, expected_vals, three_dimensions, /*opset*/ 13, /*default axis*/ -1,
          {kOpenVINOExecutionProvider});  // OpenVINO doesn't support opset-13 yet

  RunTest(input_vals_60, expected_vals, four_dimensions, /*opset*/ 13, /*default axis*/ -1,
          {kOpenVINOExecutionProvider});  // OpenVINO doesn't support opset-13 yet
}

TEST(SoftmaxOperator, ThreeAndFourDimsNegativeAxis) {
  // x = <see input_vals_60>
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
  RunTest(input_vals_60, expected_vals, three_dimensions, /*opset*/ 12, /*axis*/ -1);
  RunTest(input_vals_60, expected_vals, four_dimensions, /*opset*/ 12, /*axis*/ -1);
}

TEST(SoftmaxOperator, InvalidAxis) {
  std::vector<float> x_vals = {-1.0f, 0.0f, 1.0f};
  std::vector<float> expected_vals = {0.0f, 0.0f, 0.0f};
  std::vector<int64_t> dimensions = {1, 3};

  RunTest(x_vals,
          expected_vals,
          dimensions,
          /*opset*/ 12,
          /* invalid axis */ -10, {kTensorrtExecutionProvider},
          OpTester::ExpectResult::kExpectFailure,
          // bug in ONNX error message currently. Message should be
          // "[ShapeInferenceError] 'axis' must be in [-2 , 1]. Its actual value is: -10"
          ", 1]. Its actual value is: -10");
}

TEST(SoftmaxOperator, InvalidAxis_opset13) {
  std::vector<float> x_vals = {-1.0f, 0.0f, 1.0f};
  std::vector<float> expected_vals = {0.0f, 0.0f, 0.0f};
  std::vector<int64_t> dimensions = {1, 3};

  RunTest(x_vals,
          expected_vals,
          dimensions,
          /*opset*/ -1,  // latest opset so we get shape inferencing errors
          /* invalid axis */ -10, {kTensorrtExecutionProvider, kOpenVINOExecutionProvider},
          OpTester::ExpectResult::kExpectFailure,
          // In opset-13, Softmax is composed as afunction of several other ops,
          // and hence it breaks differently to the test above but the most important thing
          // is that it breaks and this is the right behavior
          "[ShapeInferenceError]");
}
TEST(SoftmaxOperator, DimWithZero) {
  std::vector<float> x_vals = {};
  std::vector<float> expected_vals = {};
  std::vector<int64_t> dimensions = {1, 0};  // dim with value of 0 should be handled

  RunTest(x_vals, expected_vals, dimensions, /*opset*/ -1, /*axis*/ 0,
          {kTensorrtExecutionProvider,
           kNnapiExecutionProvider,  // NNAPI softmax does not support empty input
           kQnnExecutionProvider}    // QNN doesn't support dim 0
  );
}

TEST(SoftmaxOperator, 2DInputReduceOnAxis1WithLargeDim) {
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
    expected_vals[i] = expected_vals[i] / sum;
  }

  std::vector<int64_t> dimensions = {1, 1025};

  RunTest(x_vals, expected_vals, dimensions);
}

// Regression test for NNAPI handling of a Softmax with opset < 13 where the input has been converted to NHWC.
// The NNAPI handling of the axis is different so we need to manually coerce the input to 2D, which will negate the
// layout change. Test model has a GlobalAveragePool -> Softmax which will trigger the layout change due to
// GlobalAveragePool being layout sensitive.
TEST(SoftmaxOperator, GH15949_regression_test) {
  auto model_uri = ORT_TSTR("testdata/ort_github_issue_15949.onnx");
  ModelTester tester("Opset12NhwcSoftmax", model_uri);

  tester.AddInput<float>("X", {1, 3, 2, 2},
                         {0.f, 1.f, 2.f, 3.f,
                          4.f, 5.f, 6.f, 7.f,
                          8.f, 9.f, 10.f, 11.f});
  tester.AddOutput<float>("Y", {1, 3, 1, 1},
                          {0.00032932f, 0.01798029f, 0.9816904f});

  // disable TRT as it does not support axis=0 as used by the model
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

}  // namespace test
}  // namespace onnxruntime
