// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/mlas/inc/mlas.h"

#ifdef MLAS_F16VEC_INTRINSICS_SUPPORTED

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
using namespace std;
namespace onnxruntime {
namespace test {

namespace {

struct ConvOpAndTestAttributes {
  string auto_pad;
  vector<int64_t> dilations;
  int64_t group;
  vector<int64_t> kernel_shape;
  vector<int64_t> pads;
  vector<int64_t> strides;
  std::unordered_set<std::string> excluded_providers;
};

void TestConvFp16Op(const ConvOpAndTestAttributes& attributes,
                const vector<vector<MLFloat16>>& inputs,
                const vector<vector<int64_t>>& input_shapes,
                const std::initializer_list<MLFloat16>& expected_output,
                const vector<int64_t>& expected_output_shape,
                bool weight_is_initializer = false,
                OpTester::ExpectResult expect_result = OpTester::ExpectResult::kExpectSuccess,
                const std::string& err_str = "",
                int opset = 11) {
  OpTester test("Conv", opset);
  test.AddAttribute("group", attributes.group);
  test.AddAttribute("kernel_shape", attributes.kernel_shape);

  if (!attributes.dilations.empty()) {
    test.AddAttribute("dilations", attributes.dilations);
  }

  // Only one of pads / auto_pad can be present
  if (!attributes.pads.empty()) {
    test.AddAttribute("pads", attributes.pads);
  } else {
    test.AddAttribute("auto_pad", attributes.auto_pad);
  }

  if (!attributes.strides.empty()) {
    test.AddAttribute("strides", attributes.strides);
  }

  ORT_ENFORCE(inputs.size() <= 3, "Our name array is only setup to handle 3 inputs");
  const char* szNames[] = {"X", "W", "B"};
  test.AddInput<MLFloat16>(szNames[0], input_shapes[0], inputs[0]);
  test.AddInput<MLFloat16>(szNames[1], input_shapes[1], inputs[1], weight_is_initializer);
  if (inputs.size() == 3)
    test.AddInput<MLFloat16>(szNames[2], input_shapes[2], inputs[2]);

  test.AddOutput<MLFloat16>("Y", expected_output_shape, expected_output, /*no sort*/ false, 0.002f, 0.0f);

  std::unordered_set<std::string> excluded_providers(attributes.excluded_providers);
  // Disable TensorRT because weight as input is not supported
  excluded_providers.insert(kTensorrtExecutionProvider);
  // QNN have issue with dynamic weight, auto pad with SAME_UPPER, SAME_LOWER
  if (!weight_is_initializer || attributes.auto_pad == "SAME_UPPER" || attributes.auto_pad == "SAME_LOWER") {
    excluded_providers.insert(kQnnExecutionProvider);
  }

  test.Run(expect_result, err_str, excluded_providers);
}

}  // namespace


TEST(ConvFp16Test, Conv1D_Invalid_Input_Shape) {
  ConvOpAndTestAttributes attrs = {
      "",                     // auto_pad
      vector<int64_t>{1},     // dilations
      1,                      // group
      vector<int64_t>{2},     // kernel_shape
      vector<int64_t>{0, 0},  // pads
      vector<int64_t>{1},     // strides
      {}                      // excluded EPs
  };

  vector<MLFloat16> X = vector<MLFloat16>(1, MLFloat16(1.0f));
  vector<int64_t> X_shape = {1, 1, 1};
  vector<int64_t> dummy_shape = {1, 1, 2};
  auto dummy_vals = {MLFloat16(0.0f), MLFloat16(0.0f)};
  TestConvFp16Op(attrs, {X, dummy_vals}, {X_shape, dummy_shape}, dummy_vals, dummy_shape, false,
             OpTester::ExpectResult::kExpectFailure,
             "Node:node1 Output:Y [ShapeInferenceError] Can't merge shape info. "
             "Both source and target dimension have values but they differ. Source=0 Target=2 Dimension=2",
             -1);  // use latest opset for shape inferencing errors
}

TEST(ConvFp16Test, Conv2D_Invalid_Input_Shape) {
  ConvOpAndTestAttributes attrs = {
      "",                           // auto_pad
      vector<int64_t>{1, 1},        // dilations
      1,                            // group
      vector<int64_t>{3, 3},        // kernel_shape
      vector<int64_t>{0, 0, 0, 0},  // pads
      vector<int64_t>{1, 1},        // strides
      {}                            // excluded EPs
  };

  vector<MLFloat16> X = vector<MLFloat16>(1 * 3 * 1 * 111, MLFloat16(1.0f));
  vector<int64_t> X_shape = {1, 3, 1, 111};
  vector<int64_t> dummy_shape = {2, 2, 1, 2};
  auto dummy_vals = {MLFloat16(-0.0f), MLFloat16(0.0f), MLFloat16(-0.0f), MLFloat16(-0.0f),
                     MLFloat16(-0.0f), MLFloat16(0.0f), MLFloat16(-0.0f), MLFloat16(-0.0f)};
  TestConvFp16Op(attrs, {X, dummy_vals}, {X_shape, dummy_shape}, dummy_vals, dummy_shape, false,
             OpTester::ExpectResult::kExpectFailure,
             "Node:node1 Output:Y [ShapeInferenceError] Can't merge shape info. "
             "Both source and target dimension have values but they differ. Source=1 Target=2 Dimension=0",
             -1);  // use latest opset for shape inferencing errors
}


TEST(ConvFp16Test, Conv1D_1) {
  ConvOpAndTestAttributes attrs = {
      "",                     // auto_pad
      vector<int64_t>{1},     // dilations
      1,                      // group
      vector<int64_t>{1},     // kernel_shape
      vector<int64_t>{0, 0},  // pads
      vector<int64_t>{1},     // strides
      {}                      // excluded EPs
  };

  vector<MLFloat16> X = {MLFloat16(-0.215576172f), MLFloat16(0.469238281f), MLFloat16(0.442626953f),
      MLFloat16(-0.451660156f), MLFloat16(-0.0521545410f), MLFloat16(0.290771484f), MLFloat16(0.250976562f)};
  vector<int64_t> X_shape = {1, 1, 7};
  vector<MLFloat16> W = {MLFloat16(0.244750977f)};
  vector<int64_t> W_shape = {1, 1, 1};
  vector<int64_t> Y_shape = {1, 1, 7};
  auto expected_vals = {MLFloat16(-0.0527624786f), MLFloat16(0.114846528f), MLFloat16(0.108333379f),
      MLFloat16(-0.110544264f), MLFloat16(-0.0127648748f), MLFloat16(0.0711666048f), MLFloat16(0.0614267588f)};

  TestConvFp16Op(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape);

  TestConvFp16Op(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape, true);
}


TEST(ConvFp16Test, Conv1D_1_DefaultStridesAndDilations) {
  ConvOpAndTestAttributes attrs = {
      "",                     // auto_pad
      vector<int64_t>{},      // dilations
      1,                      // group
      vector<int64_t>{1},     // kernel_shape
      vector<int64_t>{0, 0},  // pads
      vector<int64_t>{},      // strides
      {}                      // excluded EPs
  };

  vector<MLFloat16> X = {MLFloat16(-0.215576172f), MLFloat16(0.469238281f), MLFloat16(0.442626953f),
                         MLFloat16(-0.451660156f), MLFloat16(-0.0521545410f), MLFloat16(0.290771484f),
                         MLFloat16(0.250976562f)};
  vector<int64_t> X_shape = {1, 1, 7};
  vector<MLFloat16> W = {MLFloat16(0.244750977f)};
  vector<int64_t> W_shape = {1, 1, 1};
  vector<int64_t> Y_shape = {1, 1, 7};
  auto expected_vals = {MLFloat16(-0.0527624786f), MLFloat16(0.114846528f), MLFloat16(0.108333379f),
                        MLFloat16(-0.110544264f), MLFloat16(-0.0127648748f), MLFloat16(0.0711666048f),
                        MLFloat16(0.0614267588f)};

  TestConvFp16Op(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape);

  // CoreML EP requires weight to be an initializer
  TestConvFp16Op(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape, true);
}


TEST(ConvFp16Test, Conv1D_2) {
  ConvOpAndTestAttributes attrs = {
      "",                     // auto_pad
      vector<int64_t>{2},     // dilations
      1,                      // group
      vector<int64_t>{2},     // kernel_shape
      vector<int64_t>{2, 2},  // pads
      vector<int64_t>{2},     // strides
      {}                      // excluded EPs
  };

  vector<MLFloat16> X = {MLFloat16(0.112f), MLFloat16(-0.0038f), MLFloat16(0.382f), MLFloat16(0.332f),
                         MLFloat16(0.0279f), MLFloat16(-0.0836f), MLFloat16(-0.41f), MLFloat16(-0.095f),
                         MLFloat16(-0.113f), MLFloat16(-0.0254f), MLFloat16(0.369f), MLFloat16(0.352f),
                         MLFloat16(-0.349f), MLFloat16(-0.22f), MLFloat16(0.231f), MLFloat16(-0.457f),
                         MLFloat16(-0.176f), MLFloat16(-0.0603f), MLFloat16(-0.399f), MLFloat16(-0.193f),
                         MLFloat16(-0.104f), MLFloat16(-0.145f), MLFloat16(-0.319f), MLFloat16(-0.153f)};
  vector<int64_t> X_shape = {3, 1, 8};
  vector<MLFloat16> W = {MLFloat16(0.132f), MLFloat16(0.0975f), MLFloat16(0.346f), MLFloat16(0.474f)};
  vector<int64_t> W_shape = {2, 1, 2};
  vector<int64_t> Y_shape = {3, 2, 5};
  auto expected_vals = {
      MLFloat16(0.0109176636f), MLFloat16(0.0520324707f), MLFloat16(0.0531311035f), MLFloat16(-0.0362854004f),
      MLFloat16(-0.0540771484f), MLFloat16(0.0531005859f), MLFloat16(0.219848633f), MLFloat16(0.145385742f),
      MLFloat16(-0.184692383f), MLFloat16(-0.141845703f), MLFloat16(-0.0110092163f), MLFloat16(0.0210418701f),
      MLFloat16(0.0146484375f), MLFloat16(-0.0235595703f), MLFloat16(0.0304718018f), MLFloat16(-0.0535583496f),
      MLFloat16(0.135864258f), MLFloat16(-0.0379028320f), MLFloat16(-0.0112762451f), MLFloat16(0.0798950195f),
      MLFloat16(-0.0171508789f), MLFloat16(-0.0621032715f), MLFloat16(-0.0628051758f), MLFloat16(-0.0448303223f),
      MLFloat16(-0.0421142578f), MLFloat16(-0.0834350586f), MLFloat16(-0.250000000f), MLFloat16(-0.187377930f),
      MLFloat16(-0.187255859f), MLFloat16(-0.110412598f)};

  TestConvFp16Op(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape);
  TestConvFp16Op(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape, true);
}


// Conv1
TEST(ConvFp16Test, Conv1D_Bias) {
  ConvOpAndTestAttributes attrs = {
      "",                     // auto_pad
      vector<int64_t>{2},     // dilations
      1,                      // group
      vector<int64_t>{1},     // kernel_shape
      vector<int64_t>{1, 1},  // pads
      vector<int64_t>{3},     // strides
      {}                      // excluded EPs
  };

  vector<MLFloat16> X = {MLFloat16(0.458251953f), MLFloat16(0.387695312f), MLFloat16(-0.0541381836f),
                         MLFloat16(-0.301513672f), MLFloat16(0.192993164f), MLFloat16(-0.475830078f),
                         MLFloat16(0.467041016f), MLFloat16(0.407958984f), MLFloat16(0.240112305f),
                         MLFloat16(0.416503906f), MLFloat16(-0.0383300781f), MLFloat16(0.229736328f),
                         MLFloat16(0.356445312f), MLFloat16(0.128173828f), MLFloat16(0.100952148f),
                         MLFloat16(0.256835938f), MLFloat16(0.416992188f), MLFloat16(0.341064453f),
                         MLFloat16(-0.429931641f), MLFloat16(0.354492188f), MLFloat16(0.403320312f),
                         MLFloat16(0.101745605f), MLFloat16(0.457031250f), MLFloat16(0.0857543945f),
                         MLFloat16(0.380859375f), MLFloat16(0.163818359f), MLFloat16(0.123229980f),
                         MLFloat16(-0.199340820f), MLFloat16(0.260253906f), MLFloat16(-0.184082031f),
                         MLFloat16(0.311035156f), MLFloat16(0.155517578f), MLFloat16(-0.146240234f),
                         MLFloat16(-0.177978516f), MLFloat16(-0.0139007568f), MLFloat16(-0.0926513672f)};
  vector<int64_t> X_shape = {2, 2, 9};
  vector<MLFloat16> W = {MLFloat16(-0.172119141f), MLFloat16(0.323730469f)};
  vector<int64_t> W_shape = {1, 2, 1};
  vector<MLFloat16> B = {MLFloat16(0.378906250f)};
  vector<int64_t> B_shape = {1};
  vector<int64_t> Y_shape = {2, 1, 4};
  auto expected_vals = {MLFloat16(0.378906250f), MLFloat16(0.462597132f), MLFloat16(0.493487000f),
                        MLFloat16(0.447991282f), MLFloat16(0.378906250f), MLFloat16(0.249894142f),
                        MLFloat16(0.316803873f), MLFloat16(0.327701926f)};

  TestConvFp16Op(attrs, {X, W, B}, {X_shape, W_shape, B_shape}, expected_vals, Y_shape);
  TestConvFp16Op(attrs, {X, W, B}, {X_shape, W_shape, B_shape}, expected_vals, Y_shape, true);
}


TEST(ConvFp16Test, Conv2D_1) {
  ConvOpAndTestAttributes attrs = {
      "",                           // auto_pad
      vector<int64_t>{1, 1},        // dilations
      1,                            // group
      vector<int64_t>{3, 3},        // kernel_shape
      vector<int64_t>{1, 1, 1, 2},  // pads
      vector<int64_t>{3, 1},        // strides
      {}                            // excluded EPs
  };

  vector<MLFloat16> X = {MLFloat16(-0.0910644531f), MLFloat16(-0.325195312f)};
  vector<int64_t> X_shape = {2, 1, 1, 1};
  vector<MLFloat16> W = {MLFloat16(0.431152344f), MLFloat16(-0.125610352f), MLFloat16(0.448974609f),
                         MLFloat16(-0.310058594f), MLFloat16(0.135253906f), MLFloat16(-0.0679321289f),
                         MLFloat16(0.226684570f), MLFloat16(-0.173950195f), MLFloat16(-0.312988281f),
                         MLFloat16(-0.315429688f), MLFloat16(0.065612793f), MLFloat16(0.265625f),
                         MLFloat16(0.413574219f), MLFloat16(0.312255859f), MLFloat16(-0.375976562f),
                         MLFloat16(-0.00571060181f), MLFloat16(0.349121094f), MLFloat16(0.450927734f)};
  vector<int64_t> W_shape = {2, 1, 3, 3};
  vector<int64_t> Y_shape = {2, 2, 1, 2};
  auto expected_vals = {MLFloat16(-0.012316823f), MLFloat16(0.0282353163f),
                        MLFloat16(-0.0284354091f), MLFloat16(-0.0376619101f),
                        MLFloat16(-0.0439839363f), MLFloat16(0.100829601f),
                        MLFloat16(-0.101544142f), MLFloat16(-0.134492397f)};

  TestConvFp16Op(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape);

  // NNAPI/CoreML EP requires weight to be an initializer
  TestConvFp16Op(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape, true);
}

TEST(ConvFp16Test, Conv2D_2) {
  ConvOpAndTestAttributes attrs = {
      "",                           // auto_pad
      vector<int64_t>{1, 1},        // dilations
      1,                            // group
      vector<int64_t>{1, 1},        // kernel_shape
      vector<int64_t>{0, 0, 0, 0},  // pads
      vector<int64_t>{1, 1},        // strides
      {}                            // excluded EPs
  };

  vector<MLFloat16> X = {
      MLFloat16(0.452392578f), MLFloat16(0.155029297f), MLFloat16(0.111999512f),
      MLFloat16(-0.394287109f), MLFloat16(0.262695312f), MLFloat16(0.134155273f),
      MLFloat16(-0.271728516f), MLFloat16(-0.430175781f), MLFloat16(-0.268310547f),
      MLFloat16(0.389404297f), MLFloat16(-0.136352539f), MLFloat16(-0.00959014893f),
      MLFloat16(-0.487792969f), MLFloat16(-0.252685547f), MLFloat16(-0.281250000f),
      MLFloat16(0.404296875f), MLFloat16(0.0779418945f), MLFloat16(0.326904297f),
      MLFloat16(0.131103516f), MLFloat16(-0.441650391f), MLFloat16(0.124450684f),
      MLFloat16(0.367431641f), MLFloat16(0.169921875f), MLFloat16(0.200927734f),
      MLFloat16(0.233398438f), MLFloat16(0.386230469f), MLFloat16(0.111145020f),
      MLFloat16(0.387695312f), MLFloat16(0.208129883f), MLFloat16(-0.343017578f),
      MLFloat16(-0.0292510986f), MLFloat16(-0.204833984f), MLFloat16(-0.192382812f),
      MLFloat16(-0.111022949f), MLFloat16(-0.328369141f), MLFloat16(-0.0180053711f),
      MLFloat16(0.361816406f), MLFloat16(-0.409423828f), MLFloat16(-0.182495117f),
      MLFloat16(-0.334960938f), MLFloat16(-0.340820312f), MLFloat16(0.00649642944f),
      MLFloat16(0.453857422f), MLFloat16(0.0800781250f), MLFloat16(-0.147827148f),
      MLFloat16(0.0344543457f), MLFloat16(-0.333251953f), MLFloat16(0.0604858398f),
      MLFloat16(0.426269531f)};
  vector<int64_t> X_shape = {1, 1, 7, 7};
  vector<MLFloat16> W = {MLFloat16(-0.440673828f)};
  vector<int64_t> W_shape = {1, 1, 1, 1};
  vector<int64_t> Y_shape = {1, 1, 7, 7};
  auto expected_vals = {
      MLFloat16(-0.199340820f), MLFloat16(-0.0682983398f), MLFloat16(-0.0493469238f),
      MLFloat16(0.173706055f), MLFloat16(-0.115783691f), MLFloat16(-0.0591125488f),
      MLFloat16(0.119750977f), MLFloat16(0.189575195f), MLFloat16(0.118225098f),
      MLFloat16(-0.171630859f), MLFloat16(0.0600891113f), MLFloat16(0.00422668457f),
      MLFloat16(0.214965820f), MLFloat16(0.111328125f), MLFloat16(0.123962402f),
      MLFloat16(-0.178222656f), MLFloat16(-0.0343322754f), MLFloat16(-0.144042969f),
      MLFloat16(-0.0577697754f), MLFloat16(0.194580078f), MLFloat16(-0.0548400879f),
      MLFloat16(-0.161865234f), MLFloat16(-0.0748901367f), MLFloat16(-0.0885620117f),
      MLFloat16(-0.102844238f), MLFloat16(-0.170166016f), MLFloat16(-0.0489807129f),
      MLFloat16(-0.170898438f), MLFloat16(-0.0917358398f), MLFloat16(0.151123047f),
      MLFloat16(0.0128936768f), MLFloat16(0.0902709961f), MLFloat16(0.0847778320f),
      MLFloat16(0.0489196777f), MLFloat16(0.144653320f), MLFloat16(0.00793457031f),
      MLFloat16(-0.159423828f), MLFloat16(0.180419922f), MLFloat16(0.0804443359f),
      MLFloat16(0.147583008f), MLFloat16(0.150146484f), MLFloat16(-0.00286293030f),
      MLFloat16(-0.199951172f), MLFloat16(-0.0352783203f), MLFloat16(0.0651245117f),
      MLFloat16(-0.0151824951f), MLFloat16(0.146850586f), MLFloat16(-0.0266571045f),
      MLFloat16(-0.187866211f)};

  TestConvFp16Op(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape);

  // NNAPI/CoreML EP requires weight to be an initializer
  TestConvFp16Op(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape, true);
}


TEST(ConvFp16Test, Conv2D_Bias_1) {
  ConvOpAndTestAttributes attrs = {
      "",                           // auto_pad
      vector<int64_t>{1, 1},        // dilations
      1,                            // group
      vector<int64_t>{2, 2},        // kernel_shape
      vector<int64_t>{0, 0, 0, 0},  // pads
      vector<int64_t>{1, 1},        // strides
      {}                            // excluded EPs
  };

  vector<MLFloat16> X = {MLFloat16(1.0f), MLFloat16(2.0f), MLFloat16(3.0f), MLFloat16(4.0f), MLFloat16(5.0f), MLFloat16(6.0f), MLFloat16(7.0f), MLFloat16(8.0f), MLFloat16(9.0f)};
  vector<int64_t> X_shape = {1, 1, 3, 3};
  vector<MLFloat16> W = {MLFloat16(1.0f), MLFloat16(1.0f), MLFloat16(1.0f), MLFloat16(1.0f), MLFloat16(1.0f), MLFloat16(1.0f), MLFloat16(1.0f), MLFloat16(1.0f)};
  vector<int64_t> W_shape = {2, 1, 2, 2};
  vector<int64_t> Y_shape = {1, 2, 2, 2};
  vector<MLFloat16> B = {MLFloat16(1.0f), MLFloat16(-1.0f)};
  vector<int64_t> B_shape = {2};
  auto expected_vals = {MLFloat16(13.0f), MLFloat16(17.0f), MLFloat16(25.0f), MLFloat16(29.0f), MLFloat16(11.0f), MLFloat16(15.0f), MLFloat16(23.0f), MLFloat16(27.0f)};

  TestConvFp16Op(attrs, {X, W, B}, {X_shape, W_shape, B_shape}, expected_vals, Y_shape);
  TestConvFp16Op(attrs, {X, W, B}, {X_shape, W_shape, B_shape}, expected_vals, Y_shape, true);
}

// Conv48
TEST(ConvFp16Test, Conv2D_Bias_2) {
  ConvOpAndTestAttributes attrs = {
      "",                           // auto_pad
      vector<int64_t>{1, 1},        // dilations
      1,                            // group
      vector<int64_t>{4, 4},        // kernel_shape
      vector<int64_t>{1, 2, 3, 1},  // pads
      vector<int64_t>{2, 3},        // strides
      {}                            // excluded EPs
  };

  vector<MLFloat16> X = {
      MLFloat16(-0.625f), MLFloat16(0.4375f), MLFloat16(0.0625f),
      MLFloat16(-0.3125f), MLFloat16(-0.6875f), MLFloat16(0.375f),
      MLFloat16(0.0625f), MLFloat16(-0.375f), MLFloat16(0.6875f),
      MLFloat16(0.3125f), MLFloat16(-0.0625f), MLFloat16(-0.4375f),
      MLFloat16(0.625f), MLFloat16(0.25f), MLFloat16(-0.125f),
      MLFloat16(-0.5f), MLFloat16(0.5625f), MLFloat16(0.1875f),
      MLFloat16(-0.1875f), MLFloat16(-0.5625f), MLFloat16(0.5f),
      MLFloat16(0.125f), MLFloat16(-0.25f), MLFloat16(-0.625f),
      MLFloat16(0.4375f), MLFloat16(0.0625f), MLFloat16(-0.3125f),
      MLFloat16(-0.6875f), MLFloat16(0.375f), MLFloat16(0.25f),
      MLFloat16(-0.375f), MLFloat16(0.6875f), MLFloat16(0.3125f),
      MLFloat16(-0.0625f), MLFloat16(-0.4375f), MLFloat16(0.625f),
      MLFloat16(0.25f), MLFloat16(-0.125f), MLFloat16(-0.5f),
      MLFloat16(0.5625f), MLFloat16(0.1875f), MLFloat16(-0.1875f),
      MLFloat16(-0.5625f), MLFloat16(0.5f), MLFloat16(0.125f),
      MLFloat16(-0.25f), MLFloat16(-0.625f), MLFloat16(0.4375f),
      MLFloat16(0.0625f), MLFloat16(-0.3125f), MLFloat16(-0.6875f),
      MLFloat16(0.375f), MLFloat16(0.125f), MLFloat16(-0.375f),
      MLFloat16(0.6875f), MLFloat16(0.3125f), MLFloat16(-0.0625f),
      MLFloat16(-0.4375f), MLFloat16(0.625f), MLFloat16(0.25f),
      MLFloat16(-0.125f), MLFloat16(-0.5f), MLFloat16(0.5625f),
      MLFloat16(0.1875f), MLFloat16(-0.1875f), MLFloat16(-0.5625f),
      MLFloat16(0.5f), MLFloat16(0.125f), MLFloat16(-0.25f),
      MLFloat16(-0.625f), MLFloat16(0.4375f), MLFloat16(0.0625f)};
  vector<int64_t>  X_shape = {1, 2, 6, 6};
  vector<MLFloat16> W = {
      MLFloat16(-0.3125f), MLFloat16(-0.6875f), MLFloat16(0.375f), MLFloat16(0.025f),
      MLFloat16(-0.375f), MLFloat16(0.6875f), MLFloat16(0.3125f), MLFloat16(-0.0625f),
      MLFloat16(-0.4375f), MLFloat16(0.625f), MLFloat16(0.25f), MLFloat16(-0.125f),
      MLFloat16(-0.5f), MLFloat16(0.5625f), MLFloat16(0.1875f), MLFloat16(-0.1875f),
      MLFloat16(-0.5625f), MLFloat16(0.5f), MLFloat16(0.125f), MLFloat16(-0.25f),
      MLFloat16(-0.625f), MLFloat16(0.4375f), MLFloat16(0.0625f), MLFloat16(-0.3125f),
      MLFloat16(-0.6875f), MLFloat16(0.375f), MLFloat16(-0.125f), MLFloat16(-0.375f),
      MLFloat16(0.6875f), MLFloat16(0.3125f), MLFloat16(-0.0625f), MLFloat16(-0.4375f)};
  vector<int64_t>  W_shape = {1, 2, 4, 4};
  vector<MLFloat16> B = {MLFloat16(-0.8125f)};
  vector<int64_t> B_shape = {1};
  vector<int64_t> Y_shape = {1, 1, 4, 2};
  auto expected_vals = {
      MLFloat16(-0.83203125f), MLFloat16(-1.40625f), MLFloat16(-0.595312476f), MLFloat16(-1.93906248f),
      MLFloat16(-0.896875024f), MLFloat16(-1.53750002f), MLFloat16(-0.904687524f), MLFloat16(-1.65937495f)};

  TestConvFp16Op(attrs, {X, W, B}, {X_shape, W_shape, B_shape}, expected_vals, Y_shape);

  TestConvFp16Op(attrs, {X, W, B}, {X_shape, W_shape, B_shape}, expected_vals, Y_shape, true);
}


TEST(ConvFp16Test, Conv2D_AutoPad1) {
  ConvOpAndTestAttributes attrs = {
      "SAME_UPPER",           // auto_pad
      vector<int64_t>{1, 1},  // dilations
      1,                      // group
      vector<int64_t>{3, 3},  // kernel_shape
      {},                     // pads
      vector<int64_t>{1, 1},  // strides
      {}                      // excluded EPs
  };

  vector<MLFloat16> X = vector<MLFloat16>(25, MLFloat16(1.0f));
  vector<int64_t> X_shape = {1, 1, 5, 5};
  vector<MLFloat16> W = {MLFloat16(0.0f), MLFloat16(1.0f), MLFloat16(2.0f),
                     MLFloat16(3.0f), MLFloat16(4.0f), MLFloat16(5.0f),
                     MLFloat16(6.0f), MLFloat16(7.0f), MLFloat16(8.0f)};

  vector<int64_t> W_shape = {1, 1, 3, 3};
  vector<int64_t> Y_shape = {1, 1, 5, 5};
  auto expected_vals = {MLFloat16(24.0f), MLFloat16(33.0f), MLFloat16(33.0f), MLFloat16(33.0f), MLFloat16(20.0f),
                        MLFloat16(27.0f), MLFloat16(36.0f), MLFloat16(36.0f), MLFloat16(36.0f), MLFloat16(21.0f),
                        MLFloat16(27.0f), MLFloat16(36.0f), MLFloat16(36.0f), MLFloat16(36.0f), MLFloat16(21.0f),
                        MLFloat16(27.0f), MLFloat16(36.0f), MLFloat16(36.0f), MLFloat16(36.0f), MLFloat16(21.0f),
                        MLFloat16(12.0f), MLFloat16(15.0f), MLFloat16(15.0f), MLFloat16(15.0f), MLFloat16(8.0f)};
  TestConvFp16Op(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape);

  // NNAPI/CoreML EP requires weight to be an initializer
  TestConvFp16Op(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape, true);
}

TEST(ConvFp16Test, Conv2D_AutoPad2) {
  ConvOpAndTestAttributes attrs = {
      "SAME_LOWER",           // auto_pad
      vector<int64_t>{1, 1},  // dilations
      1,                      // group
      vector<int64_t>{3, 3},  // kernel_shape
      {},                     // pads
      vector<int64_t>{1, 1},  // strides
      {}                      // excluded EPs
  };

  vector<MLFloat16> X = {MLFloat16(1.0f), MLFloat16(0.0f), MLFloat16(1.0f), MLFloat16(0.0f), MLFloat16(1.0f),
                     MLFloat16(1.0f), MLFloat16(0.0f), MLFloat16(1.0f), MLFloat16(0.0f), MLFloat16(1.0f),
                     MLFloat16(1.0f), MLFloat16(0.0f), MLFloat16(1.0f), MLFloat16(0.0f), MLFloat16(1.0f),
                     MLFloat16(1.0f), MLFloat16(0.0f), MLFloat16(1.0f), MLFloat16(0.0f), MLFloat16(1.0f),
                     MLFloat16(1.0f), MLFloat16(0.0f), MLFloat16(1.0f), MLFloat16(0.0f), MLFloat16(1.0f)};
  vector<int64_t> X_shape = {1, 1, 5, 5};
  vector<MLFloat16> W = {MLFloat16(0.0f), MLFloat16(1.0f), MLFloat16(2.0f),
                     MLFloat16(3.0f), MLFloat16(4.0f), MLFloat16(5.0f),
                     MLFloat16(6.0f), MLFloat16(7.0f), MLFloat16(8.0f)};

  vector<int64_t> W_shape = {1, 1, 3, 3};
  vector<int64_t> Y_shape = {1, 1, 5, 5};
  auto expected_vals = {MLFloat16(11.0f), MLFloat16(22.0f), MLFloat16(11.0f), MLFloat16(22.0f), MLFloat16(11.0f),
                        MLFloat16(12.0f), MLFloat16(24.0f), MLFloat16(12.0f), MLFloat16(24.0f), MLFloat16(12.0f),
                        MLFloat16(12.0f), MLFloat16(24.0f), MLFloat16(12.0f), MLFloat16(24.0f), MLFloat16(12.0f),
                        MLFloat16(12.0f), MLFloat16(24.0f), MLFloat16(12.0f), MLFloat16(24.0f), MLFloat16(12.0f),
                        MLFloat16(5.0f), MLFloat16(10.0f), MLFloat16(5.0f), MLFloat16(10.0f), MLFloat16(5.0f)};
  TestConvFp16Op(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape);

  TestConvFp16Op(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape, true);
}


TEST(ConvFp16Test, Conv3D_1) {
  ConvOpAndTestAttributes attrs = {
      "",                                 // auto_pad
      vector<int64_t>{1, 1, 1},           // dilations
      1,                                  // group
      vector<int64_t>{1, 1, 1},           // kernel_shape
      vector<int64_t>{0, 0, 0, 0, 0, 0},  // pads
      vector<int64_t>{1, 1, 1},           // strides
      {}                                  // excluded EPs
  };

  vector<MLFloat16> X = {
      MLFloat16(-0.433349609f), MLFloat16(-0.483886719f), MLFloat16(-0.309570312f),
      MLFloat16(0.160766602f), MLFloat16(-0.466796875f), MLFloat16(0.465820312f),
      MLFloat16(-0.370605469f), MLFloat16(0.406005859f), MLFloat16(-0.0354919434f),
      MLFloat16(-0.312500000f), MLFloat16(0.426757812f), MLFloat16(0.398437500f),
      MLFloat16(-0.390625000f), MLFloat16(0.259033203f), MLFloat16(-0.206420898f),
      MLFloat16(0.138183594f), MLFloat16(-0.201538086f), MLFloat16(0.100280762f),
      MLFloat16(-0.241333008f), MLFloat16(0.123107910f), MLFloat16(0.0327453613f),
      MLFloat16(0.296142578f), MLFloat16(-0.231201172f), MLFloat16(0.334472656f),
      MLFloat16(0.0256805420f), MLFloat16(0.245849609f), MLFloat16(0.117248535f)};
  vector<int64_t> X_shape = {1, 1, 3, 3, 3};
  vector<MLFloat16> W = {MLFloat16(-0.442138672f)};
  vector<int64_t> W_shape = {1, 1, 1, 1, 1};
  vector<int64_t> Y_shape = {1, 1, 3, 3, 3};
  auto expected_vals = {
      MLFloat16(0.191600621f), MLFloat16(0.213945031f), MLFloat16(0.136873007f),
      MLFloat16(-0.0710811317f), MLFloat16(0.206388950f), MLFloat16(-0.205957174f),
      MLFloat16(0.163859010f),      MLFloat16(-0.179510891f),      MLFloat16(0.0156923607f),
      MLFloat16(0.138168335f),      MLFloat16(-0.188686132f),      MLFloat16(-0.176164627f),
      MLFloat16(0.172710419f),      MLFloat16(-0.114528596f),      MLFloat16(0.0912666619f),
      MLFloat16(-0.0610963106f),      MLFloat16(0.0891077816f),      MLFloat16(-0.0443380028f),
      MLFloat16(0.106702656f),      MLFloat16(-0.0544307679f),      MLFloat16(-0.0144779906f),
      MLFloat16(-0.130936086f),      MLFloat16(0.102222979f),      MLFloat16(-0.147883296f),
      MLFloat16(-0.0113543607f),      MLFloat16(-0.108699620f),      MLFloat16(-0.0518401116f)};
  TestConvFp16Op(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape);
  TestConvFp16Op(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape, true);
}


TEST(ConvFp16Test, Conv3D_2) {
  ConvOpAndTestAttributes attrs = {
      "",                                 // auto_pad
      vector<int64_t>{1, 1, 1},           // dilations
      1,                                  // group
      vector<int64_t>{1, 1, 1},           // kernel_shape
      vector<int64_t>{2, 2, 2, 2, 2, 2},  // pads
      vector<int64_t>{2, 2, 2},           // strides
      {}                                  // excluded EPs
  };

  vector<MLFloat16> X = {
      MLFloat16(0.0107727051f), MLFloat16(-0.437988281f), MLFloat16(0.455322266f), MLFloat16(-0.286621094f),
      MLFloat16(0.456787109f), MLFloat16(-0.0320434570f), MLFloat16(0.422851562f), MLFloat16(-0.187255859f),
      MLFloat16(-0.458496094f), MLFloat16(0.0420532227f), MLFloat16(-0.133300781f), MLFloat16(-0.253662109f),
      MLFloat16(-0.238403320f), MLFloat16(0.122131348f), MLFloat16(-0.177856445f), MLFloat16(0.189208984f),
      MLFloat16(0.379638672f), MLFloat16(-0.0339965820f), MLFloat16(0.127319336f), MLFloat16(-0.0402832031f),
      MLFloat16(0.464355469f), MLFloat16(-0.226928711f), MLFloat16(0.173950195f), MLFloat16(-0.301513672f),
      MLFloat16(-0.404296875f), MLFloat16(-0.332031250f), MLFloat16(0.0465393066f), MLFloat16(-0.494873047f),
      MLFloat16(0.0755004883f), MLFloat16(0.117309570f), MLFloat16(0.470458984f), MLFloat16(0.482421875f),
      MLFloat16(-0.377441406f), MLFloat16(-0.0564880371f), MLFloat16(-0.107910156f), MLFloat16(0.0434875488f),
      MLFloat16(0.244750977f), MLFloat16(-0.409912109f), MLFloat16(0.0616149902f), MLFloat16(0.229736328f),
      MLFloat16(0.278808594f), MLFloat16(0.0814819336f), MLFloat16(0.245361328f), MLFloat16(0.0825195312f),
      MLFloat16(-0.147216797f), MLFloat16(-0.430175781f), MLFloat16(0.0271759033f), MLFloat16(0.360595703f),
      MLFloat16(0.249511719f), MLFloat16(-0.225097656f), MLFloat16(-0.362792969f), MLFloat16(-0.476806641f),
      MLFloat16(0.112731934f), MLFloat16(0.497802734f), MLFloat16(0.268554688f), MLFloat16(0.0255279541f),
      MLFloat16(-0.303710938f), MLFloat16(0.411376953f), MLFloat16(0.361572266f), MLFloat16(0.00883483887f),
      MLFloat16(-0.0795898438f), MLFloat16(0.360107422f), MLFloat16(0.173217773f), MLFloat16(-0.0120086670f)};
  vector<int64_t> X_shape = {1, 1, 4, 4, 4};
  vector<MLFloat16> W = {MLFloat16(0.328125f)};
  vector<int64_t> W_shape = {1, 1, 1, 1, 1};
  vector<int64_t> Y_shape = {1, 1, 4, 4, 4};
  auto expected_vals = {MLFloat16(), MLFloat16(), MLFloat16(), MLFloat16(), MLFloat16(), MLFloat16(), MLFloat16(), MLFloat16(),
                        MLFloat16(), MLFloat16(), MLFloat16(), MLFloat16(), MLFloat16(), MLFloat16(), MLFloat16(), MLFloat16(),
                        MLFloat16(), MLFloat16(), MLFloat16(), MLFloat16(), MLFloat16(), MLFloat16(0.00353479385f), MLFloat16(0.149402618f), MLFloat16(),
                        MLFloat16(), MLFloat16(-0.150444031f), MLFloat16(-0.0437393188f), MLFloat16(), MLFloat16(), MLFloat16(), MLFloat16(), MLFloat16(),
                        MLFloat16(), MLFloat16(), MLFloat16(), MLFloat16(), MLFloat16(), MLFloat16(-0.123847961f), MLFloat16(-0.03540802f), MLFloat16(),
                        MLFloat16(), MLFloat16(0.0914840698f), MLFloat16(0.0805091858f), MLFloat16(), MLFloat16(), MLFloat16(), MLFloat16(), MLFloat16(),
                        MLFloat16(), MLFloat16(), MLFloat16(), MLFloat16(), MLFloat16(), MLFloat16(), MLFloat16(), MLFloat16(),
                        MLFloat16(), MLFloat16(), MLFloat16(), MLFloat16(), MLFloat16(), MLFloat16(), MLFloat16(), MLFloat16()};
  TestConvFp16Op(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape);
  TestConvFp16Op(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape, true);
}


TEST(ConvFp16Test, Conv3D_Bias) {
  ConvOpAndTestAttributes attrs = {
      "",                                 // auto_pad
      vector<int64_t>{2, 2, 2},           // dilations
      1,                                  // group
      vector<int64_t>{2, 2, 2},           // kernel_shape
      vector<int64_t>{2, 2, 2, 2, 2, 2},  // pads
      vector<int64_t>{2, 2, 2},           // strides
      {}                                  // excluded EPs
  };

  vector<MLFloat16> X = {
      MLFloat16(0.468017578f), MLFloat16(-0.461425781f), MLFloat16(0.335205078f), MLFloat16(-0.401123047f),
      MLFloat16(0.417236328f), MLFloat16(-0.0481262207f), MLFloat16(0.204101562f), MLFloat16(0.0318908691f),
      MLFloat16(-0.0477905273f), MLFloat16(-0.0795288086f), MLFloat16(0.498779297f), MLFloat16(0.350585938f),
      MLFloat16(0.480712891f), MLFloat16(0.269775391f), MLFloat16(-0.246337891f), MLFloat16(0.190429688f),
      MLFloat16(-0.118286133f), MLFloat16(-0.257568359f), MLFloat16(-0.339355469f), MLFloat16(-0.258056641f),
      MLFloat16(-0.0828247070f), MLFloat16(0.351318359f), MLFloat16(-0.291259766f), MLFloat16(-0.433593750f),
      MLFloat16(-0.134277344f), MLFloat16(0.440429688f), MLFloat16(0.0530700684f), MLFloat16(-0.350097656f),
      MLFloat16(-0.284667969f), MLFloat16(-0.442138672f), MLFloat16(-0.0741577148f), MLFloat16(-0.109191895f),
      MLFloat16(0.284423828f), MLFloat16(0.349853516f), MLFloat16(-0.193115234f), MLFloat16(0.326171875f),
      MLFloat16(0.488037109f), MLFloat16(0.0557556152f), MLFloat16(-0.464599609f), MLFloat16(-0.0252380371f),
      MLFloat16(-0.187866211f), MLFloat16(-0.147216797f), MLFloat16(0.207641602f), MLFloat16(0.471679688f),
      MLFloat16(-0.0556640625f), MLFloat16(-0.498779297f), MLFloat16(0.227416992f), MLFloat16(0.458984375f),
      MLFloat16(-0.472412109f), MLFloat16(-0.435791016f), MLFloat16(0.284179688f), MLFloat16(-0.270263672f),
      MLFloat16(0.342285156f), MLFloat16(0.335693359f), MLFloat16(-0.194824219f), MLFloat16(-0.276855469f),
      MLFloat16(-0.423828125f), MLFloat16(-0.438476562f), MLFloat16(0.437255859f), MLFloat16(0.306396484f),
      MLFloat16(0.457031250f), MLFloat16(0.0529174805f), MLFloat16(-0.0236206055f), MLFloat16(-0.186035156f),
      MLFloat16(0.0866699219f), MLFloat16(0.325439453f), MLFloat16(0.184570312f), MLFloat16(-0.198486328f),
      MLFloat16(-0.275390625f), MLFloat16(0.320068359f), MLFloat16(-0.348388672f), MLFloat16(0.0999755859f),
      MLFloat16(-0.113769531f), MLFloat16(0.212280273f), MLFloat16(-0.0231475830f), MLFloat16(0.167114258f),
      MLFloat16(0.223144531f), MLFloat16(0.0361022949f), MLFloat16(-0.158691406f), MLFloat16(0.0599975586f),
      MLFloat16(-0.0395202637f), MLFloat16(-0.484130859f), MLFloat16(0.329101562f), MLFloat16(-0.231201172f),
      MLFloat16(0.394531250f), MLFloat16(-0.355468750f), MLFloat16(-0.170288086f), MLFloat16(-0.0550842285f),
      MLFloat16(0.158569336f), MLFloat16(-0.418457031f), MLFloat16(-0.247436523f), MLFloat16(0.0360412598f),
      MLFloat16(-0.283691406f), MLFloat16(0.460205078f), MLFloat16(0.291015625f), MLFloat16(-0.199340820f),
      MLFloat16(0.380859375f), MLFloat16(-0.138427734f), MLFloat16(-0.238403320f), MLFloat16(-0.190673828f),
      MLFloat16(-0.110595703f), MLFloat16(-0.0871582031f), MLFloat16(0.244506836f), MLFloat16(-0.147216797f),
      MLFloat16(0.143676758f), MLFloat16(0.395507812f), MLFloat16(-0.125366211f), MLFloat16(0.115905762f),
      MLFloat16(0.459716797f), MLFloat16(-0.300048828f), MLFloat16(-0.465820312f), MLFloat16(-0.339599609f),
      MLFloat16(-0.267089844f), MLFloat16(0.361083984f), MLFloat16(-0.114257812f), MLFloat16(-0.0838012695f),
      MLFloat16(-0.318115234f), MLFloat16(0.145141602f), MLFloat16(0.315673828f), MLFloat16(0.331787109f),
      MLFloat16(-0.255859375f), MLFloat16(0.118896484f), MLFloat16(0.128295898f), MLFloat16(-0.331054688f),
      MLFloat16(0.254882812f), MLFloat16(-0.467529297f), MLFloat16(-0.119812012f), MLFloat16(0.183471680f)};
  vector<int64_t> X_shape = {2, 1, 4, 4, 4};
  vector<MLFloat16> W = {
      MLFloat16(0.388183594f), MLFloat16(-0.163696289f),
      MLFloat16(-0.428710938f), MLFloat16(0.427734375f),
      MLFloat16(0.215209961f), MLFloat16(0.00791168213f),
      MLFloat16(0.338867188f), MLFloat16(0.218383789f),
      MLFloat16(0.341064453f), MLFloat16(-0.170410156f),
      MLFloat16(-0.0135726929f), MLFloat16(-0.267822266f),
      MLFloat16(-0.348632812f), MLFloat16(-0.267333984f),
      MLFloat16(-0.366943359f), MLFloat16(0.373046875f)};
  vector<int64_t> W_shape = {2, 1, 2, 2, 2};
  vector<MLFloat16> B = {MLFloat16(0.430908203f), MLFloat16(-0.456298828f)};
  vector<int64_t> B_shape = {2};
  vector<int64_t> Y_shape = {2, 2, 3, 3, 3};

  auto expected_vals = {
      MLFloat16(0.533115625f), MLFloat16(0.662707329f), MLFloat16(0.544498205f),
      MLFloat16(0.424174339f), MLFloat16(0.627012968f), MLFloat16(0.672067642f),
      MLFloat16(0.430530101f), MLFloat16(0.424569398f), MLFloat16(0.538250446f),
      MLFloat16(0.693208933f), MLFloat16(0.427851349f), MLFloat16(0.221761703f),
      MLFloat16(0.295077145f), MLFloat16(0.832913339f), MLFloat16(0.375999779f),
      MLFloat16(0.437245011f), MLFloat16(0.291920483f), MLFloat16(0.669212699f),
      MLFloat16(0.552566051f), MLFloat16(0.226370573f), MLFloat16(0.513698816f),
      MLFloat16(0.303992242f), MLFloat16(0.742284894f), MLFloat16(0.266925812f),
      MLFloat16(0.461661220f), MLFloat16(0.323991477f), MLFloat16(0.511511266f),
      MLFloat16(-0.281706333f), MLFloat16(-0.502987564f), MLFloat16(-0.579300106f),
      MLFloat16(-0.599243939f), MLFloat16(-0.505472362f), MLFloat16(-0.756186068f),
      MLFloat16(-0.443522811f), MLFloat16(-0.572978139f), MLFloat16(-0.630189657f),
      MLFloat16(-0.475540936f), MLFloat16(-0.728834927f), MLFloat16(-0.389986098f),
      MLFloat16(-0.669373453f), MLFloat16(-0.387869477f), MLFloat16(-0.357608467f),
      MLFloat16(-0.397931814f), MLFloat16(-0.547608852f), MLFloat16(-0.358573616f),
      MLFloat16(-0.532473862f), MLFloat16(-0.408438683f), MLFloat16(-0.453677744f),
      MLFloat16(-0.454452783f), MLFloat16(-0.379444361f), MLFloat16(-0.524981856f),
      MLFloat16(-0.424284518f), MLFloat16(-0.555757523f), MLFloat16(-0.385479659f),
      MLFloat16(0.449835509f), MLFloat16(0.500584960f), MLFloat16(0.493453026f),
      MLFloat16(0.406748474f), MLFloat16(0.407412887f), MLFloat16(0.462785602f),
      MLFloat16(0.430008084f), MLFloat16(0.406240731f), MLFloat16(0.425926626f),
      MLFloat16(0.551153421f), MLFloat16(0.549696267f), MLFloat16(0.270993829f),
      MLFloat16(0.402447432f), MLFloat16(0.574599743f), MLFloat16(0.418689728f),
      MLFloat16(0.450668573f), MLFloat16(0.420462728f), MLFloat16(0.394942641f),
      MLFloat16(0.593814850f), MLFloat16(0.165656328f), MLFloat16(0.533114314f),
      MLFloat16(0.430018425f), MLFloat16(0.502558053f), MLFloat16(0.392109811f),
      MLFloat16(0.407388866f), MLFloat16(0.507203162f), MLFloat16(0.382243097f),
      MLFloat16(-0.423966885f), MLFloat16(-0.419248402f), MLFloat16(-0.524025679f),
      MLFloat16(-0.521910012f), MLFloat16(-0.502744913f), MLFloat16(-0.512152255f),
      MLFloat16(-0.425884366f), MLFloat16(-0.410446912f), MLFloat16(-0.448228836f),
      MLFloat16(-0.337432563f), MLFloat16(-0.735596657f), MLFloat16(-0.371323436f),
      MLFloat16(-0.488816738f), MLFloat16(-0.618983328f), MLFloat16(-0.263916761f),
      MLFloat16(-0.475321025f), MLFloat16(-0.507732749f), MLFloat16(-0.420486867f),
      MLFloat16(-0.558301449f), MLFloat16(-0.397618413f), MLFloat16(-0.453063041f),
      MLFloat16(-0.559680939f), MLFloat16(-0.254149109f), MLFloat16(-0.535908163f),
      MLFloat16(-0.480782807f), MLFloat16(-0.385932118f), MLFloat16(-0.499056786f)};
  TestConvFp16Op(attrs, {X, W, B}, {X_shape, W_shape, B_shape}, expected_vals, Y_shape);
  TestConvFp16Op(attrs, {X, W, B}, {X_shape, W_shape, B_shape}, expected_vals, Y_shape, true);
}


TEST(ConvFp16Test, Conv2D_group) {
  ConvOpAndTestAttributes attrs = {
      "",                           // auto_pad
      vector<int64_t>{1, 1},        // dilations
      2,                            // group
      vector<int64_t>{1, 1},        // kernel_shape
      vector<int64_t>{0, 0, 0, 0},  // pads
      vector<int64_t>{1, 1},        // strides
      {}                            // excluded EPs
  };

  vector<MLFloat16> X = {
      MLFloat16(0.0f), MLFloat16(1.0f), MLFloat16(2.0f), MLFloat16(3.0f),
      MLFloat16(4.0f), MLFloat16(5.0f), MLFloat16(6.0f), MLFloat16(7.0f),
      MLFloat16(8.0f), MLFloat16(9.0f), MLFloat16(10.0f), MLFloat16(11.0f),
      MLFloat16(12.0f), MLFloat16(13.0f), MLFloat16(14.0f), MLFloat16(15.0f),
      MLFloat16(16.0f), MLFloat16(17.0f)};
  vector<int64_t> X_shape = {1, 2, 3, 3};
  vector<MLFloat16> W = {MLFloat16(1.0f), MLFloat16(2.0f)};
  vector<int64_t> W_shape = {2, 1, 1, 1};
  vector<int64_t> Y_shape = {1, 2, 3, 3};
  auto expected_vals = {
      MLFloat16(0.0f), MLFloat16(1.0f), MLFloat16(2.0f), MLFloat16(3.0f),
      MLFloat16(4.0f), MLFloat16(5.0f), MLFloat16(6.0f), MLFloat16(7.0f),
      MLFloat16(8.0f), MLFloat16(18.0f), MLFloat16(20.0f), MLFloat16(22.0f),
      MLFloat16(24.0f), MLFloat16(26.0f), MLFloat16(28.0f), MLFloat16(30.0f),
      MLFloat16(32.0f), MLFloat16(34.0f)};

  TestConvFp16Op(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape);
  TestConvFp16Op(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape, true);
}

TEST(ConvFp16Test, ConvDimWithZero) {
  ConvOpAndTestAttributes attrs = {
      "",                           // auto_pad
      vector<int64_t>{1, 1},        // dilations
      1,                            // group
      vector<int64_t>{1, 1},        // kernel_shape
      vector<int64_t>{0, 0, 0, 0},  // pads
      vector<int64_t>{1, 1},        // strides
      {}                            // excluded EPs
  };

  vector<MLFloat16> X;
  vector<int64_t> X_shape = {0, 2, 4, 4};  // N of 0 should be handled
  vector<MLFloat16> W = {MLFloat16(1.0f), MLFloat16(2.0f), MLFloat16(1.0f), MLFloat16(2.0f)};
  vector<int64_t> W_shape = {2, 2, 1, 1};
  vector<int64_t> out_shape = {0, 2, 4, 4};

  TestConvFp16Op(attrs, {X, W}, {X_shape, W_shape}, {}, out_shape);
}

TEST(ConvFp16Test, Conv1D_asymmetric_padding) {
  ConvOpAndTestAttributes attrs = {
      "",                     // auto_pad
      vector<int64_t>{1},     // dilations
      1,                      // group
      vector<int64_t>{3},     // kernel_shape
      vector<int64_t>{1, 0},  // pads
      vector<int64_t>{1},     // strides
      {}                      // excluded EPs
  };

  vector<MLFloat16> X = {MLFloat16(1.f), MLFloat16(2.f), MLFloat16(3.f)};
  vector<int64_t> X_shape = {1, 1, 3};
  vector<MLFloat16> W = {MLFloat16(1.f), MLFloat16(1.f), MLFloat16(1.f)};
  vector<int64_t> W_shape = {1, 1, 3};
  vector<MLFloat16> B = {MLFloat16()};
  vector<int64_t> B_shape = {1};
  vector<int64_t> Y_shape = {1, 1, 2};
  auto expected_vals = {MLFloat16(3.f), MLFloat16(6.f)};

  TestConvFp16Op(attrs, {X, W, B}, {X_shape, W_shape, B_shape}, expected_vals, Y_shape);
  TestConvFp16Op(attrs, {X, W, B}, {X_shape, W_shape, B_shape}, expected_vals, Y_shape, true);
}

TEST(ConvFp16Test, Conv_AutoPad_with_non_default_strides) {
  ConvOpAndTestAttributes attrs = {
      "SAME_LOWER",           // auto_pad
      vector<int64_t>{1, 1},  // dilations
      1,                      // group
      vector<int64_t>{3, 3},  // kernel_shape
      vector<int64_t>{},      // pads
      vector<int64_t>{2, 2},  // strides
      {}                      // excluded EPs
  };

  vector<MLFloat16> X = {
      MLFloat16(0.0f), MLFloat16(1.0f), MLFloat16(2.0f), MLFloat16(3.0f), MLFloat16(4.0f),
      MLFloat16(5.0f), MLFloat16(6.0f), MLFloat16(7.0f), MLFloat16(8.0f), MLFloat16(9.0f),
      MLFloat16(10.0f), MLFloat16(11.0f), MLFloat16(12.0f), MLFloat16(13.0f), MLFloat16(14.0f),
      MLFloat16(15.0f), MLFloat16(16.0f), MLFloat16(17.0f), MLFloat16(18.0f), MLFloat16(19.0f),
      MLFloat16(20.0f), MLFloat16(21.0f), MLFloat16(22.0f), MLFloat16(23.0f), MLFloat16(24.0f)};
  vector<int64_t> X_shape = {1, 1, 5, 5};

  vector<MLFloat16> W = {MLFloat16(1.0f), MLFloat16(1.0f), MLFloat16(1.0f),
                         MLFloat16(1.0f), MLFloat16(1.0f), MLFloat16(1.0f),
                         MLFloat16(1.0f), MLFloat16(1.0f), MLFloat16(1.0f)};
  vector<int64_t> W_shape = {1, 1, 3, 3};

  auto expected_vals = {MLFloat16(12.0f), MLFloat16(27.0f), MLFloat16(24.0f),
                        MLFloat16(63.0f), MLFloat16(108.0f), MLFloat16(81.0f),
                        MLFloat16(72.0f), MLFloat16(117.0f), MLFloat16(84.0f)};
  vector<int64_t> Y_shape = {1, 1, 3, 3};

  TestConvFp16Op(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape);
  TestConvFp16Op(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape, true);
}


TEST(ConvFp16Test, Pointwise_2D) {
  ConvOpAndTestAttributes attrs = {
      "",                           // auto_pad
      vector<int64_t>{1, 1},        // dilations
      1,                            // group
      vector<int64_t>{1, 1},        // kernel_shape
      vector<int64_t>{0, 0, 0, 0},  // pads
      vector<int64_t>{1, 1},        // strides
      {}                            // excluded EPs
  };
  vector<MLFloat16> X = {
      MLFloat16(-9.f), MLFloat16(1.f), MLFloat16(2.f),
      MLFloat16(-5.f), MLFloat16(3.f), MLFloat16(-2.f),
      MLFloat16(5.f), MLFloat16(-3.f), MLFloat16(1.f),
      MLFloat16(1.f), MLFloat16(8.f), MLFloat16(-4.f),
      MLFloat16(-1.f), MLFloat16(6.f), MLFloat16(7.f),
      MLFloat16(-1.f), MLFloat16(4.f), MLFloat16(-5.f),
      MLFloat16(-9.f), MLFloat16(1.f), MLFloat16(2.f),
      MLFloat16(-5.f), MLFloat16(3.f), MLFloat16(-2.f),
      MLFloat16(5.f), MLFloat16(-3.f), MLFloat16(1.f)};
  vector<int64_t> X_shape = {1, 3, 3, 3};
  vector<MLFloat16> W = {MLFloat16(2.f), MLFloat16(-3.f), MLFloat16(0.5f),
      MLFloat16(0.25f), MLFloat16(-2.f), MLFloat16(-0.75f)};
  vector<int64_t> W_shape = {2, 3, 1, 1};
  vector<int64_t> Y_shape = {1, 2, 3, 3};
  auto expected_vals = {
      MLFloat16(-25.5f), MLFloat16(-21.5f), MLFloat16(17.f),
      MLFloat16(-9.5f), MLFloat16(-10.5f), MLFloat16(-26.f),
      MLFloat16(15.5f), MLFloat16(-19.5f), MLFloat16(17.5f),
      MLFloat16(2.5f), MLFloat16(-16.5f), MLFloat16(7.f),
      MLFloat16(4.5f), MLFloat16(-13.5f), MLFloat16(-13.f),
      MLFloat16(-0.5f), MLFloat16(-6.5f), MLFloat16(9.5f)};

  TestConvFp16Op(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape);
  TestConvFp16Op(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape, true);
}


TEST(ConvFp16Test, Pointwise_3D) {
  ConvOpAndTestAttributes attrs = {
      "",                                 // auto_pad
      vector<int64_t>{1, 1, 1},           // dilations
      1,                                  // group
      vector<int64_t>{1, 1, 1},           // kernel_shape
      vector<int64_t>{0, 0, 0, 0, 0, 0},  // pads
      vector<int64_t>{1, 1, 1},           // strides
      {}                                  // excluded EPs
  };

  vector<MLFloat16> X = {
      MLFloat16(2 / 16.f), MLFloat16(3 / 16.f), MLFloat16(4 / 16.f),
      MLFloat16(5 / 16.f), MLFloat16(6 / 16.f), MLFloat16(7 / 16.f),
      MLFloat16(8 / 16.f), MLFloat16(9 / 16.f), MLFloat16(10 / 16.f),
      MLFloat16(11 / 16.f), MLFloat16(12 / 16.f), MLFloat16(13 / 16.f),
      MLFloat16(14 / 16.f), MLFloat16(15 / 16.f), MLFloat16(16 / 16.f),
      MLFloat16(17 / 16.f), MLFloat16(18 / 16.f), MLFloat16(19 / 16.f),
      MLFloat16(20 / 16.f), MLFloat16(21 / 16.f), MLFloat16(22 / 16.f),
      MLFloat16(23 / 16.f), MLFloat16(24 / 16.f), MLFloat16(25 / 16.f),
      MLFloat16(26 / 16.f), MLFloat16(27 / 16.f), MLFloat16(28 / 16.f)};
  vector<int64_t> X_shape = {1, 1, 3, 3, 3};

  vector<MLFloat16> W = {MLFloat16(0.5f)};
  vector<int64_t> W_shape = {1, 1, 1, 1, 1};

  auto expected_vals = {
      MLFloat16(0.0625f), MLFloat16(0.09375f), MLFloat16(0.125f),
      MLFloat16(0.15625f), MLFloat16(0.1875f), MLFloat16(0.21875f),
      MLFloat16(0.25f), MLFloat16(0.28125f), MLFloat16(0.3125f),
      MLFloat16(0.34375f), MLFloat16(0.375f), MLFloat16(0.40625f),
      MLFloat16(0.4375f), MLFloat16(0.46875f), MLFloat16(0.5f),
      MLFloat16(0.53125f), MLFloat16(0.5625f), MLFloat16(0.59375f),
      MLFloat16(0.625f), MLFloat16(0.65625f), MLFloat16(0.6875f),
      MLFloat16(0.71875f), MLFloat16(0.75f), MLFloat16(0.78125f),
      MLFloat16(0.8125f), MLFloat16(0.84375f), MLFloat16(0.875f)};
  vector<int64_t> Y_shape = {1, 1, 3, 3, 3};

  // Test with weight as initializer
  TestConvFp16Op(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape);
  TestConvFp16Op(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape, true);
}

}  // namespace test
}  // namespace onnxruntime

#endif  // MLAS_F16VEC_INTRINSICS_SUPPORTED