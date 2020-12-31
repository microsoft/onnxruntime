// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <random>
#include <cmath>
#include <type_traits>
#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/providers/cpu/reduction/reduction_test_cases.h"

namespace onnxruntime {
namespace test {

const float FLOAT_INF = std::numeric_limits<float>::infinity();
const float FLOAT_NINF = -std::numeric_limits<float>::infinity();
const double DOUBLE_NINF = -std::numeric_limits<double>::infinity();

// Disable TensorRT on some of the tests because the limit in its parser: axis >=0 && axis < nbDims
template <typename OutT>
void TestReduceOp(const std::string& op,
                  int opset_version,
                  const std::vector<int64_t>& input_dims,
                  const std::vector<float>& data,
                  const std::vector<int64_t>& axes,
                  int64_t keepdims,
                  const std::vector<int64_t>& expected_dims,
                  const std::vector<OutT>& expected_data)

{
  OpTester test(op.c_str(), opset_version);

  if (!axes.empty()) {
    if (op.compare("ArgMax") == 0 || op.compare("ArgMin") == 0)
      test.AddAttribute("axis", axes[0]);
    else
      test.AddAttribute("axes", axes);
  }
  test.AddAttribute("keepdims", keepdims);
  test.AddInput<float>("data", input_dims, data);
  test.AddOutput<OutT>("reduced", expected_dims, expected_data);
#if defined(OPENVINO_CONFIG_GPU_FP32)
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kCudaExecutionProvider, kOpenVINOExecutionProvider, kTensorrtExecutionProvider});  //TensorRT,OpenVINO: result differs
#else
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kCudaExecutionProvider, kTensorrtExecutionProvider});  //TensorRT: result differs
#endif
}

//TODO:investigate why it is so slow. It need 12 seconds on an Azure Standard F48s_v2 (48 vcpus, 96 GiB memory)
// machine in RelWithDebInfo build mode, but only 2 seconds on my local dev machine(4 cores).
#ifdef NDEBUG
TEST(ReductionOpTest, ReductionVariationTest) {
  const std::vector<float>& input_data = testcases.input_data;
  const std::vector<int64_t>& input_dims = testcases.input_dims;
  OpAttributesResultMap& opAttributesResultMap = testcases.map_op_attribute_expected;

  for (auto a : opAttributesResultMap) {
    const ReductionAttribute& attributes = std::get<0>(a.second);
    const std::vector<int64_t> expected_dims = std::get<1>(a.second);
    if (a.first.compare("ArgMax") == 0 || a.first.compare("ArgMin") == 0) {
      std::vector<int64_t> expected_values;
      for (auto v : std::get<2>(a.second))
        expected_values.push_back(static_cast<int64_t>(v));
      TestReduceOp<int64_t>(a.first, 7, input_dims, input_data, attributes.axes_, attributes.keep_dims_,
                            expected_dims, expected_values);
      TestReduceOp<int64_t>(a.first, 11, input_dims, input_data, attributes.axes_, attributes.keep_dims_,
                            expected_dims, expected_values);
    } else {
      const std::vector<float> expected_values = std::get<2>(a.second);
      TestReduceOp<float>(a.first, 7, input_dims, input_data, attributes.axes_, attributes.keep_dims_,
                          expected_dims, expected_values);
      TestReduceOp<float>(a.first, 11, input_dims, input_data, attributes.axes_, attributes.keep_dims_,
                          expected_dims, expected_values);
    }
  }
}
#endif

TEST(ReductionOpTest, ReduceL1_default_axes_keepdims) {
  OpTester test("ReduceL1");
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {1, 1, 1}, {78.0f});
  test.Run();
}

TEST(ReductionOpTest, ReduceL1_do_not_keep_dims) {
  OpTester test("ReduceL1");
  test.AddAttribute("axes", std::vector<int64_t>{2});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {3, 2}, {3.0f, 7.0f, 11.0f, 15.0f, 19.0f, 23.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: full reduce without keepDimensions is not supported with explicit batch
}

TEST(ReductionOpTest, ReduceL1_do_not_keep_dims_2) {
  OpTester test("ReduceL1");
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3},
                       {1.0f, 2.0f, 3.0f});
  test.AddOutput<float>("reduced", {}, {6.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: full reduce without keepDimensions is not supported with explicit batch
}

TEST(ReductionOpTest, ReduceL1_keepdims) {
  OpTester test("ReduceL1");
  test.AddAttribute("axes", std::vector<int64_t>{2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {3, 2, 1}, {3.0f, 7.0f, 11.0f, 15.0f, 19.0f, 23.0f});
  test.Run();
}

TEST(ReductionOpTest, ReduceL1) {
  OpTester test("ReduceL1");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {1, 2, 1}, {33.0f, 45.0f});
  test.Run();
}

TEST(ReductionOpTest, ReduceL1_int32) {
  OpTester test("ReduceL1");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddInput<int32_t>("data", {3, 2, 2},
                         {1, 2,
                          3, 4,

                          5, 6,
                          7, 8,

                          9, 10,
                          11, 12});
  test.AddOutput<int32_t>("reduced", {1, 2, 1}, {33, 45});
  test.Run();
}

#if !(defined USE_TENSORRT) && !(defined USE_TVM)
TEST(ReductionOpTest, ReduceL10DTensor) {
  OpTester test("ReduceL1");
  test.AddInput<float>("data", {}, {2});
  test.AddOutput<float>("reduced", {}, {2});
  test.Run();
}
#endif  // !(defined USE_TENSORRT) && !(defined USE_TVM)

TEST(ReductionOpTest, ReduceL2_default_axes_keepdims) {
  OpTester test("ReduceL2");
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {1, 1, 1}, {25.49509757f});
  test.Run();
}

TEST(ReductionOpTest, ReduceL2_default_axes_do_not_keep_dims) {
  OpTester test("ReduceL2");
  test.AddAttribute("keepdims", static_cast<int64_t>(0));
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {}, {25.49509757f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: full reduce without keepDimensions is not supported with explicit batch
}

TEST(ReductionOpTest, ReduceL2_do_not_keepdims) {
  OpTester test("ReduceL2");
  test.AddAttribute("axes", std::vector<int64_t>{2});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {3, 2}, {2.23606798f, 5.0f, 7.81024968f, 10.63014581f, 13.45362405f, 16.2788206f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: full reduce without keepDimensions is not supported with explicit batch
}

TEST(ReductionOpTest, ReduceL2_do_not_keepdims_2) {
  OpTester test("ReduceL2");
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3},
                       {1.0f, 2.0f, 3.0f});
  test.AddOutput<float>("reduced", {}, {3.741657387f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: full reduce without keepDimensions is not supported with explicit batch
}

TEST(ReductionOpTest, ReduceL2_keepdims) {
  OpTester test("ReduceL2");
  test.AddAttribute("axes", std::vector<int64_t>{2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {3, 2, 1}, {2.23606798f, 5.0f, 7.81024968f, 10.63014581f, 13.45362405f, 16.2788206f});
  test.Run();
}

TEST(ReductionOpTest, ReduceL2) {
  OpTester test("ReduceL2");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {2}, {15.71623325f, 20.07485962f});
  test.Run();
}

TEST(ReductionOpTest, ReduceL2_int32) {
  OpTester test("ReduceL2");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<int32_t>("data", {3, 2, 2},
                         {1, 2,
                          3, 4,

                          5, 6,
                          7, 8,

                          9, 10,
                          11, 12});
  test.AddOutput<int32_t>("reduced", {2}, {15, 20});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: Int32 not allowed as input to this layer
}

#if !(defined USE_TENSORRT) && !(defined USE_TVM)
TEST(ReductionOpTest, ReduceL20DTensor) {
  OpTester test("ReduceL2");
  test.AddInput<float>("data", {}, {2});
  test.AddOutput<float>("reduced", {}, {2});
  test.Run();
}
#endif  // !(defined USE_TENSORRT) && !(defined USE_TVM)

TEST(ReductionOpTest, ReduceLogSum) {
  OpTester test("ReduceLogSum");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,
                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {3, 1, 2},
                        {1.38629436f, 1.79175949f,
                         2.48490667f, 2.6390574f,
                         2.99573231f, 3.09104252f});
  test.Run();
}

TEST(ReductionOpTest, ReduceLogSum_samesize) {
  OpTester test("ReduceLogSum");
  test.AddAttribute("axes", std::vector<int64_t>{2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 1}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  test.AddOutput<float>("reduced", {3, 2, 1},
                        {0.0f, 0.6931471f,
                         1.09861230f, 1.38629436f,
                         1.60943794f, 1.79175949f});
  test.Run();
}

TEST(ReductionOpTest, ReduceLogSum_do_not_keepdims_2) {
  OpTester test("ReduceLogSum");
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3},
                       {1.0f, 2.0f, 3.0f});
  test.AddOutput<float>("reduced", {}, {1.79175947f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: full reduce without keepDimensions is not supported with explicit batch
}

TEST(ReductionOpTest, ReduceLogSumAxes01) {
  OpTester test("ReduceLogSum");
  test.AddAttribute("axes", std::vector<int64_t>{0, 1});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3, 4, 5},
                       {0.5172141f, 0.36681905f, 0.7489675f, 0.21859895f, 0.6378839f,
                        0.6120873f, 0.74698675f, 0.87222993f, 0.23919299f, 0.4877085f,
                        0.58392614f, 0.56973755f, 0.28744474f, 0.56500393f, 0.13348383f,
                        0.06321382f, 0.20588198f, 0.08760026f, 0.9710815f, 0.6781033f,
                        0.38189054f, 0.9127731f, 0.21229997f, 0.7597165f, 0.36321816f,
                        0.18440539f, 0.44839138f, 0.888846f, 0.54862875f, 0.15642975f,
                        0.5046317f, 0.6035792f, 0.42172152f, 0.55201846f, 0.8684674f,
                        0.8725194f, 0.89469117f, 0.88513845f, 0.48750868f, 0.2512843f,
                        0.54381144f, 0.970685f, 0.44817686f, 0.7655562f, 0.64186585f,
                        0.8696393f, 0.91110307f, 0.12956737f, 0.9199235f, 0.26789218f,
                        0.25372583f, 0.6147827f, 0.67517287f, 0.74066293f, 0.6317299f,
                        0.70738846f, 0.27802366f, 0.15887405f, 0.95882577f, 0.23314993f});
  test.AddOutput<float>("reduced", {5},
                        {1.8073791f, 2.0180254f, 1.7606194f, 2.0446842f, 1.6773242f});

  test.Run();
}

#if !(defined USE_TENSORRT) && !(defined USE_TVM)
TEST(ReductionOpTest, ReduceLogSum0DTensor) {
  OpTester test("ReduceLogSum");
  test.AddInput<float>("data", {}, {2.f});
  test.AddOutput<float>("reduced", {}, {0.693147f});
  test.Run();
}
#endif  // !(defined USE_TENSORRT) && !(defined USE_TVM)

TEST(ReductionOpTest, ReduceLogSumExp_default_axes_keepdims) {
  OpTester test("ReduceLogSumExp");
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {5.0f, 1.0f,
                        20.0f, 2.0f,

                        30.0f, 1.0f,
                        40.0f, 2.0f,

                        55.0f, 1.0f,
                        60.0f, 2.0f});
  test.AddOutput<float>("reduced", {1, 1, 1}, {60.00671387f});
  test.Run();
}

TEST(ReductionOpTest, ReduceLogSumExp_default_axes_keepdims_double) {
  OpTester test("ReduceLogSumExp");
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<double>("data", {3, 2, 2},
                       {5.0, 1.0,
                        20.0, 2.0,

                        30.0, 1.0,
                        40.0, 2.0,

                        55.0, 1.0,
                        60.0, 2.0});
  test.AddOutput<double>("reduced", {1, 1, 1}, {60.00671387});
  test.Run();
}

TEST(ReductionOpTest, ReduceLogSumExp_default_axes_do_not_keep_dims) {
  OpTester test("ReduceLogSumExp");
  test.AddAttribute("keepdims", static_cast<int64_t>(0));
  test.AddInput<float>("data", {3, 2, 2},
                       {5.0f, 1.0f,
                        20.0f, 2.0f,

                        30.0f, 1.0f,
                        40.0f, 2.0f,

                        55.0f, 1.0f,
                        60.0f, 2.0f});
  test.AddOutput<float>("reduced", {}, {60.00671387f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: full reduce without keepDimensions is not supported with explicit batch
}

TEST(ReductionOpTest, ReduceLogSumExp_default_axes_do_not_keep_dims_double) {
  OpTester test("ReduceLogSumExp");
  test.AddAttribute("keepdims", static_cast<int64_t>(0));
  test.AddInput<double>("data", {3, 2, 2},
                       {5.0, 1.0,
                        20.0, 2.0,

                        30.0, 1.0,
                        40.0, 2.0,

                        55.0, 1.0,
                        60.0, 2.0});
  test.AddOutput<double>("reduced", {}, {60.00671387});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: full reduce without keepDimensions is not supported with explicit batch
}

TEST(ReductionOpTest, ReduceLogSumExp_do_not_keepdims) {
  OpTester test("ReduceLogSumExp");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3, 2, 2},
                       {5.0f, 1.0f,
                        20.0f, 2.0f,

                        30.0f, 1.0f,
                        40.0f, 2.0f,

                        55.0f, 1.0f,
                        60.0f, 2.0f});
  test.AddOutput<float>("reduced", {3, 2}, {20.0f, 2.31326175f, 40.00004578f, 2.31326175f, 60.00671387f, 2.31326175f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: full reduce without keepDimensions is not supported with explicit batch
}

TEST(ReductionOpTest, ReduceLogSumExp_do_not_keepdims_double) {
  OpTester test("ReduceLogSumExp");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<double>("data", {3, 2, 2},
                       {5.0, 1.0,
                        20.0, 2.0,

                        30.0, 1.0,
                        40.0, 2.0,

                        55.0, 1.0,
                        60.0, 2.0});
  test.AddOutput<double>("reduced", {3, 2}, {20.0, 2.31326175, 40.00004578, 2.31326175, 60.00671387, 2.31326175});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: full reduce without keepDimensions is not supported with explicit batch
}

TEST(ReductionOpTest, ReduceLogSumExp_do_not_keepdims_2) {
  OpTester test("ReduceLogSumExp");
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3},
                       {1.0f, 2.0f, 3.0f});
  test.AddOutput<float>("reduced", {}, {3.40760596f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: full reduce without keepDimensions is not supported with explicit batch
}

TEST(ReductionOpTest, ReduceLogSumExp_do_not_keepdims_2_double) {
  OpTester test("ReduceLogSumExp");
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<double>("data", {3},
                       {1.0, 2.0, 3.0});
  test.AddOutput<double>("reduced", {}, {3.40760596});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: full reduce without keepDimensions is not supported with explicit batch
}

TEST(ReductionOpTest, ReduceLogSumExp_keepdims) {
  OpTester test("ReduceLogSumExp");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {5.0f, 1.0f,
                        20.0f, 2.0f,

                        30.0f, 1.0f,
                        40.0f, 2.0f,

                        55.0f, 1.0f,
                        60.0f, 2.0f});
  test.AddOutput<float>("reduced", {3, 1, 2}, {20.0f, 2.31326175f, 40.00004578f, 2.31326175f, 60.00671387f, 2.31326175f});
  test.Run();
}

TEST(ReductionOpTest, ReduceLogSumExp_keepdims_double) {
  OpTester test("ReduceLogSumExp");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<double>("data", {3, 2, 2},
                       {5.0, 1.0,
                        20.0, 2.0,

                        30.0, 1.0,
                        40.0, 2.0,

                        55.0, 1.0,
                        60.0, 2.0});
  test.AddOutput<double>("reduced", {3, 1, 2}, {20.0, 2.31326175, 40.00004578, 2.31326175, 60.00671387, 2.31326175});
  test.Run();
}

TEST(ReductionOpTest, ReduceLogSumExp) {
  OpTester test("ReduceLogSumExp");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {1, 2, 1}, {10.33174133f, 12.33174133f});
  test.Run();
}

TEST(ReductionOpTest, ReduceLogSumExp_double) {
  OpTester test("ReduceLogSumExp");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<double>("data", {3, 2, 2},
                       {1.0, 2.0,
                        3.0, 4.0,

                        5.0, 6.0,
                        7.0, 8.0,

                        9.0, 10.0,
                        11.0, 12.0});
  test.AddOutput<double>("reduced", {1, 2, 1}, {10.33174133, 12.33174133});
  test.Run();
}

TEST(ReductionOpTest, ReduceLogSumExp_int32) {
  OpTester test("ReduceLogSumExp");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<int32_t>("data", {3, 2, 2},
                         {1, 2,
                          3, 4,

                          5, 6,
                          7, 8,

                          9, 10,
                          11, 12});
  test.AddOutput<int32_t>("reduced", {1, 2, 1}, {10, 12});
  test.Run();
}

#if !(defined USE_TENSORRT) && !(defined USE_TVM)
TEST(ReductionOpTest, ReduceLogSumExp0DTensor) {
  OpTester test("ReduceLogSumExp");
  test.AddInput<float>("data", {}, {2});
  test.AddOutput<float>("reduced", {}, {2});
  test.Run();
}

TEST(ReductionOpTest, ReduceLogSumExp0DTensor_double) {
  OpTester test("ReduceLogSumExp");
  test.AddInput<double>("data", {}, {2});
  test.AddOutput<double>("reduced", {}, {2});
  test.Run();
}
#endif  // !(defined USE_TENSORRT) && !(defined USE_TVM)

TEST(ReductionOpTest, ReduceMax_default_axes_keepdims) {
  OpTester test("ReduceMax");
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {5.0f, 1.0f,
                        20.0f, 2.0f,

                        30.0f, 1.0f,
                        40.0f, 2.0f,

                        55.0f, 1.0f,
                        60.0f, 2.0f});
  test.AddOutput<float>("reduced", {1, 1, 1}, {60.0f});
  test.Run();
}

TEST(ReductionOpTest, ReduceMax_default_axes_do_not_keep_dims) {
  OpTester test("ReduceMax");
  test.AddAttribute("keepdims", static_cast<int64_t>(0));
  test.AddInput<float>("data", {3, 2, 2},
                       {5.0f, 1.0f,
                        20.0f, 2.0f,

                        30.0f, 1.0f,
                        40.0f, 2.0f,

                        55.0f, 1.0f,
                        60.0f, 2.0f});
  test.AddOutput<float>("reduced", {}, {60.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: full reduce without keepDimensions is not supported with explicit batch
}

TEST(ReductionOpTest, ReduceMax_do_not_keepdims) {
  OpTester test("ReduceMax");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3, 2, 2},
                       {5.0f, 1.0f,
                        20.0f, 2.0f,

                        30.0f, 1.0f,
                        40.0f, 2.0f,

                        55.0f, 1.0f,
                        60.0f, 2.0f});
  test.AddOutput<float>("reduced", {3, 2}, {20.0f, 2.0f, 40.0f, 2.0f, 60.0f, 2.0f});
  test.Run();
}

TEST(ReductionOpTest, ReduceMax_do_not_keepdims_2) {
  OpTester test("ReduceMax");
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3},
                       {5.0f, 1.0f, 20.0f});
  test.AddOutput<float>("reduced", {}, {20.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: full reduce without keepDimensions is not supported with explicit batch
}

TEST(ReductionOpTest, ReduceMax_keepdims) {
  OpTester test("ReduceMax");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {5.0f, 1.0f,
                        20.0f, 2.0f,

                        30.0f, 1.0f,
                        40.0f, 2.0f,

                        55.0f, 1.0f,
                        60.0f, 2.0f});
  test.AddOutput<float>("reduced", {3, 1, 2}, {20.0f, 2.0f, 40.0f, 2.0f, 60.0f, 2.0f});
  test.Run();
}

TEST(ReductionOpTest, ReduceMax) {
  OpTester test("ReduceMax");
  test.AddAttribute("axes", std::vector<int64_t>{1, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {3, 1, 1}, {4.0f, 8.0f, 12.0f});
  test.Run();
}

TEST(ReductionOpTest, ReduceMax_int32) {
  OpTester test("ReduceMax");
  test.AddAttribute("axes", std::vector<int64_t>{1, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<int32_t>("data", {3, 2, 2},
                         {1, 2,
                          3, 4,

                          5, 6,
                          7, 8,

                          9, 10,
                          11, 12});
  test.AddOutput<int32_t>("reduced", {3, 1, 1}, {4, 8, 12});

#if defined(OPENVINO_CONFIG_GPU_FP32) || defined(OPENVINO_CONFIG_GPU_FP16) || defined(OPENVINO_CONFIG_MYRIAD)
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});  // OpenVINO: Disabled temporarily
#else
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});                          //TensorRT: axis must be 0
#endif
}

TEST(ReductionOpTest, ReduceMax_int64) {
  OpTester test("ReduceMax");
  test.AddAttribute("axes", std::vector<int64_t>{1, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<int64_t>("data", {3, 2, 2},
                         {1, 2,
                          3, 4,

                          5, 6,
                          7, 8,

                          9, 10,
                          11, 12});
  test.AddOutput<int64_t>("reduced", {3, 1, 1}, {4, 8, 12});
#if defined(OPENVINO_CONFIG_GPU_FP32) || defined(OPENVINO_CONFIG_GPU_FP16) || defined(OPENVINO_CONFIG_MYRIAD)
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});  // OpenVINO: Disabled temporarily
#else
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});                          //TensorRT: axis must be 0
#endif
}

TEST(ReductionOpTest, ReduceMax_int8) {
  OpTester test("ReduceMax", 12);
  test.AddAttribute("axes", std::vector<int64_t>{1, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<int8_t>("data", {3, 2, 2},
                        {1, 2,
                         3, 4,

                         5, 6,
                         7, 8,

                         9, 10,
                         11, 12});
  test.AddOutput<int8_t>("reduced", {3, 1, 1}, {4, 8, 12});
#if defined(OPENVINO_CONFIG_MYRIAD)
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});  // OpenVINO: Disabled temporarily
#else
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});                          //TensorRT: axis must be 0
#endif
}

TEST(ReductionOpTest, ReduceMax_uint8) {
  OpTester test("ReduceMax", 12);
  test.AddAttribute("axes", std::vector<int64_t>{1, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<uint8_t>("data", {3, 2, 2},
                         {1, 2,
                          3, 4,

                          5, 6,
                          7, 8,

                          9, 10,
                          11, 12});
  test.AddOutput<uint8_t>("reduced", {3, 1, 1}, {4, 8, 12});
#if defined(OPENVINO_CONFIG_MYRIAD)
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});  // OpenVINO: Disabled temporarily
#else
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});                          //TensorRT: axis must be 0
#endif
}

#if !(defined USE_TENSORRT) && !(defined USE_TVM)
TEST(ReductionOpTest, ReduceMax0DTensor) {
  OpTester test("ReduceMax");
  test.AddInput<float>("data", {}, {2});
  test.AddOutput<float>("reduced", {}, {2});
  test.Run();
}
#endif  // !(defined USE_TENSORRT) && !(defined USE_TVM)

TEST(ReductionOpTest, ReduceMean_default_axes_keepdims) {
  OpTester test("ReduceMean");
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {5.0f, 1.0f,
                        20.0f, 2.0f,

                        30.0f, 1.0f,
                        40.0f, 2.0f,

                        55.0f, 1.0f,
                        60.0f, 2.0f});
  test.AddOutput<float>("reduced", {1, 1, 1}, {18.25f});
  test.Run();
}

TEST(ReductionOpTest, ReduceMean_default_axes_keepdims_double) {
  OpTester test("ReduceMean");
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<double>("data", {3, 2, 2},
                       {5.0, 1.0,
                        20.0, 2.0,

                        30.0, 1.0,
                        40.0, 2.0,

                        55.0, 1.0,
                        60.0, 2.0});
  test.AddOutput<double>("reduced", {1, 1, 1}, {18.25});
  test.Run();
}

TEST(ReductionOpTest, ReduceMean_default_axes_do_not_keep_dims) {
  OpTester test("ReduceMean");
  test.AddAttribute("keepdims", static_cast<int64_t>(0));
  test.AddInput<float>("data", {3, 2, 2},
                       {5.0f, 1.0f,
                        20.0f, 2.0f,

                        30.0f, 1.0f,
                        40.0f, 2.0f,

                        55.0f, 1.0f,
                        60.0f, 2.0f});
  test.AddOutput<float>("reduced", {}, {18.25f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: full reduce without keepDimensions is not supported with explicit batch
}

TEST(ReductionOpTest, ReduceMean_default_axes_do_not_keep_dims_double) {
  OpTester test("ReduceMean");
  test.AddAttribute("keepdims", static_cast<int64_t>(0));
  test.AddInput<double>("data", {3, 2, 2},
                       {5.0, 1.0,
                        20.0, 2.0,

                        30.0, 1.0,
                        40.0, 2.0,

                        55.0, 1.0,
                        60.0, 2.0});
  test.AddOutput<double>("reduced", {}, {18.25});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: full reduce without keepDimensions is not supported with explicit batch
}

TEST(ReductionOpTest, ReduceMean_do_not_keepdims) {
  OpTester test("ReduceMean");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3, 2, 2},
                       {5.0f, 1.0f,
                        20.0f, 2.0f,

                        30.0f, 1.0f,
                        40.0f, 2.0f,

                        55.0f, 1.0f,
                        60.0f, 2.0f});
  test.AddOutput<float>("reduced", {3, 2}, {12.5f, 1.5f, 35.0f, 1.5f, 57.5f, 1.5f});

#if defined(__arm__)
  // armv7 isn't as accurate so need to add a little tolerance for the diffs
  //  expected[i] evaluates to 35,
  //  output[i] evaluates to 34.999866485595703
  test.SetOutputRelErr("reduced", 1e-5f);
#endif

  test.Run();
}

TEST(ReductionOpTest, ReduceMean_do_not_keepdims_double) {
  OpTester test("ReduceMean");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<double>("data", {3, 2, 2},
                       {5.0, 1.0,
                        20.0, 2.0,

                        30.0, 1.0f,
                        40.0, 2.0f,

                        55.0, 1.0,
                        60.0, 2.0});
  test.AddOutput<double>("reduced", {3, 2}, {12.5, 1.5, 35.0, 1.5, 57.5, 1.5});

#if defined(__arm__)
  // armv7 isn't as accurate so need to add a little tolerance for the diffs
  //  expected[i] evaluates to 35,
  //  output[i] evaluates to 34.999866485595703
  test.SetOutputRelErr("reduced", 1e-5f);
#endif

  test.Run();
}

TEST(ReductionOpTest, ReduceMean_do_not_keepdims_2) {
  OpTester test("ReduceMean");
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3},
                       {1.0f, 2.0f, 3.0f});
  test.AddOutput<float>("reduced", {}, {2.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: full reduce without keepDimensions is not supported with explicit batch
}

TEST(ReductionOpTest, ReduceMean_do_not_keepdims_2_double) {
  OpTester test("ReduceMean");
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<double>("data", {3},
                       {1.0, 2.0, 3.0});
  test.AddOutput<double>("reduced", {}, {2.0});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: full reduce without keepDimensions is not supported with explicit batch
}

TEST(ReductionOpTest, ReduceMean_keepdims) {
  OpTester test("ReduceMean");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {5.0f, 1.0f,
                        20.0f, 2.0f,

                        30.0f, 1.0f,
                        40.0f, 2.0f,

                        55.0f, 1.0f,
                        60.0f, 2.0f});
  test.AddOutput<float>("reduced", {3, 1, 2}, {12.5f, 1.5f, 35.0f, 1.5f, 57.5f, 1.5f});

#if defined(__arm__)
  // armv7 isn't as accurate so need to add a little tolerance for the diffs
  //  expected[i] evaluates to 35,
  //  output[i] evaluates to 34.999866485595703
  test.SetOutputRelErr("reduced", 1e-5f);
#endif

  test.Run();
}

TEST(ReductionOpTest, ReduceMean_keepdims_double) {
  OpTester test("ReduceMean");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<double>("data", {3, 2, 2},
                       {5.0, 1.0,
                        20.0, 2.0,

                        30.0, 1.0,
                        40.0, 2.0,

                        55.0, 1.0,
                        60.0, 2.0});
  test.AddOutput<double>("reduced", {3, 1, 2}, {12.5, 1.5, 35.0, 1.5, 57.5, 1.5});

#if defined(__arm__)
  // armv7 isn't as accurate so need to add a little tolerance for the diffs
  //  expected[i] evaluates to 35,
  //  output[i] evaluates to 34.999866485595703
  test.SetOutputRelErr("reduced", 1e-5f);
#endif

  test.Run();
}

TEST(ReductionOpTest, ReduceMean) {
  OpTester test("ReduceMean");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {1, 2, 1}, {5.5f, 7.5f});

  test.Run();
}

TEST(ReductionOpTest, ReduceMean_double) {
  OpTester test("ReduceMean");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<double>("data", {3, 2, 2},
                       {1.0, 2.0,
                        3.0, 4.0,

                        5.0, 6.0,
                        7.0, 8.0,

                        9.0, 10.0,
                        11.0, 12.0});
  test.AddOutput<double>("reduced", {1, 2, 1}, {5.5, 7.5});

  test.Run();
}

TEST(ReductionOpTest, ReduceMean_int32) {
  OpTester test("ReduceMean");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<int32_t>("data", {3, 2, 2},
                         {10, 20,
                          30, 40,

                          50, 60,
                          70, 80,

                          90, 100,
                          110, 120});
  test.AddOutput<int32_t>("reduced", {1, 2, 1}, {55, 75});
  test.Run();
}

#if !(defined USE_TENSORRT) && !(defined USE_TVM)
TEST(ReductionOpTest, ReduceMean0DTensor) {
  OpTester test("ReduceMean");
  test.AddInput<float>("data", {}, {2});
  test.AddOutput<float>("reduced", {}, {2});
  test.Run();
}

TEST(ReductionOpTest, ReduceMean0DTensor_double) {
  OpTester test("ReduceMean");
  test.AddInput<double>("data", {}, {2});
  test.AddOutput<double>("reduced", {}, {2});
  test.Run();
}
#endif  // !(defined USE_TENSORRT) && !(defined USE_TVM)

TEST(ReductionOpTest, ReduceMin_default_axes_keepdims) {
  OpTester test("ReduceMin");
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {5.0f, 1.0f,
                        20.0f, 2.0f,

                        30.0f, 1.0f,
                        40.0f, 2.0f,

                        55.0f, 1.0f,
                        60.0f, 2.0f});
  test.AddOutput<float>("reduced", {1, 1, 1}, {1.0f});
  test.Run();
}

TEST(ReductionOpTest, ReduceMin_default_axes_do_not_keep_dims) {
  OpTester test("ReduceMin");
  test.AddAttribute("keepdims", static_cast<int64_t>(0));
  test.AddInput<float>("data", {3, 2, 2},
                       {5.0f, 1.0f,
                        20.0f, 2.0f,

                        30.0f, 1.0f,
                        40.0f, 2.0f,

                        55.0f, 1.0f,
                        60.0f, 2.0f});
  test.AddOutput<float>("reduced", {}, {1.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: full reduce without keepDimensions is not supported with explicit batch
}

TEST(ReductionOpTest, ReduceMin_default_axes_do_not_keep_dims_2D) {
  OpTester test("ReduceMin");
  test.AddAttribute("keepdims", static_cast<int64_t>(0));
  test.AddInput<float>("data", {2, 2},
                       {5.0f, 1.0f,
                        20.0f, 2.0f});
  test.AddOutput<float>("reduced", {}, {1.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: full reduce without keepDimensions is not supported with explicit batch
}

TEST(ReductionOpTest, ReduceMin_do_not_keepdims) {
  OpTester test("ReduceMin");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3, 2, 2},
                       {5.0f, 1.0f,
                        20.0f, 2.0f,

                        30.0f, 1.0f,
                        40.0f, 2.0f,

                        55.0f, 1.0f,
                        60.0f, 2.0f});
  test.AddOutput<float>("reduced", {3, 2}, {5.0f, 1.0f, 30.0f, 1.0f, 55.0f, 1.0f});
  test.Run();
}

TEST(ReductionOpTest, ReduceMin_do_not_keepdims_2) {
  OpTester test("ReduceMin");
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3},
                       {5.0f, 1.0f, 20.0f});
  test.AddOutput<float>("reduced", {}, {1.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: full reduce without keepDimensions is not supported with explicit batch
}

TEST(ReductionOpTest, ReduceMin_keepdims) {
  OpTester test("ReduceMin");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {5.0f, 1.0f,
                        20.0f, 2.0f,

                        30.0f, 1.0f,
                        40.0f, 2.0f,

                        55.0f, 1.0f,
                        60.0f, 2.0f});
  test.AddOutput<float>("reduced", {3, 1, 2}, {5.0f, 1.0f, 30.0f, 1.0f, 55.0f, 1.0f});
  test.Run();
}

TEST(ReductionOpTest, ReduceMin) {
  OpTester test("ReduceMin");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {1, 2, 1}, {1.0f, 3.0f});
  test.Run();
}

TEST(ReductionOpTest, ReduceMin_int32) {
  OpTester test("ReduceMin");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<int32_t>("data", {3, 2, 2},
                         {1, 2,
                          3, 4,

                          5, 6,
                          7, 8,

                          9, 10,
                          11, 12});
  test.AddOutput<int32_t>("reduced", {1, 2, 1}, {1, 3});
  test.Run();
}

TEST(ReductionOpTest, ReduceMin_int8) {
  OpTester test("ReduceMin", 12);
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<int8_t>("data", {3, 2, 2},
                        {1, 2,
                         3, 4,

                         5, 6,
                         7, 8,

                         9, 10,
                         11, 12});
  test.AddOutput<int8_t>("reduced", {1, 2, 1}, {1, 3});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(ReductionOpTest, ReduceMin_uint8) {
  OpTester test("ReduceMin", 12);
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<uint8_t>("data", {3, 2, 2},
                         {1, 2,
                          3, 4,

                          5, 6,
                          7, 8,

                          9, 10,
                          11, 12});
  test.AddOutput<uint8_t>("reduced", {1, 2, 1}, {1, 3});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

#if !(defined USE_TENSORRT) && !(defined USE_TVM)
TEST(ReductionOpTest, ReduceMin0DTensor) {
  OpTester test("ReduceMin");
  test.AddInput<float>("data", {}, {2});
  test.AddOutput<float>("reduced", {}, {2});
  test.Run();
}
#endif  // !(defined USE_TENSORRT) && !(defined USE_TVM)

TEST(ReductionOpTest, ReduceSum) {
  OpTester test("ReduceSum");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {1, 2, 1}, {33.0f, 45.0f});
  test.Run();
}

TEST(ReductionOpTest, ReduceSum_double) {
  OpTester test("ReduceSum");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<double>("data", {3, 2, 2},
                        {1.0, 2.0,
                         3.0, 4.0,

                         5.0, 6.0,
                         7.0, 8.0,

                         9.0, 10.0,
                         11.0, 12.0});
  test.AddOutput<double>("reduced", {1, 2, 1}, {33.0, 45.0});
  test.Run();
}

TEST(ReductionOpTest, ReduceSum_axes01) {
  OpTester test("ReduceSum");
  test.AddAttribute("axes", std::vector<int64_t>{2});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {3, 2}, {3.0f, 7.0f, 11.0f, 15.0f, 19.0f, 23.0f});
  test.Run();
}

TEST(ReductionOpTest, ReduceSum_axes02) {
  OpTester test("ReduceSum");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {3, 2}, {4.0f, 6.0f, 12.0f, 14.0f, 20.0f, 22.0f});
  test.Run();
}

TEST(ReductionOpTest, ReduceSum_int32) {
  OpTester test("ReduceSum");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<int32_t>("data", {3, 2, 2},
                         {1, 2,
                          3, 4,

                          5, 6,
                          7, 8,

                          9, 10,
                          11, 12});
  test.AddOutput<int32_t>("reduced", {1, 2, 1}, {33, 45});
  test.Run();
}

#ifdef USE_CUDA
TEST(ReductionOpTest, ReduceSumHalfHalf) {
  OpTester test("ReduceSum");
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddAttribute("axes", std::vector<int64_t>{0, 1});

  std::vector<float> data = {1.0f, 2.0f,
                             3.0f, 4.0f,

                             5.0f, 6.0f,
                             7.0f, 8.0f,

                             9.0f, 10.0f,
                             11.0f, 12.0f};
  std::vector<MLFloat16> data_half(12);
  ConvertFloatToMLFloat16(data.data(), data_half.data(), 12);

  std::vector<float> result = {36.0f, 42.0f};
  std::vector<MLFloat16> result_half(2);
  ConvertFloatToMLFloat16(result.data(), result_half.data(), 2);

  test.AddInput<MLFloat16>("data", {3, 2, 2}, data_half);
  test.AddOutput<MLFloat16>("reduced", {2}, result_half);
  test.Run();
}

void test_half_reduce_sum(
    int64_t m, int64_t n) {
  OpTester test("ReduceSum");
  // Input tensor.
  std::vector<float> X(m * n, 0.0f);
  // Reduced tensor.
  std::vector<float> Y(n, 0.0f);
  // Random number generator.
  std::default_random_engine generator(0);
  std::uniform_real_distribution<float> distribution(0.0, 1.0);
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      const float value = distribution(generator) / float(m);
      X[i * n + j] = value;
      Y[j] += value;
    }
  }

  std::vector<MLFloat16> X_half(m * n);
  ConvertFloatToMLFloat16(X.data(), X_half.data(), int(m * n));

  std::vector<MLFloat16> Y_half(n);
  ConvertFloatToMLFloat16(Y.data(), Y_half.data(), int(n));

  test.AddAttribute("keepdims", (int64_t)0);
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddInput<MLFloat16>("data", {m, n}, X_half);
  test.AddOutput<MLFloat16>("reduced", {n}, Y_half);
  test.Run();
}

TEST(ReductionOpTest, ReduceSum_half_bert) {
  test_half_reduce_sum(6 * 128, 128);
  test_half_reduce_sum(8 * 128, 128);
  test_half_reduce_sum(6 * 384, 128);
  test_half_reduce_sum(8 * 384, 128);
}

// Add more UTs for half as needed
#endif
TEST(ReductionOpTest, ReduceSum_apex_reduction) {
  OpTester test("ReduceSum");
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddAttribute("axes", std::vector<int64_t>{0, 1});
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {2}, {36.0f, 42.0f});
  test.Run();
}

void test_apex_reduce_sum(
    int64_t m, int64_t n, bool exclude_openvino = false) {
  OpTester test("ReduceSum");
  // Input tensor.
  std::vector<float> X(m * n, 0.0f);
  // Reduced tensor.
  std::vector<float> Y(n, 0.0f);
  // Random number generator.
  std::default_random_engine generator(0);
  std::uniform_real_distribution<float> distribution(0.0, 1.0);
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      const float value = distribution(generator) / float(m);
      X[i * n + j] = value;
      Y[j] += value;
    }
  }

  test.AddAttribute("keepdims", (int64_t)0);
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddInput<float>("data", {m, n}, X);
  test.AddOutput<float>("reduced", {n}, Y);

  if (exclude_openvino) {
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kOpenVINOExecutionProvider});
  } else {
    test.Run();
  }
}

TEST(ReductionOpTest, ReduceSum_apex_matrix_large) {
  for (int64_t m = 1; m < 2049; m *= 8) {
    for (int64_t n = 2; n < 2049; n *= 8) {
      if (m * n > 32768) {
        continue;
      }
      test_apex_reduce_sum(m, n);
      test_apex_reduce_sum(m + 1, n);
      test_apex_reduce_sum(m + 3, n);
      test_apex_reduce_sum(m + 5, n);
      test_apex_reduce_sum(m + 23, n);
      test_apex_reduce_sum(m + 47, n);
      test_apex_reduce_sum(m + 97, n);
    }
  }
}

TEST(ReductionOpTest, ReduceSum_apex_bert) {
  test_apex_reduce_sum(6 * 128, 128);
  test_apex_reduce_sum(8 * 128, 128);
  test_apex_reduce_sum(6 * 384, 128);
  test_apex_reduce_sum(8 * 384, 128);
}

TEST(ReductionOpTest, ReduceSum_batch_by_two) {
  for (int i = 1; i < 128; ++i) {
    test_apex_reduce_sum(i, 2);
  }
}

TEST(ReductionOpTest, ReduceSum_batch_by_seq_by_128) {
  for (int i = 1; i < 16; i += 1) {
    test_apex_reduce_sum(i * 128, 128);
    test_apex_reduce_sum(i * 512, 128);
    test_apex_reduce_sum(i * 128, 768);
    test_apex_reduce_sum(i * 512, 768);
    test_apex_reduce_sum(i * 128, 1024);
    test_apex_reduce_sum(i * 512, 1024);
  }
}

#ifdef USE_CUDA
TEST(ReductionOpTest, ReduceSum_batch_by_seq_by_30528) {
  test_apex_reduce_sum(4 * 128, 30528);
  test_apex_reduce_sum(4 * 512, 30528);
}
#endif

TEST(ReductionOpTest, ReduceSum_bert_selected_batch_size) {
#if defined(OPENVINO_CONFIG_MYRIAD) || defined(OPENVINO_CONFIG_VAD_M)
  test_apex_reduce_sum(85 * 128, 768, true);
  test_apex_reduce_sum(86 * 128, 768, true);
#else
  test_apex_reduce_sum(85 * 128, 768);
  test_apex_reduce_sum(86 * 128, 768);
#endif
}

TEST(ReductionOpTest, ReduceSum_apex_more) {
  std::srand(0);
  for (int64_t m = 1; m < 16; ++m) {
    for (int64_t n = 1; n < 16; ++n) {
      const auto m_ = 2 * m;
      const auto n_ = 2 * n;
      test_apex_reduce_sum(m_, n_);
    }
  }
}

TEST(ReductionOpTest, ReduceSum_int64) {
  OpTester test("ReduceSum");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<int64_t>("data", {3, 2, 2},
                         {1, 2,
                          3, 4,

                          5, 6,
                          7, 8,

                          9, 10,
                          11, 12});
  test.AddOutput<int64_t>("reduced", {1, 2, 1}, {33, 45});
  test.Run();
}

TEST(ReductionOpTest, ReduceSum_default_axes_keepdims) {
  OpTester test("ReduceSum");
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {1, 1, 1}, {78.0f});
  test.Run();
}

TEST(ReductionOpTest, ReduceSum_default_axes_do_not_keep_dims) {
  OpTester test("ReduceSum");
  test.AddAttribute("keepdims", static_cast<int64_t>(0));
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {}, {78.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: full reduce without keepDimensions is not supported with explicit batch
}

TEST(ReductionOpTest, ReduceSum_do_not_keepdims) {
  OpTester test("ReduceSum");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {1, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f});
  test.AddOutput<float>("reduced", {1, 2}, {4.0f, 6.0f});
  test.Run();
}

TEST(ReductionOpTest, ReduceSum_do_not_keepdims_2) {
  OpTester test("ReduceSum");
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3},
                       {1.0f, 2.0f, 3.0f});
  test.AddOutput<float>("reduced", {}, {6.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: full reduce without keepDimensions is not supported with explicit batch
}

TEST(ReductionOpTest, ReduceSum_keepdims) {
  OpTester test("ReduceSum");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {3, 1, 2}, {4.0f, 6.0f, 12.0f, 14.0f, 20.0f, 22.0f});
  test.Run();
}

TEST(ReductionOpTest, ReduceSum_int32_axes_input) {
  OpTester test("ReduceSum", 13, onnxruntime::kOnnxDomain);
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<int32_t>("data", {3, 2, 2},
                         {1, 2,
                          3, 4,

                          5, 6,
                          7, 8,

                          9, 10,
                          11, 12});
  test.AddInput<int64_t>("axes", {2}, std::vector<int64_t>{0, 2}, true);
  test.AddOutput<int32_t>("reduced", {1, 2, 1}, {33, 45});
  // TODO: TensorRT and OpenVINO dont support "axes" input in opset 13, re-enable after
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(ReductionOpTest, ReduceSum_do_not_keepdims_axes_input_initializer) {
  OpTester test("ReduceSum", 13, onnxruntime::kOnnxDomain);
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {1, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f});
  test.AddInput<int64_t>("axes", {1}, std::vector<int64_t>{1}, true);
  test.AddOutput<float>("reduced", {1, 2}, {4.0f, 6.0f});
  // TODO: TensorRT and OpenVINO dont support "axes" input in opset 13, re-enable after
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(ReductionOpTest, ReduceSum_do_not_keepdims_axes_input_not_initializer) {
  OpTester test("ReduceSum", 13, onnxruntime::kOnnxDomain);
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {1, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f});
  test.AddInput<int64_t>("axes", {1}, std::vector<int64_t>{1}, false);
  test.AddOutput<float>("reduced", {1, 2}, {4.0f, 6.0f});
  // TODO: TensorRT and OpenVINO dont support "axes" input in opset 13, re-enable after
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(ReductionOpTest, ReduceSum_noop_axes_input_initializer) {
  OpTester test("ReduceSum", 13, onnxruntime::kOnnxDomain);
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddAttribute("noop_with_empty_axes", (int64_t)1);
  test.AddInput<float>("data", {1, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f});
  test.AddInput<int64_t>("axes", {0}, {}, true);
  test.AddOutput<float>("reduced", {1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  // TODO: TensorRT and OpenVINO dont support "axes" input in opset 13, re-enable after
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

#if !(defined USE_TENSORRT) && !(defined USE_TVM)
TEST(ReductionOpTest, ReduceSum0DTensor) {
  OpTester test("ReduceSum");
  test.AddInput<float>("data", {}, {2});
  test.AddOutput<float>("reduced", {}, {2});
  test.Run();
}
#endif  // !(defined USE_TENSORRT) && !(defined USE_TVM)

TEST(ReductionOpTest, ReduceSumSquare) {
  OpTester test("ReduceSumSquare");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {1, 2, 1}, {247.0f, 403.f});
  test.Run();
}

TEST(ReductionOpTest, ReduceSumSquare_double) {
  OpTester test("ReduceSumSquare");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<double>("data", {3, 2, 2},
                        {1.0, 2.0,
                         3.0, 4.0,

                         5.0, 6.0,
                         7.0, 8.0,

                         9.0, 10.0,
                         11.0, 12.0});
  test.AddOutput<double>("reduced", {1, 2, 1}, {247.0, 403.});
  test.Run();
}

TEST(ReductionOpTest, ReduceSumSquare_int32) {
  OpTester test("ReduceSumSquare");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<int32_t>("data", {3, 2, 2},
                         {1, 2,
                          3, 4,

                          5, 6,
                          7, 8,

                          9, 10,
                          11, 12});
  test.AddOutput<int32_t>("reduced", {1, 2, 1}, {247, 403});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: Int32 not allowed as input to this layer
}

TEST(ReductionOpTest, ReduceSumSquare_default_axes_keepdims) {
  OpTester test("ReduceSumSquare");
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {1, 1, 1}, {650.0f});
  test.Run();
}

TEST(ReductionOpTest, ReduceSumSquare_default_axes_do_not_keep_dims) {
  OpTester test("ReduceSumSquare");
  test.AddAttribute("keepdims", static_cast<int64_t>(0));
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {}, {650.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: full reduce without keepDimensions is not supported with explicit batch
}

TEST(ReductionOpTest, ReduceSumSquare_do_not_keepdims) {
  OpTester test("ReduceSumSquare");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {3, 2}, {10.0f, 20.0f, 74.0f, 100.0f, 202.0f, 244.0f});
  test.Run();
}

TEST(ReductionOpTest, ReduceSumSquare_do_not_keepdims_2) {
  OpTester test("ReduceSumSquare");
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3},
                       {1.0f, 2.0f, 3.0f});
  test.AddOutput<float>("reduced", {}, {14.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: full reduce without keepDimensions is not supported with explicit batch
}

TEST(ReductionOpTest, ReduceSumSquare_keepdims) {
  OpTester test("ReduceSumSquare");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {3, 1, 2}, {10.0f, 20.0f, 74.0f, 100.0f, 202.0f, 244.0f});
  test.Run();
}

#if !(defined USE_TENSORRT) && !(defined USE_TVM)
TEST(ReductionOpTest, ReduceSumSquare0DTensor) {
  OpTester test("ReduceSumSquare");
  test.AddInput<float>("data", {}, {2});
  test.AddOutput<float>("reduced", {}, {4});
  test.Run();
}
#endif  // !(defined USE_TENSORRT) && !(defined USE_TVM)

TEST(ReductionOpTest, ReduceProd_default_axes_keepdims) {
  OpTester test("ReduceProd");
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {1, 1, 1}, {479001600.f});
  test.Run();
}

TEST(ReductionOpTest, ReduceProd_default_axes_do_not_keep_dims) {
  OpTester test("ReduceProd");
  test.AddAttribute("keepdims", static_cast<int64_t>(0));
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {}, {479001600.f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: full reduce without keepDimensions is not supported with explicit batch
}

TEST(ReductionOpTest, ReduceProd_do_not_keepdims) {
  OpTester test("ReduceProd");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {3, 2}, {3.f, 8.f, 35.f, 48.f, 99.f, 120.f});
  test.Run();
}

TEST(ReductionOpTest, ReduceProd_do_not_keepdims_2) {
  OpTester test("ReduceProd");
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3},
                       {1.0f, 2.0f, 3.0f});
  test.AddOutput<float>("reduced", {}, {6.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: full reduce without keepDimensions is not supported with explicit batch
}

TEST(ReductionOpTest, ReduceProd_keepdims) {
  OpTester test("ReduceProd");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {3, 1, 2}, {3.f, 8.f, 35.f, 48.f, 99.f, 120.f});
  test.Run();
}

TEST(ReductionOpTest, ReduceProd) {
  OpTester test("ReduceProd");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<float>("reduced", {1, 2, 1}, {5400.f, 88704.f});
  test.Run();
}

TEST(ReductionOpTest, ReduceProd_int32) {
  OpTester test("ReduceProd");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddInput<int32_t>("data", {3, 2, 2},
                         {1, 2,
                          3, 4,

                          5, 6,
                          7, 8,

                          9, 10,
                          11, 12});
  test.AddOutput<int32_t>("reduced", {1, 2, 1}, {5400, 88704});
  test.Run();
}

TEST(ReductionOpTest, ReduceProd_int64) {
  OpTester test("ReduceProd");
  test.AddAttribute("axes", std::vector<int64_t>{0, 2});
  test.AddInput<int64_t>("data", {3, 2, 2},
                         {1, 2,
                          3, 4,

                          5, 6,
                          7, 8,

                          9, 10,
                          11, 12});
  test.AddOutput<int64_t>("reduced", {1, 2, 1}, {5400, 88704});
  test.Run();
}

#if !(defined USE_TENSORRT) && !(defined USE_TVM)
TEST(ReductionOpTest, ReduceProd0DTensor) {
  OpTester test("ReduceProd");
  test.AddInput<float>("data", {}, {2});
  test.AddOutput<float>("reduced", {}, {2});
  test.Run();
}
#endif  // (!defined USE_TENSORRT) && (!defined USE_TVM)

TEST(ReductionOpTest, ArgMax) {
  OpTester test("ArgMax");
  test.AddAttribute("axis", (int64_t)1);
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<int64_t>("reduced", {3, 1, 2},
                          {1, 1,
                           1, 1,
                           1, 1});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: axis must be 0
}

TEST(ReductionOpTest, ArgMax_Double_Type) {
  OpTester test("ArgMax", 11);
  test.AddAttribute("axis", static_cast<int64_t>(1));
  test.AddAttribute("keepdims", static_cast<int64_t>(1));
  test.AddInput<double>("data", {3, 2, 2},
                        {1.0, 2.0,
                         3.0, 4.0,

                         5.0, 6.0,
                         7.0, 8.0,

                         9.0, 10.0,
                         11.0, 12.0});
  test.AddOutput<int64_t>("reduced", {3, 1, 2},
                          {1, 1,
                           1, 1,
                           1, 1});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: axis must be 0
}

TEST(ReductionOpTest, ArgMax_do_not_keepdims) {
  OpTester test("ArgMax");
  test.AddAttribute("axis", (int64_t)1);
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<int64_t>("reduced", {3, 2},
                          {1, 1,
                           1, 1,
                           1, 1});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: axis must be 0
}

TEST(ReductionOpTest, ArgMax_do_not_keepdims_2) {
  OpTester test("ArgMax");
  test.AddAttribute("axis", (int64_t)0);
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3},
                       {1.0f, 2.0f, 3.0f});
  test.AddOutput<int64_t>("reduced", {},
                          {2});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: node1: at least 2 dimensions are required for input
}

TEST(ReductionOpTest, ArgMax_int32) {
  OpTester test("ArgMax");
  test.AddAttribute("axis", (int64_t)1);
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<int32_t>("data", {3, 2, 2},
                         {1, 2,
                          3, 4,

                          5, 6,
                          7, 8,

                          9, 10,
                          11, 12});
  test.AddOutput<int64_t>("reduced", {3, 1, 2},
                          {1, 1,
                           1, 1,
                           1, 1});
  test.Run();
}

TEST(ReductionOpTest, ArgMax_int32_last_index_nodups) {
  OpTester test("ArgMax", 12);
  test.AddAttribute("axis", (int64_t)1);
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddAttribute("select_last_index", (int64_t)1);

  test.AddInput<int32_t>("data", {3, 2, 2},
                         {2, 2,
                          3, 4,

                          5, 6,
                          7, 8,

                          10, 10,
                          11, 12});
  test.AddOutput<int64_t>("reduced", {3, 1, 2},
                          {1, 1,
                           1, 1,
                           1, 1});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(ReductionOpTest, ArgMax_int32_last_index_dups) {
  OpTester test("ArgMax", 12);
  test.AddAttribute("axis", (int64_t)1);
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddAttribute("select_last_index", (int64_t)1);

  test.AddInput<int32_t>("data", {3, 2, 2},
                         {2, 4,
                          3, 4,

                          8, 6,
                          7, 8,

                          9, 13,
                          11, 12});
  test.AddOutput<int64_t>("reduced", {3, 1, 2},
                          {1, 1,
                           0, 1,
                           1, 0});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(ReductionOpTest, ArgMax_int32_neg_axis) {
  OpTester test("ArgMax");
  test.AddAttribute("axis", (int64_t)(-2));
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<int32_t>("data", {3, 2, 2},
                         {1, 2,
                          3, 4,

                          5, 6,
                          7, 8,

                          9, 10,
                          11, 12});
  test.AddOutput<int64_t>("reduced", {3, 1, 2},
                          {1, 1,
                           1, 1,
                           1, 1});

  test.Run();
}

TEST(ReductionOpTest, ArgMax2D) {
  OpTester test("ArgMax");
  test.AddAttribute("axis", (int64_t)1);
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2},
                       {1.0f, 2.0f,
                        6.0f, 5.0f,
                        9.0f, 10.0f});
  test.AddOutput<int64_t>("reduced", {3, 1},
                          {1, 0, 1});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: axis must be 0
}

TEST(ReductionOpTest, ArgMax2D_select_last) {
  OpTester test("ArgMax", 12);
  test.AddAttribute("axis", (int64_t)1);
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddAttribute("select_last_index", (int64_t)1);
  test.AddInput<float>("data", {3, 2},
                       {1.0f, 2.0f,
                        6.0f, 5.0f,
                        9.0f, 10.0f});
  test.AddOutput<int64_t>("reduced", {3, 1},
                          {1, 0, 1});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(ReductionOpTest, ArgMin) {
  OpTester test("ArgMin");
  test.AddAttribute("axis", (int64_t)0);
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<int64_t>("reduced", {1, 2, 2},
                          {0, 0,
                           0, 0});
  test.Run();
}

TEST(ReductionOpTest, ArgMin_Double_Type) {
  OpTester test("ArgMin", 11);
  test.AddAttribute("axis", (int64_t)1);
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<double>("data", {3, 2, 2},
                        {1.0, 2.0,
                         3.0, 4.0,

                         5.0, 6.0,
                         7.0, 8.0,

                         9.0, 10.0,
                         11.0, 12.0});
  test.AddOutput<int64_t>("reduced", {3, 1, 2},
                          {0, 0,
                           0, 0,
                           0, 0});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: axis must be 0
}

TEST(ReductionOpTest, ArgMin_Double_Precision) {
  OpTester test("ArgMin", 11);
  test.AddAttribute("axis", (int64_t)1);
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<double>("data", {3, 2, 2},
                        {1.0 + 1e-10, 3.0,
                         1.0, 3.0 - 0.5e-10,

                         1.0, 3.0 + 1e-10,
                         1 + 1e-10, 3 - 1e-10,

                         1.0 + 2e-10, 3.0 - 1e-10,
                         1 - 1e-10, 3.0});
  test.AddOutput<int64_t>("reduced", {3, 1, 2},
                          {1, 1,
                           0, 1,
                           1, 0});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: axis must be 0
}

TEST(ReductionOpTest, ArgMin_do_not_keepdims) {
  OpTester test("ArgMin");
  test.AddAttribute("axis", (int64_t)0);
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddOutput<int64_t>("reduced", {2, 2},
                          {0, 0,
                           0, 0});
  test.Run();
}

TEST(ReductionOpTest, ArgMin_do_not_keepdims_2) {
  OpTester test("ArgMin");
  test.AddAttribute("axis", (int64_t)0);
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3},
                       {1.0f, 2.0f, 3.0f});
  test.AddOutput<int64_t>("reduced", {}, {0});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: node1: at least 2 dimensions are required for input
}

TEST(ReductionOpTest, ArgMin_do_not_keepdims_2_select_last) {
  OpTester test("ArgMin", 12);
  test.AddAttribute("axis", (int64_t)0);
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddAttribute("select_last_index", (int64_t)1);
  test.AddInput<float>("data", {3},
                       {1.0f, 2.0f, 3.0f});
  test.AddOutput<int64_t>("reduced", {}, {0});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(ReductionOpTest, ArgMin_int32) {
  OpTester test("ArgMin");
  test.AddAttribute("axis", (int64_t)0);
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<int32_t>("data", {3, 2, 2},
                         {1, 2,
                          3, 4,

                          5, 6,
                          7, 8,

                          9, 10,
                          11, 12});
  test.AddOutput<int64_t>("reduced", {2, 2},
                          {0, 0,
                           0, 0});
  test.Run();
}

TEST(ReductionOpTest, ArgMin_int32_select_last) {
  OpTester test("ArgMin", 12);
  test.AddAttribute("axis", (int64_t)0);
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddAttribute("select_last_index", (int64_t)1);
  test.AddInput<int32_t>("data", {3, 2, 2},
                         {1, 2,
                          3, 4,

                          1, 6,
                          7, 8,

                          9, 10,
                          11, 12});
  test.AddOutput<int64_t>("reduced", {2, 2},
                          {1, 0,
                           0, 0});

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(ReductionOpTest, ArgMin_int32_neg_axis) {
  OpTester test("ArgMin");
  test.AddAttribute("axis", (int64_t)(-3));
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<int32_t>("data", {3, 2, 2},
                         {1, 2,
                          3, 4,

                          5, 6,
                          7, 8,

                          9, 10,
                          11, 12});
  test.AddOutput<int64_t>("reduced", {2, 2},
                          {0, 0,
                           0, 0});

  test.Run();
}

// test that PrepareForReduce handles this case. Called by all reduction ops so any op can be used in the test
TEST(ReductionOpTest, ReduceDimWithZero) {
  auto run = [](OpTester& tester, const std::string& error_msg = "") {
    auto expect = error_msg.empty() ? OpTester::ExpectResult::kExpectSuccess
                                    : OpTester::ExpectResult::kExpectFailure;

    // exclude OpenVINO and TensorRT as this isn't handled by those EPs
    tester.Run(expect, error_msg, {kTensorrtExecutionProvider, kOpenVINOExecutionProvider, kNupharExecutionProvider});
  };

  // reduce on all axes keeping dims. should allow the 0 to be the reduced value
  OpTester test("ReduceSum", 10);
  test.AddAttribute("keepdims", int64_t(1));
  test.AddShapeToTensorData(true, 1);  // make second dim symbolic so that we don't break during shape inferencing
  test.AddInput<float>("data", {3, 0, 2}, {});
  test.AddOutput<float>("reduced", {1, 0, 1}, {});
  run(test);

  // reduction without keeping dims on all axes. can't reduce on an axis with value of 0
  OpTester test2("ReduceSum", 10);
  test2.AddAttribute("keepdims", int64_t(0));
  test2.AddShapeToTensorData(true, 1);
  test2.AddInput<float>("data", {3, 0, 2}, {});
  test2.AddOutput<float>("reduced", {}, {0.f});
  run(test2,
      "Can't reduce on dim with value of 0 if 'keepdims' is false. "
      "Invalid output shape would be produced. input_shape:{3,0,2}");

  // reduction is possible without keeping dims if we only reduce on non-zero dims
  OpTester test3("ReduceSum", 10);
  test3.AddAttribute("keepdims", int64_t(0));
  test3.AddAttribute("axes", std::vector<int64_t>{2});
  test3.AddShapeToTensorData(true, 1);
  test3.AddInput<float>("data", {3, 0, 2}, {});
  test3.AddOutput<float>("reduced", {3, 0}, {});
  run(test3);
}

TEST(ReductionOpTest, ReduceInfMax) {
  OpTester test("ReduceMax");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {6, 2},
                       {1.0f, FLOAT_NINF,
                        FLOAT_NINF, 4.0f,
                        FLOAT_INF, FLOAT_NINF,
                        FLOAT_NINF, FLOAT_INF,
                        1.0f, FLOAT_INF,
                        FLOAT_INF, 4.0f});
  test.AddOutput<float>("reduced", {6},
                        {1.0f, 4.0f,
                         FLOAT_INF, FLOAT_INF,
                         FLOAT_INF, FLOAT_INF});
  test.Run();
}

TEST(ReductionOpTest, ReduceInfMin) {
  OpTester test("ReduceMin");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {6, 2},
                       {1.0f, FLOAT_INF,
                        FLOAT_INF, 4.0f,
                        FLOAT_INF, FLOAT_NINF,
                        FLOAT_NINF, FLOAT_INF,
                        1.0f, FLOAT_NINF,
                        FLOAT_NINF, 4.0f});
  test.AddOutput<float>("reduced", {6},
                        {1.0f, 4.0f,
                         FLOAT_NINF, FLOAT_NINF,
                         FLOAT_NINF, FLOAT_NINF});
  test.Run();
}

TEST(ReductionOpTest, ReduceInfSum) {
  OpTester test("ReduceSum");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {6, 2},
                       {1.0f, FLOAT_INF,
                        FLOAT_INF, 4.0f,
                        FLOAT_INF, FLOAT_NINF,
                        FLOAT_NINF, FLOAT_INF,
                        1.0f, FLOAT_NINF,
                        FLOAT_NINF, 4.0f});
  test.AddOutput<float>("reduced", {6},
                        {FLOAT_INF, FLOAT_INF,
                         std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(),
                         FLOAT_NINF, FLOAT_NINF});
  test.Run();
}

TEST(ReductionOpTest, ReduceInfLogSum) {
  OpTester test("ReduceLogSum");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {6, 2},
                       {1.0f, FLOAT_INF,
                        FLOAT_INF, 1.0f,
                        FLOAT_INF, FLOAT_NINF,
                        FLOAT_NINF, FLOAT_INF,
                        1.0f, FLOAT_NINF,
                        FLOAT_NINF, 1.0f});
  test.AddOutput<float>("reduced", {6},
                        {FLOAT_INF, FLOAT_INF,
                         -std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(),
                         std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN()});
  test.Run();
}

TEST(ReductionOpTest, ReduceInfLogSumExp) {
  OpTester test("ReduceLogSumExp");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {2, 2}, {1.0f, FLOAT_NINF, FLOAT_NINF, 1.0f});
  test.AddOutput<float>("reduced", {2}, {1.0f, 1.0f});
  test.Run();
}

TEST(ReductionOpTest, ReduceInfLogSumExp_double) {
  OpTester test("ReduceLogSumExp");
  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<double>("data", {2, 2}, {1.0, DOUBLE_NINF, DOUBLE_NINF, 1.0});
  test.AddOutput<double>("reduced", {2}, {1.0, 1.0});
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
