// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <type_traits>
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/providers/cpu/reduction/reduction_test_cases.h"

namespace onnxruntime {
namespace test {

// Disable TensorRT on some of the tests because the limit in its parser: axis >=0 && axis < nbDims

template <typename OutT>
void TestReduceOp(const std::string& op,
                  const std::vector<int64_t>& input_dims,
                  const std::vector<float>& data,
                  const std::vector<int64_t>& axes,
                  int64_t keepdims,
                  const std::vector<int64_t>& expected_dims,
                  const std::vector<OutT>& expected_data)

{
  OpTester test(op.c_str());
  if (!axes.empty()) {
    if (op.compare("ArgMax") == 0 || op.compare("ArgMin") == 0)
      test.AddAttribute("axis", axes[0]);
    else
      test.AddAttribute("axes", axes);
  }
  test.AddAttribute("keepdims", keepdims);
  test.AddInput<float>("data", input_dims, data);
  test.AddOutput<OutT>("reduced", expected_dims, expected_data);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kCudaExecutionProvider, kTensorrtExecutionProvider}); //TensorRT: result differs
}

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
      TestReduceOp<int64_t>(a.first, input_dims, input_data, attributes.axes_, attributes.keep_dims_,
                            expected_dims, expected_values);
    } else {
      const std::vector<float> expected_values = std::get<2>(a.second);
      TestReduceOp<float>(a.first, input_dims, input_data, attributes.axes_, attributes.keep_dims_,
                          expected_dims, expected_values);
    }
  }
}

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

TEST(ReductionOpTest, ReduceL1_do_not_keepdims) {
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
  test.Run();
}

TEST(ReductionOpTest, ReduceL1_do_not_keepdims_2) {
  OpTester test("ReduceL1");
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3},
                       {1.0f, 2.0f, 3.0f});
  test.AddOutput<float>("reduced", {}, {6.0f});
  test.Run();
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
  test.Run();
}

TEST(ReductionOpTest, ReduceL2_do_not_keepdims_2) {
  OpTester test("ReduceL2");
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3},
                       {1.0f, 2.0f, 3.0f});
  test.AddOutput<float>("reduced", {}, {3.741657387f});
  test.Run();
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
  test.Run();
}

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
  test.Run();
}

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
  test.Run();
}

TEST(ReductionOpTest, ReduceLogSumExp_do_not_keepdims_2) {
  OpTester test("ReduceLogSumExp");
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3},
                       {1.0f, 2.0f, 3.0f});
  test.AddOutput<float>("reduced", {}, {3.40760596f});
  test.Run();
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
  test.Run();
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
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider}); //TensorRT: axis must be 0
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
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: axis must be 0
}

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
  test.Run();
}

TEST(ReductionOpTest, ReduceMean_do_not_keepdims_2) {
  OpTester test("ReduceMean");
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3},
                       {1.0f, 2.0f, 3.0f});
  test.AddOutput<float>("reduced", {}, {2.0f});
  test.Run();
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
  test.Run();
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
  int64_t m, int64_t n) {
  OpTester test("ReduceSum");
  // Input tensor.
  std::vector<float> X(m * n, 0.0f);
  // Reduced tensor.
  std::vector<float> Y(n, 0.0f);
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      const float value = float(i + j);
      X[i * n + j] = value; 
      Y[j] += value;
    }
  }

  test.AddAttribute("keepdims", (int64_t)0);
  test.AddAttribute("axes", std::vector<int64_t>{0});
  test.AddInput<float>("data", {m, n}, X);
  test.AddOutput<float>("reduced", {n}, Y);

  test.Run();
}

TEST(ReductionOpTest, ReduceSum_apex_matrix) {
  for (int64_t m = 1; m < 2048; m *= 8) {
    for (int64_t n = 1; n < 2048; n *= 8)  {
      if (m * n <= 32768) {
        test_apex_reduce_sum(m, n);
      }
    }
  }
}

TEST(ReductionOpTest, ReduceSum_apex_bert) {
  test_apex_reduce_sum(6, 2);
  test_apex_reduce_sum(8, 2);
  test_apex_reduce_sum(6 * 128, 128);
  test_apex_reduce_sum(8 * 128, 128);
  test_apex_reduce_sum(6 * 384, 128);
  test_apex_reduce_sum(8 * 384, 128);
  test_apex_reduce_sum(6 * 512, 128);
  test_apex_reduce_sum(8 * 512, 128);
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

TEST( ReductionOpTest, ReduceSum_int64 )
{
   OpTester test( "ReduceSum" );
   test.AddAttribute( "axes", std::vector<int64_t>{0, 2} );
   test.AddAttribute( "keepdims", ( int64_t ) 1 );
   test.AddInput<int64_t>( "data", { 3, 2, 2 },
      { 1, 2,
       3, 4,

       5, 6,
       7, 8,

       9, 10,
       11, 12 } );
   test.AddOutput<int64_t>( "reduced", { 1, 2, 1 }, { 33, 45 } );
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
  test.Run();
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
  test.Run();
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
  test.Run();
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
  test.Run();
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
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider}); //TensorRT: axis must be 0
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
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider}); //TensorRT: axis must be 0
}

TEST(ReductionOpTest, ArgMax_do_not_keepdims_2) {
  OpTester test("ArgMax");
  test.AddAttribute("axis", (int64_t)0);
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {3},
                       {1.0f, 2.0f, 3.0f});
  test.AddOutput<int64_t>("reduced", {},
                          {2});
  test.Run();
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
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider}); //TensorRT: axis must be 0
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
  test.Run();
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

#ifdef USE_CUDA
void test_all_1d_true(size_t size) {
  std::unique_ptr<bool[]> p_data(new bool[size]);
  for (size_t i = 0; i < size; ++i) {
      p_data[i] = true;
  }

  OpTester test("All", 9);
  test.AddInput<bool>("data", {static_cast<int64_t>(size)}, p_data.get(), size);
  test.AddOutput<bool>("result", {}, {true});
  test.Run();
}

void test_all_1d_false(size_t size) {
  std::unique_ptr<bool[]> p_data(new bool[size]);
  for (size_t i = 0; i < size; ++i) {
      p_data[i] = false;
  }

  OpTester test("All", 9);
  test.AddInput<bool>("data", {static_cast<int64_t>(size)}, p_data.get(), size);
  test.AddOutput<bool>("result", {}, {false});
  test.Run();
}

void test_all_1d_first_false(size_t size) {
  std::unique_ptr<bool[]> p_data(new bool[size]);
  for (size_t i = 0; i < size; ++i) {
      p_data[i] = true;
  }
  p_data[0] = false;

  OpTester test("All", 9);
  test.AddInput<bool>("data", {static_cast<int64_t>(size)}, p_data.get(), size);
  test.AddOutput<bool>("result", {}, {false});
  test.Run();
}

void test_all_1d_last_false(size_t size) {
  std::unique_ptr<bool[]> p_data(new bool[size]);
  for (size_t i = 0; i < size; ++i) {
      p_data[i] = true;
  }
  p_data[size - 1] = false;

  OpTester test("All", 9);
  test.AddInput<bool>("data", {static_cast<int64_t>(size)}, p_data.get(), size);
  test.AddOutput<bool>("result", {}, {false});
  test.Run();
}

TEST(AllOpTest, All_1d_small) {
  for (size_t i = 1; i < 256; ++i) {
    test_all_1d_false(i);
    test_all_1d_first_false(i);
    test_all_1d_last_false(i);
    test_all_1d_true(i);
  }
}

TEST(AllOpTest, All_1d_large) {
  std::vector<int> centers = {1228, 8877};
  for (auto it = centers.begin(); it != centers.end(); ++it) {
    for (int j = -32; j <= 32; ++j) {
      test_all_1d_first_false(*it + j);
      test_all_1d_last_false(*it + j);
    }
  }
}

#endif

}  // namespace test
}  // namespace onnxruntime
