// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/common/tensor_op_test_utils.h"

namespace onnxruntime {
namespace test {

template <typename T>
class ConcatOpTest : public ::testing::Test {
};

using ConcatOpTestTypes = ::testing::Types<float, MLFloat16>;
TYPED_TEST_SUITE(ConcatOpTest, ConcatOpTestTypes);

// Some of the tests can't run on TensorrtExecutionProvider because of unsupported data types or limits
// in its parser: axis >=0 && axis < nbDims. Those Tests will fallback to other EPs

TEST(ConcatOpTest, Concat1D_string) {
  OpTester test("Concat");
  test.AddAttribute("axis", int64_t{0});

  test.AddInput<std::string>("input1", {1}, {"1"});
  test.AddInput<std::string>("input2", {2}, {"2", "3"});
  test.AddInput<std::string>("input3", {4}, {"4", "5", "6", "7"});
  test.AddOutput<std::string>("concat_result", {7}, {"1", "2", "3", "4", "5", "6", "7"});
  test.Run();
}

TEST(ConcatOpTest, Concat1D_int32) {
  OpTester test("Concat");
  test.AddAttribute("axis", int64_t{0});

  test.AddInput<int32_t>("input1", {1}, {1});
  test.AddInput<int32_t>("input2", {2}, {2, 3});
  test.AddInput<int32_t>("input3", {4}, {4, 5, 6, 7});
  test.AddOutput<int32_t>("concat_result", {7}, {1, 2, 3, 4, 5, 6, 7});
  test.Run();
}

TEST(ConcatOpTest, Concat1D_int32_negative_axis) {
  OpTester test("Concat");
  test.AddAttribute("axis", int64_t{-1});

  test.AddInput<int32_t>("input1", {1}, {1});
  test.AddInput<int32_t>("input2", {2}, {2, 3});
  test.AddInput<int32_t>("input3", {4}, {4, 5, 6, 7});
  test.AddOutput<int32_t>("concat_result", {7}, {1, 2, 3, 4, 5, 6, 7});
  test.Run();
}

TEST(ConcatOpTest, Concat1D_1) {
  OpTester test("Concat");
  test.AddAttribute("axis", int64_t{0});

  test.AddInput<float>("input1", {1}, {1.0f});
  test.AddInput<float>("input2", {2}, {2.0f, 3.0f});
  test.AddInput<float>("input3", {4}, {4.0f, 5.0f, 6.0f, 7.0f});
  test.AddOutput<float>("concat_result", {7}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f});
  test.Run();
}

TEST(ConcatOpTest, Concat1D_2) {
  OpTester test("Concat");
  test.AddAttribute("axis", int64_t{0});

  test.AddInput<float>("input1", {1}, {1.0f});
  test.AddInput<float>("input2", {2}, {2.0f, 3.0f});
  test.AddInput<float>("input3", {0}, {});
  test.AddOutput<float>("concat_result", {3}, {1.0f, 2.0f, 3.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider,  // TensorRT: no support for dynamic shape tensor
            kNnapiExecutionProvider,     // NNAPI: concat does not support 0 size input
            kQnnExecutionProvider});     // QNN: not support dynamic shape tensor
}

TYPED_TEST(ConcatOpTest, Concat2D_1) {
  OpTester test("Concat");
  test.AddAttribute("axis", int64_t{0});

  std::vector<int64_t> dims{1, 4};
  test.AddInput<TypeParam>("input1", dims, GetTypedArray<TypeParam>({11.0f, 12.0f, 13.0f, 14.0f}));
  test.AddInput<TypeParam>("input2", dims, GetTypedArray<TypeParam>({21.0f, 22.0f, 23.0f, 24.0f}));
  test.AddInput<TypeParam>("input3", dims, GetTypedArray<TypeParam>({31.0f, 32.0f, 33.0f, 34.0f}));
  test.AddOutput<TypeParam>("concat_result", {3, 4},
                            GetTypedArray<TypeParam>({11.0f, 12.0f, 13.0f, 14.0f,
                                                      21.0f, 22.0f, 23.0f, 24.0f,
                                                      31.0f, 32.0f, 33.0f, 34.0f}));
  test.Run();
}

TYPED_TEST(ConcatOpTest, Concat2D_2) {
  OpTester test("Concat");
  test.AddAttribute("axis", int64_t{1});

  std::vector<int64_t> dims{4, 1};
  test.AddInput<TypeParam>("input1", dims, GetTypedArray<TypeParam>({11.0f, 21.0f, 31.0f, 41.0f}));
  test.AddInput<TypeParam>("input2", {4, 2}, GetTypedArray<TypeParam>({12.0f, 13.0f, 22.0f, 23.0f, 32.0f, 33.0f, 42.0f, 43.0f}));
  test.AddInput<TypeParam>("input3", dims, GetTypedArray<TypeParam>({14.0f, 24.0f, 34.0f, 44.0f}));
  test.AddOutput<TypeParam>("concat_result", {4, 4},
                            GetTypedArray<TypeParam>({11.0f, 12.0f, 13.0f, 14.0f,
                                                      21.0f, 22.0f, 23.0f, 24.0f,
                                                      31.0f, 32.0f, 33.0f, 34.0f,
                                                      41.0f, 42.0f, 43.0f, 44.0f}));
  test.Run();
}

TEST(ConcatOpTest, Concat2D_3) {
  OpTester test("Concat");
  test.AddAttribute("axis", int64_t{1});

  test.AddInput<float>("input1", {1, 0}, {});
  test.AddInput<float>("input2", {1, 0}, {});
  test.AddInput<float>("input3", {1, 0}, {});
  test.AddOutput<float>("concat_result", {1, 0}, {});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider,  // TensorRT: no support for dynamic shape tensor
            kNnapiExecutionProvider,     // NNAPI: concat does not support 0 size input
            kQnnExecutionProvider});     // QNN: not support dynamic shape tensor
}

// Test Concat of tensors when one of them has dynamic shape
// This is useful for testing EP's own shape inferencing, such as NNAPI EP
TEST(ConcatOpTest, Concat2D_4) {
  OpTester test("Concat");
  test.AddAttribute("axis", int64_t{1});

  std::vector<int64_t> dims{4, 1};
  std::vector<std::string> dim_params{"batch", "seq"};
  test.AddInput<float>("input1", dims, {11.0f, 21.0f, 31.0f, 41.0f});
  test.AddInput<float>("input2", {4, 2}, {12.0f, 13.0f, 22.0f, 23.0f, 32.0f, 33.0f, 42.0f, 43.0f}, false, &dim_params);
  test.AddInput<float>("input3", dims, {14.0f, 24.0f, 34.0f, 44.0f});
  test.AddOutput<float>("concat_result", {4, 4},
                        {11.0f, 12.0f, 13.0f, 14.0f,
                         21.0f, 22.0f, 23.0f, 24.0f,
                         31.0f, 32.0f, 33.0f, 34.0f,
                         41.0f, 42.0f, 43.0f, 44.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider});  // TensorRT: no support for dynamic shape tensor
}

TEST(ConcatOpTest, Concat2D_5) {
  OpTester test("Concat");
  test.AddAttribute("axis", int64_t{0});

  std::vector<int64_t> dims{2, 2};
  test.AddInput<double>("input1", dims,
                        {111.0f, 112.0f,
                         121.0f, 122.0f});
  test.AddInput<double>("input2", dims,
                        {211.0f, 212.0f,
                         221.0f, 222.0f});
  test.AddOutput<double>("concat_result", {4, 2},
                         {111.0f, 112.0f,
                          121.0f, 122.0f,
                          211.0f, 212.0f,
                          221.0f, 222.0f});
  test.Run();
}

TEST(ConcatOpTest, Concat3D_1) {
  OpTester test("Concat");
  test.AddAttribute("axis", int64_t{0});

  std::vector<int64_t> dims{1, 3, 3};
  test.AddInput<float>("input1", dims,
                       {111.0f, 112.0f, 113.0f,
                        121.0f, 122.0f, 123.0f,
                        131.0f, 132.0f, 133.0f});
  test.AddInput<float>("input2", dims,
                       {211.0f, 212.0f, 213.0f,
                        221.0f, 222.0f, 223.0f,
                        231.0f, 232.0f, 233.0f});
  test.AddInput<float>("input3", dims,
                       {311.0f, 312.0f, 313.0f,
                        321.0f, 322.0f, 323.0f,
                        331.0f, 332.0f, 333.0f});
  test.AddOutput<float>("concat_result", {3, 3, 3},
                        {111.0f, 112.0f, 113.0f,
                         121.0f, 122.0f, 123.0f,
                         131.0f, 132.0f, 133.0f,

                         211.0f, 212.0f, 213.0f,
                         221.0f, 222.0f, 223.0f,
                         231.0f, 232.0f, 233.0f,

                         311.0f, 312.0f, 313.0f,
                         321.0f, 322.0f, 323.0f,
                         331.0f, 332.0f, 333.0f});
  test.Run();
}

TEST(ConcatOpTest, Concat3D_1_negative_axis) {
  OpTester test("Concat");
  test.AddAttribute("axis", int64_t{-3});

  std::vector<int64_t> dims{1, 3, 3};
  test.AddInput<float>("input1", dims,
                       {111.0f, 112.0f, 113.0f,
                        121.0f, 122.0f, 123.0f,
                        131.0f, 132.0f, 133.0f});
  test.AddInput<float>("input2", dims,
                       {211.0f, 212.0f, 213.0f,
                        221.0f, 222.0f, 223.0f,
                        231.0f, 232.0f, 233.0f});
  test.AddInput<float>("input3", dims,
                       {311.0f, 312.0f, 313.0f,
                        321.0f, 322.0f, 323.0f,
                        331.0f, 332.0f, 333.0f});
  test.AddOutput<float>("concat_result", {3, 3, 3},
                        {111.0f, 112.0f, 113.0f,
                         121.0f, 122.0f, 123.0f,
                         131.0f, 132.0f, 133.0f,

                         211.0f, 212.0f, 213.0f,
                         221.0f, 222.0f, 223.0f,
                         231.0f, 232.0f, 233.0f,

                         311.0f, 312.0f, 313.0f,
                         321.0f, 322.0f, 323.0f,
                         331.0f, 332.0f, 333.0f});
  test.Run();
}

TEST(ConcatOpTest, Concat3D_2) {
  OpTester test("Concat");
  test.AddAttribute("axis", int64_t{1});

  std::vector<int64_t> dims{3, 1, 3};
  test.AddInput<float>("input1", dims,
                       {111.0f, 112.0f, 113.0f,
                        211.0f, 212.0f, 213.0f,
                        311.0f, 312.0f, 313.0f});
  test.AddInput<float>("input2", dims,
                       {121.0f, 122.0f, 123.0f,
                        221.0f, 222.0f, 223.0f,
                        321.0f, 322.0f, 323.0f});
  test.AddInput<float>("input3", dims,
                       {131.0f, 132.0f, 133.0f,
                        231.0f, 232.0f, 233.0f,
                        331.0f, 332.0f, 333.0f});
  test.AddOutput<float>("concat_result", {3, 3, 3},
                        {111.0f, 112.0f, 113.0f,
                         121.0f, 122.0f, 123.0f,
                         131.0f, 132.0f, 133.0f,

                         211.0f, 212.0f, 213.0f,
                         221.0f, 222.0f, 223.0f,
                         231.0f, 232.0f, 233.0f,

                         311.0f, 312.0f, 313.0f,
                         321.0f, 322.0f, 323.0f,
                         331.0f, 332.0f, 333.0f});
  test.Run();
}

TEST(ConcatOpTest, Concat3D_3) {
  OpTester test("Concat");
  test.AddAttribute("axis", int64_t{1});

  std::vector<int64_t> dims{2, 2, 2};
  test.AddInput<float>("input1", dims,
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f});
  test.AddInput<float>("input2", dims,
                       {9.0f, 10.0f,
                        11.0f, 12.0f,

                        13.0f, 14.0f,
                        15.0f, 16.0f});
  test.AddOutput<float>("concat_result", {2, 4, 2},
                        {1.0f, 2.0f,
                         3.0f, 4.0f,
                         9.0f, 10.0f,
                         11.0f, 12.0f,

                         5.0f, 6.0f,
                         7.0f, 8.0f,
                         13.0f, 14.0f,
                         15.0f, 16.0f});
  test.Run();
}

TEST(ConcatOpTest, Concat3D_4) {
  OpTester test("Concat");
  test.AddAttribute("axis", int64_t{2});

  test.AddInput<float>("input1", {1, 3, 3},
                       {111.0f, 112.0f, 113.0f,
                        121.0f, 122.0f, 123.0f,
                        131.0f, 132.0f, 133.0f});
  test.AddInput<float>("input2", {1, 3, 4},
                       {211.0f, 212.0f, 213.0f, 214.0f,
                        221.0f, 222.0f, 223.0f, 224.0f,
                        231.0f, 232.0f, 233.0f, 234.0f});
  test.AddInput<float>("input3", {1, 3, 5},
                       {311.0f, 312.0f, 313.0f, 314.0f, 315.0f,
                        321.0f, 322.0f, 323.0f, 324.0f, 325.0f,
                        331.0f, 332.0f, 333.0f, 334.0f, 335.0f});
  test.AddOutput<float>("concat_result", {1, 3, 12},
                        {111.0f, 112.0f, 113.0f, 211.0f, 212.0f, 213.0f, 214.0f, 311.0f, 312.0f, 313.0f, 314.0f, 315.0f,
                         121.0f, 122.0f, 123.0f, 221.0f, 222.0f, 223.0f, 224.0f, 321.0f, 322.0f, 323.0f, 324.0f, 325.0f,
                         131.0f, 132.0f, 133.0f, 231.0f, 232.0f, 233.0f, 234.0f, 331.0f, 332.0f, 333.0f, 334.0f, 335.0f});
  test.Run();
}

TEST(ConcatOpTest, Concat3D_5) {
  OpTester test("Concat");
  test.AddAttribute("axis", int64_t{1});

  test.AddInput<float>("input1", {1, 3, 2},
                       {111.0f, 112.0f,
                        121.0f, 122.0f,
                        131.0f, 132.0f});
  test.AddInput<float>("input2", {1, 2, 2},
                       {211.0f, 212.0f,
                        221.0f, 222.0f});
  test.AddInput<float>("input3", {1, 4, 2},
                       {311.0f, 312.0f,
                        321.0f, 322.0f,
                        331.0f, 332.0f,
                        341.0f, 342.0f});
  test.AddOutput<float>("concat_result", {1, 9, 2},
                        {111.0f, 112.0f,
                         121.0f, 122.0f,
                         131.0f, 132.0f,
                         211.0f, 212.0f,
                         221.0f, 222.0f,
                         311.0f, 312.0f,
                         321.0f, 322.0f,
                         331.0f, 332.0f,
                         341.0f, 342.0f});
  test.Run();
}

TEST(ConcatOpTest, Concat4D_1) {
  OpTester test("Concat");
  test.AddAttribute("axis", int64_t{1});

  std::vector<int64_t> dims{1, 1, 3, 3};
  test.AddInput<float>("input1", dims,
                       {111.0f, 112.0f, 113.0f,
                        121.0f, 122.0f, 123.0f,
                        131.0f, 132.0f, 133.0f});
  test.AddInput<float>("input2", dims,
                       {211.0f, 212.0f, 213.0f,
                        221.0f, 222.0f, 223.0f,
                        231.0f, 232.0f, 233.0f});
  test.AddInput<float>("input3", dims,
                       {311.0f, 312.0f, 313.0f,
                        321.0f, 322.0f, 323.0f,
                        331.0f, 332.0f, 333.0f});
  test.AddOutput<float>("concat_result", {1, 3, 3, 3},
                        {111.0f, 112.0f, 113.0f,
                         121.0f, 122.0f, 123.0f,
                         131.0f, 132.0f, 133.0f,

                         211.0f, 212.0f, 213.0f,
                         221.0f, 222.0f, 223.0f,
                         231.0f, 232.0f, 233.0f,

                         311.0f, 312.0f, 313.0f,
                         321.0f, 322.0f, 323.0f,
                         331.0f, 332.0f, 333.0f});
  test.Run();
}

TEST(ConcatOpTest, Concat4D_1_negative_axis) {
  OpTester test("Concat");
  test.AddAttribute("axis", int64_t{-3});

  std::vector<int64_t> dims{1, 1, 3, 3};
  test.AddInput<float>("input1", dims,
                       {111.0f, 112.0f, 113.0f,
                        121.0f, 122.0f, 123.0f,
                        131.0f, 132.0f, 133.0f});
  test.AddInput<float>("input2", dims,
                       {211.0f, 212.0f, 213.0f,
                        221.0f, 222.0f, 223.0f,
                        231.0f, 232.0f, 233.0f});
  test.AddInput<float>("input3", dims,
                       {311.0f, 312.0f, 313.0f,
                        321.0f, 322.0f, 323.0f,
                        331.0f, 332.0f, 333.0f});
  test.AddOutput<float>("concat_result", {1, 3, 3, 3},
                        {111.0f, 112.0f, 113.0f,
                         121.0f, 122.0f, 123.0f,
                         131.0f, 132.0f, 133.0f,

                         211.0f, 212.0f, 213.0f,
                         221.0f, 222.0f, 223.0f,
                         231.0f, 232.0f, 233.0f,

                         311.0f, 312.0f, 313.0f,
                         321.0f, 322.0f, 323.0f,
                         331.0f, 332.0f, 333.0f});
  test.Run();
}

TEST(ConcatOpTest, Concat4D_2) {
  OpTester test("Concat");
  test.AddAttribute("axis", int64_t{2});

  std::vector<int64_t> dims{1, 3, 1, 3};
  test.AddInput<float>("input1", dims,
                       {111.0f, 112.0f, 113.0f,
                        211.0f, 212.0f, 213.0f,
                        311.0f, 312.0f, 313.0f});
  test.AddInput<float>("input2", dims,
                       {121.0f, 122.0f, 123.0f,
                        221.0f, 222.0f, 223.0f,
                        321.0f, 322.0f, 323.0f});
  test.AddInput<float>("input3", dims,
                       {131.0f, 132.0f, 133.0f,
                        231.0f, 232.0f, 233.0f,
                        331.0f, 332.0f, 333.0f});
  test.AddOutput<float>("concat_result", {1, 3, 3, 3},
                        {111.0f, 112.0f, 113.0f,
                         121.0f, 122.0f, 123.0f,
                         131.0f, 132.0f, 133.0f,

                         211.0f, 212.0f, 213.0f,
                         221.0f, 222.0f, 223.0f,
                         231.0f, 232.0f, 233.0f,

                         311.0f, 312.0f, 313.0f,
                         321.0f, 322.0f, 323.0f,
                         331.0f, 332.0f, 333.0f});
  test.Run();
}

#ifdef USE_WEBGPU
TEST(ConcatOpTest, Concat1D_exceed_maxStorageBuffersPerShaderStage) {
  // maxStorageBuffersPerShaderStage==8
  OpTester test("Concat");
  test.AddAttribute("axis", int64_t{0});

  test.AddInput<int32_t>("input1", {1}, {1});
  test.AddInput<int32_t>("input2", {1}, {2});
  test.AddInput<int32_t>("input3", {1}, {3});
  test.AddInput<int32_t>("input4", {1}, {4});
  test.AddInput<int32_t>("input5", {1}, {5});
  test.AddInput<int32_t>("input6", {1}, {6});
  test.AddInput<int32_t>("input7", {1}, {7});
  test.AddInput<int32_t>("input8", {1}, {8});
  test.AddInput<int32_t>("input9", {1}, {9});
  test.AddOutput<int32_t>("concat_result", {9}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  test.Run();
}

TEST(ConcatOpTest, Concat2D_exceed_maxStorageBuffersPerShaderStage_axis0) {
  // maxStorageBuffersPerShaderStage==8
  OpTester test("Concat");
  test.AddAttribute("axis", int64_t{0});

  test.AddInput<int32_t>("input1", {1, 2}, {1, 1});
  test.AddInput<int32_t>("input2", {1, 2}, {2, 2});
  test.AddInput<int32_t>("input3", {1, 2}, {3, 3});
  test.AddInput<int32_t>("input4", {1, 2}, {4, 4});
  test.AddInput<int32_t>("input5", {1, 2}, {5, 5});
  test.AddInput<int32_t>("input6", {1, 2}, {6, 6});
  test.AddInput<int32_t>("input7", {1, 2}, {7, 7});
  test.AddInput<int32_t>("input8", {1, 2}, {8, 8});
  test.AddInput<int32_t>("input9", {1, 2}, {9, 9});
  test.AddOutput<int32_t>("concat_result", {9, 2}, {1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9});
  test.Run();
}

TEST(ConcatOpTest, Concat2D_exceed_maxStorageBuffersPerShaderStage_axis1) {
  // maxStorageBuffersPerShaderStage==8
  OpTester test("Concat");
  test.AddAttribute("axis", int64_t{1});

  test.AddInput<int32_t>("input1", {1, 2}, {1, 1});
  test.AddInput<int32_t>("input2", {1, 2}, {2, 2});
  test.AddInput<int32_t>("input3", {1, 2}, {3, 3});
  test.AddInput<int32_t>("input4", {1, 2}, {4, 4});
  test.AddInput<int32_t>("input5", {1, 2}, {5, 5});
  test.AddInput<int32_t>("input6", {1, 2}, {6, 6});
  test.AddInput<int32_t>("input7", {1, 2}, {7, 7});
  test.AddInput<int32_t>("input8", {1, 2}, {8, 8});
  test.AddInput<int32_t>("input9", {1, 2}, {9, 9});
  test.AddOutput<int32_t>("concat_result", {1, 18}, {1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9});
  test.Run();
}

TEST(ConcatOpTest, Concat3D_exceed_maxStorageBuffersPerShaderStage) {
  // maxStorageBuffersPerShaderStage==8
  OpTester test("Concat");
  test.AddAttribute("axis", int64_t{1});

  test.AddInput<int32_t>("input1", {2, 2, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  test.AddInput<int32_t>("input2", {2, 3, 3}, {13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30});
  test.AddInput<int32_t>("input3", {2, 4, 3}, {31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54});
  test.AddInput<int32_t>("input4", {2, 5, 3}, {55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84});
  test.AddInput<int32_t>("input5", {2, 6, 3}, {85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120});
  test.AddInput<int32_t>("input6", {2, 7, 3}, {121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162});
  test.AddInput<int32_t>("input7", {2, 8, 3}, {163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210});
  test.AddInput<int32_t>("input8", {2, 9, 3}, {211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264});
  test.AddInput<int32_t>("input9", {2, 10, 3}, {265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324});
  test.AddOutput<int32_t>("concat_result", {2, 54, 3}, {
    // First batch (batch 0)
    1, 2, 3, 4, 5, 6,                    // input1 batch 0
    13, 14, 15, 16, 17, 18, 19, 20, 21,  // input2 batch 0
    31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,  // input3 batch 0
    55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,  // input4 batch 0
    85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102,  // input5 batch 0
    121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141,  // input6 batch 0
    163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186,  // input7 batch 0
    211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237,  // input8 batch 0
    265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294,  // input9 batch 0
    // Second batch (batch 1)
    7, 8, 9, 10, 11, 12,                  // input1 batch 1
    22, 23, 24, 25, 26, 27, 28, 29, 30,   // input2 batch 1
    43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,  // input3 batch 1
    70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,  // input4 batch 1
    103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120,  // input5 batch 1
    142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162,  // input6 batch 1
    187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210,  // input7 batch 1
    238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264,  // input8 batch 1
    295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324   // input9 batch 1
  });
  test.Run();
}


TEST(ConcatOpTest, Concat3D_exceed_maxStorageBuffersPerShaderStage_small) {
  // maxStorageBuffersPerShaderStage==8
  OpTester test("Concat");
  test.AddAttribute("axis", int64_t{1});

  test.AddInput<int32_t>("input1", {2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
  test.AddInput<int32_t>("input2", {2, 2, 2}, {9, 10, 11, 12, 13, 14, 15, 16});
  test.AddInput<int32_t>("input3", {2, 2, 2}, {17, 18, 19, 20, 21, 22, 23, 24});
  test.AddInput<int32_t>("input4", {2, 2, 2}, {25, 26, 27, 28, 29, 30, 31, 32});
  test.AddInput<int32_t>("input5", {2, 2, 2}, {33, 34, 35, 36, 37, 38, 39, 40});
  test.AddInput<int32_t>("input6", {2, 2, 2}, {41, 42, 43, 44, 45, 46, 47, 48});
  test.AddInput<int32_t>("input7", {2, 2, 2}, {49, 50, 51, 52, 53, 54, 55, 56});
  test.AddInput<int32_t>("input8", {2, 2, 2}, {57, 58, 59, 60, 61, 62, 63, 64});
  test.AddInput<int32_t>("input9", {2, 2, 2}, {65, 66, 67, 68, 69, 70, 71, 72});
  test.AddOutput<int32_t>("concat_result", {2, 18, 2}, {
    // First batch (batch 0)
    1, 2, 3, 4, 9, 10, 11, 12, 17, 18, 19, 20, 25, 26, 27, 28, 33, 34, 35, 36, 41, 42, 43, 44, 49, 50, 51, 52, 57, 58, 59, 60, 65, 66, 67, 68,
    // Second batch (batch 1)
    5, 6, 7, 8, 13, 14, 15, 16, 21, 22, 23, 24, 29, 30, 31, 32, 37, 38, 39, 40, 45, 46, 47, 48, 53, 54, 55, 56, 61, 62, 63, 64, 69, 70, 71, 72
  });
  test.Run();
}
#endif  // USE_WEBGPU

}  // namespace test
}  // namespace onnxruntime
