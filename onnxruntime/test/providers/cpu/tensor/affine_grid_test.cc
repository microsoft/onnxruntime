// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/util/math.h"
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {
TEST(AffineGridTest, 2d) {
  OpTester test("AffineGrid", 20);
  test.AddInput<float>("theta", {1, 2, 3}, {1.0f, 0.0, 0.0f, 0.0f, 1.0, 0.0f});
  test.AddInput<int64_t>("size", {4}, {1, 1, 2, 3});
  test.AddOutput<float>("grid", {1, 2, 3, 2},
                        {-0.6667f, -0.5000f, 0.0000f, -0.5000f, 0.6667f, -0.5000f, -0.6667f, 0.5000f, 0.0000f, 0.5000f, 0.6667f, 0.5000f});
  test.Run();
}

// following tests code is generated with:
// python onnxruntime/test/providers/cpu/tensor/affine_grid_test_gen.py
TEST(AffineGridTest, test_2d_0) {
  OpTester test("AffineGrid", 20);
  test.AddAttribute("align_corners", (int64_t)0);
  test.AddInput<float>("theta", {1, 2, 3}, {1.477212f, -0.173648f, 0.300000f, 0.173648f, 0.492404f, -0.500000f});
  test.AddInput<int64_t>("size", {4}, {1, 1, 3, 2});
  test.AddOutput<float>("grid", {1, 3, 2, 2}, {-0.3228f, -0.9151f, 1.1544f, -0.7414f, -0.4386f, -0.5868f, 1.0386f, -0.4132f, -0.5544f, -0.2586f, 0.9228f, -0.0849f});
  test.Run();
}

TEST(AffineGridTest, test_2d_1) {
  OpTester test("AffineGrid", 20);
  test.AddAttribute("align_corners", (int64_t)0);
  test.AddInput<float>("theta", {2, 2, 3}, {1.477212f, -0.173648f, 0.300000f, 0.173648f, 0.492404f, -0.500000f, 1.477212f, -0.173648f, 0.300000f, 0.173648f, 0.492404f, -0.500000f});
  test.AddInput<int64_t>("size", {4}, {2, 10, 2, 3});
  test.AddOutput<float>("grid", {2, 2, 3, 2}, {-0.5980f, -0.8620f, 0.3868f, -0.7462f, 1.3716f, -0.6304f, -0.7716f, -0.3696f, 0.2132f, -0.2538f, 1.1980f, -0.1380f, -0.5980f, -0.8620f, 0.3868f, -0.7462f, 1.3716f, -0.6304f, -0.7716f, -0.3696f, 0.2132f, -0.2538f, 1.1980f, -0.1380f});
  test.Run();
}

TEST(AffineGridTest, test_2d_2) {
  OpTester test("AffineGrid", 20);
  test.AddAttribute("align_corners", (int64_t)0);
  test.AddInput<float>("theta", {1, 2, 3}, {1.500000f, -0.866025f, -0.500000f, 0.866025f, 2.750000f, -0.500000f});
  test.AddInput<int64_t>("size", {4}, {1, 1, 3, 2});
  test.AddOutput<float>("grid", {1, 3, 2, 2}, {-0.6726f, -2.7663f, 0.8274f, -1.9003f, -1.2500f, -0.9330f, 0.2500f, -0.0670f, -1.8274f, 0.9003f, -0.3274f, 1.7663f});
  test.Run();
}

TEST(AffineGridTest, test_2d_3) {
  OpTester test("AffineGrid", 20);
  test.AddAttribute("align_corners", (int64_t)0);
  test.AddInput<float>("theta", {2, 2, 3}, {1.500000f, -0.866025f, -0.500000f, 0.866025f, 2.750000f, -0.500000f, 1.500000f, -0.866025f, -0.500000f, 0.866025f, 2.750000f, -0.500000f});
  test.AddInput<int64_t>("size", {4}, {2, 10, 2, 3});
  test.AddOutput<float>("grid", {2, 2, 3, 2}, {-1.0670f, -2.4524f, -0.0670f, -1.8750f, 0.9330f, -1.2976f, -1.9330f, 0.2976f, -0.9330f, 0.8750f, 0.0670f, 1.4524f, -1.0670f, -2.4524f, -0.0670f, -1.8750f, 0.9330f, -1.2976f, -1.9330f, 0.2976f, -0.9330f, 0.8750f, 0.0670f, 1.4524f});
  test.Run();
}

TEST(AffineGridTest, test_2d_4) {
  OpTester test("AffineGrid", 20);
  test.AddAttribute("align_corners", (int64_t)1);
  test.AddInput<float>("theta", {1, 2, 3}, {1.477212f, -0.173648f, 0.300000f, 0.173648f, 0.492404f, -0.500000f});
  test.AddInput<int64_t>("size", {4}, {1, 1, 3, 2});
  test.AddOutput<float>("grid", {1, 3, 2, 2}, {-1.0036f, -1.1661f, 1.9509f, -0.8188f, -1.1772f, -0.6736f, 1.7772f, -0.3264f, -1.3509f, -0.1812f, 1.6036f, 0.1661f});
  test.Run();
}

TEST(AffineGridTest, test_2d_5) {
  OpTester test("AffineGrid", 20);
  test.AddAttribute("align_corners", (int64_t)1);
  test.AddInput<float>("theta", {2, 2, 3}, {1.477212f, -0.173648f, 0.300000f, 0.173648f, 0.492404f, -0.500000f, 1.477212f, -0.173648f, 0.300000f, 0.173648f, 0.492404f, -0.500000f});
  test.AddInput<int64_t>("size", {4}, {2, 10, 2, 3});
  test.AddOutput<float>("grid", {2, 2, 3, 2}, {-1.0036f, -1.1661f, 0.4736f, -0.9924f, 1.9509f, -0.8188f, -1.3509f, -0.1812f, 0.1264f, -0.0076f, 1.6036f, 0.1661f, -1.0036f, -1.1661f, 0.4736f, -0.9924f, 1.9509f, -0.8188f, -1.3509f, -0.1812f, 0.1264f, -0.0076f, 1.6036f, 0.1661f});
  test.Run();
}

TEST(AffineGridTest, test_2d_6) {
  OpTester test("AffineGrid", 20);
  test.AddAttribute("align_corners", (int64_t)1);
  test.AddInput<float>("theta", {1, 2, 3}, {1.500000f, -0.866025f, -0.500000f, 0.866025f, 2.750000f, -0.500000f});
  test.AddInput<int64_t>("size", {4}, {1, 1, 3, 2});
  test.AddOutput<float>("grid", {1, 3, 2, 2}, {-1.1340f, -4.1160f, 1.8660f, -2.3840f, -2.0000f, -1.3660f, 1.0000f, 0.3660f, -2.8660f, 1.3840f, 0.1340f, 3.1160f});
  test.Run();
}

TEST(AffineGridTest, test_2d_7) {
  OpTester test("AffineGrid", 20);
  test.AddAttribute("align_corners", (int64_t)1);
  test.AddInput<float>("theta", {2, 2, 3}, {1.500000f, -0.866025f, -0.500000f, 0.866025f, 2.750000f, -0.500000f, 1.500000f, -0.866025f, -0.500000f, 0.866025f, 2.750000f, -0.500000f});
  test.AddInput<int64_t>("size", {4}, {2, 10, 2, 3});
  test.AddOutput<float>("grid", {2, 2, 3, 2}, {-1.1340f, -4.1160f, 0.3660f, -3.2500f, 1.8660f, -2.3840f, -2.8660f, 1.3840f, -1.3660f, 2.2500f, 0.1340f, 3.1160f, -1.1340f, -4.1160f, 0.3660f, -3.2500f, 1.8660f, -2.3840f, -2.8660f, 1.3840f, -1.3660f, 2.2500f, 0.1340f, 3.1160f});
  test.Run();
}

TEST(AffineGridTest, test_3d_0) {
  OpTester test("AffineGrid", 20);
  test.AddAttribute("align_corners", (int64_t)0);
  test.AddInput<float>("theta", {1, 3, 4}, {1.409539f, 0.000000f, 0.513030f, 0.300000f, 0.118782f, 1.969615f, -0.326352f, -0.500000f, -0.168412f, 0.086824f, 0.462708f, 1.800000f});
  test.AddInput<int64_t>("size", {5}, {1, 1, 3, 2, 2});
  test.AddOutput<float>("grid", {1, 3, 2, 2, 3}, {-0.7468f, -1.3266f, 1.5323f, 0.6627f, -1.2078f, 1.3639f, -0.7468f, 0.6430f, 1.6191f, 0.6627f, 0.7618f, 1.4507f, -0.4048f, -1.5442f, 1.8408f, 1.0048f, -1.4254f, 1.6724f, -0.4048f, 0.4254f, 1.9276f, 1.0048f, 0.5442f, 1.7592f, -0.0627f, -1.7618f, 2.1493f, 1.3468f, -1.6430f, 1.9809f, -0.0627f, 0.2078f, 2.2361f, 1.3468f, 0.3266f, 2.0677f});
  test.Run();
}

TEST(AffineGridTest, test_3d_1) {
  OpTester test("AffineGrid", 20);
  test.AddAttribute("align_corners", (int64_t)0);
  test.AddInput<float>("theta", {2, 3, 4}, {1.409539f, 0.000000f, 0.513030f, 0.300000f, 0.118782f, 1.969615f, -0.326352f, -0.500000f, -0.168412f, 0.086824f, 0.462708f, 1.800000f, 1.409539f, 0.000000f, 0.513030f, 0.300000f, 0.118782f, 1.969615f, -0.326352f, -0.500000f, -0.168412f, 0.086824f, 0.462708f, 1.800000f});
  test.AddInput<int64_t>("size", {5}, {2, 10, 2, 2, 3});
  test.AddOutput<float>("grid", {2, 2, 2, 3, 3}, {-0.8962f, -1.4008f, 1.6375f, 0.0435f, -1.3216f, 1.5252f, 0.9832f, -1.2424f, 1.4130f, -0.8962f, 0.5688f, 1.7243f, 0.0435f, 0.6480f, 1.6121f, 0.9832f, 0.7272f, 1.4998f, -0.3832f, -1.7272f, 2.1002f, 0.5565f, -1.6480f, 1.9879f, 1.4962f, -1.5688f, 1.8757f, -0.3832f, 0.2424f, 2.1870f, 0.5565f, 0.3216f, 2.0748f, 1.4962f, 0.4008f, 1.9625f, -0.8962f, -1.4008f, 1.6375f, 0.0435f, -1.3216f, 1.5252f, 0.9832f, -1.2424f, 1.4130f, -0.8962f, 0.5688f, 1.7243f, 0.0435f, 0.6480f, 1.6121f, 0.9832f, 0.7272f, 1.4998f, -0.3832f, -1.7272f, 2.1002f, 0.5565f, -1.6480f, 1.9879f, 1.4962f, -1.5688f, 1.8757f, -0.3832f, 0.2424f, 2.1870f, 0.5565f, 0.3216f, 2.0748f, 1.4962f, 0.4008f, 1.9625f});
  test.Run();
}

TEST(AffineGridTest, test_3d_2) {
  OpTester test("AffineGrid", 20);
  test.AddAttribute("align_corners", (int64_t)0);
  test.AddInput<float>("theta", {1, 3, 4}, {0.259808f, 0.000000f, -0.150000f, -0.500000f, -1.299038f, 1.500000f, -2.250000f, -0.500000f, 1.375000f, 4.763140f, 2.381570f, 0.300000f});
  test.AddInput<int64_t>("size", {5}, {1, 1, 3, 2, 2});
  test.AddOutput<float>("grid", {1, 3, 2, 2, 3}, {-0.5299f, 0.8995f, -4.3568f, -0.2701f, -0.3995f, -2.9818f, -0.5299f, 2.3995f, 0.4064f, -0.2701f, 1.1005f, 1.7814f, -0.6299f, -0.6005f, -2.7691f, -0.3701f, -1.8995f, -1.3941f, -0.6299f, 0.8995f, 1.9941f, -0.3701f, -0.3995f, 3.3691f, -0.7299f, -2.1005f, -1.1814f, -0.4701f, -3.3995f, 0.1936f, -0.7299f, -0.6005f, 3.5818f, -0.4701f, -1.8995f, 4.9568f});
  test.Run();
}

TEST(AffineGridTest, test_3d_3) {
  OpTester test("AffineGrid", 20);
  test.AddAttribute("align_corners", (int64_t)0);
  test.AddInput<float>("theta", {2, 3, 4}, {0.259808f, 0.000000f, -0.150000f, -0.500000f, -1.299038f, 1.500000f, -2.250000f, -0.500000f, 1.375000f, 4.763140f, 2.381570f, 0.300000f, 0.259808f, 0.000000f, -0.150000f, -0.500000f, -1.299038f, 1.500000f, -2.250000f, -0.500000f, 1.375000f, 4.763140f, 2.381570f, 0.300000f});
  test.AddInput<int64_t>("size", {5}, {2, 10, 2, 2, 3});
  test.AddOutput<float>("grid", {2, 2, 2, 3, 3}, {-0.5982f, 0.7410f, -4.1890f, -0.4250f, -0.1250f, -3.2724f, -0.2518f, -0.9910f, -2.3557f, -0.5982f, 2.2410f, 0.5741f, -0.4250f, 1.3750f, 1.4908f, -0.2518f, 0.5090f, 2.4075f, -0.7482f, -1.5090f, -1.8075f, -0.5750f, -2.3750f, -0.8908f, -0.4018f, -3.2410f, 0.0259f, -0.7482f, -0.0090f, 2.9557f, -0.5750f, -0.8750f, 3.8724f, -0.4018f, -1.7410f, 4.7890f, -0.5982f, 0.7410f, -4.1890f, -0.4250f, -0.1250f, -3.2724f, -0.2518f, -0.9910f, -2.3557f, -0.5982f, 2.2410f, 0.5741f, -0.4250f, 1.3750f, 1.4908f, -0.2518f, 0.5090f, 2.4075f, -0.7482f, -1.5090f, -1.8075f, -0.5750f, -2.3750f, -0.8908f, -0.4018f, -3.2410f, 0.0259f, -0.7482f, -0.0090f, 2.9557f, -0.5750f, -0.8750f, 3.8724f, -0.4018f, -1.7410f, 4.7890f});
  test.Run();
}

TEST(AffineGridTest, test_3d_4) {
  OpTester test("AffineGrid", 20);
  test.AddAttribute("align_corners", (int64_t)1);
  test.AddInput<float>("theta", {1, 3, 4}, {1.409539f, 0.000000f, 0.513030f, 0.300000f, 0.118782f, 1.969615f, -0.326352f, -0.500000f, -0.168412f, 0.086824f, 0.462708f, 1.800000f});
  test.AddInput<int64_t>("size", {5}, {1, 1, 3, 2, 2});
  test.AddOutput<float>("grid", {1, 3, 2, 2, 3}, {-1.6226f, -2.2620f, 1.4189f, 1.1965f, -2.0245f, 1.0821f, -1.6226f, 1.6772f, 1.5925f, 1.1965f, 1.9147f, 1.2557f, -1.1095f, -2.5884f, 1.8816f, 1.7095f, -2.3508f, 1.5448f, -1.1095f, 1.3508f, 2.0552f, 1.7095f, 1.5884f, 1.7184f, -0.5965f, -2.9147f, 2.3443f, 2.2226f, -2.6772f, 2.0075f, -0.5965f, 1.0245f, 2.5179f, 2.2226f, 1.2620f, 2.1811f});
  test.Run();
}

TEST(AffineGridTest, test_3d_5) {
  OpTester test("AffineGrid", 20);
  test.AddAttribute("align_corners", (int64_t)1);
  test.AddInput<float>("theta", {2, 3, 4}, {1.409539f, 0.000000f, 0.513030f, 0.300000f, 0.118782f, 1.969615f, -0.326352f, -0.500000f, -0.168412f, 0.086824f, 0.462708f, 1.800000f, 1.409539f, 0.000000f, 0.513030f, 0.300000f, 0.118782f, 1.969615f, -0.326352f, -0.500000f, -0.168412f, 0.086824f, 0.462708f, 1.800000f});
  test.AddInput<int64_t>("size", {5}, {2, 10, 2, 2, 3});
  test.AddOutput<float>("grid", {2, 2, 2, 3, 3}, {-1.6226f, -2.2620f, 1.4189f, -0.2130f, -2.1433f, 1.2505f, 1.1965f, -2.0245f, 1.0821f, -1.6226f, 1.6772f, 1.5925f, -0.2130f, 1.7960f, 1.4241f, 1.1965f, 1.9147f, 1.2557f, -0.5965f, -2.9147f, 2.3443f, 0.8130f, -2.7960f, 2.1759f, 2.2226f, -2.6772f, 2.0075f, -0.5965f, 1.0245f, 2.5179f, 0.8130f, 1.1433f, 2.3495f, 2.2226f, 1.2620f, 2.1811f, -1.6226f, -2.2620f, 1.4189f, -0.2130f, -2.1433f, 1.2505f, 1.1965f, -2.0245f, 1.0821f, -1.6226f, 1.6772f, 1.5925f, -0.2130f, 1.7960f, 1.4241f, 1.1965f, 1.9147f, 1.2557f, -0.5965f, -2.9147f, 2.3443f, 0.8130f, -2.7960f, 2.1759f, 2.2226f, -2.6772f, 2.0075f, -0.5965f, 1.0245f, 2.5179f, 0.8130f, 1.1433f, 2.3495f, 2.2226f, 1.2620f, 2.1811f});
  test.Run();
}

TEST(AffineGridTest, test_3d_6) {
  OpTester test("AffineGrid", 20);
  test.AddAttribute("align_corners", (int64_t)1);
  test.AddInput<float>("theta", {1, 3, 4}, {0.259808f, 0.000000f, -0.150000f, -0.500000f, -1.299038f, 1.500000f, -2.250000f, -0.500000f, 1.375000f, 4.763140f, 2.381570f, 0.300000f});
  test.AddInput<int64_t>("size", {5}, {1, 1, 3, 2, 2});
  test.AddOutput<float>("grid", {1, 3, 2, 2, 3}, {-0.6098f, 1.5490f, -8.2197f, -0.0902f, -1.0490f, -5.4697f, -0.6098f, 4.5490f, 1.3066f, -0.0902f, 1.9510f, 4.0566f, -0.7598f, -0.7010f, -5.8381f, -0.2402f, -3.2990f, -3.0881f, -0.7598f, 2.2990f, 3.6881f, -0.2402f, -0.2990f, 6.4381f, -0.9098f, -2.9510f, -3.4566f, -0.3902f, -5.5490f, -0.7066f, -0.9098f, 0.0490f, 6.0697f, -0.3902f, -2.5490f, 8.8197f});
  test.Run();
}

TEST(AffineGridTest, test_3d_7) {
  OpTester test("AffineGrid", 20);
  test.AddAttribute("align_corners", (int64_t)1);
  test.AddInput<float>("theta", {2, 3, 4}, {0.259808f, 0.000000f, -0.150000f, -0.500000f, -1.299038f, 1.500000f, -2.250000f, -0.500000f, 1.375000f, 4.763140f, 2.381570f, 0.300000f, 0.259808f, 0.000000f, -0.150000f, -0.500000f, -1.299038f, 1.500000f, -2.250000f, -0.500000f, 1.375000f, 4.763140f, 2.381570f, 0.300000f});
  test.AddInput<int64_t>("size", {5}, {2, 10, 2, 2, 3});
  test.AddOutput<float>("grid", {2, 2, 2, 3, 3}, {-0.6098f, 1.5490f, -8.2197f, -0.3500f, 0.2500f, -6.8447f, -0.0902f, -1.0490f, -5.4697f, -0.6098f, 4.5490f, 1.3066f, -0.3500f, 3.2500f, 2.6816f, -0.0902f, 1.9510f, 4.0566f, -0.9098f, -2.9510f, -3.4566f, -0.6500f, -4.2500f, -2.0816f, -0.3902f, -5.5490f, -0.7066f, -0.9098f, 0.0490f, 6.0697f, -0.6500f, -1.2500f, 7.4447f, -0.3902f, -2.5490f, 8.8197f, -0.6098f, 1.5490f, -8.2197f, -0.3500f, 0.2500f, -6.8447f, -0.0902f, -1.0490f, -5.4697f, -0.6098f, 4.5490f, 1.3066f, -0.3500f, 3.2500f, 2.6816f, -0.0902f, 1.9510f, 4.0566f, -0.9098f, -2.9510f, -3.4566f, -0.6500f, -4.2500f, -2.0816f, -0.3902f, -5.5490f, -0.7066f, -0.9098f, 0.0490f, 6.0697f, -0.6500f, -1.2500f, 7.4447f, -0.3902f, -2.5490f, 8.8197f});
  test.Run();
}
}  // namespace test
}  // namespace onnxruntime
