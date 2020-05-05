// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(CDistOpTest, Euclidean) {
  OpTester test("CDist", 1, onnxruntime::kMSDomain);
  test.AddAttribute("metric", "euclidean");

  test.AddInput<float>("A", {4, 2},
                       {-1.0856307f, 0.99734545f,
                        0.2829785f, -1.5062947f,
                        -0.5786002f, 1.6514366f,
                        -2.4266791f, -0.42891264f});
  test.AddInput<float>("B", {3, 2},
                       {1.2659363f, -0.8667404f,
                        -0.6788862f, -0.09470897f,
                        1.4913896f, -0.638902f});

  test.AddOutput<float>("y", {4, 3},
                        {3.0007803f, 1.1653428f, 3.0525956f,
                         1.1727045f, 1.7081447f, 1.4874904f,
                         3.1214628f, 1.749023f, 3.0871522f,
                         3.718481f, 1.7794584f, 3.923692f});
  test.Run();
}

TEST(CDistOpTest, Sqeuclidean) {
  OpTester test("CDist", 1, onnxruntime::kMSDomain);
  test.AddAttribute("metric", "sqeuclidean");

  test.AddInput<float>("A", {4, 2},
                       {-1.0856307f, 0.99734545f,
                        0.2829785f, -1.5062947f,
                        -0.5786002f, 1.6514366f,
                        -2.4266791f, -0.42891264f});
  test.AddInput<float>("B", {3, 2},
                       {1.2659363f, -0.8667404f,
                        -0.6788862f, -0.09470897f,
                        1.4913896f, -0.638902f});

  test.AddOutput<float>("y", {4, 3},
                        {9.004683f, 1.3580238f, 9.318338f,
                         1.3752356f, 2.917758f, 2.2126276f,
                         9.74353f, 3.0590816f, 9.530509f,
                         13.827101f, 3.1664724f, 15.395359f});
  test.Run();
}

TEST(CDistOpTest, DoubleEuclidean) {
  OpTester test("CDist", 1, onnxruntime::kMSDomain);
  test.AddAttribute("metric", "euclidean");

  test.AddInput<double>("A", {2, 3},
                        {0.17251948, 1.6354825, 0.0373364,
                         -0.8841497, -1.1431923, -0.621366});
  test.AddInput<double>("B", {3, 3},
                        {-1.3486496, -0.81973106, -0.1342539,
                         1.5996001, -0.28360364, -0.5063398,
                         0.06890842, 1.4522595, -1.6390957});

  test.AddOutput<double>("y", {2, 3},
                         {2.8933496, 2.4525568, 1.6895947,
                          0.7467701, 2.6308053, 2.9462626});
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
