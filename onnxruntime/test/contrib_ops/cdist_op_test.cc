// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(CDistOpTest, Basic) {
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

}  // namespace test
}  // namespace onnxruntime
