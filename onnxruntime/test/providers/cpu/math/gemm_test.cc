// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(MathOpTest, GemmNoTrans) {
  OpTester test("Gemm");

  test.AddAttribute("transA", (int64_t)0);
  test.AddAttribute("transB", (int64_t)0);
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 1.0f);

  test.AddInput<float>("A", {2, 4},
                       {1.0f, 2.0f, 3.0f, 4.0f,
                        -1.0f, -2.0f, -3.0f, -4.0f});
  test.AddInput<float>("B", {4, 3}, std::vector<float>(12, 1.0f));
  test.AddInput<float>("C", {2, 3}, std::vector<float>(6, 1.0f));
  test.AddOutput<float>("Y", {2, 3},
                        {11.0f, 11.0f, 11.0f,
                         -9.0f, -9.0f, -9.0f});
  test.Run();
}

TEST(MathOpTest, GemmBroadcast) {
  OpTester test("Gemm");

  test.AddAttribute("transA", (int64_t)0);
  test.AddAttribute("transB", (int64_t)0);
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 1.0f);

  test.AddInput<float>("A", {2, 4},
                       {1.0f, 2.0f, 3.0f, 4.0f,
                        -1.0f, -2.0f, -3.0f, -4.0f});
  test.AddInput<float>("B", {4, 3}, std::vector<float>(12, 1.0f));
  test.AddInput<float>("C", {3}, std::vector<float>{1.0f, 2.0f, 3.0f});
  test.AddOutput<float>("Y", {2, 3},
                        {11.0f, 12.0f, 13.0f,
                         -9.0f, -8.0f, -7.0f});
  test.Run();
}

TEST(MathOpTest, GemmTrans) {
  OpTester test("Gemm");

  test.AddAttribute("transA", (int64_t)1);
  test.AddAttribute("transB", (int64_t)1);
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 1.0f);

  test.AddInput<float>("A", {4, 2},
                       {1.0f, -1.0f,
                        2.0f, -2.0f,
                        3.0f, -3.0f,
                        4.0f, -4.0f});
  test.AddInput<float>("B", {3, 4}, std::vector<float>(12, 1.0f));
  test.AddInput<float>("C", {3}, std::vector<float>(3, 1.0f));
  test.AddOutput<float>("Y", {2, 3},
                        {11.0f, 11.0f, 11.0f,
                         -9.0f, -9.0f, -9.0f});
  test.Run();
}

TEST(MathOpTest, GemmAlphaBeta) {
  OpTester test("Gemm");

  test.AddAttribute("transA", (int64_t)0);
  test.AddAttribute("transB", (int64_t)0);
  test.AddAttribute("alpha", 0.5f);
  test.AddAttribute("beta", 2.0f);

  test.AddInput<float>("A", {2, 4},
                       {1.0f, 2.0f, 3.0f, 4.0f,
                        -1.0f, -2.0f, -3.0f, -4.0f});
  test.AddInput<float>("B", {4, 3}, std::vector<float>(12, 1.0f));
  test.AddInput<float>("C", {3}, std::vector<float>(3, 1.0f));
  test.AddOutput<float>("Y", {2, 3},
                        {7.0f, 7.0f, 7.0f,
                         -3.0f, -3.0f, -3.0f});
  test.Run();
}

TEST(MathOpTest, GemmNaN) {
  OpTester test("Gemm");

  test.AddAttribute("transA", (int64_t)0);
  test.AddAttribute("transB", (int64_t)0);
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 0.0f);

  test.AddInput<float>("A", {2, 4},
                       {1.0f, 2.0f, 3.0f, 4.0f,
                        -1.0f, -2.0f, -3.0f, -4.0f});
  test.AddInput<float>("B", {4, 3}, std::vector<float>(12, 1.0f));
  test.AddInput<float>("C", {2, 3}, std::vector<float>(6, 1.0f));
  test.AddOutput<float>("Y", {2, 3},
                        {10.0f, 10.0f, 10.0f,
                         -10.0f, -10.0f, -10.0f});
  test.Run();
}

TEST(MathOpTest, GemmScalarBroadcast) {
  OpTester test("Gemm");

  test.AddAttribute("transA", (int64_t)0);
  test.AddAttribute("transB", (int64_t)0);
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 1.0f);

  test.AddInput<float>("A", {2, 4},
                       {1.0f, 2.0f, 3.0f, 4.0f,
                        -1.0f, -2.0f, -3.0f, -4.0f});
  test.AddInput<float>("B", {4, 3}, std::vector<float>(12, 1.0f));
  test.AddInput<float>("C", {1}, std::vector<float>{1.0f});
  test.AddOutput<float>("Y", {2, 3},
                        {11.0f, 11.0f, 11.0f,
                         -9.0f, -9.0f, -9.0f});
  test.Run();
}

TEST(MathOpTest, Gemm2DBroadcast) {
  OpTester test("Gemm");

  test.AddAttribute("transA", (int64_t)0);
  test.AddAttribute("transB", (int64_t)0);
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 1.0f);

  test.AddInput<float>("A", {2, 4},
                       {1.0f, 2.0f, 3.0f, 4.0f,
                        -1.0f, -2.0f, -3.0f, -4.0f});
  test.AddInput<float>("B", {4, 3}, std::vector<float>(12, 1.0f));
  test.AddInput<float>("C", {2, 1}, std::vector<float>{1.0f, 2.0f});
  test.AddOutput<float>("Y", {2, 3},
                        {11.0f, 11.0f, 11.0f,
                         -8.0f, -8.0f, -8.0f});
  test.Run();
}

TEST(MathOpTest, GemmFalseBroadcast) {
  OpTester test("Gemm");

  test.AddAttribute("transA", (int64_t)0);
  test.AddAttribute("transB", (int64_t)0);
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 1.0f);

  test.AddInput<float>("A", {2, 4},
                       {1.0f, 2.0f, 3.0f, 4.0f,
                        -1.0f, -2.0f, -3.0f, -4.0f});
  test.AddInput<float>("B", {4, 3}, std::vector<float>(12, 1.0f));
  test.AddInput<float>("C", {2, 3}, std::vector<float>{1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f});
  test.AddOutput<float>("Y", {2, 3},
                        {11.0f, 11.0f, 11.0f,
                         -8.0f, -8.0f, -8.0f});
  test.Run();
}

TEST(MathOpTest, GemmEmptyTensor) {
  OpTester test("Gemm");

  test.AddAttribute("transA", static_cast<int64_t>(0));
  test.AddAttribute("transB", static_cast<int64_t>(0));
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 1.0f);

  test.AddInput<float>("A", {0, 4},
                       {});
  test.AddInput<float>("B", {4, 3}, std::vector<float>(12, 1.0f));
  test.AddInput<float>("C", {3}, std::vector<float>(3, 1.0f));
  test.AddOutput<float>("Y", {0, 3},
                        {});
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
