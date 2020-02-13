// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

template <typename T>
static void RunTest(
    T start,
    T limit,
    T delta,
    const std::vector<int64_t>& output_dims,
    const std::vector<T>& output) {
  // ONNX domain opset-11
  OpTester test1("Range", 11);
  test1.AddInput<T>("start", {}, {start});
  test1.AddInput<T>("limit", {}, {limit});
  test1.AddInput<T>("delta", {}, {delta});
  test1.AddOutput<T>("output", output_dims, output);
  // NGraph and TensorRT do not yet support opset-11 and builds break on this test, hence exclude the EP
  test1.Run(OpTester::ExpectResult::kExpectSuccess, "", {kNGraphExecutionProvider, kTensorrtExecutionProvider});

#ifndef DISABLE_CONTRIB_OPS

  // MSFT domain opset-1 (contrib op)
  OpTester test2("Range", 1, kMSDomain);
  test2.AddInput<T>("start", {}, {start});
  test2.AddInput<T>("limit", {}, {limit});

  if (delta != T{1})  // only contrib schema allows optional 'delta' input
    test2.AddInput<T>("delta", {}, {delta});

  test2.AddOutput<T>("output", output_dims, output);
  // TensorRT doesn't fully support opset 11 yet
  test2.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});

#endif
}  // namespace test

TEST(RangeTest, Int32_DeltaDefault) {
  RunTest<int32_t>(0, 5, 1, {5}, {0, 1, 2, 3, 4});
}

TEST(RangeTest, Int64_DeltaDefault) {
  RunTest<int64_t>(0, 5, 1, {5}, {0, 1, 2, 3, 4});
}

TEST(RangeTest, Float_DeltaDefault) {
  RunTest<float>(0.f, 5.f, 1.f, {5}, {0.f, 1.f, 2.f, 3.f, 4.f});
}

TEST(RangeTest, Double_DeltaDefault) {
  RunTest<double>(0., 5., 1., {5}, {0., 1., 2., 3., 4.});
}

TEST(RangeTest, Int32_Delta_NonDefault) {
  RunTest<int32_t>(0, 10, 2, {5}, {0, 2, 4, 6, 8});
}

TEST(RangeTest, Int64_Delta_NonDefault_0) {
  RunTest<int64_t>(0, 9, 2, {5}, {0, 2, 4, 6, 8});
}

TEST(RangeTest, Int64_Delta_NonDefault_1) {
  RunTest<int64_t>(1, 2, 2, {1}, {1});
}

TEST(RangeTest, Int32_NegativeDelta_0) {
  RunTest<int32_t>(2, -9, -2, {6}, {2, 0, -2, -4, -6, -8});
}

TEST(RangeTest, Int32_NegativeDelta_1) {
  RunTest<int32_t>(2, 9, -2, {0}, {});
}

TEST(RangeTest, Float_NegativeDelta_0) {
  RunTest<float>(2.0f, -8.1f, -2.0f, {6}, {2.0f, 0.0f, -2.0f, -4.0f, -6.0f, -8.0f});
}

TEST(RangeTest, Float_SameStartAndLimit) {
  RunTest<float>(2.0f, 2.0f, 1, {0}, {});
}

TEST(RangeTest, AlmostSameStartAndLimitHighDelta) {
  RunTest<float>(2.0f, 2.01f, 1000000.0f, {1}, {2.0f});
}

}  // namespace test
}  // namespace onnxruntime
