// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "core/util/math.h"

namespace onnxruntime {
namespace test {
TEST(SliceGradOpTest, SliceGrad_basic) {
  OpTester test("SliceGrad", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("grad", {2}, {5, 7});
  test.AddInput<int64_t>("shape", {2}, {2, 4});
  test.AddInput<int64_t>("starts", {2}, {1LL, 0LL});
  test.AddInput<int64_t>("ends", {2}, {2LL, 3LL});
  test.AddInput<int64_t>("axes", {2}, {0LL, 1LL});
  test.AddInput<int64_t>("steps", {2}, {1LL, 2LL});
  test.AddOutput<float>("output", {2, 4}, {0, 0, 0, 0, 5, 0, 7, 0});
  test.Run();
}

TEST(SliceGradOpTest, SliceGrad_basic_double) {
  OpTester test("SliceGrad", 1, onnxruntime::kMSDomain);
  test.AddInput<double>("grad", {2}, {5, 7});
  test.AddInput<int64_t>("shape", {2}, {2, 4});
  test.AddInput<int64_t>("starts", {2}, {1LL, 0LL});
  test.AddInput<int64_t>("ends", {2}, {2LL, 3LL});
  test.AddInput<int64_t>("axes", {2}, {0LL, 1LL});
  test.AddInput<int64_t>("steps", {2}, {1LL, 2LL});
  test.AddOutput<double>("output", {2, 4}, {0, 0, 0, 0, 5, 0, 7, 0});
  test.Run();
}

TEST(SliceGradOpTest, SliceGrad_blockcopy) {
  OpTester test("SliceGrad", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("grad", {2, 4}, {1, 2, 3, 4, 5, 6, 7, 8});
  test.AddInput<int64_t>("shape", {2}, {2, 4});
  test.AddInput<int64_t>("starts", {2}, {0LL, 0LL});
  test.AddInput<int64_t>("ends", {2}, {2LL, 4LL});
  test.AddInput<int64_t>("axes", {2}, {0LL, 1LL});
  test.AddInput<int64_t>("steps", {2}, {1LL, 1LL});
  test.AddOutput<float>("output", {2, 4}, {1, 2, 3, 4, 5, 6, 7, 8});
  test.Run();
}

TEST(SliceGradOpTest, SliceGrad_blockcopy_double) {
  OpTester test("SliceGrad", 1, onnxruntime::kMSDomain);
  test.AddInput<double>("grad", {2, 4}, {1, 2, 3, 4, 5, 6, 7, 8});
  test.AddInput<int64_t>("shape", {2}, {2, 4});
  test.AddInput<int64_t>("starts", {2}, {0LL, 0LL});
  test.AddInput<int64_t>("ends", {2}, {2LL, 4LL});
  test.AddInput<int64_t>("axes", {2}, {0LL, 1LL});
  test.AddInput<int64_t>("steps", {2}, {1LL, 1LL});
  test.AddOutput<double>("output", {2, 4}, {1, 2, 3, 4, 5, 6, 7, 8});
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
