// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/math/gemm_helper.h"
#include <gtest/gtest.h>
#include "test/util/include/asserts.h"

namespace onnxruntime {
namespace test {
namespace {
TEST(GemmHelperTest, no_trans) {
  std::unique_ptr<GemmHelper> h;
  TensorShape left{2, 4};
  TensorShape right{4, 3};
  TensorShape bias{2, 3};
  ASSERT_STATUS_OK(GemmHelper::Create(left, false, right, false, bias, h));
}

TEST(GemmHelperTest, negative_dim) {
  // The Create function will fail if any of the dim is negative
  std::array<int64_t, 6> dims = {2, 4, 4, 3, 2, 3};
  for (size_t i = 0; i != dims.size(); ++i) {
    std::array<int64_t, 6> dims_copy = dims;
    // Set a dim to negative
    dims_copy[i] = -1;
    std::unique_ptr<GemmHelper> h;
    int64_t* p = dims_copy.data();
    TensorShape left(p, 2);
    p += 2;
    TensorShape right{p, 2};
    p += 2;
    TensorShape bias{p, 2};
    ASSERT_STATUS_NOT_OK(GemmHelper::Create(left, false, right, false, bias, h));
  }
}

TEST(GemmHelperTest, scalar_bias) {
  std::unique_ptr<GemmHelper> h;
  TensorShape left{2, 4};
  TensorShape right{4, 3};
  TensorShape bias{1};
  ASSERT_STATUS_OK(GemmHelper::Create(left, false, right, false, bias, h));
}

TEST(GemmHelperTest, scalar_bias2) {
  std::unique_ptr<GemmHelper> h;
  TensorShape left{2, 4};
  TensorShape right{4, 3};
  TensorShape bias;
  ASSERT_STATUS_OK(GemmHelper::Create(left, false, right, false, bias, h));
}

TEST(GemmHelperTest, one_column_bias) {
  std::unique_ptr<GemmHelper> h;
  TensorShape left{2, 4};
  TensorShape right{4, 3};
  TensorShape bias{1, 3};
  ASSERT_STATUS_OK(GemmHelper::Create(left, false, right, false, bias, h));
}

TEST(GemmHelperTest, one_column_bias2) {
  std::unique_ptr<GemmHelper> h;
  TensorShape left{2, 4};
  TensorShape right{4, 3};
  TensorShape bias{2, 1};
  ASSERT_STATUS_OK(GemmHelper::Create(left, false, right, false, bias, h));
}

TEST(GemmHelperTest, one_column_bias_wrong_shape) {
  std::unique_ptr<GemmHelper> h;
  TensorShape left{2, 4};
  TensorShape right{4, 3};
  TensorShape bias{9, 1};
  ASSERT_STATUS_NOT_OK(GemmHelper::Create(left, false, right, false, bias, h));
}

TEST(GemmHelperTest, transA_wrong_shape) {
  std::unique_ptr<GemmHelper> h;
  TensorShape left{2, 4};
  TensorShape right{4, 3};
  TensorShape bias{2, 3};
  ASSERT_STATUS_NOT_OK(GemmHelper::Create(left, true, right, false, bias, h));
}

TEST(GemmHelperTest, transA) {
  std::unique_ptr<GemmHelper> h;
  TensorShape left{4, 2};
  TensorShape right{4, 3};
  TensorShape bias{2, 3};
  ASSERT_STATUS_OK(GemmHelper::Create(left, true, right, false, bias, h));
}

TEST(GemmHelperTest, transB) {
  std::unique_ptr<GemmHelper> h;
  TensorShape left{2, 4};
  TensorShape right{3, 4};
  TensorShape bias{2, 3};
  ASSERT_STATUS_OK(GemmHelper::Create(left, false, right, true, bias, h));
}

TEST(GemmHelperTest, trans_A_and_B) {
  std::unique_ptr<GemmHelper> h;
  TensorShape left{4, 2};
  TensorShape right{3, 4};
  TensorShape bias{2, 3};
  ASSERT_STATUS_OK(GemmHelper::Create(left, true, right, true, bias, h));
}
}  // namespace
}  // namespace test
}  // namespace onnxruntime