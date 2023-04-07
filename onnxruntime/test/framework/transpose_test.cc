// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "core/framework/transpose_helper.h"

namespace onnxruntime {
namespace test {

TEST(TransposeIsMovingSingleAxis, test1) {
  std::array<size_t, 4> perm{1, 2, 3, 0};
  size_t from = 0, to = 0;
  ASSERT_TRUE(IsTransposeMovingSingleAxis(perm, from, to));
  ASSERT_EQ(from, static_cast<size_t>(0));
  ASSERT_EQ(to, static_cast<size_t>(3));
}

TEST(TransposeIsMovingSingleAxis, test2) {
  std::array<size_t, 4> perm{0, 2, 3, 1};
  size_t from = 0, to = 0;
  ASSERT_TRUE(IsTransposeMovingSingleAxis(perm, from, to));
  ASSERT_EQ(from, static_cast<size_t>(1));
  ASSERT_EQ(to, static_cast<size_t>(3));
}

TEST(TransposeIsMovingSingleAxis, test3) {
  std::array<size_t, 4> perm{3, 1, 0, 2};
  size_t from = 0, to = 0;
  ASSERT_FALSE(IsTransposeMovingSingleAxis(perm, from, to));
}

TEST(TransposeIsMovingSingleAxis, test4) {
  std::array<size_t, 4> perm{0, 1, 2, 3};
  size_t from = 0, to = 0;
  ASSERT_FALSE(IsTransposeMovingSingleAxis(perm, from, to));
}
}  // namespace test
}  // namespace onnxruntime
