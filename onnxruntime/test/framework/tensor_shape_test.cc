// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensor_shape.h"

#include <vector>

#include "gtest/gtest.h"

namespace onnxruntime {
namespace utils {
namespace test {

static void TestShapeWithVector(const std::vector<int64_t>& vector) {

  // Test constructing from a vector
  TensorShape shape{vector};
  EXPECT_EQ(shape, vector);

  // Test copying to a new shape
  TensorShape shape_copy{shape};
  EXPECT_EQ(shape, shape_copy);

  // Test copying to itself
  TensorShape &shape2=shape;
  shape = shape2;
  EXPECT_EQ(shape, shape_copy);
}

TEST(TensorShapeTest, VariousSizes) {

  // Test various sizes of copying between vectors
  TestShapeWithVector({10});
  TestShapeWithVector({10, 20});
  TestShapeWithVector({10, 20, 30});
  TestShapeWithVector({10, 20, 30, 40});
  TestShapeWithVector({12, 23, 34, 45, 56, 67, 78, 89, 90});

  // Test assigning a shape to a large then a small vector (causing it to switch from small block to large, then back to small)
  std::vector<int64_t> small{1, 2, 3};
  std::vector<int64_t> large{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  TensorShape shape{small};
  EXPECT_EQ(shape, small);

  shape=TensorShape{large};
  EXPECT_EQ(shape, large);

  shape=TensorShape{small};
  EXPECT_EQ(shape, small);

}

}  // namespace test
}  // namespace utils
}  // namespace onnxruntime
