// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensor_shape.h"

#include <vector>

#include "gtest/gtest.h"

namespace onnxruntime {
namespace utils {
namespace test {

static void TestShapeWithVector(const TensorShapeVector& vector) {

  // Test constructing from a vector
  TensorShape shape{vector};
  EXPECT_EQ(shape, gsl::make_span(vector));

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
  TestShapeWithVector({});
  TestShapeWithVector({10});
  TestShapeWithVector({10, 20});
  TestShapeWithVector({10, 20, 30});
  TestShapeWithVector({10, 20, 30, 40});
  TestShapeWithVector({12, 23, 34, 45, 56, 67, 78, 89, 90});

  // Test assigning a shape to a large then a small vector (causing it to switch from small block to large, then back to small)
  TensorShapeVector small{1, 2, 3};
  TensorShapeVector large{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  TensorShape shape{small};
  EXPECT_EQ(shape.GetDims(), gsl::make_span(small));

  shape=TensorShape{large};
  EXPECT_EQ(shape.GetDims(), gsl::make_span(large));

  shape=TensorShape{small};
  EXPECT_EQ(shape.GetDims(), gsl::make_span(small));
}

TEST(TensorShapeTest, FromExistingBuffer) {

  std::vector<int64_t> buffer{12, 23, 34, 45, 56, 67, 78, 89};
  auto shape = TensorShape::FromExistingBuffer(buffer);
  auto shape_copy=shape;

  // Pointers and sizes should match as they're the same buffer
  EXPECT_EQ(gsl::make_span(buffer).begin(), shape.GetDims().begin());
  EXPECT_EQ(gsl::make_span(buffer).size(), shape.GetDims().size());

  // Pointers should not match as they're no longer the same buffer
  EXPECT_NE(gsl::make_span(buffer).begin(), shape_copy.GetDims().begin());
  // Size should still match
  EXPECT_EQ(gsl::make_span(buffer).size(), shape_copy.GetDims().size());

  EXPECT_EQ(shape, shape_copy);

  // Test assigning from an empty shape
  TensorShape empty_shape;
  shape_copy=empty_shape;

  EXPECT_EQ(shape_copy, empty_shape);
}

}  // namespace test
}  // namespace utils
}  // namespace onnxruntime
