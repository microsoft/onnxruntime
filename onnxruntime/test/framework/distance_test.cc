// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/util/distance.h"
#include <gtest/gtest.h>
using testing::Types;

namespace onnxruntime {

template <typename T>
class SqeuclideanTest : public ::testing::Test {
 public:
  T func;
};

using MyTypes = Types<Sqeuclidean<double>, SqeuclideanWithEigen<double> >;

TYPED_TEST_SUITE(SqeuclideanTest, MyTypes);

TYPED_TEST(SqeuclideanTest, test1) {
  double a = 10;
  double b = 5;
  ASSERT_DOUBLE_EQ(25, this->func(&a, &b, 1));
}
}  // namespace onnxruntime
