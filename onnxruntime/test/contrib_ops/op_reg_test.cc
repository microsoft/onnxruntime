// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>
#include "core/graph/model.h"
#include "core/graph/op.h"
#include "core/graph/onnx_protobuf.h"
#include "gtest/gtest.h"

using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace test {

TEST(OpRegistrationTest, AffineOp) {
  auto op = OpSchemaRegistry::Schema("Affine");
  EXPECT_TRUE(nullptr != op);
  size_t input_size = op->inputs().size();
  EXPECT_EQ(input_size, 1u);
  EXPECT_EQ(op->inputs()[0].GetTypes(), op->outputs()[0].GetTypes());
  size_t attr_size = op->attributes().size();
  EXPECT_EQ(attr_size, 2u);
  auto attr_alpha = op->attributes().find("alpha")->second;
  EXPECT_EQ(attr_alpha.name, "alpha");
  EXPECT_EQ(attr_alpha.type, AttrType::AttributeProto_AttributeType_FLOAT);
  auto attr_beta = op->attributes().find("beta")->second;
  EXPECT_EQ(attr_beta.name, "beta");
  EXPECT_EQ(attr_beta.type, AttrType::AttributeProto_AttributeType_FLOAT);
}
}  // namespace test
}  // namespace onnxruntime
