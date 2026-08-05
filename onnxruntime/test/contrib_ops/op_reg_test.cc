// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>
#include "core/graph/constants.h"
#include "core/graph/contrib_ops/contrib_defs.h"
#include "core/graph/model.h"
#include "core/graph/op.h"
#include "core/graph/onnx_protobuf.h"
#include "core/graph/schema_registry.h"
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

TEST(OpRegistrationTest, DeepSeekV4StandaloneOps) {
  static bool contrib_schemas_registered = false;
  if (!contrib_schemas_registered) {
    onnxruntime::contrib::RegisterContribSchemas();
    contrib_schemas_registered = true;
  }

  struct ExpectedSchema {
    const char* name;
    size_t inputs;
    size_t outputs;
  };

  constexpr ExpectedSchema expected_schemas[] = {
      {"HeavilyCompressedAttention", 11, 5},
      {"CompressedSparseAttention", 13, 6},
      {"LightningIndexer", 16, 6},
      {"CompressedAttention", 6, 1},
      {"HashRouter", 4, 3},
  };

  SchemaRegistryManager registry_manager;
  for (const auto& registry : IOnnxRuntimeOpSchemaRegistryList()) {
    registry_manager.RegisterRegistry(registry);
  }

  for (const auto& expected : expected_schemas) {
    const auto* schema = registry_manager.GetSchema(expected.name, 1, onnxruntime::kMSDomain);
    ASSERT_NE(schema, nullptr) << expected.name;
    EXPECT_EQ(schema->inputs().size(), expected.inputs) << expected.name;
    EXPECT_EQ(schema->outputs().size(), expected.outputs) << expected.name;
  }
}
}  // namespace test
}  // namespace onnxruntime
