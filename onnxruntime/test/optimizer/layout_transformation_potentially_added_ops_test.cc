// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/layout_transformation/layout_transformation_potentially_added_ops.h"

#include "gtest/gtest.h"

#include "onnx/defs/schema.h"

#include "core/graph/constants.h"

namespace onnxruntime::test {

// This test is to ensure the latest opset version for ops which can be added
// during layout transformation step are added. If this test fails then it means
// there is a new version available for one of the ops in the map.
TEST(LayoutTransformationPotentiallyAddedOpsTests, OpsHaveLatestVersions) {
  const auto* schema_registry = ONNX_NAMESPACE::OpSchemaRegistry::Instance();

  // kLayoutTransformationPotentiallyAddedOps is sorted in increasing order of <domain, op_type, since_version>
  // iterate backwards and check the latest since_version of each domain, op_type
  std::string_view prev_domain, prev_op_type{};

  for (auto it = std::rbegin(kLayoutTransformationPotentiallyAddedOps),
            end = std::rend(kLayoutTransformationPotentiallyAddedOps);
       it != end; ++it) {
    if (prev_domain != it->domain || prev_op_type != it->op_type) {
      const auto* schema = schema_registry->GetSchema(std::string{it->op_type}, INT_MAX, std::string{it->domain});
      ASSERT_NE(schema, nullptr);
      EXPECT_EQ(schema->SinceVersion(), it->since_version)
          << "A new version for op " << it->op_type << " (" << schema->SinceVersion()
          << ") is available. Please update kLayoutTransformationPotentiallyAddedOps to include it.";
      prev_domain = it->domain;
      prev_op_type = it->op_type;
    }
  }
}

}  // namespace onnxruntime::test
