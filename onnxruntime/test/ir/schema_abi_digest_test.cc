// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/schema_abi_digest.h"

#include <vector>

#include "gtest/gtest.h"
#include "onnx/defs/schema.h"

namespace onnxruntime::test {
namespace {

ONNX_NAMESPACE::OpSchema MakeSchema(const char* doc,
                                    ONNX_NAMESPACE::OpSchema::FormalParameterOption second_input_option,
                                    int64_t attribute_default,
                                    std::vector<std::string> allowed_types = {"tensor(float)", "tensor(float16)"}) {
  ONNX_NAMESPACE::OpSchema schema("DigestTest", __FILE__, __LINE__);
  schema.SetName("DigestTest")
      .SetDomain("com.microsoft")
      .SinceVersion(2)
      .SetDoc(doc)
      .Input(0, "X", "input documentation", "T")
      .Input(1, "Y", "more input documentation", "T", second_input_option)
      .Output(0, "Z", "output documentation", "T")
      .TypeConstraint("T", allowed_types, "type constraint documentation")
      .Attr("axis", "attribute documentation", ONNX_NAMESPACE::AttributeProto::INT, attribute_default);
  return schema;
}

SchemaAbiDigest Digest(const ONNX_NAMESPACE::OpSchema& schema) {
  SchemaAbiDigest result{};
  const auto status = ComputeSchemaAbiDigest(schema, result);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  return result;
}

TEST(SchemaAbiDigestTest, IgnoresDocumentationAndTypeConstraintOrdering) {
  const auto first = MakeSchema("first documentation",
                                ONNX_NAMESPACE::OpSchema::Optional, 1,
                                {"tensor(float)", "tensor(float16)"});
  const auto second = MakeSchema("completely different documentation",
                                 ONNX_NAMESPACE::OpSchema::Optional, 1,
                                 {"tensor(float16)", " tensor( float ) "});

  EXPECT_EQ(Digest(first), Digest(second));
}

TEST(SchemaAbiDigestTest, ChangesWhenExecutionContractChanges) {
  const auto baseline = MakeSchema("documentation", ONNX_NAMESPACE::OpSchema::Optional, 1);
  const auto required_input = MakeSchema("documentation", ONNX_NAMESPACE::OpSchema::Single, 1);
  const auto different_default = MakeSchema("documentation", ONNX_NAMESPACE::OpSchema::Optional, 2);

  EXPECT_NE(Digest(baseline), Digest(required_input));
  EXPECT_NE(Digest(baseline), Digest(different_default));
}

}  // namespace
}  // namespace onnxruntime::test
