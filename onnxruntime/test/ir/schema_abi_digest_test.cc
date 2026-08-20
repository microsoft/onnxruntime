// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/schema_abi_digest.h"

#include <algorithm>
#include <cstring>
#include <vector>

#include "core/graph/constants.h"
#include "core/graph/contrib_ops/ms_opset.h"
#include "core/graph/contrib_ops/ms_schema_abi_manifest.h"
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

TEST(SchemaAbiDigestTest, MSManifestMatchesSchemas) {
  std::vector<ONNX_NAMESPACE::OpSchema> schemas;
  contrib::OpSet_Microsoft_ver1::ForEachSchema(
      [&](ONNX_NAMESPACE::OpSchema&& schema) { schemas.push_back(std::move(schema)); });

  constexpr size_t manifest_size = sizeof(contrib::kMSDomainSchemaAbiManifest) /
                                   sizeof(contrib::kMSDomainSchemaAbiManifest[0]);
  ASSERT_EQ(schemas.size(), manifest_size);

  for (const auto& entry : contrib::kMSDomainSchemaAbiManifest) {
    ASSERT_STREQ(entry.domain, kMSDomain);
    const auto schema = std::find_if(schemas.begin(), schemas.end(), [&](const auto& candidate) {
      return candidate.Name() == entry.op_type && candidate.since_version() == entry.since_version;
    });
    ASSERT_NE(schema, schemas.end()) << entry.op_type << "@" << entry.since_version;

    const auto digest = Digest(*schema);
    EXPECT_EQ(std::memcmp(digest.data(), entry.schema_abi_digest, digest.size()), 0)
        << entry.op_type << "@" << entry.since_version
        << " changed without regenerating the com.microsoft schema ABI manifest";
  }
}

}  // namespace
}  // namespace onnxruntime::test
