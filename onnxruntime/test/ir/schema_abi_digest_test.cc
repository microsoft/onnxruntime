// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/schema_abi_digest.h"

#include <algorithm>
#include <cstring>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "core/graph/constants.h"
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

#if defined(DISABLE_CONTRIB_OPS)
TEST(SchemaAbiDigestTest, MSManifestSchemasAreUnavailableWhenContribOpsAreDisabled) {
  for (const auto& entry : contrib::kMSDomainSchemaAbiManifest) {
    EXPECT_EQ(ONNX_NAMESPACE::OpSchemaRegistry::Schema(
                  entry.op_type, entry.since_version, entry.domain),
              nullptr)
        << entry.op_type << "@" << entry.since_version;
  }
}
#else
TEST(SchemaAbiDigestTest, MSManifestMatchesSchemas) {
  auto schemas = ONNX_NAMESPACE::OpSchemaRegistry::get_all_schemas_with_history();
  schemas.erase(std::remove_if(schemas.begin(), schemas.end(), [](const auto& schema) {
                  const auto& schema_file = schema.file();
                  const bool is_test_schema = schema_file.find("/test/") != std::string::npos ||
                                              schema_file.find("\\test\\") != std::string::npos;
                  return schema.domain() != kMSDomain || is_test_schema;
                }),
                schemas.end());

#if !defined(ENABLE_TRAINING_OPS) && !defined(ORT_USE_NCCL)
  // The checked-in manifest is the inference plugin catalog. Training and NCCL
  // builds register additional com.microsoft schemas that the plugin does not
  // claim and therefore are not required to appear in this manifest.
  constexpr size_t manifest_size = sizeof(contrib::kMSDomainSchemaAbiManifest) /
                                   sizeof(contrib::kMSDomainSchemaAbiManifest[0]);
  EXPECT_EQ(schemas.size(), manifest_size);
#endif

  using SchemaKey = std::tuple<std::string, std::string, int>;
  std::set<SchemaKey> manifest_keys;
  for (const auto& entry : contrib::kMSDomainSchemaAbiManifest) {
    ASSERT_STREQ(entry.domain, kMSDomain);
    ASSERT_TRUE(manifest_keys.emplace(entry.domain, entry.op_type, entry.since_version).second)
        << "Duplicate manifest entry: " << entry.op_type << "@" << entry.since_version;

    const auto schema = std::find_if(schemas.begin(), schemas.end(), [&](const auto& candidate) {
      return candidate.Name() == entry.op_type && candidate.since_version() == entry.since_version;
    });
    ASSERT_NE(schema, schemas.end()) << entry.op_type << "@" << entry.since_version;

    const auto digest = Digest(*schema);
    EXPECT_EQ(std::memcmp(digest.data(), entry.schema_abi_digest, digest.size()), 0)
        << entry.op_type << "@" << entry.since_version
        << " changed without regenerating the com.microsoft schema ABI manifest";
  }

#if !defined(ENABLE_TRAINING_OPS) && !defined(ORT_USE_NCCL)
  for (const auto& schema : schemas) {
    EXPECT_NE(manifest_keys.find(SchemaKey{kMSDomain, schema.Name(), schema.since_version()}),
              manifest_keys.end())
        << "Missing manifest entry: " << schema.Name() << "@" << schema.since_version();
  }
#endif
}
#endif  // defined(DISABLE_CONTRIB_OPS)

}  // namespace
}  // namespace onnxruntime::test
