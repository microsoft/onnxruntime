// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/schema_abi_digest.h"

#include <algorithm>
#include <cstdio>
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
constexpr const char* kRegenerateHint =
    "Regenerate onnxruntime/core/graph/contrib_ops/ms_schema_abi_manifest.inc with "
    "'onnxruntime_test_all --gtest_also_run_disabled_tests "
    "--gtest_filter=SchemaAbiDigestTest.DISABLED_GenerateMSManifest'.";

// Returns the finalized com.microsoft schemas that the checked-in manifest must cover,
// excluding schemas registered by unit tests.
std::vector<ONNX_NAMESPACE::OpSchema> CollectMSDomainSchemas() {
  auto schemas = ONNX_NAMESPACE::OpSchemaRegistry::get_all_schemas_with_history();
  schemas.erase(std::remove_if(schemas.begin(), schemas.end(), [](const auto& schema) {
                  const auto& schema_file = schema.file();
                  const bool is_test_schema = schema_file.find("/test/") != std::string::npos ||
                                              schema_file.find("\\test\\") != std::string::npos;
                  return schema.domain() != kMSDomain || is_test_schema;
                }),
                schemas.end());
  std::sort(schemas.begin(), schemas.end(), [](const auto& lhs, const auto& rhs) {
    if (lhs.Name() != rhs.Name()) {
      return lhs.Name() < rhs.Name();
    }
    return lhs.since_version() < rhs.since_version();
  });
  return schemas;
}

// Disabled by default: prints the regenerated manifest body to stdout so it can be
// redirected into ms_schema_abi_manifest.inc.
TEST(SchemaAbiDigestTest, DISABLED_GenerateMSManifest) {
  std::printf("// Generated file. Do not edit individual digest entries by hand.\n");
  for (const auto& schema : CollectMSDomainSchemas()) {
    const auto digest = Digest(schema);
    std::printf("{kMSDomain, \"%s\", %d, {", schema.Name().c_str(), schema.since_version());
    for (size_t i = 0; i < digest.size(); ++i) {
      std::printf("%s0x%02x", i == 0 ? "" : ", ", digest[i]);
    }
    std::printf("}},\n");
  }
}

TEST(SchemaAbiDigestTest, MSManifestMatchesSchemas) {
  const auto schemas = CollectMSDomainSchemas();

#if !defined(ENABLE_TRAINING_OPS) && !defined(ORT_USE_NCCL)
  // The checked-in manifest is the inference plugin catalog. Training and NCCL
  // builds register additional com.microsoft schemas that the plugin does not
  // claim and therefore are not required to appear in this manifest.
  constexpr size_t manifest_size = sizeof(contrib::kMSDomainSchemaAbiManifest) /
                                   sizeof(contrib::kMSDomainSchemaAbiManifest[0]);
  EXPECT_EQ(schemas.size(), manifest_size) << kRegenerateHint;
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
        << " changed without regenerating the com.microsoft schema ABI manifest. "
        << kRegenerateHint;
  }

#if !defined(ENABLE_TRAINING_OPS) && !defined(ORT_USE_NCCL)
  for (const auto& schema : schemas) {
    EXPECT_NE(manifest_keys.find(SchemaKey{kMSDomain, schema.Name(), schema.since_version()}),
              manifest_keys.end())
        << "Missing manifest entry: " << schema.Name() << "@" << schema.since_version() << ". "
        << kRegenerateHint;
  }
#endif
}
#endif  // defined(DISABLE_CONTRIB_OPS)

}  // namespace
}  // namespace onnxruntime::test
