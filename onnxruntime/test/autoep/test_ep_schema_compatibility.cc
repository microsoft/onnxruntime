// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/plugin_ep/ep_schema_compatibility.h"

#include <cstring>
#include <memory>
#include <vector>

#include "core/graph/constants.h"
#include "core/graph/schema_abi_digest.h"
#include "gtest/gtest.h"
#include "onnx/defs/schema.h"
#include "test/test_environment.h"
#include "test/util/include/asserts.h"

namespace onnxruntime::test {
namespace {

struct TestFactory {
  TestFactory() {
    api.ort_version_supported = 30;
    api.GetName = GetName;
    api.GetOperatorCompatibilityInfo = GetCompatibilityInfo;
  }

  static const char* ORT_API_CALL GetName(const OrtEpFactory*) noexcept { return "SchemaCompatibilityTestEP"; }

  static OrtStatus* ORT_API_CALL GetCompatibilityInfo(
      OrtEpFactory* this_ptr,
      const OrtEpOperatorCompatibilityInfo** entries,
      size_t* num_entries) noexcept {
    auto& factory = *reinterpret_cast<TestFactory*>(this_ptr);
    *entries = factory.entries.empty() ? nullptr : factory.entries.data();
    *num_entries = factory.entries.size();
    return nullptr;
  }

  // Keep OrtEpFactory first so the callback can recover the enclosing test object.
  OrtEpFactory api{};
  std::vector<OrtEpOperatorCompatibilityInfo> entries;
};

OrtEpOperatorCompatibilityInfo MakeGqaEntry() {
  const auto* schema = ONNX_NAMESPACE::OpSchemaRegistry::Schema(
      "GroupQueryAttention", 1, kMSDomain);
  EXPECT_NE(schema, nullptr);

  SchemaAbiDigest digest{};
  const auto status = ComputeSchemaAbiDigest(*schema, digest);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();

  OrtEpOperatorCompatibilityInfo entry{kMSDomain, "GroupQueryAttention", 1, {}};
  std::memcpy(entry.schema_abi_digest, digest.data(), digest.size());
  return entry;
}

TEST(PluginEpSchemaCompatibilityTest, MissingCallbackIsPermissiveDuringTransition) {
  TestFactory factory;
  factory.api.GetOperatorCompatibilityInfo = nullptr;

  std::shared_ptr<const PluginEpSchemaCompatibility> compatibility;
  ASSERT_STATUS_OK(PluginEpSchemaCompatibility::Create(
      factory.api, DefaultLoggingManager().DefaultLogger(), compatibility));

  EXPECT_FALSE(compatibility->IsNegotiated());
  EXPECT_TRUE(compatibility->IsCompatible(kMSDomain, "GroupQueryAttention", 999));
}

TEST(PluginEpSchemaCompatibilityTest, AcceptsMatchingEntryAndQuarantinesMismatch) {
  TestFactory factory;
  factory.entries.push_back(MakeGqaEntry());

  std::shared_ptr<const PluginEpSchemaCompatibility> compatibility;
  ASSERT_STATUS_OK(PluginEpSchemaCompatibility::Create(
      factory.api, DefaultLoggingManager().DefaultLogger(), compatibility));
  ASSERT_TRUE(compatibility->IsNegotiated());
  EXPECT_TRUE(compatibility->IsCompatible(kMSDomain, "GroupQueryAttention", 1));

  factory.entries[0].schema_abi_digest[0] ^= 1;
  ASSERT_STATUS_OK(PluginEpSchemaCompatibility::Create(
      factory.api, DefaultLoggingManager().DefaultLogger(), compatibility));
  EXPECT_FALSE(compatibility->IsCompatible(kMSDomain, "GroupQueryAttention", 1));
}

TEST(PluginEpSchemaCompatibilityTest, MissingEntryDoesNotAffectStandardOnnxDomain) {
  TestFactory factory;

  std::shared_ptr<const PluginEpSchemaCompatibility> compatibility;
  ASSERT_STATUS_OK(PluginEpSchemaCompatibility::Create(
      factory.api, DefaultLoggingManager().DefaultLogger(), compatibility));

  EXPECT_FALSE(compatibility->IsCompatible(kMSDomain, "GroupQueryAttention", 1));
  EXPECT_TRUE(compatibility->IsCompatible(kOnnxDomain, "Add", 14));
}

}  // namespace
}  // namespace onnxruntime::test
