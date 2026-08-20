// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/plugin_ep/ep_schema_compatibility.h"

#include <cstring>
#include <iterator>
#include <memory>
#include <vector>

#include "core/framework/kernel_def_builder.h"
#include "core/framework/kernel_registry.h"
#include "core/graph/constants.h"
#include "core/graph/contrib_ops/ms_schema_abi_manifest.h"
#include "core/graph/schema_abi_digest.h"
#include "core/session/plugin_ep/ep_kernel_registration.h"
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

struct TestEp : OrtEp {
  explicit TestEp(const KernelRegistry& kernel_registry) : OrtEp{}, kernel_registry_{kernel_registry} {
    ort_version_supported = 30;
    GetKernelRegistry = GetKernelRegistryImpl;
  }

  static OrtStatus* ORT_API_CALL GetKernelRegistryImpl(
      OrtEp* this_ptr, const OrtKernelRegistry** kernel_registry) noexcept {
    const auto& ep = *static_cast<const TestEp*>(this_ptr);
    *kernel_registry = reinterpret_cast<const OrtKernelRegistry*>(&ep.kernel_registry_);
    return nullptr;
  }

  const KernelRegistry& kernel_registry_;
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

TEST(PluginEpSchemaCompatibilityTest, QuarantinesDuplicateEntry) {
  TestFactory factory;
  factory.entries.push_back(MakeGqaEntry());
  factory.entries.push_back(factory.entries.front());

  std::shared_ptr<const PluginEpSchemaCompatibility> compatibility;
  ASSERT_STATUS_OK(PluginEpSchemaCompatibility::Create(
      factory.api, DefaultLoggingManager().DefaultLogger(), compatibility));

  ASSERT_TRUE(compatibility->IsNegotiated());
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

TEST(PluginEpSchemaCompatibilityTest, AcceptsPublishedMSDomainManifest) {
  const auto& domain_versions = ONNX_NAMESPACE::OpSchemaRegistry::DomainToVersionRange::Instance();
  const auto ms_domain_range = domain_versions.Map().find(kMSDomain);
  const auto ms_domain_last_release = domain_versions.LastReleaseVersionMap().find(kMSDomain);
  ASSERT_NE(ms_domain_range, domain_versions.Map().end());
  ASSERT_NE(ms_domain_last_release, domain_versions.LastReleaseVersionMap().end());
  EXPECT_EQ(ms_domain_range->second.second, kMSDomainOpsetVersion);
  EXPECT_EQ(ms_domain_last_release->second, kMSDomainOpsetVersionLastReleased);

  TestFactory factory;
  factory.entries.assign(std::begin(contrib::kMSDomainSchemaAbiManifest),
                         std::end(contrib::kMSDomainSchemaAbiManifest));

  std::shared_ptr<const PluginEpSchemaCompatibility> compatibility;
  ASSERT_STATUS_OK(PluginEpSchemaCompatibility::Create(
      factory.api, DefaultLoggingManager().DefaultLogger(), compatibility));
  ASSERT_TRUE(compatibility->IsNegotiated());

  for (const auto& entry : contrib::kMSDomainSchemaAbiManifest) {
    EXPECT_TRUE(compatibility->IsCompatible(entry.domain, entry.op_type, entry.since_version))
        << entry.op_type << "@" << entry.since_version;
  }
}

TEST(PluginEpSchemaCompatibilityTest, KeepsCoreVisiblePartOfNewerKernelRanges) {
  TestFactory factory;
  factory.entries.assign(std::begin(contrib::kMSDomainSchemaAbiManifest),
                         std::end(contrib::kMSDomainSchemaAbiManifest));

  std::shared_ptr<const PluginEpSchemaCompatibility> compatibility;
  ASSERT_STATUS_OK(PluginEpSchemaCompatibility::Create(
      factory.api, DefaultLoggingManager().DefaultLogger(), compatibility));

  KernelRegistry source_registry;
  auto visible_and_future = KernelDefBuilder()
                                .SetName("GroupQueryAttention")
                                .SetDomain(kMSDomain)
                                .SinceVersion(1, kMSDomainOpsetVersion + 1)
                                .Provider("SchemaCompatibilityTestEP")
                                .Build();
  ASSERT_STATUS_OK(source_registry.Register(
      KernelCreateInfo(std::move(visible_and_future), KernelCreateFn{})));

  TestEp ep{source_registry};
  std::shared_ptr<KernelRegistry> effective_registry;
  ASSERT_STATUS_OK(GetPluginEpKernelRegistry(
      ep, *compatibility, DefaultLoggingManager().DefaultLogger(), effective_registry));
  ASSERT_NE(effective_registry, nullptr);
  EXPECT_EQ(effective_registry->GetKernelCreateMap().size(), 1u);

  KernelRegistry future_registry;
  auto future_only = KernelDefBuilder()
                         .SetName("GroupQueryAttention")
                         .SetDomain(kMSDomain)
                         .SinceVersion(kMSDomainOpsetVersion + 1, kMSDomainOpsetVersion + 1)
                         .Provider("SchemaCompatibilityTestEP")
                         .Build();
  ASSERT_STATUS_OK(future_registry.Register(
      KernelCreateInfo(std::move(future_only), KernelCreateFn{})));

  TestEp future_ep{future_registry};
  ASSERT_STATUS_OK(GetPluginEpKernelRegistry(
      future_ep, *compatibility, DefaultLoggingManager().DefaultLogger(), effective_registry));
  ASSERT_NE(effective_registry, nullptr);
  EXPECT_TRUE(effective_registry->GetKernelCreateMap().empty());
}

}  // namespace
}  // namespace onnxruntime::test
