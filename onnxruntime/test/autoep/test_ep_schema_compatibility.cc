// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/plugin_ep/ep_schema_compatibility.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <exception>
#include <iterator>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/framework/kernel_def_builder.h"
#include "core/framework/kernel_registry.h"
#include "core/graph/constants.h"
#include "core/graph/contrib_ops/ms_schema_abi_manifest.h"
#include "core/graph/model.h"
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

constexpr const char* kSchemaExperimentProvider = "SchemaCompatibilityTestEP";

#if !defined(DISABLE_CONTRIB_OPS)
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

constexpr const char* kSchemaExperimentOp = "PluginSchemaCompatibilityExperiment";

ONNX_NAMESPACE::OpSchema MakeSchemaExperimentOp(int since_version, bool use_v2_contract) {
  ONNX_NAMESPACE::OpSchema schema(kSchemaExperimentOp, __FILE__, __LINE__);
  schema.SetName(kSchemaExperimentOp)
      .SetDomain(kMSDomain)
      .SinceVersion(since_version)
      .SetDoc("Synthetic schema used to exercise plugin EP compatibility negotiation.")
      .Input(0, "X", "Primary input.", "T")
      .Output(0, "Y", "Primary output.", "T")
      .TypeConstraint("T",
                      use_v2_contract ? std::vector<std::string>{"tensor(float)", "tensor(int32)"}
                                      : std::vector<std::string>{"tensor(float)"},
                      "Allowed tensor types.");

  if (use_v2_contract) {
    schema.Input(1, "Scale", "Input added in v2.", "T")
        .Output(1, "Metadata", "Output added in v2.", "T")
        .Attr("axis", "Attribute added in v2.", ONNX_NAMESPACE::AttributeProto::INT,
              /*required=*/true);
  }

  return schema;
}

SchemaAbiDigest DigestSchema(const ONNX_NAMESPACE::OpSchema& schema) {
  SchemaAbiDigest digest{};
  const auto status = ComputeSchemaAbiDigest(schema, digest);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  return digest;
}

class ScopedSchemaExperimentRegistration {
 public:
  ScopedSchemaExperimentRegistration() {
    auto& versions = ONNX_NAMESPACE::OpSchemaRegistry::DomainToVersionRange::Instance();
    const auto range = versions.Map().at(kMSDomain);
    original_min_version_ = range.first;
    original_max_version_ = range.second;
    original_last_release_version_ = versions.LastReleaseVersionMap().at(kMSDomain);

    versions.UpdateDomainToVersion(kMSDomain, original_min_version_,
                                   std::max(original_max_version_, 2),
                                   std::max(original_last_release_version_, 2));
    ONNX_NAMESPACE::RegisterSchema(MakeSchemaExperimentOp(1, /*use_v2_contract=*/false),
                                   /*opset_version_to_load=*/0,
                                   /*fail_duplicate_schema=*/true,
                                   /*fail_with_exception=*/true);
    registered_v1_ = true;
    ONNX_NAMESPACE::RegisterSchema(MakeSchemaExperimentOp(2, /*use_v2_contract=*/true),
                                   /*opset_version_to_load=*/0,
                                   /*fail_duplicate_schema=*/true,
                                   /*fail_with_exception=*/true);
    registered_v2_ = true;
  }

  ~ScopedSchemaExperimentRegistration() {
    if (registered_v2_) {
      ONNX_NAMESPACE::DeregisterSchema(kSchemaExperimentOp, 2, kMSDomain);
    }
    if (registered_v1_) {
      ONNX_NAMESPACE::DeregisterSchema(kSchemaExperimentOp, 1, kMSDomain);
    }

    ONNX_NAMESPACE::OpSchemaRegistry::DomainToVersionRange::Instance().UpdateDomainToVersion(
        kMSDomain, original_min_version_, original_max_version_, original_last_release_version_);
  }

  ScopedSchemaExperimentRegistration(const ScopedSchemaExperimentRegistration&) = delete;
  ScopedSchemaExperimentRegistration& operator=(const ScopedSchemaExperimentRegistration&) = delete;

  void SetCoreMaxOpset(int max_version) const {
    ONNX_NAMESPACE::OpSchemaRegistry::DomainToVersionRange::Instance().UpdateDomainToVersion(
        kMSDomain, original_min_version_, max_version, max_version);
  }

 private:
  int original_min_version_ = 0;
  int original_max_version_ = 0;
  int original_last_release_version_ = 0;
  bool registered_v1_ = false;
  bool registered_v2_ = false;
};

OrtEpOperatorCompatibilityInfo MakeSchemaExperimentEntry(int since_version) {
  const auto* schema = ONNX_NAMESPACE::OpSchemaRegistry::Schema(
      kSchemaExperimentOp, since_version, kMSDomain);
  EXPECT_NE(schema, nullptr);
  EXPECT_EQ(schema == nullptr ? 0 : schema->since_version(), since_version);

  const auto digest = schema == nullptr ? SchemaAbiDigest{} : DigestSchema(*schema);
  OrtEpOperatorCompatibilityInfo entry{kMSDomain, kSchemaExperimentOp, since_version, {}};
  std::memcpy(entry.schema_abi_digest, digest.data(), digest.size());
  return entry;
}

KernelRegistry MakeSchemaExperimentKernelRegistry(gsl::span<const int> versions,
                                                  bool use_open_ended_v1 = false) {
  KernelRegistry registry;
  for (const int version : versions) {
    KernelDefBuilder builder;
    builder.SetName(kSchemaExperimentOp)
        .SetDomain(kMSDomain)
        .Provider(kSchemaExperimentProvider);
    if (use_open_ended_v1) {
      EXPECT_EQ(version, 1);
      builder.SinceVersion(1);
    } else {
      builder.SinceVersion(version, version);
    }

    const auto status = registry.Register(
        KernelCreateInfo(builder.Build(), KernelCreateFn{}));
    EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  }

  return registry;
}

bool SchemaExperimentModelLoads(int opset_version) {
  ONNX_NAMESPACE::ModelProto model_proto;
  model_proto.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  auto* opset = model_proto.add_opset_import();
  opset->set_domain(kMSDomain);
  opset->set_version(opset_version);

  auto* graph = model_proto.mutable_graph();
  graph->set_name("plugin_schema_compatibility_experiment");

  const int elem_type = opset_version == 1
                            ? ONNX_NAMESPACE::TensorProto_DataType_FLOAT
                            : ONNX_NAMESPACE::TensorProto_DataType_INT32;
  const auto add_value_info = [graph, elem_type](bool is_input, const char* name) {
    auto* value_info = is_input ? graph->add_input() : graph->add_output();
    value_info->set_name(name);
    auto* tensor_type = value_info->mutable_type()->mutable_tensor_type();
    tensor_type->set_elem_type(elem_type);
    tensor_type->mutable_shape()->add_dim()->set_dim_value(1);
  };

  add_value_info(/*is_input=*/true, "X");
  add_value_info(/*is_input=*/false, "Y");
  if (opset_version == 2) {
    add_value_info(/*is_input=*/true, "Scale");
    add_value_info(/*is_input=*/false, "Metadata");
  }

  auto* node = graph->add_node();
  node->set_name("experiment_node");
  node->set_op_type(kSchemaExperimentOp);
  node->set_domain(kMSDomain);
  node->add_input("X");
  node->add_output("Y");
  if (opset_version == 2) {
    node->add_input("Scale");
    node->add_output("Metadata");
    auto* attribute = node->add_attribute();
    attribute->set_name("axis");
    attribute->set_type(ONNX_NAMESPACE::AttributeProto::INT);
    attribute->set_i(0);
  }

  try {
    std::shared_ptr<Model> model;
    return Model::Load(std::move(model_proto), model, nullptr,
                       DefaultLoggingManager().DefaultLogger())
        .IsOK();
  } catch (const std::exception&) {
    return false;
  }
}

bool RegistryHasSchemaExperimentKernel(const KernelRegistry& registry, int version) {
  const KernelCreateInfo* kernel = nullptr;
  const KernelRegistry::TypeConstraintMap no_type_constraints;
  return registry.TryFindKernel(kSchemaExperimentProvider, kSchemaExperimentOp, kMSDomain,
                                version, no_type_constraints,
                                DefaultLoggingManager().DefaultLogger(), &kernel)
             .IsOK() &&
         kernel != nullptr;
}
#endif  // !defined(DISABLE_CONTRIB_OPS)

TEST(PluginEpSchemaCompatibilityTest, MissingCallbackIsPermissiveDuringTransition) {
  TestFactory factory;
  factory.api.GetOperatorCompatibilityInfo = nullptr;

  std::shared_ptr<const PluginEpSchemaCompatibility> compatibility;
  ASSERT_STATUS_OK(PluginEpSchemaCompatibility::Create(
      factory.api, DefaultLoggingManager().DefaultLogger(), compatibility));

  EXPECT_FALSE(compatibility->IsNegotiated());
  EXPECT_TRUE(compatibility->IsCompatible(kMSDomain, "GroupQueryAttention", 999));
}

#if !defined(DISABLE_CONTRIB_OPS)
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
#endif  // !defined(DISABLE_CONTRIB_OPS)

TEST(PluginEpSchemaCompatibilityTest, MissingEntryDoesNotAffectStandardOnnxDomain) {
  TestFactory factory;

  std::shared_ptr<const PluginEpSchemaCompatibility> compatibility;
  ASSERT_STATUS_OK(PluginEpSchemaCompatibility::Create(
      factory.api, DefaultLoggingManager().DefaultLogger(), compatibility));

  EXPECT_FALSE(compatibility->IsCompatible(kMSDomain, "GroupQueryAttention", 1));
  EXPECT_TRUE(compatibility->IsCompatible(kOnnxDomain, "Add", 14));
}

#if defined(DISABLE_CONTRIB_OPS)
TEST(PluginEpSchemaCompatibilityTest, FiltersPublishedContribKernelWhenSchemasAreDisabled) {
  TestFactory factory;
  factory.entries.push_back(
      OrtEpOperatorCompatibilityInfo{kMSDomain, "GroupQueryAttention", 1, {}});

  std::shared_ptr<const PluginEpSchemaCompatibility> compatibility;
  ASSERT_STATUS_OK(PluginEpSchemaCompatibility::Create(
      factory.api, DefaultLoggingManager().DefaultLogger(), compatibility));
  ASSERT_TRUE(compatibility->IsNegotiated());
  EXPECT_FALSE(compatibility->IsCompatible(kMSDomain, "GroupQueryAttention", 1));

  KernelRegistry source_registry;
  auto kernel = KernelDefBuilder()
                    .SetName("GroupQueryAttention")
                    .SetDomain(kMSDomain)
                    .SinceVersion(1, 1)
                    .Provider(kSchemaExperimentProvider)
                    .Build();
  ASSERT_STATUS_OK(source_registry.Register(
      KernelCreateInfo(std::move(kernel), KernelCreateFn{})));

  TestEp ep{source_registry};
  std::shared_ptr<KernelRegistry> effective_registry;
  ASSERT_STATUS_OK(GetPluginEpKernelRegistry(
      ep, *compatibility, DefaultLoggingManager().DefaultLogger(), effective_registry));
  ASSERT_NE(effective_registry, nullptr);
  EXPECT_TRUE(effective_registry->GetKernelCreateMap().empty());
}
#else
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

TEST(PluginEpSchemaCompatibilityTest, VersionedContribSchemaCompatibilityMatrix) {
  ScopedSchemaExperimentRegistration schemas;

  struct CoreVariant {
    std::string_view name;
    uint32_t api_version;
    int max_ms_opset;
  };
  const std::array<CoreVariant, 3> cores = {{{"ORT0", 29, 1},
                                             {"ORT1", 30, 1},
                                             {"ORT2", 30, 2}}};

  struct PluginVariant {
    std::string_view name;
    uint32_t api_version;
    std::vector<int> manifest_versions;
    std::vector<int> kernel_versions;
    bool open_ended_v1_kernel;
  };
  const std::array<PluginVariant, 3> plugins = {{
      {"CUDA0", 29, {}, {1}, true},
      // Existing contrib registrations use the open-ended form, which the
      // kernel registry treats as an exact match for the start version.
      {"CUDA1", 30, {1}, {1}, true},
      // A v2 plugin retains its v1 contract and kernel for old models and cores.
      {"CUDA2", 30, {1, 2}, {1, 2}, false},
  }};

  enum class Outcome {
    kModelRejected,
    kCudaKernel,
    kFallback,
  };
  const Outcome expected[3][3][2] = {
      {// ORT0: the pre-API-30 core ignores schema metadata and cannot load OP2.
       {Outcome::kCudaKernel, Outcome::kModelRejected},
       {Outcome::kCudaKernel, Outcome::kModelRejected},
       {Outcome::kCudaKernel, Outcome::kModelRejected}},
      {// ORT1: API-30 negotiation is active, but the core still cannot load OP2.
       {Outcome::kCudaKernel, Outcome::kModelRejected},
       {Outcome::kCudaKernel, Outcome::kModelRejected},
       {Outcome::kCudaKernel, Outcome::kModelRejected}},
      {// ORT2: OP2 loads, but only CUDA2 has its matching contract and kernel.
       {Outcome::kCudaKernel, Outcome::kFallback},
       {Outcome::kCudaKernel, Outcome::kFallback},
       {Outcome::kCudaKernel, Outcome::kCudaKernel}},
  };

  for (size_t core_index = 0; core_index < cores.size(); ++core_index) {
    const auto& core = cores[core_index];
    schemas.SetCoreMaxOpset(core.max_ms_opset);

    const std::array<bool, 2> model_loads = {
        SchemaExperimentModelLoads(1),
        SchemaExperimentModelLoads(2),
    };
    EXPECT_TRUE(model_loads[0]) << core.name << " failed to load OP1";
    EXPECT_EQ(model_loads[1], core.max_ms_opset >= 2)
        << core.name << " returned an unexpected OP2 model-load result";

    for (size_t plugin_index = 0; plugin_index < plugins.size(); ++plugin_index) {
      const auto& plugin = plugins[plugin_index];
      SCOPED_TRACE(std::string(core.name) + " x " + std::string(plugin.name));

      TestFactory factory;
      factory.api.ort_version_supported = plugin.api_version;
      for (const int version : plugin.manifest_versions) {
        factory.entries.push_back(MakeSchemaExperimentEntry(version));
      }

      if (core.api_version < 30) {
        // Exercise the behavior of a pre-API-30 core: it has no appended callback
        // field to invoke, even if a newer plugin binary contains that field.
        factory.api.ort_version_supported = core.api_version;
        factory.api.GetOperatorCompatibilityInfo = nullptr;
      } else if (plugin.api_version < 30) {
        factory.api.GetOperatorCompatibilityInfo = nullptr;
      }

      std::shared_ptr<const PluginEpSchemaCompatibility> compatibility;
      ASSERT_STATUS_OK(PluginEpSchemaCompatibility::Create(
          factory.api, DefaultLoggingManager().DefaultLogger(), compatibility));
      const bool expect_negotiated = core.api_version >= 30 && plugin.api_version >= 30;
      EXPECT_EQ(compatibility->IsNegotiated(), expect_negotiated);

      KernelRegistry source_registry = MakeSchemaExperimentKernelRegistry(
          gsl::make_span(plugin.kernel_versions), plugin.open_ended_v1_kernel);
      TestEp ep{source_registry};
      std::shared_ptr<KernelRegistry> effective_registry;
      ASSERT_STATUS_OK(GetPluginEpKernelRegistry(
          ep, *compatibility, DefaultLoggingManager().DefaultLogger(), effective_registry));
      ASSERT_NE(effective_registry, nullptr);

      for (int model_version = 1; model_version <= 2; ++model_version) {
        const bool has_kernel = RegistryHasSchemaExperimentKernel(*effective_registry, model_version);
        const bool source_has_kernel =
            std::find(plugin.kernel_versions.begin(), plugin.kernel_versions.end(), model_version) !=
            plugin.kernel_versions.end();
        const bool plugin_published_contract =
            std::find(plugin.manifest_versions.begin(), plugin.manifest_versions.end(), model_version) !=
            plugin.manifest_versions.end();
        const bool expected_has_kernel =
            source_has_kernel &&
            (!expect_negotiated ||
             (model_version <= core.max_ms_opset && plugin_published_contract));
        EXPECT_EQ(has_kernel, expected_has_kernel)
            << "Unexpected effective-registry result for OP" << model_version;

        const Outcome actual = !model_loads[model_version - 1]
                                   ? Outcome::kModelRejected
                               : has_kernel ? Outcome::kCudaKernel
                                            : Outcome::kFallback;
        EXPECT_EQ(actual, expected[core_index][plugin_index][model_version - 1])
            << "Unexpected result for OP" << model_version;
      }
    }
  }
}

TEST(PluginEpSchemaCompatibilityTest, SameVersionContractChangeIsQuarantined) {
  ScopedSchemaExperimentRegistration schemas;
  schemas.SetCoreMaxOpset(1);

  // This simulates the original failure mode: a plugin changes inputs, outputs,
  // attributes, and allowed data types, but incorrectly continues to identify the
  // changed contract as schema version 1.
  const auto incompatible_digest =
      DigestSchema(MakeSchemaExperimentOp(1, /*use_v2_contract=*/true));
  OrtEpOperatorCompatibilityInfo incompatible_entry{kMSDomain, kSchemaExperimentOp, 1, {}};
  std::memcpy(incompatible_entry.schema_abi_digest, incompatible_digest.data(),
              incompatible_digest.size());

  TestFactory factory;
  factory.entries.push_back(incompatible_entry);
  std::shared_ptr<const PluginEpSchemaCompatibility> compatibility;
  ASSERT_STATUS_OK(PluginEpSchemaCompatibility::Create(
      factory.api, DefaultLoggingManager().DefaultLogger(), compatibility));
  ASSERT_TRUE(compatibility->IsNegotiated());
  EXPECT_FALSE(compatibility->IsCompatible(kMSDomain, kSchemaExperimentOp, 1));

  const std::array<int, 1> v1 = {1};
  KernelRegistry source_registry = MakeSchemaExperimentKernelRegistry(gsl::make_span(v1));
  TestEp ep{source_registry};
  std::shared_ptr<KernelRegistry> effective_registry;
  ASSERT_STATUS_OK(GetPluginEpKernelRegistry(
      ep, *compatibility, DefaultLoggingManager().DefaultLogger(), effective_registry));
  ASSERT_NE(effective_registry, nullptr);
  EXPECT_FALSE(RegistryHasSchemaExperimentKernel(*effective_registry, 1));

  // The temporary legacy policy cannot detect this historical same-version
  // mismatch. This is why published v1 schemas must remain frozen and every
  // post-API-30 contract change must get a new schema version.
  factory.api.ort_version_supported = 29;
  factory.api.GetOperatorCompatibilityInfo = nullptr;
  ASSERT_STATUS_OK(PluginEpSchemaCompatibility::Create(
      factory.api, DefaultLoggingManager().DefaultLogger(), compatibility));
  ASSERT_FALSE(compatibility->IsNegotiated());
  ASSERT_STATUS_OK(GetPluginEpKernelRegistry(
      ep, *compatibility, DefaultLoggingManager().DefaultLogger(), effective_registry));
  ASSERT_NE(effective_registry, nullptr);
  EXPECT_TRUE(RegistryHasSchemaExperimentKernel(*effective_registry, 1));
}
#endif  // defined(DISABLE_CONTRIB_OPS)

}  // namespace
}  // namespace onnxruntime::test
