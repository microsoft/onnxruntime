// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_CONTRIB_OPS)

#include <algorithm>
#include <array>
#include <tuple>
#include <utility>

#include "gtest/gtest.h"
#include "contrib_ops/cuda/fusion/cuda_source_replacements.h"
#include "core/framework/op_kernel.h"
#include "core/graph/contrib_ops/source_replacement_legacy.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "core/optimizer/cuda_runtime_fusion_internal.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "test/util/include/asserts.h"
#include "test/util/include/test_environment.h"

namespace onnxruntime::test {
namespace {
using namespace contrib;
using namespace contrib::cuda::fusion;

constexpr char kMS[] = "com.microsoft";
const OpKey kNew{kMS, "NewSourcePolicyTestOp", 1};

InlinedVector<ExecutableSource> Executables() {
  InlinedVector<ExecutableSource> sources;
  for (const auto* source : CudaSourceReplacements()) sources.push_back(source->Metadata());
  return sources;
}

InlinedVector<OpKey> SchemasMissingFromInventory(gsl::span<const ONNX_NAMESPACE::OpSchema> registered,
                                                 gsl::span<const OpKey> schemas) {
  InlinedVector<OpKey> missing;
  for (const auto& schema : registered) {
    auto file = schema.file();
    std::replace(file.begin(), file.end(), '\\', '/');
    if (file.find("/test/") != std::string::npos) continue;
    const bool authored = file.starts_with("onnxruntime/core/graph/contrib_ops/") ||
                          file.find("/onnxruntime/core/graph/contrib_ops/") != std::string::npos ||
                          file.starts_with("orttraining/orttraining/core/graph/") ||
                          file.find("/orttraining/orttraining/core/graph/") != std::string::npos;
    if (schema.domain() == kMSDomain || schema.domain() == kMSNchwcDomain ||
        (authored && (schema.domain() == kOnnxDomain || schema.domain() == kPytorchAtenDomain))) {
      const auto key = SourceOpKey(schema);
      if (std::find(schemas.begin(), schemas.end(), key) == schemas.end()) missing.push_back(key);
    }
  }
  return missing;
}

TEST(ContribSourceReplacementPolicyTest, FullRegisteredInventory) {
  const auto inference_legacy = LegacyInferenceSourcePolicyKeys();
  const auto training_legacy = LegacyTrainingSourcePolicyKeys();
  const auto legacy = LegacySourcePolicyKeys();
  ASSERT_EQ(inference_legacy.size(), 194u);
  ASSERT_EQ(training_legacy.size(), 101u);
  ASSERT_EQ(legacy.size(), 295u);  // Currently disjoint; domain/version are part of each key.
  for (const auto keys : {inference_legacy, training_legacy, legacy}) {
    for (size_t i = 1; i < keys.size(); ++i) {
      EXPECT_LT(std::tie(keys[i - 1].domain, keys[i - 1].name, keys[i - 1].since_version),
                std::tie(keys[i].domain, keys[i].name, keys[i].since_version));
    }
  }
  const auto schemas = GetContribSchemaKeys();
  const auto policies = GetSourcePolicies();
  const auto sources = CudaSourceReplacements();
  ASSERT_GT(schemas.size(), 100u);  // A full inference build, not a reduced inventory claim.
  ASSERT_EQ(sources.size(), 3u);
  // Verify that collected keys correspond to exact, real registrations.
  const auto registered = ONNX_NAMESPACE::OpSchemaRegistry::get_all_schemas_with_history();
  for (const auto& key : schemas) {
    EXPECT_TRUE(std::any_of(registered.begin(), registered.end(),
                            [&](const auto& schema) { return SourceOpKey(schema) == key; }))
        << key.domain << "." << key.name << "-" << key.since_version;
  }
  // Catch schemas that bypass the normal workflow, without interpreting docstrings.
  for (const auto& key : SchemasMissingFromInventory(registered, schemas)) {
    ADD_FAILURE() << key.domain << "." << key.name << "-" << key.since_version;
  }
  ASSERT_STATUS_OK(ValidateCudaSourceReplacements(sources, schemas, policies, LegacySourcePolicyKeys()));
}

TEST(ContribSourceReplacementPolicyTest, InventoryUsesPortableSchemaFileMetadata) {
  const std::pair<const char*, bool> paths[]{
      {"/src/onnxruntime/test/ir/graph_test.cc", true},
      {R"(C:\src\onnxruntime\test\ir\graph_test.cc)", true},
      {R"(C:\src\onnxruntime/test\ir/graph_test.cc)", true},
      {"/src/onnxruntime/core/graph/contrib_ops.cc", false},
      {R"(C:\src\onnxruntime\core\graph\contrib_ops.cc)", false},
      {"/src/onnxruntime/testdata/contrib_ops.cc", false},
      {R"(C:\src\onnxruntime\testdata\contrib_ops.cc)", false},
      {"", false},
  };
  for (const auto* domain : {kMSDomain, kMSNchwcDomain}) {
    for (const auto& [file, is_test] : paths) {
      SCOPED_TRACE(MakeString(domain, ": ", file));
      const std::array registered{
          ONNX_NAMESPACE::OpSchema("Fake_FunctionOp", file, 1).SetDomain(domain).SinceVersion(1),
          ONNX_NAMESPACE::OpSchema("Fake_Sub", file, 2).SetDomain(domain).SinceVersion(1)};
      const std::array keys{SourceOpKey(registered[0]), SourceOpKey(registered[1])};
      const auto missing = SchemasMissingFromInventory(registered, {});
      if (is_test) {
        EXPECT_TRUE(missing.empty());
      } else {
        ASSERT_EQ(missing.size(), keys.size());
        EXPECT_EQ(missing[0], keys[0]);
        EXPECT_EQ(missing[1], keys[1]);
      }
      EXPECT_TRUE(SchemasMissingFromInventory(registered, keys).empty());
    }
  }
}

TEST(ContribSourceReplacementPolicyTest, InventoryIncludesAuthoredOnnxAndAtenSchemas) {
  const std::pair<const char*, bool> paths[]{
      {"/src/onnxruntime/core/graph/contrib_ops/contrib_defs.cc", true},
      {R"(C:\src\onnxruntime\core\graph\contrib_ops\contrib_defs.cc)", true},
      {"onnxruntime/core/graph/contrib_ops/new_defs.cc", true},
      {"/src/orttraining/orttraining/core/graph/training_op_defs.cc", true},
      {R"(C:\src\orttraining\orttraining\core\graph\training_op_defs.cc)", true},
      {"orttraining/orttraining/core/graph/training_op_defs.cc", true},
      {"/src/onnx/onnx/defs/math/defs.cc", false},
      {"/src/onnxruntime/core/graph/contrib_ops.cc", false},
      {"/src/onnxruntime/core/optimizer/nhwc_transformer.cc", false},
      {"/src/onnxruntime/test/ir/graph_test.cc", false},
      {"", false},
  };
  for (const auto* domain : {kOnnxDomain, kPytorchAtenDomain}) {
    for (const auto& [file, authored] : paths) {
      SCOPED_TRACE(MakeString(domain, ": ", file));
      const std::array registered{
          ONNX_NAMESPACE::OpSchema("NewPolicyTestOp", file, 1).SetDomain(domain).SinceVersion(1)};
      const auto missing = SchemasMissingFromInventory(registered, {});
      if (authored) {
        ASSERT_EQ(missing.size(), 1u);
        EXPECT_EQ(missing.front(), SourceOpKey(registered.front()));
      } else {
        EXPECT_TRUE(missing.empty());
      }
      const std::array keys{SourceOpKey(registered.front())};
      EXPECT_TRUE(SchemasMissingFromInventory(registered, keys).empty());
    }
  }
}

TEST(ContribSourceReplacementPolicyTest, NewKeyNeedsExplicitPolicyAndExactVersion) {
  const std::array schemas{kNew};
  EXPECT_FALSE(ValidateSourceDeclarations(schemas, {}, {}, LegacySourcePolicyKeys()).IsOK());
  const std::array pending{SourcePolicyRecord{kNew, {NonExecutableState::PendingTodo, "Needs qualification", {}}}};
  ASSERT_STATUS_OK(ValidateSourceDeclarations(schemas, pending, {}, LegacySourcePolicyKeys()));
  EXPECT_EQ(ResolveSourceStatus(kNew, kCudaExecutionProvider, "new", pending, {}, {}),
            SourceStatus::PendingTodo);
  const std::array special{SourcePolicyRecord{kNew, {NonExecutableState::Special, "Not a source fusion", "issue"}}};
  ASSERT_STATUS_OK(ValidateSourceDeclarations(schemas, special, {}, {}));
  EXPECT_EQ(ResolveSourceStatus(kNew, kCpuExecutionProvider, "new", special, {}, {}), SourceStatus::Special);
  const OpKey next{kMS, "ScaledSiLU", 2};
  const std::array versions{next};
  EXPECT_FALSE(ValidateSourceDeclarations(versions, {}, {}, LegacySourcePolicyKeys()).IsOK());
  EXPECT_EQ(ResolveSourceStatus(next, kCudaExecutionProvider, "", {}, {}, LegacySourcePolicyKeys()),
            SourceStatus::Missing);
  EXPECT_EQ(ResolveSourceStatus({kMS, "ScaledSiLU", 1}, kCpuExecutionProvider, "", {}, {},
                                LegacySourcePolicyKeys()),
            SourceStatus::LegacyPending);
  EXPECT_EQ(ResolveSourceStatus({kMS, "ReluGrad", 1}, kCudaExecutionProvider, "", {}, {},
                                LegacySourcePolicyKeys()),
            SourceStatus::LegacyPending);
  const std::array training_versions{OpKey{kMS, "ReluGrad", 2}};
  EXPECT_FALSE(ValidateSourceDeclarations(training_versions, {}, {}, LegacySourcePolicyKeys()).IsOK());
}

TEST(ContribSourceReplacementPolicyTest, RegistrationCollectsFinalizedDefaultVersion) {
  const auto* schema = ONNX_NAMESPACE::OpSchemaRegistry::Schema("AttnLSTM", 1, kMSDomain);
  ASSERT_NE(schema, nullptr);
  ASSERT_EQ(schema->SinceVersion(), 1);
  const auto schemas = GetContribSchemaKeys();
  EXPECT_NE(std::find(schemas.begin(), schemas.end(), SourceOpKey(*schema)), schemas.end());
  EXPECT_TRUE(std::all_of(schemas.begin(), schemas.end(), [](const auto& key) { return key.since_version > 0; }));
}

TEST(ContribSourceReplacementPolicyTest, NewUpstreamSchemasHaveExactOwnedPendingPolicies) {
  const auto schemas = GetContribSchemaKeys();
  const auto policies = GetSourcePolicies();
  const auto sources = Executables();
  const auto legacy = LegacySourcePolicyKeys();
  for (const auto* name : {"MoEGroupRouter", "GatherPacked", "MatMulPacked", "MoEPacked",
                           "MatMulNBitsQLoRAMlp", "MatMulNBitsQLoRAQkv"}) {
    SCOPED_TRACE(name);
    const OpKey key{kMS, name, 1};
    const auto* registered = ONNX_NAMESPACE::OpSchemaRegistry::Schema(name, 1, kMS);
    ASSERT_NE(registered, nullptr);
    EXPECT_EQ(SourceOpKey(*registered), key);
    EXPECT_NE(std::find(schemas.begin(), schemas.end(), key), schemas.end());
    EXPECT_EQ(std::find(legacy.begin(), legacy.end(), key), legacy.end());
    const auto policy = std::find_if(policies.begin(), policies.end(),
                                     [&](const auto& record) { return record.key == key; });
    EXPECT_NE(policy, policies.end());
    if (policy == policies.end()) continue;
    EXPECT_EQ(policy->policy.state, NonExecutableState::PendingTodo);
    EXPECT_NE(policy->policy.reason.find("source pattern/numerical qualification not yet provided"), std::string::npos);
    for (const auto* ep : {kCpuExecutionProvider, kCudaExecutionProvider, kWebGpuExecutionProvider}) {
      for (const auto* variant : {"", "unqualified-variant"}) {
        EXPECT_EQ(ResolveSourceStatus(key, ep, variant, policies, sources, legacy), SourceStatus::PendingTodo);
        EXPECT_EQ(ResolveSourceStatus({kMS, name, 2}, ep, variant, policies, sources, legacy), SourceStatus::Missing);
      }
    }
    auto without_owner = policies;
    without_owner.erase(std::remove_if(without_owner.begin(), without_owner.end(),
                                       [&](const auto& record) { return record.key == key; }),
                        without_owner.end());
    EXPECT_FALSE(ValidateSourceDeclarations(schemas, without_owner, sources, legacy).IsOK());
    EXPECT_EQ(ResolveSourceStatus(key, kCudaExecutionProvider, "", without_owner, sources, legacy),
              SourceStatus::Missing);
  }
}

TEST(ContribSourceReplacementPolicyTest, RejectsInvalidPoliciesConflictsAndOrphans) {
  const std::array schemas{kNew};
  for (const auto& reason : {"", " \t\n"}) {
    const std::array policies{SourcePolicyRecord{kNew, {NonExecutableState::PendingTodo, reason, {}}}};
    EXPECT_FALSE(ValidateSourceDeclarations(schemas, policies, {}, {}).IsOK());
  }
  const SourcePolicyRecord pending{kNew, {NonExecutableState::PendingTodo, "Pending", {}}};
  const std::array identical{pending, pending};
  ASSERT_STATUS_OK(ValidateSourceDeclarations(schemas, identical, {}, {}));
  auto changed = pending;
  changed.policy.state = NonExecutableState::Special;
  const std::array conflicts{pending, changed};
  EXPECT_FALSE(ValidateSourceDeclarations(schemas, conflicts, {}, {}).IsOK());
  const std::array orphan{pending};
  EXPECT_FALSE(ValidateSourceDeclarations({}, orphan, {}, {}).IsOK());
  changed.policy.state = static_cast<NonExecutableState>(99);
  const std::array invalid_state{changed};
  EXPECT_FALSE(ValidateSourceDeclarations(schemas, invalid_state, {}, {}).IsOK());
  changed.policy = {NonExecutableState::PendingTodo, "Pending", ""};
  const std::array empty_issue{changed};
  EXPECT_FALSE(ValidateSourceDeclarations(schemas, empty_issue, {}, {}).IsOK());
  const std::array invalid_key{OpKey{kMS, "", 1}};
  EXPECT_FALSE(ValidateSourceDeclarations(invalid_key, {}, {}, invalid_key).IsOK());
  const std::array duplicate_key{kNew, kNew};
  EXPECT_FALSE(ValidateSourceDeclarations(duplicate_key, {}, {}, duplicate_key).IsOK());
}

TEST(ContribSourceReplacementPolicyTest, RecordsOwnedSchemaKeysAndIdempotentPolicies) {
  const auto global_before = GetSourcePolicies();
  InlinedVector<SourcePolicyRecord> records;
  const auto* registered = ONNX_NAMESPACE::OpSchemaRegistry::Schema("ScaledSiLU", 1, kMS);
  ASSERT_NE(registered, nullptr);
  auto schema = *registered;
  SourcePolicy policy{NonExecutableState::PendingTodo, "Future source variants require qualification", "test-issue"};
  ASSERT_STATUS_OK(RecordSourcePolicy(records, schema, policy));
  auto invalid = policy;
  invalid.reason = " \t";
  EXPECT_FALSE(RecordSourcePolicy(records, schema, invalid).IsOK());
  ASSERT_STATUS_OK(RecordSourcePolicy(records, schema, policy));
  auto conflict = policy;
  conflict.reason = "Conflicting reason";
  EXPECT_FALSE(RecordSourcePolicy(records, schema, conflict).IsOK());
  policy.reason.clear();
  schema.SetDomain("changed-test-domain");
  const auto found = std::find_if(records.begin(), records.end(), [&](const auto& record) {
    return record.key == SourceOpKey(*registered);
  });
  ASSERT_NE(found, records.end());
  EXPECT_EQ(found->policy.reason, "Future source variants require qualification");
  EXPECT_EQ(found->policy.issue, "test-issue");
  EXPECT_EQ(std::count_if(records.begin(), records.end(), [&](const auto& record) {
              return record.key == SourceOpKey(*registered);
            }),
            1);
  const auto global_after = GetSourcePolicies();
  ASSERT_EQ(global_after.size(), global_before.size());
  for (size_t i = 0; i < global_before.size(); ++i) {
    EXPECT_EQ(global_after[i].key, global_before[i].key);
    EXPECT_EQ(global_after[i].policy, global_before[i].policy);
  }
}

TEST(ContribSourceReplacementPolicyTest, AvailableIsEPAndVariantScopedAndCanCoexistWithPending) {
  const auto sources = Executables();
  const auto schemas = GetContribSchemaKeys();
  auto pending = GetSourcePolicies();
  ASSERT_STATUS_OK(ValidateSourceDeclarations(schemas, pending, sources, LegacySourcePolicyKeys()));
  const auto swish_sources = ScaledSiLUSources();
  const auto& swish = swish_sources.front();
  pending.push_back({swish.target, {NonExecutableState::PendingTodo, "Future variants", {}}});
  ASSERT_STATUS_OK(ValidateSourceDeclarations(schemas, pending, sources, LegacySourcePolicyKeys()));
  EXPECT_EQ(ResolveSourceStatus(swish.target, kCudaExecutionProvider, swish.rule_id, pending, sources, {}),
            SourceStatus::Available);
  EXPECT_EQ(ResolveSourceStatus(swish.target, kCpuExecutionProvider, swish.rule_id, pending, sources, {}),
            SourceStatus::PendingTodo);
  EXPECT_EQ(ResolveSourceStatus(swish.target, kCudaExecutionProvider, "unqualified-variant", {}, sources, {}),
            SourceStatus::Missing);
  EXPECT_NE(swish.body.numerical.guarantee, CausalConvWithStateSources().front().body.numerical.guarantee);
  EXPECT_EQ(swish.body.initializer_transformation, InitializerTransformation::None);
  EXPECT_EQ(CausalConvWithStateSources().front().body.initializer_transformation,
            InitializerTransformation::OwnedFlattenedBiasCopy);
}

TEST(ContribSourceReplacementPolicyTest, MoEExactEPVariantVersionAndNumericalContract) {
  const auto sources = Executables();
  const auto policies = GetSourcePolicies();
  const auto legacy = LegacySourcePolicyKeys();
  const OpKey moe{kMS, "MoE", 1};
  constexpr const char* variant = "moe_fp32_dense_swiglu_v1";
  EXPECT_EQ(ResolveSourceStatus(moe, kCudaExecutionProvider, variant, policies, sources, legacy),
            SourceStatus::Available);
  EXPECT_EQ(ResolveSourceStatus(moe, kCpuExecutionProvider, variant, policies, sources, legacy),
            SourceStatus::LegacyPending);
  EXPECT_EQ(ResolveSourceStatus(moe, kCudaExecutionProvider, "other", policies, sources, legacy),
            SourceStatus::LegacyPending);
  EXPECT_EQ(ResolveSourceStatus({kMS, "MoE", 2}, kCudaExecutionProvider, variant, policies, sources, legacy),
            SourceStatus::Missing);
  const auto catalog = CudaSourceReplacements();
  const auto found = std::find_if(catalog.begin(), catalog.end(),
                                  [&](const auto* source) { return source->target == moe; });
  ASSERT_NE(found, catalog.end());
  const auto& body = (*found)->body;
  EXPECT_EQ(body.numerical.name, "cuda_moe_fp32_mathematical_v1");
  EXPECT_EQ(body.numerical.guarantee, NumericalGuarantee::MathematicalSemanticsWithTolerance);
  EXPECT_EQ(body.initializer_transformation, InitializerTransformation::None);
  EXPECT_EQ(ScaledSiLUSources().front().body.numerical.guarantee,
            NumericalGuarantee::NonNaNBitsAndNaNClassification);
  EXPECT_EQ(CausalConvWithStateSources().front().body.numerical.guarantee,
            NumericalGuarantee::FiniteInputAnalyticalBudgetAndExactState);
  EXPECT_NE(body.numerical.guarantee, ScaledSiLUSources().front().body.numerical.guarantee);
  EXPECT_NE(body.numerical.guarantee, CausalConvWithStateSources().front().body.numerical.guarantee);
  ASSERT_STATUS_OK(ValidateCudaSourceReplacements(catalog, GetContribSchemaKeys(), policies, legacy));
  for (const char* blank : {"", " \t\n"}) {
    for (bool name : {false, true}) {
      auto invalid = **found;
      (name ? invalid.body.numerical.name : invalid.body.numerical.scope) = blank;
      const std::array<const SourceReplacement*, 1> single{&invalid};
      EXPECT_FALSE(ValidateCudaSourceReplacements(single, GetContribSchemaKeys(), policies, legacy).IsOK());
    }
  }
  auto invalid = **found;
  invalid.body.numerical.guarantee = static_cast<NumericalGuarantee>(99);
  const std::array<const SourceReplacement*, 1> single{&invalid};
  EXPECT_FALSE(ValidateCudaSourceReplacements(single, GetContribSchemaKeys(), policies, legacy).IsOK());
}

TEST(ContribSourceReplacementPolicyTest, RejectsUncoveredAndDuplicateOpsetDomains) {
  const auto original = ScaledSiLUSources().front();
  const std::array schemas{original.target};
  auto check_invalid = [&](const SourceReplacement& source, std::string_view reason) {
    const std::array metadata{source.Metadata()};
    const std::array<const SourceReplacement*, 1> sources{&source};
    for (const auto& status : {ValidateSourceDeclarations(schemas, {}, metadata, {}),
                               ValidateCudaSourceReplacements(sources, schemas, {}, {})}) {
      EXPECT_FALSE(status.IsOK());
      EXPECT_NE(status.ErrorMessage().find(reason), std::string::npos) << status.ErrorMessage();
    }
  };
  auto source = original;
  source.body.rule.opsets = {{kMSDomain, 1, 1}};
  check_invalid(source, "Missing rule opset requirement for domain:");

  source = original;
  source.body.rule.replacement =
      rewrite::Replacement("(x)=>{a=other.test::Identity(x); return com.microsoft.ScaledSiLU(a)}");
  check_invalid(source, "Missing rule opset requirement for replacement domain: other.test");

  source = original;
  const rewrite::PatternLibrary library(R"(
    interface Unary(x) -> (y)
    pattern first implements Unary { a=Swish(x); return (y=a) }
    pattern second implements Unary { a=other.test::Identity(x); return (y=a) }
    pattern Forms = first | second
  )");
  source.body.rule.pattern = library.GetPattern("Forms");
  check_invalid(source, "Missing rule opset requirement for domain: other.test");

  source = original;
  source.body.rule.opsets.push_back({kOnnxDomainAlias, 24, 24});
  check_invalid(source, "Duplicate rule opset domain:");
}

TEST(ContribSourceReplacementPolicyTest, AcceptsOnnxOpsetAlias) {
  auto source = ScaledSiLUSources().front();
  for (auto& requirement : source.body.rule.opsets) {
    if (requirement.domain == kOnnxDomain) requirement.domain = kOnnxDomainAlias;
  }
  const std::array schemas{source.target};
  const std::array metadata{source.Metadata()};
  const std::array<const SourceReplacement*, 1> sources{&source};
  ASSERT_STATUS_OK(ValidateSourceDeclarations(schemas, {}, metadata, {}));
  ASSERT_STATUS_OK(ValidateCudaSourceReplacements(sources, schemas, {}, {}));
  EXPECT_NO_THROW(rewrite::PatternRewriter({source.body.rule}));
}

TEST(ContribSourceReplacementPolicyTest, RejectsInvalidReplacementBindingsBeforeDiscovery) {
  for (const auto* replacement : {
           "(missing)=>com.microsoft.ScaledSiLU(missing)",
           "(a)=>com.microsoft.ScaledSiLU(a)",
           "(x,a)=>com.microsoft.ScaledSiLU(x,alpha=a)"}) {
    SCOPED_TRACE(replacement);
    auto source = ScaledSiLUSources().front();
    source.body.rule.pattern = rewrite::Pattern("(x)=>{a=Swish(x);return a}");
    source.body.rule.replacement = rewrite::Replacement(replacement);
    const std::array schemas{source.target};
    const std::array metadata{source.Metadata()};
    const std::array<const SourceReplacement*, 1> sources{&source};
    EXPECT_FALSE(ValidateSourceDeclarations(schemas, {}, metadata, {}).IsOK());
    EXPECT_FALSE(ValidateCudaSourceReplacements(sources, schemas, {}, {}).IsOK());
    EXPECT_THROW(rewrite::PatternRewriter({source.body.rule}), OnnxRuntimeException);
  }
}

TEST(ContribSourceReplacementPolicyTest, RejectsReplacementReturnCountBeforeDiscovery) {
  const std::pair<const char*, const char*> cases[]{
      {"(x)=>{a=Identity(x);return a;}",
       "(x)=>{a=com.microsoft.ScaledSiLU(x);b=Identity(a);return a,b;}"},
      {"(x)=>{a,b=Split(x,axis=0);return a,b;}",
       "(x)=>com.microsoft.ScaledSiLU(x)"},
  };
  for (const auto& [pattern, replacement] : cases) {
    SCOPED_TRACE(MakeString(pattern, " -> ", replacement));
    auto source = ScaledSiLUSources().front();
    source.body.rule.pattern = rewrite::Pattern(pattern);
    source.body.rule.replacement = rewrite::Replacement(replacement);
    const std::array schemas{source.target};
    const std::array metadata{source.Metadata()};
    const std::array<const SourceReplacement*, 1> sources{&source};
    for (const auto& status : {rewrite::PatternRewriter::ValidateRule(source.body.rule),
                               ValidateSourceDeclarations(schemas, {}, metadata, {}),
                               ValidateCudaSourceReplacements(sources, schemas, {}, {})}) {
      EXPECT_FALSE(status.IsOK());
      EXPECT_NE(status.ErrorMessage().find("Replacement return count"), std::string::npos) << status.ErrorMessage();
    }
    EXPECT_THROW(rewrite::PatternRewriter({source.body.rule}), OnnxRuntimeException);
    EXPECT_NO_THROW(rewrite::PatternRewriter({source.body.rule}, {.match_only = true}));
  }
}

TEST(ContribSourceReplacementPolicyTest, LegacyCannotExemptMalformedExecutableMetadata) {
  const auto schemas = GetContribSchemaKeys();
  const auto policies = GetSourcePolicies();
  ASSERT_STATUS_OK(ValidateCudaSourceReplacements(CudaSourceReplacements(), schemas, policies, LegacySourcePolicyKeys()));
  auto check_invalid = [&](SourceReplacement source) {
    const std::array<const SourceReplacement*, 1> sources{&source};
    EXPECT_FALSE(ValidateCudaSourceReplacements(sources, schemas, policies, LegacySourcePolicyKeys()).IsOK());
  };
  const auto swish_sources = ScaledSiLUSources();
  const auto& original = swish_sources.front();
  auto source = original;
  source.body.numerical.name.clear();
  check_invalid(source);
  source = original;
  source.body.numerical.scope.clear();
  check_invalid(source);
  source = original;
  source.body.numerical.guarantee = static_cast<NumericalGuarantee>(99);
  check_invalid(source);
  source = original;
  source.body.rule.check_and_bind = {};
  check_invalid(source);
  source = original;
  source.body.rule.replacement = nullptr;
  check_invalid(source);
  source = original;
  source.body.rule.replacement = [](const Graph&, const rewrite::Match&) {
    return rewrite::Replacement("(x)=>com.microsoft.ScaledSiLU(x,alpha=1.0)");
  };
  const std::array callback_metadata{source.Metadata()};
  ASSERT_STATUS_OK(ValidateSourceDeclarations(schemas, policies, callback_metadata, LegacySourcePolicyKeys()));
  check_invalid(source);  // Common callback bodies are valid; this CUDA adapter cannot consume them.
  source = original;
  source.body.rule.opsets.clear();
  check_invalid(source);
  source = original;
  source.body.rule.opsets.back().max_version = 2;
  check_invalid(source);
  source = original;
  source.body.rule.replacement = rewrite::Replacement("(x)=>com.microsoft.Gelu(x)");
  check_invalid(source);
  source = original;
  source.rule_id = "different-id";
  check_invalid(source);
  source = original;
  source.config.clear();
  check_invalid(source);
  source = original;
  source.discover = nullptr;
  check_invalid(source);
  source = original;
  source.target = kNew;
  check_invalid(source);
  source = CausalConvWithStateSources().front();
  source.body.transformation_description.clear();
  check_invalid(source);
  const std::array<const SourceReplacement*, 2> duplicate{&original, &original};
  EXPECT_FALSE(ValidateCudaSourceReplacements(duplicate, schemas, policies, LegacySourcePolicyKeys()).IsOK());
  const std::array<const SourceReplacement*, 1> missing{nullptr};
  EXPECT_FALSE(ValidateCudaSourceReplacements(missing, schemas, policies, LegacySourcePolicyKeys()).IsOK());
  const std::array orphan{original.Metadata()};
  EXPECT_FALSE(ValidateSourceDeclarations({}, {}, orphan, LegacySourcePolicyKeys()).IsOK());
}

class NoProbeLookup final : public IExecutionProvider::IKernelLookup {
 public:
  mutable size_t calls = 0;
  const KernelCreateInfo* LookUpKernel(const Node&) const override {
    ++calls;
    return nullptr;
  }
};

std::unique_ptr<Model> UnitSwishModel() {
  auto model = std::make_unique<Model>("declaration-body", false, ModelMetaData(), PathString(),
                                       IOnnxRuntimeOpSchemaRegistryList(),
                                       std::unordered_map<std::string, int>{{kOnnxDomain, 24}, {kMSDomain, 1}},
                                       std::vector<ONNX_NAMESPACE::FunctionProto>{},
                                       DefaultLoggingManager().DefaultLogger());
  ONNX_NAMESPACE::TypeProto type;
  type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(17);
  auto& graph = model->MainGraph();
  auto& x = graph.GetOrCreateNodeArg("x", &type);
  auto& y = graph.GetOrCreateNodeArg("y", &type);
  graph.AddNode("swish", "Swish", "", {&x}, {&y});
  graph.SetInputs({&x});
  graph.SetOutputs({&y});
  ORT_THROW_IF_ERROR(graph.Resolve());
  return model;
}

class PresentTargetLookup final : public IExecutionProvider::IKernelLookup {
 public:
  mutable size_t calls = 0;
  KernelCreateInfo kernel;
  const KernelCreateInfo* LookUpKernel(const Node&) const override {
    ++calls;
    return &kernel;
  }
};

TEST(ContribSourceReplacementPolicyTest, DeclaredGuardDrivesDiscoveryAndOwnedAcceptance) {
  auto model = UnitSwishModel();
  auto& graph = model->MainGraph();
  const auto before = graph.ToGraphProto().SerializeAsString();
  auto source = ScaledSiLUSources().front();
  PresentTargetLookup lookup;
  std::vector<std::unique_ptr<ComputeCapability>> offers;
  const auto& logger = DefaultLoggingManager().DefaultLogger();
  source.body.rule.check_and_bind = [](const Graph&, rewrite::Match&) { return false; };
  ASSERT_STATUS_OK(source.discover(source, GraphViewer(graph), lookup, logger, offers));
  EXPECT_TRUE(offers.empty());
  EXPECT_EQ(lookup.calls, 0u);
  EXPECT_EQ(graph.ToGraphProto().SerializeAsString(), before);
  size_t guard_calls = 0;
  const auto original_guard = ScaledSiLUSources().front().body.rule.check_and_bind;
  source.body.rule.check_and_bind = [original_guard, &guard_calls](const Graph& live, rewrite::Match& match) {
    ++guard_calls;
    return original_guard(live, match);
  };
  ASSERT_STATUS_OK(source.discover(source, GraphViewer(graph), lookup, logger, offers));
  ASSERT_EQ(offers.size(), 1u);
  EXPECT_GT(guard_calls, 0u);
  EXPECT_EQ(lookup.calls, 1u);
  EXPECT_EQ(graph.ToGraphProto().SerializeAsString(), before);
  const auto discovery_calls = guard_calls;
  // An offer owns its rule and snapshot; changing this test's declaration copy cannot replace it.
  source.body.rule.check_and_bind = [](const Graph&, rewrite::Match&) { return false; };
  SessionOptions options;
  CPUExecutionProvider cpu{CPUExecutionProviderInfo{}};
  GraphOptimizerRegistry registry(&options, &cpu, &logger);
  auto& selected = *offers.front()->nodes_to_optimize.front();
  ASSERT_STATUS_OK(selected.optimization_func(graph, selected, *offers.front(), registry));
  EXPECT_GT(guard_calls, discovery_calls);
  ASSERT_EQ(graph.NumberOfNodes(), 1);
  EXPECT_EQ(graph.GetProducerNode("y")->OpType(), "ScaledSiLU");
}

TEST(ContribSourceReplacementPolicyTest, MultiTargetDAGDeclinesBeforeMutationOrKernelLookup) {
  rewrite::Replacement recipe("(x)=>{a=Identity(x); y=com.microsoft.ScaledSiLU(a); return y}");
  EXPECT_EQ(recipe.Plan()->size(), 2u);
  EXPECT_FALSE(SupportsCudaReplacement(recipe, 1));
  EXPECT_TRUE(SupportsCudaReplacement(*CausalConvWithStateSources().front().body.rule.replacement.Parsed(), 2));
  EXPECT_FALSE(SupportsCudaReplacement(*ScaledSiLUSources().front().body.rule.replacement.Parsed(), 2));
  Model model("multi-target-probe", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
              {{kOnnxDomain, 24}, {kMSDomain, 1}}, {}, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();
  NoProbeLookup lookup;
  bool available = true;
  ASSERT_STATUS_OK(cuda_runtime_fusion::detail::ProbeTarget(graph, recipe, {}, {}, 1, lookup, available));
  EXPECT_FALSE(available);
  EXPECT_EQ(graph.NumberOfNodes(), 0);
  EXPECT_TRUE(graph.GetInputs().empty());
  EXPECT_TRUE(graph.GetOutputs().empty());
  EXPECT_EQ(lookup.calls, 0u);
  // The common declaration still accepts a valid general DAG; CUDA admission is separate.
  auto source = ScaledSiLUSources().front();
  source.body.rule.replacement = recipe;
  const std::array sources{source.Metadata()};
  const std::array schemas{source.target};
  ASSERT_STATUS_OK(ValidateSourceDeclarations(schemas, {}, sources, {}));
  const std::array<const SourceReplacement*, 1> cuda_sources{&source};
  EXPECT_FALSE(ValidateCudaSourceReplacements(cuda_sources, schemas, {}, {}).IsOK());
  auto swish = UnitSwishModel();
  const auto before = swish->MainGraph().ToGraphProto().SerializeAsString();
  std::vector<std::unique_ptr<ComputeCapability>> offers;
  ASSERT_STATUS_OK(source.discover(source, GraphViewer(swish->MainGraph()), lookup,
                                   DefaultLoggingManager().DefaultLogger(), offers));
  EXPECT_TRUE(offers.empty());
  EXPECT_EQ(lookup.calls, 0u);
  EXPECT_EQ(swish->MainGraph().ToGraphProto().SerializeAsString(), before);
}

TEST(ContribSourceReplacementPolicyTest, NonIdentityAndExtraTargetOutputsDeclineBeforeMutationOrLookup) {
  Model model("invalid-output-probe", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
              {{kOnnxDomain, 24}, {kMSDomain, 1}}, {}, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();
  const auto before = graph.ToGraphProto().SerializeAsString();
  NoProbeLookup lookup;
  for (const auto* text : {
           "(x,w,bias,h,dilation)=>{a,s=com.microsoft.CausalConvWithState(x,w,bias,h,dilation=dilation);return s,a}",
           "(x,w,bias,h,dilation)=>{a,s,unused=com.microsoft.CausalConvWithState(x,w,bias,h,dilation=dilation);return a,s}"}) {
    SCOPED_TRACE(text);
    auto source = CausalConvWithStateSources().front();
    source.body.rule.replacement = rewrite::Replacement(text);
    const auto& recipe = *source.body.rule.replacement.Parsed();
    EXPECT_FALSE(SupportsCudaReplacement(recipe, 2));
    const std::array schemas{source.target};
    const std::array metadata{source.Metadata()};
    const std::array<const SourceReplacement*, 1> sources{&source};
    ASSERT_STATUS_OK(ValidateSourceDeclarations(schemas, {}, metadata, {}));
    EXPECT_FALSE(ValidateCudaSourceReplacements(sources, schemas, {}, {}).IsOK());
    bool available = true;
    ASSERT_STATUS_OK(cuda_runtime_fusion::detail::ProbeTarget(graph, recipe, {}, {}, 2, lookup, available));
    EXPECT_FALSE(available);
    EXPECT_EQ(lookup.calls, 0u);
    EXPECT_EQ(graph.ToGraphProto().SerializeAsString(), before);
  }
}

}  // namespace
}  // namespace onnxruntime::test
#endif
