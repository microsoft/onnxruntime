// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/graph/contrib_ops/source_replacement_policy.h"

#if !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_CONTRIB_OPS)

#include <algorithm>
#include <cctype>
#include <mutex>

#include "core/common/common.h"
#include "core/graph/contrib_ops/source_replacement.h"

namespace onnxruntime::contrib {
namespace {

struct AuthoredMetadata {
  std::mutex mutex;
  InlinedVector<OpKey> schemas;
  InlinedVector<SourcePolicyRecord> policies;
};

AuthoredMetadata& Metadata() {
  static AuthoredMetadata metadata;
  return metadata;
}

bool HasText(std::string_view text) {
  return std::any_of(text.begin(), text.end(), [](unsigned char c) { return !std::isspace(c); });
}

bool ValidKey(const OpKey& key) {
  return HasText(key.name) && key.since_version > 0;
}

bool Contains(gsl::span<const OpKey> keys, const OpKey& key) {
  return std::find(keys.begin(), keys.end(), key) != keys.end();
}

void RecordRegisteredSchema(const std::string& name, const std::string& domain, int since_version) {
  const auto* registered = ONNX_NAMESPACE::OpSchemaRegistry::Schema(name, since_version, domain);
  if (registered && registered->SinceVersion() == since_version) RecordContribSchema(*registered);
}

Status ValidatePolicy(const SourcePolicy& policy) {
  ORT_RETURN_IF_NOT(policy.state == NonExecutableState::PendingTodo || policy.state == NonExecutableState::Special,
                    "Invalid non-executable source policy state.");
  ORT_RETURN_IF_NOT(HasText(policy.reason), "Source policy requires a nonempty reason.");
  ORT_RETURN_IF(policy.issue && !HasText(*policy.issue), "Source policy issue, when supplied, must be nonempty.");
  return Status::OK();
}

Status ValidateBody(const ExecutableSource& source) {
  ORT_RETURN_IF_NOT(ValidKey(source.target) && HasText(source.ep) && HasText(source.variant) && source.body,
                    "Malformed executable source declaration.");
  const auto& body = *source.body;
  const auto& rule = body.rule;
  ORT_RETURN_IF_NOT(HasText(rule.name) && !rule.pattern.Nodes().empty() && rule.replacement &&
                        rule.check_and_bind && !rule.opsets.empty(),
                    "Executable source requires a named body, replacement, guard and opsets.");
  ORT_RETURN_IF_NOT(HasText(body.numerical.name) && HasText(body.numerical.scope),
                    "Executable source requires a named, scoped numerical profile.");
  ORT_RETURN_IF_NOT(body.numerical.guarantee == NumericalGuarantee::NonNaNBitsAndNaNClassification ||
                        body.numerical.guarantee == NumericalGuarantee::FiniteInputAnalyticalBudgetAndExactState ||
                        body.numerical.guarantee == NumericalGuarantee::MathematicalSemanticsWithTolerance,
                    "Invalid numerical guarantee.");
  ORT_RETURN_IF_NOT(body.initializer_transformation == InitializerTransformation::None ||
                        body.initializer_transformation == InitializerTransformation::OwnedFlattenedBiasCopy,
                    "Invalid initializer transformation.");
  ORT_RETURN_IF(body.initializer_transformation != InitializerTransformation::None &&
                    !HasText(body.transformation_description),
                "Initializer transformation requires a description.");
  ORT_RETURN_IF_ERROR(rewrite::PatternRewriter::ValidateOpsets(rule));
  bool target_opset = false;
  for (const auto& opset : rule.opsets) {
    if (opset.domain == source.target.domain && opset.min_version == source.target.since_version &&
        opset.max_version == source.target.since_version) target_opset = true;
  }
  ORT_RETURN_IF_NOT(target_opset, "Executable source must specify the exact target opset.");
  if (const auto* recipe = rule.replacement.Parsed()) {
    ORT_RETURN_IF_NOT(recipe->Plan() && !recipe->Plan()->empty() && !recipe->Outputs().empty(),
                      "Executable source has an empty replacement DAG.");
    const bool has_target = std::any_of(recipe->Plan()->begin(), recipe->Plan()->end(), [&](const auto& node) {
      return node.op == source.target.name && node.domain == source.target.domain;
    });
    ORT_RETURN_IF_NOT(has_target, "Executable source replacement does not contain its declared target.");
  }
  return Status::OK();
}
}  // namespace

OpKey SourceOpKey(const ONNX_NAMESPACE::OpSchema& schema) {
  return {schema.domain(), schema.Name(), schema.SinceVersion()};
}

Status RecordSourcePolicy(const ONNX_NAMESPACE::OpSchema& schema, SourcePolicy policy) {
  auto& metadata = Metadata();
  std::lock_guard lock(metadata.mutex);
  return RecordSourcePolicy(metadata.policies, schema, std::move(policy));
}

Status RecordSourcePolicy(InlinedVector<SourcePolicyRecord>& records,
                          const ONNX_NAMESPACE::OpSchema& schema, SourcePolicy policy) {
  const auto key = SourceOpKey(schema);
  ORT_RETURN_IF_NOT(ValidKey(key), "Invalid source policy schema key.");
  ORT_RETURN_IF_ERROR(ValidatePolicy(policy));
  for (const auto& record : records) {
    if (record.key != key) continue;
    ORT_RETURN_IF_NOT(record.policy == policy, "Conflicting source policy for ", key.domain, ".", key.name,
                      "-", key.since_version, ".");
    return Status::OK();
  }
  records.push_back({key, std::move(policy)});
  return Status::OK();
}

InlinedVector<SourcePolicyRecord> GetSourcePolicies() {
  auto& metadata = Metadata();
  std::lock_guard lock(metadata.mutex);
  return metadata.policies;
}

void RecordContribSchema(const ONNX_NAMESPACE::OpSchema& schema) {
  auto& metadata = Metadata();
  std::lock_guard lock(metadata.mutex);
  const auto key = SourceOpKey(schema);
  if (!Contains(metadata.schemas, key)) metadata.schemas.push_back(key);
}

InlinedVector<OpKey> GetContribSchemaKeys() {
  auto& metadata = Metadata();
  std::lock_guard lock(metadata.mutex);
  return metadata.schemas;
}

std::function<void(ONNX_NAMESPACE::OpSchema&&)> CollectContribSchemas(
    std::function<void(ONNX_NAMESPACE::OpSchema&&)> register_schema) {
  return [register_schema = std::move(register_schema)](ONNX_NAMESPACE::OpSchema&& schema) {
    const auto name = schema.Name();
    const auto domain = schema.domain();
    register_schema(std::move(schema));
    // ONNX finalizes/normalizes SinceVersion before moving: the scalar survives,
    // unlike Name/domain, which must be captured before registration.
    RecordRegisteredSchema(name, domain, schema.SinceVersion());
  };
}

ContribSchemaRegisterOnce::ContribSchemaRegisterOnce(ONNX_NAMESPACE::OpSchema schema) {
  const auto name = schema.Name();
  const auto domain = schema.domain();
  // ONNX normalizes the version through its rvalue-reference registration before moving
  // the schema. The scalar since-version survives that move; names must be captured first.
  ONNX_NAMESPACE::OpSchemaRegistry::OpSchemaRegisterOnce::OpSchemaRegisterNoExcept(std::move(schema));
  RecordRegisteredSchema(name, domain, schema.SinceVersion());
}

Status ValidateSourceDeclarations(gsl::span<const OpKey> schemas,
                                  gsl::span<const SourcePolicyRecord> policies,
                                  gsl::span<const ExecutableSource> sources,
                                  gsl::span<const OpKey> legacy) {
  for (size_t i = 0; i < legacy.size(); ++i) {
    ORT_RETURN_IF_NOT(ValidKey(legacy[i]), "Invalid legacy schema key.");
    ORT_RETURN_IF(Contains(legacy.subspan(0, i), legacy[i]), "Duplicate legacy schema key.");
  }
  for (size_t i = 0; i < schemas.size(); ++i) {
    ORT_RETURN_IF_NOT(ValidKey(schemas[i]), "Invalid contrib schema key.");
    ORT_RETURN_IF(Contains(schemas.subspan(0, i), schemas[i]), "Duplicate contrib schema key.");
  }
  for (size_t i = 0; i < policies.size(); ++i) {
    const auto& record = policies[i];
    ORT_RETURN_IF_ERROR(ValidatePolicy(record.policy));
    ORT_RETURN_IF_NOT(Contains(schemas, record.key), "Orphan source policy for ", record.key.name, ".");
    for (size_t j = 0; j < i; ++j) {
      ORT_RETURN_IF(policies[j].key == record.key && policies[j].policy != record.policy,
                    "Conflicting source policy for ", record.key.name, ".");
    }
  }
  for (size_t i = 0; i < sources.size(); ++i) {
    const auto& source = sources[i];
    ORT_RETURN_IF_ERROR(ValidateBody(source));
    ORT_RETURN_IF_NOT(Contains(schemas, source.target), "Orphan executable source for ", source.target.name, ".");
    for (size_t j = 0; j < i; ++j) {
      ORT_RETURN_IF(sources[j].body->rule.name == source.body->rule.name,
                    "Duplicate executable source rule id: ", source.body->rule.name, ".");
    }
  }
  for (const auto& key : schemas) {
    const bool authored = std::any_of(policies.begin(), policies.end(),
                                      [&](const auto& record) { return record.key == key; });
    const bool executable = std::any_of(sources.begin(), sources.end(),
                                        [&](const auto& source) { return source.target == key; });
    ORT_RETURN_IF_NOT(authored || executable || Contains(legacy, key),
                      "Missing source replacement declaration for ", key.domain, ".", key.name,
                      "-", key.since_version, "; author PendingTodo/Special or an executable EP source.");
  }
  return Status::OK();
}

SourceStatus ResolveSourceStatus(const OpKey& key, std::string_view ep, std::string_view variant,
                                 gsl::span<const SourcePolicyRecord> policies,
                                 gsl::span<const ExecutableSource> sources,
                                 gsl::span<const OpKey> legacy) {
  for (const auto& source : sources) {
    if (source.target == key && source.ep == ep && source.variant == variant) return SourceStatus::Available;
  }
  for (const auto& record : policies) {
    if (record.key == key) {
      return record.policy.state == NonExecutableState::PendingTodo ? SourceStatus::PendingTodo : SourceStatus::Special;
    }
  }
  return Contains(legacy, key) ? SourceStatus::LegacyPending : SourceStatus::Missing;
}
}  // namespace onnxruntime::contrib
#endif
