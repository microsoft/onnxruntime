// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/graph/contrib_ops/source_replacement_policy.h"

#if !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_CONTRIB_OPS) && !defined(_OPSCHEMA_LIB_)

#include <algorithm>
#include <cctype>
#include <mutex>

#include "core/common/common.h"
#include "core/graph/constants.h"

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

}  // namespace

Status ValidateSourcePolicy(const SourcePolicy& policy) {
  ORT_RETURN_IF_NOT(policy.state == NonExecutableState::PendingTodo || policy.state == NonExecutableState::Special,
                    "Invalid non-executable source policy state.");
  ORT_RETURN_IF_NOT(HasText(policy.reason), "Source policy requires a nonempty reason.");
  ORT_RETURN_IF(policy.issue && !HasText(*policy.issue), "Source policy issue, when supplied, must be nonempty.");
  return Status::OK();
}

OpKey SourceOpKey(const ONNX_NAMESPACE::OpSchema& schema) {
  return {schema.domain(), schema.Name(), schema.SinceVersion()};
}

Status RecordSourcePolicy(const ONNX_NAMESPACE::OpSchema& schema, SourcePolicy policy) {
  auto& metadata = Metadata();
  std::lock_guard lock(metadata.mutex);
  return RecordSourcePolicy(metadata.policies, schema, std::move(policy));
}

void RecordPendingSourcePolicy(ONNX_NAMESPACE::OpSchema& schema, const char* name, int since_version,
                               std::string_view reason) {
  schema.SetName(name).SetDomain(kMSDomain).SinceVersion(since_version);
  schema.Finalize();
  ORT_THROW_IF_ERROR(RecordSourcePolicy(
      schema, {NonExecutableState::PendingTodo, std::string(reason), std::nullopt}));
}

Status RecordSourcePolicy(InlinedVector<SourcePolicyRecord>& records,
                          const ONNX_NAMESPACE::OpSchema& schema, SourcePolicy policy) {
  const auto key = SourceOpKey(schema);
  ORT_RETURN_IF_NOT(ValidKey(key), "Invalid source policy schema key.");
  ORT_RETURN_IF_ERROR(ValidateSourcePolicy(policy));
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
