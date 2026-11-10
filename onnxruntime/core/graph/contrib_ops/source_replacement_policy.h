// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#if !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_CONTRIB_OPS)

#include <functional>
#include <optional>
#include <string>
#include <string_view>

#include "core/common/inlined_containers.h"
#include "core/common/common.h"
#include "core/graph/onnx_protobuf.h"
#include "gsl/gsl"

namespace onnxruntime::contrib {

struct OpKey {
  std::string domain;
  std::string name;
  int since_version;
  bool operator==(const OpKey&) const = default;
};

OpKey SourceOpKey(const ONNX_NAMESPACE::OpSchema& schema);

enum class NonExecutableState { PendingTodo,
                                Special };

struct SourcePolicy {
  NonExecutableState state;
  std::string reason;
  std::optional<std::string> issue;
  bool operator==(const SourcePolicy&) const = default;
};

struct SourcePolicyRecord {
  OpKey key;
  SourcePolicy policy;
};

Status ValidateSourcePolicy(const SourcePolicy& policy);

// Copies authored metadata. Identical duplicates are idempotent; conflicts are errors.
// Call during schema registration, not through a static/linker registration mechanism.
Status RecordSourcePolicy(const ONNX_NAMESPACE::OpSchema& schema, SourcePolicy policy);
#if defined(_OPSCHEMA_LIB_)
static inline void RecordPendingSourcePolicy(ONNX_NAMESPACE::OpSchema&, const char*, int, std::string_view) {}
#else
// Schema builders run before the enclosing macro supplies the finalized schema key.
void RecordPendingSourcePolicy(ONNX_NAMESPACE::OpSchema& schema, const char* name, int since_version,
                               std::string_view reason);
#endif
// The same recorder over an explicitly owned collection; no process-global test state.
Status RecordSourcePolicy(InlinedVector<SourcePolicyRecord>& records,
                          const ONNX_NAMESPACE::OpSchema& schema, SourcePolicy policy);
InlinedVector<SourcePolicyRecord> GetSourcePolicies();

// This is the schema workflow's inventory, not an EP support registry.
void RecordContribSchema(const ONNX_NAMESPACE::OpSchema& schema);
InlinedVector<OpKey> GetContribSchemaKeys();
std::function<void(ONNX_NAMESPACE::OpSchema&&)> CollectContribSchemas(
    std::function<void(ONNX_NAMESPACE::OpSchema&&)> register_schema);

class ContribSchemaRegisterOnce {
 public:
  // Intentionally implicit for the existing schema-registration builder macros.
  ContribSchemaRegisterOnce(ONNX_NAMESPACE::OpSchema schema);

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ContribSchemaRegisterOnce);
};

struct SourceBody;

// A view derived from an EP's executable companion; never independently registered.
struct ExecutableSource {
  OpKey target;
  std::string_view ep;
  std::string_view variant;
  const SourceBody* body;
};

enum class SourceStatus { Missing,
                          LegacyPending,
                          PendingTodo,
                          Special,
                          Available };

Status ValidateSourceDeclarations(gsl::span<const OpKey> schemas,
                                  gsl::span<const SourcePolicyRecord> policies,
                                  gsl::span<const ExecutableSource> sources,
                                  gsl::span<const OpKey> legacy);
// Resolve only validated metadata. Available names a declaration, not current kernel/device availability.
SourceStatus ResolveSourceStatus(const OpKey& key, std::string_view ep, std::string_view variant,
                                 gsl::span<const SourcePolicyRecord> policies,
                                 gsl::span<const ExecutableSource> sources,
                                 gsl::span<const OpKey> legacy);

}  // namespace onnxruntime::contrib
#endif
