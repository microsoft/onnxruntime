// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/plugin_ep/ep_schema_compatibility.h"

#include <cstring>
#include <set>
#include <utility>

#include "core/common/logging/logging.h"
#include "core/framework/error_code_helper.h"
#include "core/graph/constants.h"
#include "core/graph/graph.h"
#include "core/graph/schema_abi_digest.h"
#include "onnx/defs/schema.h"

namespace onnxruntime {

Status PluginEpSchemaCompatibility::Create(
    OrtEpFactory& factory, const logging::Logger& logger,
    std::shared_ptr<const PluginEpSchemaCompatibility>& result) {
  auto compatibility = std::make_shared<PluginEpSchemaCompatibility>();

  // Missing metadata remains permissive during the transition. API 30 is the
  // first ABI version containing GetOperatorCompatibilityInfo.
  if (factory.ort_version_supported < 30 || factory.GetOperatorCompatibilityInfo == nullptr) {
    LOGS(logger, WARNING) << "Plugin EP factory '" << factory.GetName(&factory)
                          << "' does not publish operator schema compatibility information. "
                             "com.microsoft kernels remain enabled under the temporary legacy policy.";
    result = std::move(compatibility);
    return Status::OK();
  }

  const OrtEpOperatorCompatibilityInfo* entries = nullptr;
  size_t num_entries = 0;
  ORT_RETURN_IF_ERROR(ToStatusAndRelease(
      factory.GetOperatorCompatibilityInfo(&factory, &entries, &num_entries)));
  ORT_RETURN_IF(entries == nullptr && num_entries != 0,
                "OrtEpFactory::GetOperatorCompatibilityInfo returned a null array with ",
                num_entries, " entries.");

  compatibility->is_negotiated_ = true;
  std::set<OperatorKey> seen;
  std::set<OperatorKey> conflicted;

  for (size_t i = 0; i < num_entries; ++i) {
    const auto& entry = entries[i];
    if (entry.domain == nullptr || entry.op_type == nullptr || entry.domain[0] == '\0' ||
        entry.op_type[0] == '\0' || entry.since_version <= 0) {
      LOGS(logger, WARNING) << "Ignoring malformed operator compatibility entry " << i
                            << " from plugin EP factory '" << factory.GetName(&factory) << "'.";
      continue;
    }

    // Negotiation is initially enforced for the independently shipped ORT-private
    // contrib domain. Standard ONNX domains retain their published compatibility rules.
    if (std::strcmp(entry.domain, kMSDomain) != 0) {
      continue;
    }

    OperatorKey key{entry.domain, entry.op_type, entry.since_version};
    if (!seen.insert(key).second) {
      conflicted.insert(key);
      compatibility->compatible_operators_.erase(key);
      LOGS(logger, WARNING) << "Ignoring duplicate operator compatibility entry for "
                            << entry.domain << ":" << entry.op_type << "@" << entry.since_version
                            << " from plugin EP factory '" << factory.GetName(&factory) << "'.";
      continue;
    }

    const auto* schema = ONNX_NAMESPACE::OpSchemaRegistry::Schema(
        entry.op_type, entry.since_version, entry.domain);
    if (schema == nullptr || schema->since_version() != entry.since_version) {
      LOGS(logger, WARNING) << "Plugin EP factory '" << factory.GetName(&factory)
                            << "' reported unavailable schema " << entry.domain << ":"
                            << entry.op_type << "@" << entry.since_version << ".";
      continue;
    }

    SchemaAbiDigest core_digest{};
    ORT_RETURN_IF_ERROR(ComputeSchemaAbiDigest(*schema, core_digest));
    if (std::memcmp(core_digest.data(), entry.schema_abi_digest, core_digest.size()) != 0) {
      LOGS(logger, WARNING) << "Plugin EP schema digest does not match ORT for "
                            << entry.domain << ":" << entry.op_type << "@" << entry.since_version
                            << "; this operator is quarantined for factory '"
                            << factory.GetName(&factory) << "'.";
      continue;
    }

    compatibility->compatible_operators_.insert(std::move(key));
  }

  for (const auto& key : conflicted) {
    compatibility->compatible_operators_.erase(key);
  }

  result = std::move(compatibility);
  return Status::OK();
}

bool PluginEpSchemaCompatibility::IsCompatible(const std::string& domain,
                                               const std::string& op_type,
                                               int since_version) const {
  if (domain != kMSDomain || !is_negotiated_) {
    return true;
  }
  return compatible_operators_.find(OperatorKey{domain, op_type, since_version}) !=
         compatible_operators_.end();
}

bool PluginEpSchemaCompatibility::IsCompatible(const Node& node) const {
  return IsCompatible(node.Domain(), node.OpType(), node.SinceVersion());
}

}  // namespace onnxruntime
