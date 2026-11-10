// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/graph/contrib_ops/source_replacement.h"

#if !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_CONTRIB_OPS)

#include <algorithm>
#include <cctype>

namespace onnxruntime::contrib {
namespace {
bool HasText(std::string_view text) {
  return std::any_of(text.begin(), text.end(), [](unsigned char c) { return !std::isspace(c); });
}

bool ValidKey(const OpKey& key) {
  return HasText(key.name) && key.since_version > 0;
}

bool Contains(gsl::span<const OpKey> keys, const OpKey& key) {
  return std::find(keys.begin(), keys.end(), key) != keys.end();
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
                        body.numerical.guarantee == NumericalGuarantee::FiniteInputAnalyticalBudgetAndExactState,
                    "Invalid numerical guarantee.");
  ORT_RETURN_IF_NOT(body.initializer_transformation == InitializerTransformation::None ||
                        body.initializer_transformation == InitializerTransformation::OwnedFlattenedBiasCopy,
                    "Invalid initializer transformation.");
  ORT_RETURN_IF(body.initializer_transformation != InitializerTransformation::None &&
                    !HasText(body.transformation_description),
                "Initializer transformation requires a description.");
  ORT_RETURN_IF_ERROR(rewrite::PatternRewriter::ValidateRule(rule));
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
    ORT_RETURN_IF_ERROR(ValidateSourcePolicy(record.policy));
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
}  // namespace onnxruntime::contrib
#endif
