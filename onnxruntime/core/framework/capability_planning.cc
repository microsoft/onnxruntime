// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/capability_planning.h"

#include <algorithm>
#include <utility>

#include "core/common/common.h"
#include "core/framework/compute_capability.h"

namespace onnxruntime {

CapabilityPlanningQueryResult::CapabilityPlanningQueryResult() = default;
CapabilityPlanningQueryResult::~CapabilityPlanningQueryResult() = default;
CapabilityPlanningQueryResult::CapabilityPlanningQueryResult(CapabilityPlanningQueryResult&&) noexcept = default;
CapabilityPlanningQueryResult& CapabilityPlanningQueryResult::operator=(CapabilityPlanningQueryResult&&) noexcept = default;

namespace {

// Planning diagnostics belong to the caller, not Run-error telemetry.
Status ValidateQuery(const std::shared_ptr<const CapabilityPlanningQuery>& query) {
  if (!query || query->mode != CapabilityQueryMode::Planning || query->query_id == 0 ||
      query->graph_id.empty() || query->source_view_id.empty() || query->ep_instance_id.empty() ||
      !query->execution_device.has_value()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Planning requires an identified, owned query and device.");
  }

  if ((query->max_concurrent_runs.has_value() && *query->max_concurrent_runs == 0) ||
      (query->max_compute_streams.has_value() && *query->max_compute_streams == 0)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Planning concurrency bounds must be positive when supplied.");
  }

  InlinedHashSet<std::string> bound_names;
  bound_names.reserve(query->workload_bounds.size());
  for (const auto& bound : query->workload_bounds) {
    if (bound.name.empty() || !bound_names.insert(bound.name).second || bound.lower_bound < 0 ||
        (bound.upper_bound.has_value() && *bound.upper_bound < bound.lower_bound)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid or duplicate workload bound: ", bound.name);
    }
  }

  InlinedHashSet<size_t> constrained_nodes;
  constrained_nodes.reserve(query->node_constraints.size());
  for (const auto& constraint : query->node_constraints) {
    if (!constrained_nodes.insert(constraint.node_index).second ||
        (constraint.required_ep_instance_id.has_value() && constraint.required_ep_instance_id->empty())) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid or duplicate normalized node constraint.");
    }
  }

  InlinedHashSet<size_t> assigned_nodes;
  assigned_nodes.reserve(query->tentative_assignments.size());
  for (const auto& assignment : query->tentative_assignments) {
    if (assignment.ep_instance_id.empty() || !assigned_nodes.insert(assignment.node_index).second) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid or duplicate tentative node assignment.");
    }
  }

  return Status::OK();
}

Status ValidateResourceReference(const CapabilityResourceReference& reference) {
  if (reference.resource_id.empty() ||
      (reference.kind != CapabilityResourceKind::GraphValue &&
       reference.kind != CapabilityResourceKind::ImplementationAllocation)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Allocation relationship has an unresolved resource identity.");
  }
  if (reference.kind == CapabilityResourceKind::ImplementationAllocation &&
      (reference.ep_instance_id.empty() || reference.candidate_id.empty())) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Implementation resource references require an EP and candidate.");
  }
  if (reference.kind == CapabilityResourceKind::GraphValue &&
      (!reference.ep_instance_id.empty() || !reference.candidate_id.empty())) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Graph-value references are scoped by the query graph.");
  }
  return Status::OK();
}

Status ValidateBoundedRequirements(const CapabilityExecutionContract& contract) {
  if (!contract.allocation_requirements.has_value() || !contract.unresolved_terms.empty()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Bounded contract must declare a complete requirement list: ", contract.contract_id);
  }

  InlinedHashSet<std::string> resource_ids;
  resource_ids.reserve(contract.allocation_requirements->size());
  for (const auto& requirement : *contract.allocation_requirements) {
    if (requirement.resource_id.empty() || !resource_ids.insert(requirement.resource_id).second ||
        requirement.kind == CapabilityAllocationKind::Unresolved ||
        requirement.lifetime == CapabilityAllocationLifetime::Unresolved ||
        !requirement.device.has_value() || !requirement.size_bound_bytes.has_value() ||
        !requirement.alignment_bytes.has_value()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Bounded contract has an incomplete or duplicate allocation: ", requirement.resource_id);
    }
    if (requirement.alias_of.has_value()) {
      if (auto status = ValidateResourceReference(requirement.alias_of->target); !status.IsOK()) {
        return status;
      }
      if (!requirement.alias_of->offset_bytes.has_value()) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Bounded allocation alias requires a known offset.");
      }
    }
    for (const auto& reference : requirement.non_overlap_with) {
      if (auto status = ValidateResourceReference(reference); !status.IsOK()) {
        return status;
      }
    }
  }
  return Status::OK();
}

Status ValidateCapability(
    const std::shared_ptr<const CapabilityPlanningQuery>& query,
    const ComputeCapability& capability) {
  if (!capability.planning) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "Legacy capability has no planning metadata; requirements are not known zero.");
  }

  const auto& planning = *capability.planning;
  if (planning.query != query) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Capability belongs to a different or stale query snapshot.");
  }
  if (!capability.sub_graph || capability.sub_graph->nodes.empty() || planning.candidate_id.empty() ||
      planning.implementation_id.empty() || planning.recipe.recipe_id.empty() || planning.alternative_view_ids.empty() ||
      planning.contracts.empty()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Planning capability has incomplete source or offer identity.");
  }

  InlinedHashSet<size_t> source_nodes;
  source_nodes.reserve(capability.sub_graph->nodes.size());
  for (const auto node_index : capability.sub_graph->nodes) {
    if (!source_nodes.insert(node_index).second) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Planning capability has duplicate source-node membership.");
    }
  }

  InlinedHashSet<std::string> view_ids;
  view_ids.reserve(planning.alternative_view_ids.size());
  for (const auto& view_id : planning.alternative_view_ids) {
    if (view_id.empty() || !view_ids.insert(view_id).second) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid or duplicate alternative view identity.");
    }
  }

  if (planning.recipe.kind == CapabilityRecipeKind::Unresolved && planning.recipe.unresolved_terms.empty()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unresolved recipe requires an explicit diagnostic.");
  }

  const auto& effects = planning.recipe.graph_effects;
  if (effects.kind == CapabilityGraphEffectsKind::Unresolved && effects.unresolved_terms.empty()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unresolved graph effects require an explicit diagnostic.");
  }
  if (effects.kind == CapabilityGraphEffectsKind::Unchanged &&
      (!effects.introduced_values.empty() || !effects.removed_values.empty() || !effects.value_aliases.empty())) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unchanged recipe cannot declare graph rewrites or alias changes.");
  }

  InlinedHashSet<std::string> contract_ids;
  contract_ids.reserve(planning.contracts.size());
  for (const auto& contract : planning.contracts) {
    if (contract.contract_id.empty() || !contract_ids.insert(contract.contract_id).second) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid or duplicate contract identity.");
    }
    if (contract.eligibility != CapabilityContractEligibility::Eligible && contract.eligibility_reason.empty()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unresolved or ineligible contract requires a reason.");
    }
    if (contract.bound_status == CapabilityRequirementBoundStatus::Bounded) {
      if (auto status = ValidateBoundedRequirements(contract); !status.IsOK()) {
        return status;
      }
    } else if (contract.unresolved_terms.empty()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unresolved requirements require an explicit diagnostic.");
    }
  }
  return Status::OK();
}

}  // namespace

Status ValidateCapabilityPlanningResult(
    const std::shared_ptr<const CapabilityPlanningQuery>& query,
    const CapabilityPlanningQueryResult& result) {
  if (auto status = ValidateQuery(query); !status.IsOK()) {
    return status;
  }
  if (result.support != CapabilityPlanningSupport::PlanningSafe) {
    if (!result.capabilities.empty()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Non-participating EP cannot return planning offers.");
    }
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "EP does not provide planning-safe discovery. ", result.diagnostic);
  }

  InlinedHashSet<std::string> candidate_ids;
  candidate_ids.reserve(result.capabilities.size());
  for (const auto& capability : result.capabilities) {
    if (!capability) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Planning result contains a null capability.");
    }
    if (auto status = ValidateCapability(query, *capability); !status.IsOK()) {
      return status;
    }
    if (!candidate_ids.insert(capability->planning->candidate_id).second) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Planning result contains duplicate candidate identities.");
    }
  }
  return Status::OK();
}

Status CreateCapabilityPlanningBinding(
    const std::shared_ptr<const CapabilityPlanningQuery>& query,
    const ComputeCapability& capability,
    const CapabilityPlanningSelection& selection,
    std::shared_ptr<const CapabilityPlanningBinding>& binding) {
  if (auto status = ValidateQuery(query); !status.IsOK()) {
    return status;
  }
  if (auto status = ValidateCapability(query, capability); !status.IsOK()) {
    return status;
  }
  const auto& planning = *capability.planning;
  if (selection.plan_generation == 0 || selection.candidate_id != planning.candidate_id ||
      selection.recipe_id != planning.recipe.recipe_id ||
      std::find(planning.alternative_view_ids.begin(), planning.alternative_view_ids.end(),
                selection.alternative_view_id) == planning.alternative_view_ids.end()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Selection does not identify this candidate, recipe, and view.");
  }

  const auto& recipe = planning.recipe;
  if (recipe.kind == CapabilityRecipeKind::Unresolved || !recipe.unresolved_terms.empty() ||
      recipe.graph_effects.kind == CapabilityGraphEffectsKind::Unresolved ||
      !recipe.graph_effects.unresolved_terms.empty()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "Selected implementation recipe or graph effects are unresolved.");
  }
  if ((recipe.kind == CapabilityRecipeKind::CompiledRegion ||
       recipe.graph_effects.kind == CapabilityGraphEffectsKind::Described) &&
      !recipe.prepared_state) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Selected lowering or rewrite requires retained prepared state.");
  }

  const auto contract_it = std::find_if(
      planning.contracts.begin(), planning.contracts.end(),
      [&selection](const CapabilityExecutionContract& contract) {
        return contract.contract_id == selection.contract_id;
      });
  if (contract_it == planning.contracts.end()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Selection identifies an unknown contract.");
  }
  const auto& contract = *contract_it;
  if (contract.eligibility != CapabilityContractEligibility::Eligible) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Selected contract is not eligible: ", contract.eligibility_reason);
  }
  if (contract.bound_status != CapabilityRequirementBoundStatus::Bounded) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "Selected contract has unresolved allocation requirements.");
  }
  if (!contract.resolved_state || contract.workload_coverage.empty()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Selected contract requires resolved state and workload coverage.");
  }

  for (const auto& constraint : query->node_constraints) {
    if (std::find(capability.sub_graph->nodes.begin(), capability.sub_graph->nodes.end(),
                  constraint.node_index) == capability.sub_graph->nodes.end()) {
      continue;
    }
    if (constraint.required_ep_instance_id.has_value() &&
        *constraint.required_ep_instance_id != query->ep_instance_id) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Selection violates a required EP assignment.");
    }
    if (constraint.permitted_policies.has_value() &&
        std::find(constraint.permitted_policies->begin(), constraint.permitted_policies->end(),
                  contract.policy) == constraint.permitted_policies->end()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Selection violates the permitted policy set.");
    }
  }

  CapabilityPlanningBinding selected;
  selected.query = query;
  selected.plan_generation = selection.plan_generation;
  selected.candidate_id = planning.candidate_id;
  selected.implementation_id = planning.implementation_id;
  selected.alternative_view_id = selection.alternative_view_id;
  selected.source_nodes.assign(capability.sub_graph->nodes.begin(), capability.sub_graph->nodes.end());
  selected.compatibility = planning.compatibility;
  selected.recipe = recipe;
  selected.contract = contract;
  binding = std::make_shared<const CapabilityPlanningBinding>(std::move(selected));
  return Status::OK();
}

Status ValidateCapabilityPlanningBindingIdentity(
    const std::shared_ptr<const CapabilityPlanningQuery>& query,
    uint64_t plan_generation,
    const CapabilityPlanningBinding& binding) {
  if (auto status = ValidateQuery(query); !status.IsOK()) {
    return status;
  }
  if (binding.query != query || plan_generation == 0 || binding.plan_generation != plan_generation) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Selected binding belongs to a different query or plan generation.");
  }
  return Status::OK();
}

}  // namespace onnxruntime
