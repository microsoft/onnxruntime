// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "core/common/common.h"
#include "core/framework/capability_planning.h"
#include "core/framework/level1_memory_estimate.h"

namespace onnxruntime {

enum class Level1MemoryEstimateScope {
  Unresolved,
  ImplementationAllocations,
  GraphInclusive,
};

struct Level1MemoryAllocation {
  std::string resource_id;
  std::optional<OrtDevice> device;
  std::optional<size_t> size_bound_bytes;
  std::optional<size_t> alignment_bytes;
  std::optional<CapabilityAllocationAlias> alias_of;
  InlinedVector<CapabilityResourceReference> non_overlap_with;
};

struct Level1MemoryCategoryEvidence {
  // Present and diagnostic-free asserts a complete decomposition. Empty explicitly proves absence.
  std::optional<InlinedVector<Level1MemoryAllocation, 1>> allocations;
  std::string diagnostic;
};

struct Level1MemoryAllocationBreakdown {
  Level1MemoryCategoryEvidence execution_workspace;
  Level1MemoryCategoryEvidence persistent_prepack;
  Level1MemoryCategoryEvidence initialization_scratch;
  Level1MemoryCategoryEvidence runtime_preparation;
  // The legacy runtime-preparation scalar is a complete peak, not an additional allocation.
  bool runtime_preparation_allocations_exclude_workspace = false;
};

struct CapabilityLevel1MemoryEstimate {
  CapabilityAllocationProvenance provenance;
  Level1MemoryEstimateScope scope = Level1MemoryEstimateScope::Unresolved;
  std::optional<Level1MemoryEstimate> estimate;
  Level1MemoryAllocationBreakdown breakdown;
  std::string diagnostic;
};

class CapabilityLevel1MemoryEstimateState final : public CapabilityPlanningState {
 public:
  explicit CapabilityLevel1MemoryEstimateState(CapabilityLevel1MemoryEstimate estimate)
      : source(std::move(estimate)) {}

  const CapabilityLevel1MemoryEstimate source;
};

inline CapabilityAllocationProvenance MakeLevel1MemoryEstimateProvenance(
    const CapabilityPlanningInfo& planning, const CapabilityExecutionContract& contract,
    std::string estimator_id, CapabilityPlanningStatePtr sizing_context) {
  return {planning.query, planning.candidate_id, planning.implementation_id, planning.recipe.recipe_id,
          contract.contract_id, contract.policy, contract.workload_coverage, std::move(estimator_id), std::move(sizing_context)};
}

// Replaces only the allocation fields and their diagnostics, not eligibility, policy or runtime state.
// The producer supplies allocation/decomposition proofs; scalar peaks alone never create buffers.
// Invalid provenance or duplicate resource identities leave contract unchanged.
inline Status AdaptLevel1MemoryEstimate(
    const CapabilityLevel1MemoryEstimate& source, const CapabilityPlanningInfo& planning,
    CapabilityExecutionContract& contract) {
  if (auto status = ValidateCapabilityPlanningQuery(source.provenance.query); !status.IsOK()) {
    return status;
  }
  const auto& provenance = source.provenance;
  if (provenance.query != planning.query ||
      provenance.candidate_id.empty() || provenance.candidate_id != planning.candidate_id ||
      provenance.implementation_id.empty() || provenance.implementation_id != planning.implementation_id ||
      provenance.recipe_id.empty() || provenance.recipe_id != planning.recipe.recipe_id ||
      provenance.contract_id.empty() || provenance.contract_id != contract.contract_id ||
      provenance.policy != contract.policy || provenance.workload_coverage != contract.workload_coverage ||
      provenance.estimator_id.empty() || !provenance.sizing_context) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "L1 sizing requires the original query and implementation context.");
  }

  auto adapted = contract;
  adapted.bound_status = CapabilityRequirementBoundStatus::Unresolved;
  adapted.allocation_requirements.reset();
  adapted.unresolved_terms.clear();
  auto retained_provenance = provenance;
  retained_provenance.sizing_context = std::make_shared<const CapabilityLevel1MemoryEstimateState>(source);
  adapted.allocation_provenance = std::make_shared<const CapabilityAllocationProvenance>(std::move(retained_provenance));

  if (!source.diagnostic.empty()) {
    adapted.unresolved_terms.push_back(source.diagnostic);
  }
  if (source.scope != Level1MemoryEstimateScope::ImplementationAllocations) {
    adapted.unresolved_terms.push_back("Graph-inclusive or unclassified totals are not implementation allocation estimates.");
    contract = std::move(adapted);
    return Status::OK();
  }
  if (!source.estimate.has_value()) {
    adapted.unresolved_terms.push_back("L1 estimator is missing, failed, or cannot size this context.");
    contract = std::move(adapted);
    return Status::OK();
  }

  struct Category {
    std::string_view name;
    const Level1MemoryCategoryEvidence& evidence;
    std::optional<size_t> scalar;
    CapabilityAllocationKind kind;
    CapabilityAllocationLifetime lifetime;
  };
  const auto& estimate = *source.estimate;
  const std::array<Category, 4> categories{{
      {"runtime_workspace_bytes", source.breakdown.execution_workspace, estimate.runtime_workspace_bytes,
       CapabilityAllocationKind::ExecutionWorkspace, CapabilityAllocationLifetime::Operation},
      {"persistent_prepack_bytes", source.breakdown.persistent_prepack, estimate.persistent_prepack_bytes,
       CapabilityAllocationKind::PersistentPrepack, CapabilityAllocationLifetime::Session},
      {"initialization_scratch_bytes", source.breakdown.initialization_scratch, estimate.initialization_scratch_bytes,
       CapabilityAllocationKind::InitializationScratch, CapabilityAllocationLifetime::Initialization},
      {"runtime_transient_bytes", source.breakdown.runtime_preparation, estimate.runtime_transient_bytes,
       CapabilityAllocationKind::RuntimePreparation, CapabilityAllocationLifetime::Operation},
  }};

  InlinedVector<CapabilityAllocationRequirement, 1> requirements;
  InlinedHashSet<std::string> resource_ids;
  bool has_category_evidence = false;
  for (const auto& category : categories) {
    const auto unresolved = [&](std::string_view reason) {
      adapted.unresolved_terms.push_back(std::string{category.name} + ": " + std::string{reason});
    };
    if (!category.evidence.diagnostic.empty()) {
      unresolved(category.evidence.diagnostic);
    }
    if (!category.scalar.has_value()) {
      unresolved("the legacy estimate is unknown.");
      continue;
    }
    if (!category.evidence.allocations.has_value()) {
      unresolved("a scalar/default zero does not establish a complete allocation decomposition.");
      continue;
    }
    if (category.kind == CapabilityAllocationKind::RuntimePreparation && *category.scalar > 0 &&
        !source.breakdown.runtime_preparation_allocations_exclude_workspace) {
      unresolved("preparation allocations have not been separated from retained execution workspace.");
      continue;
    }

    has_category_evidence = true;
    bool has_positive_allocation = false;
    bool has_unknown_size = false;
    for (const auto& allocation : *category.evidence.allocations) {
      if (allocation.resource_id.empty() || !resource_ids.insert(allocation.resource_id).second) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "L1 allocation identities must be nonempty and unique.");
      }
      if (!allocation.device.has_value() || !allocation.size_bound_bytes.has_value() ||
          !allocation.alignment_bytes.has_value()) {
        unresolved("allocation device, size or alignment is unresolved: " + allocation.resource_id);
      }
      has_unknown_size |= !allocation.size_bound_bytes.has_value();
      has_positive_allocation |= allocation.size_bound_bytes.value_or(0) > 0;
      const auto reference_resolved = [](const CapabilityResourceReference& reference) {
        return !reference.resource_id.empty() &&
               ((reference.kind == CapabilityResourceKind::GraphValue &&
                 reference.ep_instance_id.empty() && reference.candidate_id.empty()) ||
                (reference.kind == CapabilityResourceKind::ImplementationAllocation &&
                 !reference.ep_instance_id.empty() && !reference.candidate_id.empty()));
      };
      if (allocation.alias_of.has_value() &&
          (!reference_resolved(allocation.alias_of->target) || !allocation.alias_of->offset_bytes.has_value())) {
        unresolved("allocation alias is unresolved: " + allocation.resource_id);
      }
      for (const auto& reference : allocation.non_overlap_with) {
        if (!reference_resolved(reference)) {
          unresolved("allocation non-overlap reference is unresolved: " + allocation.resource_id);
        }
      }

      CapabilityAllocationRequirement requirement;
      requirement.resource_id = allocation.resource_id;
      requirement.kind = category.kind;
      requirement.device = allocation.device;
      requirement.size_bound_bytes = allocation.size_bound_bytes;
      requirement.alignment_bytes = allocation.alignment_bytes;
      requirement.lifetime = category.lifetime;
      requirement.alias_of = allocation.alias_of;
      requirement.non_overlap_with = allocation.non_overlap_with;
      requirements.push_back(std::move(requirement));
    }
    if (*category.scalar == 0 && has_positive_allocation) {
      unresolved("a zero scalar contradicts the supplied allocations.");
    }
    if (*category.scalar > 0 && !has_positive_allocation && !has_unknown_size) {
      unresolved("a positive scalar cannot prove absence of allocations.");
    }
  }
  if (has_category_evidence) {
    adapted.allocation_requirements = std::move(requirements);
  }
  if (adapted.unresolved_terms.empty()) {
    adapted.bound_status = CapabilityRequirementBoundStatus::Bounded;
  }
  contract = std::move(adapted);
  return Status::OK();
}

}  // namespace onnxruntime
