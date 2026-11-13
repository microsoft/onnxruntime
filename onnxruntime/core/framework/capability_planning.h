// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "core/common/inlined_containers.h"
#include "core/common/status.h"
#include "core/framework/kernel_dispatch_policy.h"
#include "core/framework/max_shape_override.h"
#include "core/framework/ortdevice.h"

namespace onnxruntime {

struct ComputeCapability;

enum class CapabilityQueryMode {
  Legacy,
  Planning,
};

struct CapabilityWorkloadBound {
  std::string name;
  int64_t lower_bound = 0;
  std::optional<int64_t> upper_bound;
};

struct CapabilityNodeConstraint {
  size_t node_index = 0;
  std::optional<std::string> required_ep_instance_id;
  // Unset preserves the configured policy; an explicitly empty set permits none.
  std::optional<InlinedVector<KernelDispatchPolicy>> permitted_policies;
};

struct CapabilityNodeAssignment {
  size_t node_index = 0;
  std::string ep_instance_id;
};

// Publish a const, owned snapshot. A new query is required when any relevant fact changes.
struct CapabilityPlanningQuery {
  CapabilityQueryMode mode = CapabilityQueryMode::Legacy;
  uint64_t query_id = 0;
  std::string graph_id;
  uint64_t graph_revision = 0;
  std::string source_view_id;
  std::string ep_instance_id;
  std::optional<OrtDevice> execution_device;

  MaxShapeOverrideMap max_input_shapes;
  InlinedVector<CapabilityWorkloadBound> workload_bounds;
  InlinedVector<CapabilityNodeConstraint> node_constraints;
  InlinedVector<CapabilityNodeAssignment> tentative_assignments;
  InlinedHashMap<std::string, std::string> ep_configuration;
  std::optional<size_t> max_concurrent_runs;
  std::optional<size_t> max_compute_streams;
};

// EP-private host metadata, not borrowed graph views or a materialized device kernel.
class CapabilityPlanningState {
 public:
  virtual ~CapabilityPlanningState() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(CapabilityPlanningState);

 protected:
  CapabilityPlanningState() = default;
};

using CapabilityPlanningStatePtr = std::shared_ptr<const CapabilityPlanningState>;

enum class CapabilityResourceKind {
  Unresolved,
  GraphValue,
  ImplementationAllocation,
};

struct CapabilityResourceReference {
  CapabilityResourceKind kind = CapabilityResourceKind::Unresolved;
  std::string resource_id;
  // Required for implementation allocations; graph values are scoped by the query.
  std::string ep_instance_id;
  std::string candidate_id;
};

struct CapabilityAllocationAlias {
  CapabilityResourceReference target;
  std::optional<size_t> offset_bytes;
};

enum class CapabilityAllocationKind {
  Unresolved,
  ExecutionWorkspace,
  PersistentPrepack,
  InitializationScratch,
  RuntimePreparation,
  Other,
};

enum class CapabilityAllocationLifetime {
  Unresolved,
  Initialization,
  Session,
  Run,
  Operation,
};

struct CapabilityAllocationRequirement {
  std::string resource_id;
  CapabilityAllocationKind kind = CapabilityAllocationKind::Unresolved;
  std::optional<OrtDevice> device;
  std::optional<size_t> size_bound_bytes;
  // A known zero requests allocator-default alignment, as in WorkspaceRequirement.
  std::optional<size_t> alignment_bytes;
  CapabilityAllocationLifetime lifetime = CapabilityAllocationLifetime::Unresolved;
  std::optional<CapabilityAllocationAlias> alias_of;
  InlinedVector<CapabilityResourceReference> non_overlap_with;
};

struct CapabilityTensorDescription {
  std::string value_name;
  // ONNX TensorProto element-type value; zero is unresolved.
  int32_t element_type = 0;
  std::optional<TensorShapeVector> shape_bounds;
  bool is_initializer = false;
};

struct CapabilityValueAlias {
  std::string value_name;
  CapabilityAllocationAlias alias;
};

enum class CapabilityGraphEffectsKind {
  Unresolved,
  Unchanged,
  Described,
};

struct CapabilityGraphEffects {
  CapabilityGraphEffectsKind kind = CapabilityGraphEffectsKind::Unresolved;
  std::optional<InlinedVector<std::string>> boundary_inputs;
  std::optional<InlinedVector<std::string>> boundary_outputs;
  InlinedVector<CapabilityTensorDescription> introduced_values;
  InlinedVector<std::string> removed_values;
  InlinedVector<CapabilityValueAlias> value_aliases;
  InlinedVector<std::string> unresolved_terms;
};

struct CapabilityCandidateReference {
  std::string ep_instance_id;
  std::string candidate_id;
};

struct CapabilityBoundaryConstraint {
  std::string value_name;
  std::string layout_id;
  std::optional<OrtDevice> device;
};

struct CapabilityCompatibilityConstraints {
  InlinedVector<CapabilityCandidateReference> required_candidates;
  InlinedVector<CapabilityCandidateReference> excluded_candidates;
  InlinedVector<CapabilityBoundaryConstraint> boundary_constraints;
};

enum class CapabilityRecipeKind {
  Unresolved,
  Standalone,
  RegisteredFusion,
  CompiledRegion,
};

struct CapabilityImplementationRecipe {
  std::string recipe_id;
  CapabilityRecipeKind kind = CapabilityRecipeKind::Unresolved;
  CapabilityPlanningStatePtr prepared_state;
  CapabilityGraphEffects graph_effects;
  InlinedVector<std::string> unresolved_terms;
};

enum class CapabilityContractEligibility {
  Unresolved,
  Eligible,
  Ineligible,
};

enum class CapabilityRequirementBoundStatus {
  Unresolved,
  Bounded,
};

struct CapabilityExecutionContract {
  std::string contract_id;
  KernelDispatchPolicy policy = KernelDispatchPolicy::Auto;
  CapabilityContractEligibility eligibility = CapabilityContractEligibility::Unresolved;
  std::string eligibility_reason;
  std::string workload_coverage;
  InlinedVector<std::string> explicit_failure_conditions;
  CapabilityPlanningStatePtr resolved_state;

  CapabilityRequirementBoundStatus bound_status = CapabilityRequirementBoundStatus::Unresolved;
  // Unset means the requirement list is unknown, not that there are no allocations.
  std::optional<InlinedVector<CapabilityAllocationRequirement, 1>> allocation_requirements;
  InlinedVector<std::string> unresolved_terms;
};

struct CapabilityPlanningInfo {
  std::shared_ptr<const CapabilityPlanningQuery> query;
  std::string candidate_id;
  std::string implementation_id;
  InlinedVector<std::string> alternative_view_ids;
  CapabilityCompatibilityConstraints compatibility;
  CapabilityImplementationRecipe recipe;
  InlinedVector<CapabilityExecutionContract, 1> contracts;
};

enum class CapabilityPlanningSupport {
  Unsupported,
  PlanningSafe,
};

// A dispatcher returns Status separately: successful empty discovery is not an EP error.
struct CapabilityPlanningQueryResult {
  CapabilityPlanningQueryResult();
  ~CapabilityPlanningQueryResult();
  CapabilityPlanningQueryResult(CapabilityPlanningQueryResult&&) noexcept;
  CapabilityPlanningQueryResult& operator=(CapabilityPlanningQueryResult&&) noexcept;
  CapabilityPlanningQueryResult(const CapabilityPlanningQueryResult&) = delete;
  CapabilityPlanningQueryResult& operator=(const CapabilityPlanningQueryResult&) = delete;

  CapabilityPlanningSupport support = CapabilityPlanningSupport::Unsupported;
  std::string diagnostic;
  InlinedVector<std::unique_ptr<ComputeCapability>> capabilities;
};

struct CapabilityPlanningSelection {
  std::string candidate_id;
  std::string recipe_id;
  std::string alternative_view_id;
  std::string contract_id;
  uint64_t plan_generation = 0;
};

// Published as const; copies retain only this recipe and contract, not the offer's menu.
struct CapabilityPlanningBinding {
  std::shared_ptr<const CapabilityPlanningQuery> query;
  uint64_t plan_generation = 0;
  std::string candidate_id;
  std::string implementation_id;
  std::string alternative_view_id;
  InlinedVector<size_t> source_nodes;
  CapabilityCompatibilityConstraints compatibility;
  CapabilityImplementationRecipe recipe;
  CapabilityExecutionContract contract;
};

Status ValidateCapabilityPlanningResult(
    const std::shared_ptr<const CapabilityPlanningQuery>& query,
    const CapabilityPlanningQueryResult& result);

// Checks local identity and bounded requirements, not whole-plan feasibility or admission.
// Failure leaves binding unchanged.
Status CreateCapabilityPlanningBinding(
    const std::shared_ptr<const CapabilityPlanningQuery>& query,
    const ComputeCapability& capability,
    const CapabilityPlanningSelection& selection,
    std::shared_ptr<const CapabilityPlanningBinding>& binding);

Status ValidateCapabilityPlanningBindingIdentity(
    const std::shared_ptr<const CapabilityPlanningQuery>& query,
    uint64_t plan_generation,
    const CapabilityPlanningBinding& binding);

}  // namespace onnxruntime
