// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/capability_planning.h"
#include "core/framework/compute_capability.h"

#include <array>
#include <memory>
#include <utility>

#include "gtest/gtest.h"
#include "test/util/include/asserts.h"

namespace onnxruntime {
namespace test {
namespace {

struct TestPlanningState final : CapabilityPlanningState {
  explicit TestPlanningState(InlinedVector<std::string> permitted_routes)
      : permissions(std::move(permitted_routes)) {}

  InlinedVector<std::string> permissions;
};

OrtDevice CudaDevice() {
  return OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, OrtDevice::VendorIds::NVIDIA, 0);
}

CapabilityPlanningQuery MakeQuery() {
  CapabilityPlanningQuery query;
  query.mode = CapabilityQueryMode::Planning;
  query.query_id = 1;
  query.graph_id = "decoder";
  query.graph_revision = 17;
  query.source_view_id = "V_source";
  query.ep_instance_id = "CUDA:0";
  query.execution_device = CudaDevice();
  query.workload_bounds = {{"batch", 1, 1}, {"query_length", 1, 4}, {"total_kv", 0, 64}, {"kv_capacity", 64, 64}};
  query.node_constraints = {{42, std::nullopt,
                             InlinedVector<KernelDispatchPolicy>{KernelDispatchPolicy::Memory,
                                                                 KernelDispatchPolicy::Latency}}};
  query.ep_configuration = {{"flash_enabled", "true"}, {"mea_enabled", "true"}, {"unfused_enabled", "true"}, {"xqa_enabled", "false"}, {"cudnn_enabled", "false"}};
  query.max_concurrent_runs = 1;
  query.max_compute_streams = 1;
  return query;
}

CapabilityAllocationRequirement MakeRequirement(
    std::string resource_id, CapabilityAllocationKind kind, CapabilityAllocationLifetime lifetime, size_t bytes) {
  CapabilityAllocationRequirement requirement;
  requirement.resource_id = std::move(resource_id);
  requirement.kind = kind;
  requirement.device = CudaDevice();
  requirement.size_bound_bytes = bytes;
  requirement.alignment_bytes = 0;
  requirement.lifetime = lifetime;
  return requirement;
}

CapabilityPlanningInfo MakePlanningInfo(const std::shared_ptr<const CapabilityPlanningQuery>& query) {
  CapabilityPlanningInfo planning;
  planning.query = query;
  planning.candidate_id = "GQA_42";
  planning.implementation_id = "GQA_cuda_fp16";
  planning.alternative_view_ids = {"V_source"};
  planning.recipe.recipe_id = "R_GQA";
  planning.recipe.kind = CapabilityRecipeKind::Standalone;
  planning.recipe.graph_effects.kind = CapabilityGraphEffectsKind::Unchanged;

  CapabilityExecutionContract memory;
  memory.contract_id = "C_memory";
  memory.policy = KernelDispatchPolicy::Memory;
  memory.eligibility = CapabilityContractEligibility::Eligible;
  memory.workload_coverage = "Declared prefill, chunked-prefill, and decode envelope";
  memory.explicit_failure_conditions = {"No permitted bounded route for the actual request"};
  memory.resolved_state = std::make_shared<const TestPlanningState>(
      InlinedVector<std::string>{"Flash_prefill", "unfused_covering_fallback"});
  memory.bound_status = CapabilityRequirementBoundStatus::Bounded;
  memory.allocation_requirements = InlinedVector<CapabilityAllocationRequirement, 1>{
      MakeRequirement("workspace", CapabilityAllocationKind::ExecutionWorkspace,
                      CapabilityAllocationLifetime::Operation, 213248),
      MakeRequirement("prepack", CapabilityAllocationKind::PersistentPrepack,
                      CapabilityAllocationLifetime::Session, 0),
      MakeRequirement("init_scratch", CapabilityAllocationKind::InitializationScratch,
                      CapabilityAllocationLifetime::Initialization, 0),
      MakeRequirement("runtime_preparation", CapabilityAllocationKind::RuntimePreparation,
                      CapabilityAllocationLifetime::Operation, 0)};

  CapabilityExecutionContract latency = memory;
  latency.contract_id = "C_latency";
  latency.policy = KernelDispatchPolicy::Latency;
  latency.resolved_state = std::make_shared<const TestPlanningState>(
      InlinedVector<std::string>{"Flash", "MEA", "unfused"});
  latency.allocation_requirements->front().size_bound_bytes = 1245440;
  planning.contracts.push_back(std::move(memory));
  planning.contracts.push_back(std::move(latency));
  return planning;
}

std::unique_ptr<ComputeCapability> MakeCapability(CapabilityPlanningInfo planning) {
  auto sub_graph = std::make_unique<IndexedSubGraph>();
  sub_graph->nodes = {42};
  auto capability = std::make_unique<ComputeCapability>(std::move(sub_graph));
  capability->planning = std::make_shared<const CapabilityPlanningInfo>(std::move(planning));
  return capability;
}

CapabilityPlanningSelection MemorySelection() {
  return {"GQA_42", "R_GQA", "V_source", "C_memory", 7};
}

}  // namespace

TEST(CapabilityPlanningTest, DiscoveryDistinguishesUnsupportedEmptyUnresolvedAndLegacy) {
  const auto query = std::make_shared<const CapabilityPlanningQuery>(MakeQuery());
  CapabilityPlanningQueryResult result;
  EXPECT_EQ(ValidateCapabilityPlanningResult(query, result).Code(), common::NOT_IMPLEMENTED);

  result.support = CapabilityPlanningSupport::PlanningSafe;
  ASSERT_STATUS_OK(ValidateCapabilityPlanningResult(query, result));

  auto planning = MakePlanningInfo(query);
  planning.contracts.front().bound_status = CapabilityRequirementBoundStatus::Unresolved;
  planning.contracts.front().allocation_requirements.reset();
  planning.contracts.front().unresolved_terms = {"Execution workspace bound is unavailable"};
  result.capabilities.push_back(MakeCapability(std::move(planning)));
  ASSERT_STATUS_OK(ValidateCapabilityPlanningResult(query, result));

  std::shared_ptr<const CapabilityPlanningBinding> binding;
  EXPECT_EQ(CreateCapabilityPlanningBinding(query, *result.capabilities.front(), MemorySelection(), binding).Code(),
            common::NOT_IMPLEMENTED);
  EXPECT_EQ(binding, nullptr);

  result.capabilities.front()->planning.reset();
  EXPECT_EQ(ValidateCapabilityPlanningResult(query, result).Code(), common::NOT_IMPLEMENTED);
  EXPECT_EQ(CreateCapabilityPlanningBinding(query, *result.capabilities.front(), MemorySelection(), binding).Code(),
            common::NOT_IMPLEMENTED);
}

TEST(CapabilityPlanningTest, SelectionKeepsResolvedContractAndAllL1CategoriesTogether) {
  const auto query = std::make_shared<const CapabilityPlanningQuery>(MakeQuery());
  const auto capability = MakeCapability(MakePlanningInfo(query));
  const auto& memory = capability->planning->contracts.front();
  const auto& latency = capability->planning->contracts.back();
  std::shared_ptr<const CapabilityPlanningBinding> binding;
  ASSERT_STATUS_OK(CreateCapabilityPlanningBinding(query, *capability, MemorySelection(), binding));
  ASSERT_STATUS_OK(ValidateCapabilityPlanningBindingIdentity(query, MemorySelection().plan_generation, *binding));
  EXPECT_EQ(ValidateCapabilityPlanningBindingIdentity(query, MemorySelection().plan_generation + 1, *binding).Code(),
            common::INVALID_ARGUMENT);

  EXPECT_EQ(binding->query, query);
  EXPECT_EQ(binding->contract.resolved_state, memory.resolved_state);
  EXPECT_NE(binding->contract.resolved_state, latency.resolved_state);
  ASSERT_TRUE(binding->contract.allocation_requirements.has_value());
  ASSERT_EQ(binding->contract.allocation_requirements->size(), memory.allocation_requirements->size());
  for (size_t i = 0; i < memory.allocation_requirements->size(); ++i) {
    const auto& selected = (*binding->contract.allocation_requirements)[i];
    const auto& original = (*memory.allocation_requirements)[i];
    EXPECT_EQ(selected.kind, original.kind);
    EXPECT_EQ(selected.lifetime, original.lifetime);
    EXPECT_EQ(selected.size_bound_bytes, original.size_bound_bytes);
  }

  auto selection = MemorySelection();
  selection.contract_id = latency.contract_id;
  ASSERT_STATUS_OK(CreateCapabilityPlanningBinding(query, *capability, selection, binding));
  EXPECT_EQ(binding->contract.resolved_state, latency.resolved_state);
  EXPECT_EQ(binding->contract.allocation_requirements->front().size_bound_bytes,
            latency.allocation_requirements->front().size_bound_bytes);
}

TEST(CapabilityPlanningTest, DiscardedAlternativesReleaseWithoutInvalidatingSelectedState) {
  const auto query = std::make_shared<const CapabilityPlanningQuery>(MakeQuery());
  std::weak_ptr<const CapabilityPlanningState> memory_state;
  std::weak_ptr<const CapabilityPlanningState> latency_state;
  std::weak_ptr<const CapabilityPlanningState> recipe_state;
  std::shared_ptr<const CapabilityPlanningBinding> binding;
  {
    auto planning = MakePlanningInfo(query);
    planning.recipe.prepared_state = std::make_shared<const TestPlanningState>(
        InlinedVector<std::string>{"Prepared GQA implementation parameters"});
    memory_state = planning.contracts.front().resolved_state;
    latency_state = planning.contracts.back().resolved_state;
    recipe_state = planning.recipe.prepared_state;
    const auto capability = MakeCapability(std::move(planning));
    ASSERT_STATUS_OK(CreateCapabilityPlanningBinding(query, *capability, MemorySelection(), binding));

    auto rejected_selection = MemorySelection();
    rejected_selection.contract_id = "not_offered";
    std::shared_ptr<const CapabilityPlanningBinding> rejected_binding;
    EXPECT_EQ(CreateCapabilityPlanningBinding(query, *capability, rejected_selection, rejected_binding).Code(),
              common::INVALID_ARGUMENT);
    EXPECT_EQ(rejected_binding, nullptr);
  }

  EXPECT_FALSE(memory_state.expired());
  EXPECT_FALSE(recipe_state.expired());
  EXPECT_TRUE(latency_state.expired());
  binding.reset();
  EXPECT_TRUE(memory_state.expired());
  EXPECT_TRUE(recipe_state.expired());
}

TEST(CapabilityPlanningTest, InvalidSelectionDoesNotReplaceAnExistingBinding) {
  const auto query = std::make_shared<const CapabilityPlanningQuery>(MakeQuery());
  const auto capability = MakeCapability(MakePlanningInfo(query));
  std::shared_ptr<const CapabilityPlanningBinding> binding;
  ASSERT_STATUS_OK(CreateCapabilityPlanningBinding(query, *capability, MemorySelection(), binding));
  const auto original = binding;

  std::array<CapabilityPlanningSelection, 5> invalid;
  invalid.fill(MemorySelection());
  invalid[0].candidate_id = "another_candidate";
  invalid[1].recipe_id = "another_recipe";
  invalid[2].alternative_view_id = "another_view";
  invalid[3].contract_id = "another_contract";
  invalid[4].plan_generation = 0;
  for (const auto& selection : invalid) {
    EXPECT_EQ(CreateCapabilityPlanningBinding(query, *capability, selection, binding).Code(), common::INVALID_ARGUMENT);
    EXPECT_EQ(binding, original);
  }
}

TEST(CapabilityPlanningTest, ChangedQuerySnapshotInvalidatesOldOffers) {
  const auto query = std::make_shared<const CapabilityPlanningQuery>(MakeQuery());
  const auto capability = MakeCapability(MakePlanningInfo(query));
  std::shared_ptr<const CapabilityPlanningBinding> original_binding;
  ASSERT_STATUS_OK(CreateCapabilityPlanningBinding(query, *capability, MemorySelection(), original_binding));
  auto changed = MakeQuery();
  changed.graph_revision += 1;
  changed.workload_bounds.front().upper_bound = 2;
  changed.ep_configuration["flash_enabled"] = "false";
  const auto changed_query = std::make_shared<const CapabilityPlanningQuery>(std::move(changed));
  std::shared_ptr<const CapabilityPlanningBinding> binding;
  EXPECT_EQ(CreateCapabilityPlanningBinding(changed_query, *capability, MemorySelection(), binding).Code(),
            common::INVALID_ARGUMENT);
  EXPECT_EQ(binding, nullptr);
  EXPECT_EQ(ValidateCapabilityPlanningBindingIdentity(changed_query, MemorySelection().plan_generation,
                                                      *original_binding)
                .Code(),
            common::INVALID_ARGUMENT);

  const auto copied_query = std::make_shared<const CapabilityPlanningQuery>(*query);
  EXPECT_EQ(CreateCapabilityPlanningBinding(copied_query, *capability, MemorySelection(), binding).Code(),
            common::INVALID_ARGUMENT);
}

TEST(CapabilityPlanningTest, KnownZeroAndEmptyRequirementsAreNotUnknown) {
  const auto query = std::make_shared<const CapabilityPlanningQuery>(MakeQuery());
  auto planning = MakePlanningInfo(query);
  planning.contracts.front().allocation_requirements->front().size_bound_bytes = 0;
  auto capability = MakeCapability(planning);
  std::shared_ptr<const CapabilityPlanningBinding> binding;
  ASSERT_STATUS_OK(CreateCapabilityPlanningBinding(query, *capability, MemorySelection(), binding));

  planning.contracts.front().allocation_requirements = InlinedVector<CapabilityAllocationRequirement, 1>{};
  capability = MakeCapability(planning);
  ASSERT_STATUS_OK(CreateCapabilityPlanningBinding(query, *capability, MemorySelection(), binding));

  planning.contracts.front().allocation_requirements.reset();
  capability = MakeCapability(std::move(planning));
  const auto original = binding;
  EXPECT_EQ(CreateCapabilityPlanningBinding(query, *capability, MemorySelection(), binding).Code(),
            common::INVALID_ARGUMENT);
  EXPECT_EQ(binding, original);
}

TEST(CapabilityPlanningTest, BoundedClaimsRequireCompleteAllocationDescriptors) {
  const auto query = std::make_shared<const CapabilityPlanningQuery>(MakeQuery());
  using Mutator = void (*)(CapabilityExecutionContract&);
  const std::array<Mutator, 8> incomplete = {
      [](CapabilityExecutionContract& c) { c.allocation_requirements->front().size_bound_bytes.reset(); },
      [](CapabilityExecutionContract& c) { c.allocation_requirements->front().alignment_bytes.reset(); },
      [](CapabilityExecutionContract& c) { c.allocation_requirements->front().device.reset(); },
      [](CapabilityExecutionContract& c) {
        c.allocation_requirements->front().kind = CapabilityAllocationKind::Unresolved;
      },
      [](CapabilityExecutionContract& c) {
        c.allocation_requirements->front().lifetime = CapabilityAllocationLifetime::Unresolved;
      },
      [](CapabilityExecutionContract& c) {
        c.allocation_requirements->push_back(c.allocation_requirements->front());
      },
      [](CapabilityExecutionContract& c) { c.unresolved_terms = {"Contradicts the bounded claim"}; },
      [](CapabilityExecutionContract& c) {
        c.allocation_requirements->front().alias_of =
            CapabilityAllocationAlias{{CapabilityResourceKind::GraphValue, "Q", "", ""}, std::nullopt};
      }};
  for (const auto mutate : incomplete) {
    auto planning = MakePlanningInfo(query);
    mutate(planning.contracts.front());
    const auto capability = MakeCapability(std::move(planning));
    std::shared_ptr<const CapabilityPlanningBinding> binding;
    EXPECT_EQ(CreateCapabilityPlanningBinding(query, *capability, MemorySelection(), binding).Code(),
              common::INVALID_ARGUMENT);
    EXPECT_EQ(binding, nullptr);
  }
}

TEST(CapabilityPlanningTest, SelectionPreservesNormalizedEpPinsAndPolicyPermissions) {
  auto query_value = MakeQuery();
  query_value.node_constraints.front().permitted_policies = InlinedVector<KernelDispatchPolicy>{
      KernelDispatchPolicy::Latency};
  auto query = std::make_shared<const CapabilityPlanningQuery>(query_value);
  auto capability = MakeCapability(MakePlanningInfo(query));
  std::shared_ptr<const CapabilityPlanningBinding> binding;
  EXPECT_EQ(CreateCapabilityPlanningBinding(query, *capability, MemorySelection(), binding).Code(),
            common::INVALID_ARGUMENT);

  auto latency_selection = MemorySelection();
  latency_selection.contract_id = "C_latency";
  ASSERT_STATUS_OK(CreateCapabilityPlanningBinding(query, *capability, latency_selection, binding));
  const auto original = binding;

  query_value.node_constraints.front().permitted_policies = InlinedVector<KernelDispatchPolicy>{};
  query = std::make_shared<const CapabilityPlanningQuery>(query_value);
  capability = MakeCapability(MakePlanningInfo(query));
  EXPECT_EQ(CreateCapabilityPlanningBinding(query, *capability, latency_selection, binding).Code(),
            common::INVALID_ARGUMENT);
  EXPECT_EQ(binding, original);

  query_value.node_constraints.front().permitted_policies.reset();
  query_value.node_constraints.front().required_ep_instance_id = "CPU:0";
  query = std::make_shared<const CapabilityPlanningQuery>(std::move(query_value));
  capability = MakeCapability(MakePlanningInfo(query));
  EXPECT_EQ(CreateCapabilityPlanningBinding(query, *capability, latency_selection, binding).Code(),
            common::INVALID_ARGUMENT);
  EXPECT_EQ(binding, original);
}

TEST(CapabilityPlanningTest, FusionBindingPreservesSourceEffectsAndPreparedStateWithoutApplying) {
  const auto query = std::make_shared<const CapabilityPlanningQuery>(MakeQuery());
  auto planning = MakePlanningInfo(query);
  planning.candidate_id = "fused_AB";
  planning.implementation_id = "fused_AB_cuda";
  planning.alternative_view_ids = {"V_optimized"};
  planning.recipe.recipe_id = "R_AB";
  planning.recipe.kind = CapabilityRecipeKind::RegisteredFusion;
  planning.recipe.prepared_state = std::make_shared<const TestPlanningState>(
      InlinedVector<std::string>{"Prepared source match A,B and packed-weight layout"});
  auto& effects = planning.recipe.graph_effects;
  effects.kind = CapabilityGraphEffectsKind::Described;
  effects.boundary_inputs = InlinedVector<std::string>{"X", "W"};
  effects.boundary_outputs = InlinedVector<std::string>{"Y"};
  effects.removed_values = {"T"};
  effects.introduced_values = {{"P", 1, TensorShapeVector{8, 16}, true}};
  planning.compatibility.boundary_constraints = {{"Y", "row_major", CudaDevice()}};
  auto& contract = planning.contracts.front();
  contract.contract_id = "C_AB";
  contract.allocation_requirements->at(1).resource_id = "P_device_prepack";
  contract.allocation_requirements->at(1).size_bound_bytes = 512;
  contract.allocation_requirements->at(2).size_bound_bytes = 128;
  contract.allocation_requirements->at(2).non_overlap_with = {
      {CapabilityResourceKind::ImplementationAllocation, "workspace", query->ep_instance_id, planning.candidate_id}};
  planning.contracts.pop_back();

  auto capability = MakeCapability(std::move(planning));
  capability->sub_graph->nodes = {42, 43};
  int applications = 0;
  capability->optimization_func = [&applications](Graph&, const ComputeCapability&, ComputeCapability&,
                                                  const GraphOptimizerRegistry&) {
    ++applications;
    return Status::OK();
  };
  CapabilityPlanningSelection selection{"fused_AB", "R_AB", "V_optimized", "C_AB", 7};
  std::shared_ptr<const CapabilityPlanningBinding> binding;
  ASSERT_STATUS_OK(CreateCapabilityPlanningBinding(query, *capability, selection, binding));
  EXPECT_EQ(applications, 0);
  EXPECT_EQ(binding->source_nodes, (InlinedVector<size_t>{42, 43}));
  EXPECT_EQ(binding->recipe.prepared_state, capability->planning->recipe.prepared_state);
  EXPECT_EQ(binding->recipe.graph_effects.removed_values, capability->planning->recipe.graph_effects.removed_values);
  EXPECT_EQ(binding->recipe.graph_effects.introduced_values.front().shape_bounds,
            capability->planning->recipe.graph_effects.introduced_values.front().shape_bounds);
  EXPECT_EQ(binding->compatibility.boundary_constraints.front().layout_id,
            capability->planning->compatibility.boundary_constraints.front().layout_id);
}

}  // namespace test
}  // namespace onnxruntime
