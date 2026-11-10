// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "contrib_ops/cuda/quantization/matmul_nbits_memory_estimate.h"
#include "core/framework/compute_capability.h"
#include "core/framework/level1_memory_requirements.h"
#include "gtest/gtest.h"
#include "test/util/include/asserts.h"

namespace onnxruntime::test {
namespace {

struct TestSizingInputs final : CapabilityPlanningState {
  explicit TestSizingInputs(std::string description) : description(std::move(description)) {}
  const std::string description;
};

CapabilityPlanningInfo MakePlanning() {
  CapabilityPlanningQuery query;
  query.mode = CapabilityQueryMode::Planning;
  query.query_id = 1;
  query.graph_id = "matmul";
  query.source_view_id = "source";
  query.ep_instance_id = "CUDA:0";
  query.execution_device = OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, OrtDevice::VendorIds::NVIDIA, 0);
  query.max_concurrent_runs = 1;
  query.max_compute_streams = 1;

  CapabilityPlanningInfo planning;
  planning.query = std::make_shared<const CapabilityPlanningQuery>(std::move(query));
  planning.candidate_id = "matmul_7";
  planning.implementation_id = "sizing_fixture";
  planning.alternative_view_ids = {"source"};
  planning.recipe.recipe_id = "standalone";
  planning.recipe.kind = CapabilityRecipeKind::Standalone;
  planning.recipe.graph_effects.kind = CapabilityGraphEffectsKind::Unchanged;
  auto& contract = planning.contracts.emplace_back();
  contract.contract_id = "configured";
  contract.eligibility = CapabilityContractEligibility::Eligible;
  contract.workload_coverage = "Exact sizing fixture inputs";
  contract.resolved_state = std::make_shared<const TestSizingInputs>("Runtime state is not replaced by allocation sizing");
  return planning;
}

Level1MemoryAllocation Allocation(const CapabilityPlanningInfo& planning, std::string id, size_t bytes) {
  Level1MemoryAllocation allocation;
  allocation.resource_id = std::move(id);
  allocation.device = planning.query->execution_device;
  allocation.size_bound_bytes = bytes;
  allocation.alignment_bytes = 0;
  return allocation;
}

CapabilityLevel1MemoryEstimate MakeSource(const CapabilityPlanningInfo& planning) {
  CapabilityLevel1MemoryEstimate source;
  source.provenance = MakeLevel1MemoryEstimateProvenance(
      planning, planning.contracts.front(), "Test.L1",
      std::make_shared<const TestSizingInputs>("Explicit allocation/decomposition evidence"));
  source.scope = Level1MemoryEstimateScope::ImplementationAllocations;
  source.estimate.emplace();
  return source;
}

void ProveZeroCategories(Level1MemoryAllocationBreakdown& breakdown) {
  breakdown.execution_workspace.allocations.emplace();
  breakdown.persistent_prepack.allocations.emplace();
  breakdown.initialization_scratch.allocations.emplace();
  breakdown.runtime_preparation.allocations.emplace();
}

CapabilityLevel1MemoryEstimate MakeDecomposedSource(const CapabilityPlanningInfo& planning) {
  auto source = MakeSource(planning);
  source.estimate->runtime_workspace_bytes = 128;
  source.estimate->persistent_prepack_bytes = 72;
  source.estimate->initialization_scratch_bytes = 256;
  source.estimate->runtime_transient_bytes = 512;
  source.breakdown.execution_workspace.allocations =
      InlinedVector<Level1MemoryAllocation, 1>{Allocation(planning, "workspace", 128)};
  source.breakdown.persistent_prepack.allocations =
      InlinedVector<Level1MemoryAllocation, 1>{Allocation(planning, "packed_a", 32), Allocation(planning, "packed_b", 40)};
  source.breakdown.initialization_scratch.allocations =
      InlinedVector<Level1MemoryAllocation, 1>{Allocation(planning, "conversion", 256)};
  source.breakdown.runtime_preparation.allocations =
      InlinedVector<Level1MemoryAllocation, 1>{Allocation(planning, "profiling", 384)};
  source.breakdown.runtime_preparation_allocations_exclude_workspace = true;
  return source;
}

std::unique_ptr<ComputeCapability> MakeCapability(CapabilityPlanningInfo planning) {
  auto graph = std::make_unique<IndexedSubGraph>();
  graph->nodes = {7};
  auto capability = std::make_unique<ComputeCapability>(std::move(graph));
  capability->planning = std::make_shared<const CapabilityPlanningInfo>(std::move(planning));
  return capability;
}

const CapabilityAllocationRequirement* FindAllocation(const CapabilityExecutionContract& contract, std::string_view id) {
  if (!contract.allocation_requirements.has_value()) {
    return nullptr;
  }
  const auto& requirements = *contract.allocation_requirements;
  const auto it = std::find_if(requirements.begin(), requirements.end(),
                               [id](const auto& requirement) { return requirement.resource_id == id; });
  return it == requirements.end() ? nullptr : &*it;
}

using namespace onnxruntime::contrib::cuda;

std::shared_ptr<MatMulNBitsMemorySizingContext> MakeMatMulContext(
    const CapabilityPlanningInfo& planning, int64_t weight_prepacked = kMatMulNBitsWeightNotPrepacked,
    bool has_zero_points = false) {
  auto context = std::make_shared<MatMulNBitsMemorySizingContext>();
  context->provenance = MakeLevel1MemoryEstimateProvenance(
      planning, planning.contracts.front(), "CUDA.MatMulNBits.L1", nullptr);
  context->route = MatMulNBitsMemoryRoute::FpAIntB;
  context->n = 128;
  context->k = 256;
  context->bits = 4;
  context->block_size = 32;
  context->weight_prepacked = weight_prepacked;
  context->has_zero_points = has_zero_points;
  context->estimate = ComputeMatMulNBitsPrepackMemoryEstimate(
      context->n, context->k, context->bits, context->block_size, weight_prepacked, has_zero_points);
  context->prepack_buffers = ComputeMatMulNBitsPrepackBufferSizes(
      context->n, context->k, context->bits, context->block_size, has_zero_points);
  return context;
}

}  // namespace

TEST(Level1MemoryRequirementsTest, DefaultZerosAndUnknownWorkspaceAreNotProofs) {
  auto planning = MakePlanning();
  auto& contract = planning.contracts.front();
  auto source = MakeSource(planning);
  source.estimate->runtime_workspace_bytes = 0;
  ASSERT_STATUS_OK(AdaptLevel1MemoryEstimate(source, planning, contract));
  EXPECT_EQ(contract.bound_status, CapabilityRequirementBoundStatus::Unresolved);
  EXPECT_FALSE(contract.allocation_requirements.has_value());
  EXPECT_EQ(contract.unresolved_terms.size(), 4U);

  ProveZeroCategories(source.breakdown);
  ASSERT_STATUS_OK(AdaptLevel1MemoryEstimate(source, planning, contract));
  EXPECT_EQ(contract.bound_status, CapabilityRequirementBoundStatus::Bounded);
  ASSERT_TRUE(contract.allocation_requirements.has_value());
  EXPECT_TRUE(contract.allocation_requirements->empty());

  source.estimate->runtime_workspace_bytes.reset();
  ASSERT_STATUS_OK(AdaptLevel1MemoryEstimate(source, planning, contract));
  EXPECT_EQ(contract.bound_status, CapabilityRequirementBoundStatus::Unresolved);
  EXPECT_FALSE(contract.unresolved_terms.empty());
}

TEST(Level1MemoryRequirementsTest, MapsAllocationsAndRetainsOwnedSizingProvenanceThroughBinding) {
  auto planning = MakePlanning();
  auto& contract = planning.contracts.front();
  auto source = MakeDecomposedSource(planning);
  const auto runtime_state = contract.resolved_state;
  const auto sizing_inputs = source.provenance.sizing_context;
  ASSERT_STATUS_OK(AdaptLevel1MemoryEstimate(source, planning, contract));
  EXPECT_EQ(contract.bound_status, CapabilityRequirementBoundStatus::Bounded);
  EXPECT_EQ(contract.resolved_state, runtime_state);
  ASSERT_TRUE(contract.allocation_requirements.has_value());
  ASSERT_EQ(contract.allocation_requirements->size(), 5U);
  const std::array<CapabilityAllocationKind, 5> kinds{
      CapabilityAllocationKind::ExecutionWorkspace, CapabilityAllocationKind::PersistentPrepack,
      CapabilityAllocationKind::PersistentPrepack, CapabilityAllocationKind::InitializationScratch,
      CapabilityAllocationKind::RuntimePreparation};
  const std::array<CapabilityAllocationLifetime, 5> lifetimes{
      CapabilityAllocationLifetime::Operation, CapabilityAllocationLifetime::Session,
      CapabilityAllocationLifetime::Session, CapabilityAllocationLifetime::Initialization,
      CapabilityAllocationLifetime::Operation};
  for (size_t i = 0; i < kinds.size(); ++i) {
    const auto& requirement = (*contract.allocation_requirements)[i];
    EXPECT_EQ(requirement.kind, kinds[i]);
    EXPECT_EQ(requirement.lifetime, lifetimes[i]);
    EXPECT_EQ(requirement.alignment_bytes, 0U);
    EXPECT_FALSE(requirement.alias_of.has_value());
    EXPECT_TRUE(requirement.non_overlap_with.empty());
  }
  EXPECT_EQ(FindAllocation(contract, "profiling")->size_bound_bytes, 384U);
  const auto provenance = contract.allocation_provenance;
  source.estimate->runtime_transient_bytes = 9000;
  const auto retained = std::static_pointer_cast<const CapabilityLevel1MemoryEstimateState>(provenance->sizing_context);
  EXPECT_EQ(retained->source.estimate->runtime_transient_bytes, 512U);
  EXPECT_EQ(retained->source.provenance.sizing_context, sizing_inputs);
  EXPECT_EQ(provenance->query, planning.query);
  EXPECT_EQ(provenance->candidate_id, planning.candidate_id);
  EXPECT_EQ(provenance->implementation_id, planning.implementation_id);
  EXPECT_EQ(provenance->recipe_id, planning.recipe.recipe_id);

  const auto capability = MakeCapability(planning);
  const CapabilityPlanningSelection selection{"matmul_7", "standalone", "source", "configured", 1};
  std::shared_ptr<const CapabilityPlanningBinding> binding;
  ASSERT_STATUS_OK(CreateCapabilityPlanningBinding(planning.query, *capability, selection, binding));
  EXPECT_EQ(binding->contract.allocation_provenance, provenance);

  planning.implementation_id = "different_implementation";
  const auto retagged = MakeCapability(planning);
  const auto original_binding = binding;
  EXPECT_EQ(CreateCapabilityPlanningBinding(planning.query, *retagged, selection, binding).Code(),
            common::INVALID_ARGUMENT);
  EXPECT_EQ(binding, original_binding);
}

TEST(Level1MemoryRequirementsTest, MissingFailedAndGraphInclusiveEstimatorsRemainUnresolved) {
  auto planning = MakePlanning();
  auto& contract = planning.contracts.front();
  auto source = MakeDecomposedSource(planning);
  source.estimate.reset();
  source.diagnostic = "Estimator failed for these inputs";
  ASSERT_STATUS_OK(AdaptLevel1MemoryEstimate(source, planning, contract));
  EXPECT_EQ(contract.bound_status, CapabilityRequirementBoundStatus::Unresolved);
  EXPECT_FALSE(contract.allocation_requirements.has_value());
  EXPECT_EQ(contract.unresolved_terms.front(), source.diagnostic);

  source = MakeDecomposedSource(planning);
  source.estimate->runtime_workspace_bytes = 1000000;  // A graph-inclusive total is not workspace.
  for (const auto scope : {Level1MemoryEstimateScope::GraphInclusive, Level1MemoryEstimateScope::Unresolved}) {
    source.scope = scope;
    ASSERT_STATUS_OK(AdaptLevel1MemoryEstimate(source, planning, contract));
    EXPECT_EQ(contract.bound_status, CapabilityRequirementBoundStatus::Unresolved);
    EXPECT_FALSE(contract.allocation_requirements.has_value());
  }
}

TEST(Level1MemoryRequirementsTest, AggregatePeaksNeedDecompositionAlignmentAndPreparationCoverage) {
  auto planning = MakePlanning();
  auto& contract = planning.contracts.front();
  auto source = MakeDecomposedSource(planning);
  source.breakdown.persistent_prepack.allocations.reset();
  source.breakdown.execution_workspace.allocations->front().alignment_bytes.reset();
  source.breakdown.runtime_preparation_allocations_exclude_workspace = false;
  ASSERT_STATUS_OK(AdaptLevel1MemoryEstimate(source, planning, contract));
  EXPECT_EQ(contract.bound_status, CapabilityRequirementBoundStatus::Unresolved);
  ASSERT_NE(FindAllocation(contract, "workspace"), nullptr);
  EXPECT_FALSE(FindAllocation(contract, "workspace")->alignment_bytes.has_value());
  EXPECT_EQ(FindAllocation(contract, "packed_a"), nullptr);
  EXPECT_EQ(FindAllocation(contract, "profiling"), nullptr);
  EXPECT_GE(contract.unresolved_terms.size(), 3U);
}

TEST(Level1MemoryRequirementsTest, RejectsRetaggingWithoutReplacingExistingRequirements) {
  auto planning = MakePlanning();
  auto& contract = planning.contracts.front();
  const auto source = MakeDecomposedSource(planning);
  ASSERT_STATUS_OK(AdaptLevel1MemoryEstimate(source, planning, contract));
  const auto original_provenance = contract.allocation_provenance;
  using Mutator = void (*)(CapabilityLevel1MemoryEstimate&);
  const std::array<Mutator, 9> invalid{
      [](auto& s) { s.provenance.query = std::make_shared<const CapabilityPlanningQuery>(*s.provenance.query); },
      [](auto& s) { s.provenance.candidate_id = "different"; },
      [](auto& s) { s.provenance.implementation_id = "different"; },
      [](auto& s) { s.provenance.recipe_id = "different"; },
      [](auto& s) { s.provenance.contract_id = "different"; },
      [](auto& s) { s.provenance.policy = KernelDispatchPolicy::Memory; },
      [](auto& s) { s.provenance.workload_coverage = "A larger workload"; },
      [](auto& s) { s.provenance.estimator_id.clear(); },
      [](auto& s) { s.provenance.sizing_context.reset(); },
  };
  for (const auto mutate : invalid) {
    auto invalid_source = source;
    mutate(invalid_source);
    EXPECT_EQ(AdaptLevel1MemoryEstimate(invalid_source, planning, contract).Code(), common::INVALID_ARGUMENT);
    EXPECT_EQ(contract.allocation_provenance, original_provenance);
    EXPECT_EQ(contract.bound_status, CapabilityRequirementBoundStatus::Bounded);
    EXPECT_EQ(FindAllocation(contract, "workspace")->size_bound_bytes, 128U);
  }

  auto duplicate_source = source;
  duplicate_source.breakdown.persistent_prepack.allocations->front().resource_id = "workspace";
  EXPECT_EQ(AdaptLevel1MemoryEstimate(duplicate_source, planning, contract).Code(), common::INVALID_ARGUMENT);
  EXPECT_EQ(contract.allocation_provenance, original_provenance);
}

TEST(Level1MemoryRequirementsTest, InvalidQueryDoesNotReplaceExistingContract) {
  auto planning = MakePlanning();
  auto& contract = planning.contracts.front();
  ASSERT_STATUS_OK(AdaptLevel1MemoryEstimate(MakeDecomposedSource(planning), planning, contract));
  const auto original_provenance = contract.allocation_provenance;
  const auto original_runtime_state = contract.resolved_state;
  using Mutator = void (*)(CapabilityPlanningQuery&);
  const std::array<Mutator, 4> invalid{
      [](auto& query) { query.graph_id.clear(); },
      [](auto& query) { query.source_view_id.clear(); },
      [](auto& query) { query.ep_instance_id.clear(); },
      [](auto& query) { query.workload_bounds.push_back({"m", 2, 1}); },
  };
  for (const auto mutate : invalid) {
    auto invalid_planning = planning;
    auto invalid_query = std::make_shared<CapabilityPlanningQuery>(*planning.query);
    mutate(*invalid_query);
    invalid_planning.query = std::move(invalid_query);
    const auto source = MakeDecomposedSource(invalid_planning);
    EXPECT_EQ(AdaptLevel1MemoryEstimate(source, invalid_planning, contract).Code(), common::INVALID_ARGUMENT);
    EXPECT_EQ(contract.allocation_provenance, original_provenance);
    EXPECT_EQ(contract.resolved_state, original_runtime_state);
    EXPECT_EQ(contract.bound_status, CapabilityRequirementBoundStatus::Bounded);
    EXPECT_TRUE(contract.unresolved_terms.empty());
    ASSERT_TRUE(contract.allocation_requirements.has_value());
    ASSERT_EQ(contract.allocation_requirements->size(), 5U);
    EXPECT_EQ(FindAllocation(contract, "workspace")->size_bound_bytes, 128U);
  }
}

TEST(MatMulNBitsMemoryRequirementsTest, LegacyChunkPlanSuppliesWorkspaceAndVerifiedZeroCategories) {
  for (const bool force_chunked : {false, true}) {
    for (const bool has_g_idx : {false, true}) {
      auto planning = MakePlanning();
      auto& contract = planning.contracts.front();
      auto context = std::make_shared<MatMulNBitsMemorySizingContext>();
      context->provenance = MakeLevel1MemoryEstimateProvenance(planning, contract, "CUDA.MatMulNBits.L1", nullptr);
      context->route = MatMulNBitsMemoryRoute::Legacy;
      context->n = 100;
      context->k = 65;
      context->block_size = 32;
      context->element_size = sizeof(uint16_t);
      context->force_chunked = force_chunked;
      context->has_g_idx = has_g_idx;
      context->chunk_target_rows = 16;
      context->legacy_workspace = ComputeMatMulNBitsLegacyWorkspacePlan(
          context->n, context->k, context->block_size, context->element_size, !has_g_idx, force_chunked, 16);
      ASSERT_TRUE(context->legacy_workspace.has_value());
      context->estimate.emplace();
      context->estimate->runtime_workspace_bytes = context->legacy_workspace->workspace_bytes;
      const auto source = DescribeMatMulNBitsMemoryEstimate(context);
      ASSERT_STATUS_OK(AdaptLevel1MemoryEstimate(source, planning, contract));
      EXPECT_EQ(contract.bound_status, has_g_idx ? CapabilityRequirementBoundStatus::Unresolved
                                                 : CapabilityRequirementBoundStatus::Bounded);
      ASSERT_TRUE(contract.allocation_requirements.has_value());
      ASSERT_EQ(contract.allocation_requirements->size(), 1U);
      const auto& workspace = contract.allocation_requirements->front();
      EXPECT_EQ(workspace.kind, CapabilityAllocationKind::ExecutionWorkspace);
      EXPECT_EQ(workspace.size_bound_bytes, (force_chunked && !has_g_idx ? 16U : 100U) * 96U * sizeof(uint16_t));
      EXPECT_EQ(contract.unresolved_terms.empty(), !has_g_idx);
    }
  }
}

TEST(MatMulNBitsMemoryRequirementsTest, PrepackDecomposesDestinationsAndActualConversionScratch) {
  auto planning = MakePlanning();
  auto& contract = planning.contracts.front();
  auto context = MakeMatMulContext(planning, kMatMulNBitsWeightNotPrepacked, true);
  context->facts.prepack_inputs_are_constant_initializers = true;
  ASSERT_TRUE(context->estimate.has_value());
  ASSERT_TRUE(context->prepack_buffers.has_value());
  const auto& sizes = *context->prepack_buffers;
  EXPECT_EQ(sizes.packed_weight_bytes, 16384U);
  EXPECT_EQ(sizes.scale_bytes, 2048U);
  EXPECT_EQ(context->estimate->persistent_prepack_bytes, 16384U + 2048U + 2048U);
  EXPECT_EQ(context->estimate->initialization_scratch_bytes, 16384U + 32U * sizeof(int32_t));
  ASSERT_STATUS_OK(AdaptLevel1MemoryEstimate(DescribeMatMulNBitsMemoryEstimate(context), planning, contract));
  EXPECT_EQ(contract.bound_status, CapabilityRequirementBoundStatus::Unresolved);  // Profiling is not supplied.
  ASSERT_NE(FindAllocation(contract, "packed_weight"), nullptr);
  EXPECT_EQ(FindAllocation(contract, "packed_weight")->size_bound_bytes, sizes.packed_weight_bytes);
  EXPECT_EQ(FindAllocation(contract, "packed_scales")->size_bound_bytes, sizes.scale_bytes);
  EXPECT_EQ(FindAllocation(contract, "scaled_zero_points")->size_bound_bytes, sizes.zero_point_bytes);
  EXPECT_EQ(FindAllocation(contract, "prepack_transposed_weight")->kind, CapabilityAllocationKind::InitializationScratch);
  EXPECT_EQ(FindAllocation(contract, "prepack_permutation_map")->size_bound_bytes, 64U * sizeof(int32_t));
}

TEST(MatMulNBitsMemoryRequirementsTest, OfflineWeightReuseRequiresResidencyAndInitializerProof) {
  for (const auto weight_prepacked : {kMatMulNBitsWeightPrepackedSm80, kMatMulNBitsWeightPrepackedSm90}) {
    const std::array<MatMulNBitsMemoryPlanningFacts, 4> facts{
        {{}, {true, std::nullopt, true}, {true, true, true}, {true, false, true}}};
    for (const auto& input_facts : facts) {
      auto planning = MakePlanning();
      auto& contract = planning.contracts.front();
      auto context = MakeMatMulContext(planning, weight_prepacked);
      context->facts = input_facts;
      ASSERT_STATUS_OK(AdaptLevel1MemoryEstimate(DescribeMatMulNBitsMemoryEstimate(context), planning, contract));
      EXPECT_EQ(contract.bound_status, CapabilityRequirementBoundStatus::Unresolved);
      EXPECT_EQ(FindAllocation(contract, "packed_scales") != nullptr,
                input_facts.prepack_inputs_are_constant_initializers.value_or(false));
      EXPECT_EQ(FindAllocation(contract, "packed_weight"), nullptr);
      EXPECT_EQ(FindAllocation(contract, "prepack_transposed_weight"), nullptr);
      const bool needs_copy = input_facts.prepack_inputs_are_constant_initializers.value_or(false) &&
                              input_facts.offline_weight_on_execution_device == false;
      EXPECT_EQ(FindAllocation(contract, "offline_weight_device_copy") != nullptr, needs_copy);
      if (needs_copy) {
        EXPECT_EQ(FindAllocation(contract, "offline_weight_device_copy")->size_bound_bytes,
                  context->prepack_buffers->packed_weight_bytes);
      }
    }
  }
}

TEST(MatMulNBitsMemoryRequirementsTest, PreparationDoesNotReemitRetainedWorkspaceOrSumAlternativeRoutes) {
  for (const bool dequant_route : {false, true}) {
    auto planning = MakePlanning();
    auto& contract = planning.contracts.front();
    auto context = MakeMatMulContext(planning, kMatMulNBitsWeightPrepackedSm80);
    context->facts.prepack_inputs_are_constant_initializers = true;
    context->facts.offline_weight_on_execution_device = true;
    context->facts.persistent_tactic_cache_load_disabled = true;
    // Captured profiler results exercise decomposition, not CUDA tactic/formula coverage.
    context->constructor_profile_scratch_bytes = 4096;
    context->cutlass_workspace_bytes = 128;
    context->lazy_profile_scratch_bytes = 4096;
    context->workspace_retained_during_preparation = true;
    context->estimate->runtime_workspace_bytes = 128;
    context->estimate->initialization_scratch_bytes = 4096;
    context->estimate->runtime_transient_bytes = 4096 + 128;
    if (dequant_route) {
      context->dequant_workspace_bytes = context->n * context->k * sizeof(uint16_t);
      context->estimate->runtime_workspace_bytes = context->dequant_workspace_bytes;
    }
    ASSERT_STATUS_OK(AdaptLevel1MemoryEstimate(DescribeMatMulNBitsMemoryEstimate(context), planning, contract));
    EXPECT_EQ(contract.bound_status, dequant_route ? CapabilityRequirementBoundStatus::Unresolved
                                                   : CapabilityRequirementBoundStatus::Bounded);
    ASSERT_NE(FindAllocation(contract, "lazy_profiler"), nullptr);
    EXPECT_EQ(FindAllocation(contract, "lazy_profiler")->size_bound_bytes, 4096U);
    EXPECT_FALSE(FindAllocation(contract, "lazy_profiler")->alias_of.has_value());
    EXPECT_TRUE(FindAllocation(contract, "lazy_profiler")->non_overlap_with.empty());
    if (dequant_route) {
      EXPECT_EQ(FindAllocation(contract, "cutlass_workspace"), nullptr);
      for (const auto& requirement : *contract.allocation_requirements) {
        EXPECT_NE(requirement.kind, CapabilityAllocationKind::ExecutionWorkspace);
      }
    } else {
      ASSERT_NE(FindAllocation(contract, "cutlass_workspace"), nullptr);
      EXPECT_EQ(FindAllocation(contract, "cutlass_workspace")->size_bound_bytes, 128U);
    }
  }
}

TEST(MatMulNBitsMemoryRequirementsTest, CachedTacticValidationIsNotCoveredByTheInitialProfilePeak) {
  for (const auto disabled : {std::optional<bool>{}, std::optional<bool>{false}, std::optional<bool>{true}}) {
    auto planning = MakePlanning();
    auto& contract = planning.contracts.front();
    auto context = MakeMatMulContext(planning, kMatMulNBitsWeightPrepackedSm80);
    context->facts = {true, true, disabled};
    context->constructor_profile_scratch_bytes = 4096;
    context->cutlass_workspace_bytes = 0;
    context->lazy_profile_scratch_bytes = 0;
    context->estimate->runtime_workspace_bytes = 0;
    context->estimate->initialization_scratch_bytes = 4096;
    ASSERT_STATUS_OK(AdaptLevel1MemoryEstimate(DescribeMatMulNBitsMemoryEstimate(context), planning, contract));
    EXPECT_EQ(contract.bound_status, disabled.value_or(false) ? CapabilityRequirementBoundStatus::Bounded
                                                              : CapabilityRequirementBoundStatus::Unresolved);
    ASSERT_NE(FindAllocation(contract, "constructor_profiler"), nullptr);
    EXPECT_EQ(FindAllocation(contract, "constructor_profiler")->size_bound_bytes, 4096U);
  }
}

TEST(MatMulNBitsMemoryRequirementsTest, InvalidAndOverflowingSizingRemainUnknown) {
  const auto missing_context = DescribeMatMulNBitsMemoryEstimate(nullptr);
  EXPECT_FALSE(missing_context.diagnostic.empty());
  EXPECT_FALSE(missing_context.estimate.has_value());
  EXPECT_FALSE(ComputeMatMulNBitsPrepackMemoryEstimate(0, 256, 4, 32, 0, false).has_value());
  EXPECT_FALSE(ComputeMatMulNBitsPrepackMemoryEstimate(128, 256, 3, 32, 0, false).has_value());
  EXPECT_FALSE(ComputeMatMulNBitsPrepackMemoryEstimate(
                   std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::max(), 8, 32, 0, true)
                   .has_value());
  EXPECT_FALSE(ComputeMatMulNBitsLegacyWorkspacePlan(
                   128, std::numeric_limits<int64_t>::max(), 32, sizeof(uint16_t), true, false, 16)
                   .has_value());
  EXPECT_FALSE(ComputeMatMulNBitsLegacyWorkspacePlan(
                   0, std::numeric_limits<int64_t>::max(), 1, sizeof(uint16_t), false, false, 16)
                   .has_value());
  EXPECT_FALSE(ComputeMatMulNBitsPrepackMemoryEstimate(
                   1, std::numeric_limits<int64_t>::max(), 8, 1, kMatMulNBitsWeightPrepackedSm80, true)
                   .has_value());
  auto planning = MakePlanning();
  auto& contract = planning.contracts.front();
  auto context = MakeMatMulContext(planning);
  context->estimate.reset();
  ASSERT_STATUS_OK(AdaptLevel1MemoryEstimate(DescribeMatMulNBitsMemoryEstimate(context), planning, contract));
  EXPECT_EQ(contract.bound_status, CapabilityRequirementBoundStatus::Unresolved);
  EXPECT_FALSE(contract.allocation_requirements.has_value());
}

}  // namespace onnxruntime::test
