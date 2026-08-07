// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <filesystem>
#include <iosfwd>
#include <optional>
#include <string>
#include <unordered_set>
#include <variant>

#include "core/common/common.h"
#include "core/common/inlined_containers.h"
#include "core/framework/max_shape_override.h"

namespace onnxruntime {

struct ConfigOptions;
#ifndef SHARED_PROVIDER
class Node;
#else
struct Node;
#endif

// Common holder for potentially different resource accounting
// for different EPs
using ResourceCount = std::variant<size_t>;

enum class WorkspaceEstimateSource {
  kNone,
  kFallback,
  kProfile,
  kEstimator,
  kProfileAndEstimator,
};

struct WorkspaceEstimateSourceCounts {
  size_t fallback = 0;
  size_t profile = 0;
  size_t estimator = 0;
  size_t profile_and_estimator = 0;
};

// Type-erased arithmetic for ResourceCount values.
// Implementations use std::visit so the compiler enforces exhaustive handling
// of all variant members — adding a new type to ResourceCount will produce
// build errors at each call site that must be addressed.
//
// NOTE: These functions are NOT available through the provider bridge (shared library EPs).
// Budget enforcement for bridge-based EPs (e.g., in-tree CUDA EP) will be moved to the
// graph partitioner in a follow-up PR.
ResourceCount AddResourceCounts(const ResourceCount& a, const ResourceCount& b);
bool ResourceCountExceeds(const ResourceCount& a, const ResourceCount& b);
std::string FormatResourceCount(const ResourceCount& rc);

/// <summary>
/// This class is used for graph partitioning by EPs
/// It stores the cumulative amount of the resource such as
/// memory that would be consumed by the graph nodes if it is assigned to the EP.
///
/// It provides interfaces to add, remove and query the resource consumption.
///
/// Each provider may assign its own meaning to the resource according to its constraints.
/// </summary>
class IResourceAccountant {
 protected:
  IResourceAccountant() = default;
  IResourceAccountant(const ResourceCount& threshold) : threshold_(threshold) {}

 public:
  virtual ~IResourceAccountant() = default;
  virtual ResourceCount GetConsumedAmount() const = 0;
  virtual void AddConsumedAmount(const ResourceCount& amount) = 0;
  virtual void RemoveConsumedAmount(const ResourceCount& amount) = 0;
  virtual ResourceCount ComputeResourceCount(const Node& node) = 0;

  // Combines an EP/kernel-specific Level-1 estimate with a previously computed
  // node cost. The estimator replaces fallback workspace or is maximized with
  // profiled workspace. The returned cost is used for the capability's budget.
  virtual ResourceCount UpdateResourceCountWithWorkspaceEstimate(
      size_t /*node_index*/, const ResourceCount& resource_count, size_t /*workspace_bytes*/) {
    return resource_count;
  }

  std::optional<ResourceCount> GetThreshold() const {
    return threshold_;
  }

  void SetThreshold(const ResourceCount& threshold) {
    threshold_ = threshold;
  }

  void SetStopAssignment() noexcept {
    stop_assignment_ = true;
  }

  bool IsStopIssued() const noexcept { return stop_assignment_; }

  // Called before each GetCapability pass to reset per-pass state:
  // clears the stop flag (which only applies to the pass that set it)
  // and discards pending resource tracking from a previous (discarded) pass.
  void ResetForNewPass() {
    stop_assignment_ = false;
    ResetPendingResourcesImpl();
  }

  // Called when a node's cost is committed (AccountForNode/AccountForAllNodes).
  // Commits any per-node resource breakdown tracked while ComputeResourceCount()
  // was called. Default no-op for accountants without a resource breakdown.
  virtual void CommitResourcesForNode(size_t /*node_index*/) {}

  // Returns the pending workspace estimate recorded while computing a node's cost.
  // Used when layout transformation defers committing first-pass capabilities.
  virtual size_t GetPendingWorkspaceEstimate(size_t /*node_index*/) const { return 0; }

  virtual WorkspaceEstimateSource GetPendingWorkspaceEstimateSource(size_t /*node_index*/) const {
    return WorkspaceEstimateSource::kNone;
  }

  // Commits a workspace estimate whose original pending state is no longer available.
  // Used for nodes that survive a layout-transformation second pass.
  virtual void AddCommittedWorkspaceEstimate(
      size_t /*workspace_bytes*/, WorkspaceEstimateSource /*source*/) {}

  static std::string MakeUniqueNodeName(const Node& node);

  /// Set the max shape overrides for workspace estimation.
  /// Called during graph partitioner initialization when session.max_shape_override is set.
  void SetMaxShapeOverrides(MaxShapeOverrideMap overrides) {
    max_shape_overrides_ = std::move(overrides);
  }

  const MaxShapeOverrideMap& GetMaxShapeOverrides() const {
    return max_shape_overrides_;
  }

  void SetMaxShapeInferenceResult(MaxShapeInferenceResult result) {
    max_shape_inference_result_ = std::move(result);
  }

  const MaxShapeInferenceResult& GetMaxShapeInferenceResult() const {
    return max_shape_inference_result_;
  }

  /// Returns workspace for nodes that were accepted and committed by partitioning.
  virtual size_t GetCommittedWorkspaceEstimate() const { return 0; }

  /// Returns accepted-node counts grouped by the workspace source used for budgeting.
  virtual WorkspaceEstimateSourceCounts GetWorkspaceEstimateSourceCounts() const { return {}; }

 protected:
  // Override to discard per-pass state for capabilities that were only probed.
  virtual void ResetPendingResourcesImpl() {}

 private:
  bool stop_assignment_ = false;
  std::optional<ResourceCount> threshold_;
  MaxShapeOverrideMap max_shape_overrides_;
  MaxShapeInferenceResult max_shape_inference_result_;
};

// A map of Ep Type to a resource accountant for this EP
using ResourceAccountantMap = InlinedHashMap<std::string, std::unique_ptr<IResourceAccountant>>;

// This struct keeps accounting of the memory allocation stats
// for a kernel during runtime if enabled.
// Each metric describes max value seen as a result of inference run(s)
struct NodeAllocationStats {
  // Total input sizes for the node
  size_t input_sizes = 0;
  // consumed initializer sizes
  size_t initializers_sizes = 0;
  // dynamically allocated outputs that actually occurred
  // at inference time. (usually not fixed size and not pre-allocated)
  size_t total_dynamic_sizes = 0;
  // Temporary allocations that took place at this execution.
  size_t total_temp_allocations = 0;

  NodeAllocationStats& operator+=(const NodeAllocationStats& other) {
    input_sizes += other.input_sizes;
    initializers_sizes += other.initializers_sizes;
    total_dynamic_sizes += other.total_dynamic_sizes;
    total_temp_allocations += other.total_temp_allocations;
    return *this;
  }

  void UpdateIfGreater(const NodeAllocationStats& other) {
    input_sizes = std::max(input_sizes, other.input_sizes);
    initializers_sizes = std::max(initializers_sizes, other.initializers_sizes);
    total_dynamic_sizes = std::max(total_dynamic_sizes, other.total_dynamic_sizes);
    total_temp_allocations = std::max(total_temp_allocations, other.total_temp_allocations);
  }
};

class NodeStatsRecorder {
 public:
  explicit NodeStatsRecorder(const std::filesystem::path& stats_file_name);
  ~NodeStatsRecorder();

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(NodeStatsRecorder);

  const std::filesystem::path& GetNodeStatsFileName() const noexcept;

  bool ShouldAccountFor(const std::string& input_output_name) const;

  void ResetPerRunNameDeduper();

  void ReportNodeStats(const std::string& node_name, const NodeAllocationStats& stats);

  void DumpStats(const std::filesystem::path& model_path) const;

 private:
  void DumpStats(std::ostream& os) const;

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

Status CreateAccountants(
    const ConfigOptions& config_options,
    const std::filesystem::path& model_path,
    std::optional<ResourceAccountantMap>& acc_map);

}  // namespace onnxruntime
