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
#include "core/common/inlined_containers_fwd.h"

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
  virtual ResourceCount ComputeResourceCount(const Node& node) const = 0;

  std::optional<ResourceCount> GetThreshold() const {
    return threshold_;
  }

  void SetStopAssignment() noexcept {
    stop_assignment_ = true;
  }

  bool IsStopIssued() const noexcept { return stop_assignment_; }

  static std::string MakeUniqueNodeName(const Node& node);

 private:
  bool stop_assignment_ = false;
  std::optional<ResourceCount> threshold_;
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

  [[nodiscard]] static Status CreateAccountants(
      const ConfigOptions& config_options,
      const std::filesystem::path& model_path,
      std::optional<ResourceAccountantMap>& acc_map);

 private:
  void DumpStats(std::ostream& os) const;

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace onnxruntime
