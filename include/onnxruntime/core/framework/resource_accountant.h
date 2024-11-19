// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <optional>
#include <variant>

namespace onnxruntime {
#ifndef SHARED_PROVIDER
class Graph;
#else
struct Graph;
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
  virtual ResourceCount ComputeResourceCount(const Graph&, size_t node_index) const = 0;
  std::optional<ResourceCount> GetThreshold() const {
    return threshold_;
  }

 private:
  std::optional<ResourceCount> threshold_;
};

}  // namespace onnxruntime
