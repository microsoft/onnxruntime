// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/nuphar/partition/partitioner.h"

#include "core/providers/nuphar/common/analysis/graph_stats.h"
#include "core/providers/nuphar/common/nuphar_subgraph.h"

#include <vector>

namespace onnxruntime {
namespace nuphar {

class SubgraphPartitioner : public Partitioner {
 public:
  SubgraphPartitioner()
      : Partitioner() {}

  Status Partition(
      const Node& node,
      std::vector<NupharSubgraphUnit>& subgraphs,
      const std::map<std::string, const Tensor*>& initializers);

 private:
  std::unique_ptr<codegen::OrtGraphStats> graph_stats_;

  bool IsNodeSupported(const Node& node) override;
  bool ForcePartition(const NodeIndex& node_idx,
                      const int topology_idx,
                      const Node& node,
                      const std::vector<NodeIndex>& candiates,
                      const std::vector<NodeIndex>& rejected_partitions) override;
  int Cost(const Node& node, const std::vector<NodeIndex>& candiates) const override;
  int Cost(const Node& node) const;
};

}  // namespace nuphar
}  // namespace onnxruntime
