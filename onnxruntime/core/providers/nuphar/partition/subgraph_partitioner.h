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
      FindInitializerFunc find_initializer_func);

  void SetSpecifiedNodeNames(const std::vector<std::string>& specified_names);

 private:
  std::vector<NodeIndex> sorted_partitions_;

  std::unique_ptr<OrtGraphStats> graph_stats_;

  bool IsNodeSupported(const Node& node) const override;

  bool ForcePartition(
      const onnxruntime::GraphViewer& graph,
      const Node& node,
      const std::vector<NodeIndex>& candiates,
      const std::vector<NodeIndex>& rejected_partitions) override;

  int Cost(const Node& node, const std::vector<NodeIndex>& candiates) const override;
  int Cost(const Node& node) const;

  // Some help function for ForcePartition
  bool SpecifiedNodePartition(const Node& node,
                              const std::vector<NodeIndex>& candiates,
                              const std::vector<NodeIndex>& rejected_partitions);

  // a lookup to user-guided partitioning
  std::unordered_set<std::string> specified_names_;

  std::map<std::string, NodeIndex> no_merged_args_to_nodes_;
};

}  // namespace nuphar
}  // namespace onnxruntime
