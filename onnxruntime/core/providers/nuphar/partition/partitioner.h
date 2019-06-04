// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/graph/graph_viewer.h"

#include <unordered_set>
#include <vector>

namespace onnxruntime {
namespace nuphar {

// A generic Partition data struct for Partitioner
struct PartitionMeta {
  std::vector<NodeIndex> nodes;
  std::unordered_set<NodeIndex> frontier_nodes;
  std::unordered_set<NodeIndex> rejected_nodes;
  std::unordered_set<NodeIndex> predominate_partitions;
  int cost;
  int max_topology_index;  // topology_idx that helps sort

  PartitionMeta() {}
  PartitionMeta(NodeIndex node, int topology_idx)
      : max_topology_index(topology_idx) {
    nodes.push_back(node);
  }

  inline NodeIndex Id() {
    //Use the first NodeIndex as the id for PartitionMeta
    return nodes.front();
  }
};

// Base class of Partitioner.
// Partitioner is used for GraphPartition to generate a FuseNode in Ort for the nuphar provider.
// OR used for FuncPartition to generate subgraph Function within FuseNode in nuphar itself.
class Partitioner {
 public:
  Partitioner() {}

  // Main function to perform partiton
  Status Evaluate(const onnxruntime::GraphViewer& graph);

 protected:
  // Check whether a Node is included
  virtual bool IsNodeSupported(const Node& node) = 0;

  // Force a Partition.
  // It returns false to perform default merge process.
  // Returning true avoid performing default process.
  // The customized process need to be implmented within this function
  virtual bool ForcePartition(
      const NodeIndex& /*n_idx*/,
      const int /*topology_idx*/,
      const Node& /*node*/,
      const std::vector<NodeIndex>& /*candidate_partitions*/,
      const std::vector<NodeIndex>& /*rejected_partitions*/) {
    return false;
  }

  // Cost Function interface to exstimate Cost of a PartitionMeta.
  // It can be used to trigger FocePartition or any other process.
  virtual int Cost(const Node&, const std::vector<NodeIndex>&) const { return 0; };

  // Update PartitonMeta to include a node
  void UpdateNodesInPartitionMeta(PartitionMeta& part_meta, const Node& node);

  // Merge at least two Partitions when they are connected by a node
  void MergePartitions(const NodeIndex& node_idx,
                       const int topology_idx,
                       const Node& node,
                       const std::vector<NodeIndex>& candiates,
                       const std::vector<NodeIndex>& rejected_partitions);

  std::map<NodeIndex, PartitionMeta> partitions_;

 private:
  void RejectNode(
      const onnxruntime::GraphViewer& graph,
      const NodeIndex& node_idx);

  void AcceptNode(
      const onnxruntime::GraphViewer& graph,
      const NodeIndex& node_idx,
      const int topology_idx);

  void HandleSubgraph(const onnxruntime::GraphViewer& graph);

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(Partitioner);
};

}  // namespace nuphar
}  // namespace onnxruntime
