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
  std::vector<NodeIndex> nodes;                                  // a list of NodeIndex to represent Nodes in this Partition
  std::unordered_set<std::string> frontier_node_args;            // a set of string to repsent frontier NodeArgs in this Partition
  std::unordered_set<std::string> rejected_frontiner_node_args;  // a set of string to repsent rejected frontier NodeArgs in this Partition

  std::unordered_set<NodeIndex> predecessor_partitions;            // a set of NodeIndex to represent predecessor Partitions of this Partition
  std::unordered_set<NodeIndex> immediate_predecessor_partitions;  // a set of NodeIndex to represent immediate predecessor Partitions of this Partition

  int cost;  // a cost of this Partition. It can be used to guide customized partitioning

  PartitionMeta() {}
  PartitionMeta(NodeIndex node) {
    nodes.push_back(node);
  }

  inline NodeIndex Id() {
    //Use the first NodeIndex as the id for PartitionMeta
    return nodes.front();
  }
};

// Base class of Partitioner.
// Partitioner is used for GraphPartition to generate a FuseNode in Ort for the nuphar provider.
// OR used for SubgraphPartition to generate subgraph Function within FuseNode in nuphar itself.
class Partitioner {
 public:
  Partitioner() {}

  virtual ~Partitioner() = default;

  // Main function to perform partiton
  Status Evaluate(const onnxruntime::GraphViewer& graph, bool distinguish_subgraph);

 protected:
  // Check whether a Node is included
  virtual bool IsNodeSupported(const Node& node) const = 0;

  // Force a Partition.
  // It returns false to perform default merge process.
  // Returning true avoid performing default process.
  // The customized process need to be implmented within this function
  virtual bool ForcePartition(
      const onnxruntime::GraphViewer& /*graph*/,
      const Node& /*node*/,
      const std::vector<NodeIndex>& /*candidate_partitions*/,
      const std::vector<NodeIndex>& /*immedidate_rejected_partitions*/) {
    return false;
  }

  // Cost Function interface to exstimate Cost of a PartitionMeta.
  // It can be used to trigger FocePartition or any other process.
  virtual int Cost(const Node&, const std::vector<NodeIndex>&) const { return 0; };

  // Update PartitonMeta to include a node
  void UpdateFrontiers(PartitionMeta& part_meta, const Node& node);

  void UpdatePredecessors(PartitionMeta& part_meta, const NodeIndex& node_id);

  // Merge at least two Partitions when they are connected by a node
  void MergePartitions(const Node& node,
                       const std::vector<NodeIndex>& candidates,
                       const std::vector<NodeIndex>& rejected_partitions);

  std::map<NodeIndex, PartitionMeta> partitions_;

 private:
  void RejectNode(
      const onnxruntime::GraphViewer& graph,
      const NodeIndex& node_idx);

  void AcceptNode(
      const onnxruntime::GraphViewer& graph,
      const NodeIndex& node_idx);

  virtual void HandleSubgraph(const onnxruntime::GraphViewer&) {}

 protected:
  virtual void CreateNewPartition(const Node& node, const std::vector<NodeIndex>& immedidate_rejected_partitions);

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(Partitioner);
};

}  // namespace nuphar
}  // namespace onnxruntime
