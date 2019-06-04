// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/partition/partitioner.h"

#include "core/codegen/common/common.h"

namespace onnxruntime {
namespace nuphar {

void Partitioner::UpdateNodesInPartitionMeta(PartitionMeta& part_meta, const Node& node) {
  // update frontier_nodes and rejected_nodes
  for (auto it = node.OutputEdgesBegin(); it != node.OutputEdgesEnd(); ++it) {
    const Node& dst_node = it->GetNode();
    if (IsNodeSupported(dst_node) &&
        part_meta.rejected_nodes.count(dst_node.Index()) == 0) {
      // If a child is supported and not rejected, put it to frontier_nodes
      part_meta.frontier_nodes.insert(dst_node.Index());
    } else if (part_meta.rejected_nodes.count(dst_node.Index()) == 0) {
      part_meta.rejected_nodes.insert(dst_node.Index());
    }
  }
}

void Partitioner::MergePartitions(const NodeIndex& node_idx,
                                  const int topology_idx,
                                  const Node& node,
                                  const std::vector<NodeIndex>& candiates,
                                  const std::vector<NodeIndex>& rejected_partitions) {
  std::unordered_set<NodeIndex> merged_partitions;
  PartitionMeta& part_meta = partitions_[candiates[0]];
  // update max_topology_idx
  part_meta.max_topology_index = topology_idx;
  // update cost
  part_meta.cost = Cost(node, candiates);
  // merge the rest meta
  for (size_t i = 1; i < candiates.size(); ++i) {
    PartitionMeta& other_part_meta = partitions_[candiates[i]];
    // record merged_partitions
    merged_partitions.insert(other_part_meta.Id());
    // merge nodes
    for (auto& n : other_part_meta.nodes) {
      part_meta.nodes.push_back(n);
    }
    // merge rejected_nodes
    for (auto& n : other_part_meta.rejected_nodes) {
      part_meta.rejected_nodes.insert(n);
    }
    // merge frontier_nodes
    for (auto& n : other_part_meta.frontier_nodes) {
      part_meta.frontier_nodes.insert(n);
    }
    // predominate_partitions
    for (auto& p : other_part_meta.predominate_partitions) {
      part_meta.predominate_partitions.insert(p);
    }
    // erase the partition
    partitions_.erase(other_part_meta.Id());
  }

  // update all predominate_partitions in the rest partition
  for (auto& iter : partitions_) {
    for (auto& p : iter.second.predominate_partitions) {
      if (merged_partitions.count(p) > 0) {
        iter.second.predominate_partitions.erase(p);
        iter.second.predominate_partitions.insert(candiates[0]);
      }
    }
  }

  // make this new node to this partition
  part_meta.nodes.push_back(node_idx);
  // update frontier_nodes and rejected_nodes
  UpdateNodesInPartitionMeta(part_meta, node);
  // update rejected partitions to predominate partitions
  // rejected partitions' predominate parititions also to predominate partitions
  for (auto& id : rejected_partitions) {
    part_meta.predominate_partitions.insert(id);
    for (auto& p : partitions_[id].predominate_partitions) {
      part_meta.predominate_partitions.insert(p);
    }
  }
}

void Partitioner::AcceptNode(
    const onnxruntime::GraphViewer& graph,
    const NodeIndex& node_idx,
    const int topology_idx) {
  std::vector<NodeIndex> candidate_partitions;
  std::vector<NodeIndex> rejected_partitions;
  for (auto& p : partitions_) {
    bool is_child = p.second.frontier_nodes.count(node_idx) > 0;
    bool is_rejected = p.second.rejected_nodes.count(node_idx) > 0;
    if (is_child && !is_rejected) {
      candidate_partitions.push_back(p.first);
    }
    if (is_rejected) {
      rejected_partitions.push_back(p.first);
    }
  }

  std::vector<NodeIndex> coexist_partitions;
  if (candidate_partitions.size() > 1) {
    // found multiple candidate partitions
    // remove predominate_partitions from candidate_partitions
    std::vector<bool> is_partitions_coexisted(candidate_partitions.size(), true);
    for (auto& cand_id : candidate_partitions) {
      PartitionMeta& part_meta_cand = partitions_[cand_id];
      for (size_t i = 0; i < candidate_partitions.size(); ++i) {
        if (is_partitions_coexisted[i] &&
            part_meta_cand.predominate_partitions.count(candidate_partitions[i]) > 0) {
          is_partitions_coexisted[i] = false;
        }
      }
    }

    for (size_t i = 0; i < candidate_partitions.size(); ++i) {
      if (is_partitions_coexisted[i]) {
        coexist_partitions.push_back(candidate_partitions[i]);
      }
    }

  } else {
    coexist_partitions = candidate_partitions;
  }

  const Node* node = graph.GetNode(node_idx);

  if (coexist_partitions.size() == 0) {
    // a new partition
    partitions_.insert(std::make_pair(node_idx, PartitionMeta(node_idx, topology_idx)));
    PartitionMeta& part_meta = partitions_[node_idx];
    // update cost
    part_meta.cost = Cost(*node, coexist_partitions);
    // update frontier_nodes and rejected_nodes
    UpdateNodesInPartitionMeta(part_meta, *node);
    // update rejected partitions to predominate partitions
    // rejected partitions' predominate parititions also to predominate partitions
    for (auto& id : rejected_partitions) {
      part_meta.predominate_partitions.insert(id);
      for (auto& p : partitions_[id].predominate_partitions) {
        part_meta.predominate_partitions.insert(p);
      }
    }

  } else if (!ForcePartition(node_idx, topology_idx,
                             *node, coexist_partitions, rejected_partitions)) {
    if (coexist_partitions.size() == 1) {
      // found a unique partition
      PartitionMeta& part_meta = partitions_[coexist_partitions[0]];
      // make this new node to this partition
      part_meta.nodes.push_back(node_idx);
      // update max_topology_idx
      part_meta.max_topology_index = topology_idx;
      // update cost
      part_meta.cost = Cost(*node, coexist_partitions);
      // update frontier_nodes and rejected_nodes
      UpdateNodesInPartitionMeta(part_meta, *node);
      // update rejected partitions to predominate partitions
      // rejected partitions' predominate parititions also to predominate partitions
      for (auto& id : rejected_partitions) {
        part_meta.predominate_partitions.insert(id);
        for (auto& p : partitions_[id].predominate_partitions) {
          part_meta.predominate_partitions.insert(p);
        }
      }
    } else {
      // if there are still multiple coexist partitions
      // we can fuse all coexist partitions together
      // we fuse all partitions to the first partition.
      MergePartitions(node_idx, topology_idx, *node, coexist_partitions, rejected_partitions);
    }
  }
}

void Partitioner::RejectNode(
    const onnxruntime::GraphViewer& graph,
    const NodeIndex& node_idx) {
  const Node* node = graph.GetNode(node_idx);

  for (auto& p : partitions_) {
    bool is_child = p.second.frontier_nodes.count(node_idx) > 0;
    bool is_rejected = p.second.rejected_nodes.count(node_idx) > 0;
    if (is_child || is_rejected) {
      for (auto it = node->OutputEdgesBegin(); it != node->OutputEdgesEnd(); ++it) {
        const Node& dst_node = it->GetNode();
        if (p.second.rejected_nodes.count(dst_node.Index()) == 0) {
          p.second.rejected_nodes.insert(dst_node.Index());
        }
      }
    }
  }
}

void Partitioner::HandleSubgraph(const onnxruntime::GraphViewer& graph) {
  PartitionMeta part_meta;

  for (auto& node_idx : graph.GetNodesInTopologicalOrder()) {
    const Node* node = graph.GetNode(node_idx);
    if (IsNodeSupported(*node)) {
      part_meta.nodes.push_back(node_idx);
    } else {
      return;
    }
  }

  partitions_.insert(std::make_pair(graph.GetNodesInTopologicalOrder().front(), part_meta));
}

// Partition the graph (fusing ops) based on the dependency and whether ops are supported:
//
Status Partitioner::Evaluate(const onnxruntime::GraphViewer& graph) {
  if (graph.IsSubgraph()) {
    HandleSubgraph(graph);
    return Status::OK();
  }

  int topology_idx = 0;
  for (auto& node_idx : graph.GetNodesInTopologicalOrder()) {
    const Node* node = graph.GetNode(node_idx);
    if (IsNodeSupported(*node)) {
      AcceptNode(graph, node_idx, topology_idx);
    } else {
      RejectNode(graph, node_idx);
    }
    ++topology_idx;
  }

  return Status::OK();
}
}  // namespace nuphar
}  // namespace onnxruntime
