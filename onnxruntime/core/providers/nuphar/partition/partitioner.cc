// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/partition/partitioner.h"

#include "core/codegen/common/common.h"

namespace onnxruntime {
namespace nuphar {

void Partitioner::UpdateFrontiers(PartitionMeta& part_meta, const Node& node) {
  // update frontier_nodes and rejected_frontier_nodes
  for (auto it = node.OutputEdgesBegin(); it != node.OutputEdgesEnd(); ++it) {
    const Node& dst_node = it->GetNode();
    if (IsNodeSupported(dst_node) &&
        part_meta.rejected_frontier_nodes.count(dst_node.Index()) == 0) {
      // If a child is supported and not rejected, put it to frontier_nodes
      part_meta.frontier_nodes.insert(dst_node.Index());
    } else if (part_meta.rejected_frontier_nodes.count(dst_node.Index()) == 0) {
      part_meta.rejected_frontier_nodes.insert(dst_node.Index());
    }
  }
}

void Partitioner::UpdatePredecessors(PartitionMeta& part_meta, const NodeIndex& id) {
  part_meta.predecessor_partitions.insert(id);
  part_meta.immediate_predecessor_partitions.insert(id);
  for (auto& p : partitions_[id].predecessor_partitions) {
    part_meta.predecessor_partitions.insert(p);
  }
}

void Partitioner::MergePartitions(const Node& node,
                                  const std::vector<NodeIndex>& candiates,
                                  const std::vector<NodeIndex>& rejected_partitions) {
  std::unordered_set<NodeIndex> merged_partitions;
  PartitionMeta& part_meta = partitions_[candiates[0]];
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

    // merge rejected_frontier_nodes
    for (auto& n : other_part_meta.rejected_frontier_nodes) {
      part_meta.rejected_frontier_nodes.insert(n);
    }

    // merge frontier_nodes
    for (auto& n : other_part_meta.frontier_nodes) {
      part_meta.frontier_nodes.insert(n);
    }

    // predecessor_partitions
    for (auto& p : other_part_meta.predecessor_partitions) {
      part_meta.predecessor_partitions.insert(p);
    }

    // immediate_predecessor_partitions
    for (auto& p : other_part_meta.immediate_predecessor_partitions) {
      part_meta.immediate_predecessor_partitions.insert(p);
    }

    // erase the partition
    partitions_.erase(other_part_meta.Id());
  }

  // update all predecessor_partitions in the rest partition
  // by replacing merged_partitions to candiates[0]
  for (const auto& mp : merged_partitions) {
    for (auto& iter : partitions_) {
      // replace predecessor_partitions
      if (iter.second.predecessor_partitions.count(mp) > 0) {
        iter.second.predecessor_partitions.erase(mp);
        iter.second.predecessor_partitions.insert(candiates[0]);
      }

      // replace predecessor_partitions
      if (iter.second.immediate_predecessor_partitions.count(mp) > 0) {
        iter.second.immediate_predecessor_partitions.erase(mp);
        iter.second.immediate_predecessor_partitions.insert(candiates[0]);
      }
    }
  }

  // make this new node to this partition
  const NodeIndex node_idx = node.Index();
  part_meta.nodes.push_back(node_idx);
  // update frontier_nodes and rejected_frontier_nodes
  UpdateFrontiers(part_meta, node);
  // update rejected's predecessor partitions
  for (auto& id : rejected_partitions) {
    UpdatePredecessors(part_meta, id);
  }
}

void Partitioner::AcceptNode(
    const onnxruntime::GraphViewer& graph,
    const NodeIndex& node_idx) {
  std::vector<NodeIndex> immedidate_rejected_partitions;  // immediate rejected partitions
  std::unordered_set<NodeIndex> all_rejected_partitions;  // all rejected partitions
  for (const auto& p : partitions_) {
    bool is_rejected = p.second.rejected_frontier_nodes.count(node_idx) > 0;
    if (is_rejected) {
      immedidate_rejected_partitions.push_back(p.first);
      all_rejected_partitions.insert(p.first);
      const PartitionMeta& part_meta_rejected = partitions_[p.first];
      for (const auto& r_p : part_meta_rejected.predecessor_partitions) {
        all_rejected_partitions.insert(r_p);
      }
    }
  }

  std::vector<NodeIndex> candidate_partitions;
  for (const auto& p : partitions_) {
    if (p.second.frontier_nodes.count(node_idx) > 0 &&
        all_rejected_partitions.count(p.first) == 0) {
      candidate_partitions.push_back(p.first);
    }
  }

  std::vector<NodeIndex> coexist_partitions;
  if (candidate_partitions.size() > 1) {
    // found multiple candidate partitions
    // remove a candidate from candidate_partitions
    // if it is in a predecessor_partitions of another candidate_partitions
    std::vector<bool> is_partitions_coexisted(candidate_partitions.size(), true);
    for (auto& cand_id : candidate_partitions) {
      PartitionMeta& part_meta_cand = partitions_[cand_id];
      for (size_t i = 0; i < candidate_partitions.size(); ++i) {
        if (is_partitions_coexisted[i] &&
            part_meta_cand.predecessor_partitions.count(candidate_partitions[i]) > 0) {
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

  if (ForcePartition(*node, coexist_partitions, immedidate_rejected_partitions)) {
    return;
  }

  if (coexist_partitions.size() == 0) {
    // a new partition
    partitions_.insert(std::make_pair(node_idx, PartitionMeta(node_idx)));
    PartitionMeta& part_meta = partitions_[node_idx];
    // update cost
    part_meta.cost = Cost(*node, coexist_partitions);
    // update frontier_nodes and rejected_frontier_nodes
    UpdateFrontiers(part_meta, *node);
    // update rejected's predecessor partitions
    for (auto& id : immedidate_rejected_partitions) {
      UpdatePredecessors(part_meta, id);
    }
  } else if (coexist_partitions.size() == 1) {
    // found a unique partition
    PartitionMeta& part_meta = partitions_[coexist_partitions[0]];
    // make this new node to this partition
    part_meta.nodes.push_back(node_idx);
    // update cost
    part_meta.cost = Cost(*node, coexist_partitions);
    // update frontier_nodes and rejected_frontier_nodes
    UpdateFrontiers(part_meta, *node);
    // update rejected's predecessor partitions
    for (auto& id : immedidate_rejected_partitions) {
      UpdatePredecessors(part_meta, id);
    }
  } else {
    // if there are still multiple coexist partitions
    // we can fuse all coexist partitions together
    // we fuse all partitions to the first partition.
    MergePartitions(*node, coexist_partitions, immedidate_rejected_partitions);
  }
}

void Partitioner::RejectNode(
    const onnxruntime::GraphViewer& graph,
    const NodeIndex& node_idx) {
  const Node* node = graph.GetNode(node_idx);

  for (auto& p : partitions_) {
    bool is_child = p.second.frontier_nodes.count(node_idx) > 0;
    bool is_rejected = p.second.rejected_frontier_nodes.count(node_idx) > 0;
    if (is_child || is_rejected) {
      for (auto it = node->OutputEdgesBegin(); it != node->OutputEdgesEnd(); ++it) {
        const Node& dst_node = it->GetNode();
        if (p.second.rejected_frontier_nodes.count(dst_node.Index()) == 0) {
          p.second.rejected_frontier_nodes.insert(dst_node.Index());
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
Status Partitioner::Evaluate(const onnxruntime::GraphViewer& graph, bool distinguish_subgraph) {
  if (graph.IsSubgraph() && distinguish_subgraph) {
    HandleSubgraph(graph);
    return Status::OK();
  }

  for (auto& node_idx : graph.GetNodesInTopologicalOrder()) {
    const Node* node = graph.GetNode(node_idx);
    if (IsNodeSupported(*node)) {
      AcceptNode(graph, node_idx);
    } else {
      RejectNode(graph, node_idx);
    }
  }

  return Status::OK();
}
}  // namespace nuphar
}  // namespace onnxruntime
