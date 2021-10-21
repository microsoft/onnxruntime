// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/partition/partitioner.h"

#include "core/codegen/common/common.h"

namespace onnxruntime {
namespace nuphar {

static bool CheckConnection(const NodeArg* node_arg, const std::unordered_set<std::string>& frontier_node_args) {
  if (node_arg->Exists())
    return frontier_node_args.count(node_arg->Name()) > 0;
  return false;
}

void Partitioner::UpdateFrontiers(PartitionMeta& part_meta, const Node& node) {
  // update frontier_node_args
  for (const NodeArg* output_def : node.OutputDefs()) {
    if (output_def->Exists()) {
      part_meta.frontier_node_args.insert(output_def->Name());
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
                                  const std::vector<NodeIndex>& candidates,
                                  const std::vector<NodeIndex>& rejected_partitions) {
  std::unordered_set<NodeIndex> merged_partitions;
  PartitionMeta& part_meta = partitions_[candidates[0]];
  // update cost
  part_meta.cost = Cost(node, candidates);

  // merge the rest meta
  for (size_t i = 1; i < candidates.size(); ++i) {
    PartitionMeta& other_part_meta = partitions_[candidates[i]];
    // record merged_partitions
    merged_partitions.insert(other_part_meta.Id());

    // merge nodes
    for (auto& n : other_part_meta.nodes) {
      part_meta.nodes.push_back(n);
    }

    // merge rejected_frontier_nodes
    for (auto& n : other_part_meta.rejected_frontiner_node_args) {
      part_meta.rejected_frontiner_node_args.insert(n);
    }

    // merge frontier_nodes
    for (auto& n : other_part_meta.frontier_node_args) {
      part_meta.frontier_node_args.insert(n);
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
  // by replacing merged_partitions to candidates[0]
  for (const auto& mp : merged_partitions) {
    for (auto& iter : partitions_) {
      // replace predecessor_partitions
      if (iter.second.predecessor_partitions.count(mp) > 0) {
        iter.second.predecessor_partitions.erase(mp);
        iter.second.predecessor_partitions.insert(candidates[0]);
      }

      // replace predecessor_partitions
      if (iter.second.immediate_predecessor_partitions.count(mp) > 0) {
        iter.second.immediate_predecessor_partitions.erase(mp);
        iter.second.immediate_predecessor_partitions.insert(candidates[0]);
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

  const Node* node = graph.GetNode(node_idx);

  for (const auto& p : partitions_) {
    bool is_rejected = false;
    for (const NodeArg* input_def : node->InputDefs()) {
      is_rejected = CheckConnection(input_def, p.second.rejected_frontiner_node_args);
      if (is_rejected) {
        break;
      }
    }

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
    bool is_child = false;
    for (const NodeArg* input_def : node->InputDefs()) {
      is_child = CheckConnection(input_def, p.second.frontier_node_args);
      if (is_child) {
        break;
      }
    }

    if (is_child &&
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

  if (ForcePartition(graph, *node, coexist_partitions, immedidate_rejected_partitions)) {
    return;
  }

  if (coexist_partitions.size() == 0) {
    CreateNewPartition(*node, immedidate_rejected_partitions);
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

void Partitioner::CreateNewPartition(
    const Node& node,
    const std::vector<NodeIndex>& immedidate_rejected_partitions) {
  const NodeIndex node_idx = node.Index();

  partitions_.insert(std::make_pair(node_idx, PartitionMeta(node_idx)));
  PartitionMeta& part_meta = partitions_[node_idx];
  // update cost
  part_meta.cost = Cost(node, {});
  // update frontier_nodes and rejected_frontier_nodes
  UpdateFrontiers(part_meta, node);
  // update rejected's predecessor partitions
  for (auto& id : immedidate_rejected_partitions) {
    UpdatePredecessors(part_meta, id);
  }
}

void Partitioner::RejectNode(
    const onnxruntime::GraphViewer& graph,
    const NodeIndex& node_idx) {
  const Node* node = graph.GetNode(node_idx);

  // if a node (A) is not supported.
  // its child (B) will be also in rejected_frontier_nodes of a partition (P) which holds the node (A)
  for (auto& p : partitions_) {
    bool is_child = false;
    bool is_rejected = false;
    for (const NodeArg* input_def : node->InputDefs()) {
      is_child = is_child || CheckConnection(input_def, p.second.frontier_node_args);
      is_rejected = is_rejected || CheckConnection(input_def, p.second.rejected_frontiner_node_args);
    }

    if (is_child || is_rejected) {
      for (const NodeArg* output_def : node->OutputDefs()) {
        if (output_def->Exists()) {
          const std::string& output_def_name = output_def->Name();
          if (p.second.rejected_frontiner_node_args.count(output_def_name) == 0) {
            p.second.rejected_frontiner_node_args.insert(output_def_name);
          }
        }
      }
    }
  }
}

// Partition the graph based on the dependency and whether ops are supported.
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
      LOGS_DEFAULT(INFO) << "unsupported node (" << node->Name() << ") in nuphar provider";
      RejectNode(graph, node_idx);
    }
  }

  return Status::OK();
}
}  // namespace nuphar
}  // namespace onnxruntime
