// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/partition/subgraph_partitioner.h"

#include "core/codegen/common/common.h"
#include "core/common/logging/logging.h"
#include "core/providers/nuphar/common/analysis/subgraph_partition_stats.h"
#include "core/providers/nuphar/common/nuphar_settings.h"

#include <algorithm>

namespace onnxruntime {
namespace nuphar {

// A paramter for COST_UPPER_BOUND.
// This is for mimicing the criteria of the old greedy algorithm.
// TODO remove this after rewriting Cost function and ForcePartition
constexpr int COST_UPPER_BOUND = 180;

// Here, we implement a partitioner using the new merge algorithm mimicing the criteria of the old greedy algorithm.
// It will generate the SAME subgraph partition as the old greedy algorithm did.
// This might change soon after the interface becomes stable.
// TODO: change the Cost and ForcePartition to a more complex form.

// Here we use NodeUseCount as Cost to meet the criteria of the old greedy algorithm.
// Note Cost function can use function, E.g. weight size or L2 pressure.
int SubgraphPartitioner::Cost(const Node& node) const {
  return codegen::Promote<codegen::SubgraphPartitionStats>(graph_stats_)->NodeUseCount(&node);
}

// Here we use linear summation for a Partition cost.
int SubgraphPartitioner::Cost(const Node& node, const std::vector<NodeIndex>& candidates) const {
  int cost = Cost(node);
  for (auto n_id : candidates) {
    const PartitionMeta& part_meta_cand = partitions_.at(n_id);
    cost += part_meta_cand.cost;
  }
  return cost;
}

// Node is always supported in SubgraphPartitioner, so always return true
bool SubgraphPartitioner::IsNodeSupported(const Node&) {
  return true;
}

// ForcePartition implements the logic equalivent to the old greedy algorithm using a merging algorith.
// It first check whether it is a Scan node. If so, make it a single partition.
// If not, check whether estimated cost is larger than an upperbound And current UseCount >= 2.
// If so, merge candidates with the node, and then force a partition for the merged partitions.
// If not, go the default process.
bool SubgraphPartitioner::ForcePartition(
    const NodeIndex& node_idx,
    const int topology_idx,
    const Node& node,
    const std::vector<NodeIndex>& candidates,
    const std::vector<NodeIndex>& rejected_partitions) {
  if (IsRecurrentNode(node)) {
    // a new partition
    partitions_.insert(std::make_pair(node_idx, PartitionMeta(node_idx, topology_idx)));
    PartitionMeta& part_meta = partitions_[node_idx];
    // update cost
    part_meta.cost = Cost(node, candidates);
    // update frontier_nodes and rejected_nodes
    UpdateNodesInPartitionMeta(part_meta, node);

    // update rejected predomiate partitions, all candidates become its dominators
    for (auto& id : candidates) {
      part_meta.predominate_partitions.insert(id);
    }

    // update rejected partitions to predominate partitions
    // rejected partitions' predominate parititions also to predominate partitions
    for (auto& id : rejected_partitions) {
      part_meta.predominate_partitions.insert(id);
      for (auto& p : partitions_[id].predominate_partitions) {
        part_meta.predominate_partitions.insert(p);
      }
    }

    // all children of node become current partition's rejected_nodes
    // to avoid any child be merged with the current partition in the future
    for (auto it = node.OutputEdgesBegin(); it != node.OutputEdgesEnd(); ++it) {
      const Node& dst_node = it->GetNode();
      if (part_meta.rejected_nodes.count(dst_node.Index()) == 0) {
        part_meta.rejected_nodes.insert(dst_node.Index());
      }
    }

    return true;
  }

  // estimated the cost < COST_UPPER_BOUND
  if (Cost(node, candidates) < COST_UPPER_BOUND) {
    return false;
  }

  int use_cnt = codegen::Promote<codegen::SubgraphPartitionStats>(graph_stats_)->NodeUseCount(&node);

  if (use_cnt >= 2) {
    // Here the old algorithm-equalivent. Merge two partitions
    MergePartitions(node_idx, topology_idx, node, candidates, rejected_partitions);

    PartitionMeta& part_meta = partitions_[candidates[0]];

    // all children of node become current partition's rejected_nodes
    // to avoid any child be merged with current partition
    for (auto it = node.OutputEdgesBegin(); it != node.OutputEdgesEnd(); ++it) {
      const Node& dst_node = it->GetNode();
      if (part_meta.rejected_nodes.count(dst_node.Index()) == 0) {
        part_meta.rejected_nodes.insert(dst_node.Index());
      }
    }
    return true;
  }
  return false;
}

// Main interface for Partition
Status SubgraphPartitioner::Partition(
    const Node& node,
    std::vector<NupharSubgraphUnit>& results,
    const std::map<std::string, const Tensor*>& initializers) {
  const Graph* onnx_subgraph = GetSubgraph(node);

  // Handle single node
  if (nullptr == onnx_subgraph) {
    NupharSubgraphUnit subgraph;
    // set node
    subgraph.nodes.push_back(&node);

    for (const auto def : node.InputDefs()) {
      auto iter = initializers.find(def->Name());
      if (iter != initializers.end()) {
        // set initializers
        subgraph.initializers.emplace(iter->first, iter->second);
      }
      // set real inputs
      subgraph.inputs.push_back(def);
    }

    // set real outputs
    for (const auto def : node.OutputDefs()) {
      subgraph.outputs.push_back(def);
    }

    // push back
    results.push_back(subgraph);
    return Status::OK();
  }

  //////////////////////
  // The rest code handles a subgraph
  //////////////////////
  const onnxruntime::GraphViewer& graph_viewer = GraphViewer(*onnx_subgraph);
  std::unordered_set<std::string> real_output_names;
  node.ForEachWithIndex(
      node.OutputDefs(),
      [&real_output_names](const onnxruntime::NodeArg& def, size_t) {
        real_output_names.insert(def.Name());
        return Status::OK();
      });

  // shape infernece here
  std::shared_ptr<ShapeExprContext> whole_partition_shape_infer = std::make_shared<ShapeExprContext>();
  ORT_RETURN_IF_ERROR(ShapeInference(graph_viewer, *whole_partition_shape_infer));

  // construct graph stats
  graph_stats_ = std::make_unique<codegen::SubgraphPartitionStats>();
  codegen::Promote<codegen::SubgraphPartitionStats>(graph_stats_)->SetShapeInference(whole_partition_shape_infer);
  graph_stats_->Evaluate(graph_viewer);

  // perform partition
  ORT_RETURN_IF_ERROR(Evaluate(graph_viewer));

  // A simplified group topology sort using max_topology_idx within each group
  std::vector<std::pair<NodeIndex, int>> sorted_proxies;
  for (const auto& iter : partitions_) {
    sorted_proxies.push_back(std::make_pair(iter.first, iter.second.max_topology_index));
  }

  // call std::sort
  std::sort(sorted_proxies.begin(), sorted_proxies.end(),
            [](std::pair<NodeIndex, int> a, std::pair<NodeIndex, int> b) {
              return a.second < b.second;
            });

  // create results
  for (const auto& proxy : sorted_proxies) {
    const PartitionMeta& meta = partitions_.at(proxy.first);

    NupharSubgraphUnit subgraph;
    std::unordered_set<NodeIndex> node_indices;

    for (auto& n_idx : meta.nodes) {
      // set node
      const Node* n = graph_viewer.GetNode(n_idx);
      subgraph.nodes.push_back(n);
      node_indices.insert(n_idx);
    }

    for (auto& n_idx : meta.nodes) {
      const Node* n = graph_viewer.GetNode(n_idx);

      // handle current graph's inputs
      n->ForEachWithIndex(
          n->InputDefs(),
          [&subgraph, &n, &node_indices, &initializers](const onnxruntime::NodeArg& def, size_t) {
            const onnxruntime::Node* input_node = GetInputNode(*n, &def);
            bool input_from_subgraph = (nullptr != input_node && node_indices.count(input_node->Index()) > 0);
            auto ini_iter = initializers.find(def.Name());

            if (!input_from_subgraph && ini_iter == initializers.end()) {
              // input is from weights or outside of graph
              subgraph.inputs.push_back(&def);
            }

            if (ini_iter != initializers.end()) {
              subgraph.initializers.emplace(ini_iter->first, ini_iter->second);

              // a intializer is an input
              subgraph.inputs.push_back(&def);
            }

            return Status::OK();
          });

      // Handle outouts
      // three cases are considerd as outputs
      // 1. Output NodeArg is not used by any Node
      // 2. Output NodeArg is used by at least one Node out of this subgraph.
      //    Note a NodeArg can be used by Nodes in and out of the subgraph at the same time.
      // 3. Output NodeArg is one of real outputs of an Ort subgraph.
      //    Note if a NodeArg was the case 2 during Ort graph partition,
      //    that NodeArg will disapear in the Ort subgraph,
      //    and becomes only visible in FuseNode's real outputs.
      //    This is not an Ort bug. This is due to ONNX limitation that
      //    no NodeArg can be both internal and external at the same time.
      //    That is what the case 3 to handle.

      auto InsertOutputToSubgraph = [&subgraph](const NodeArg* def) {
        if (std::find(subgraph.outputs.begin(), subgraph.outputs.end(), def) ==
            subgraph.outputs.end()) {
          subgraph.outputs.push_back(def);
        }
      };

      std::unordered_set<std::string> input_names_from_the_output_node;

      for (auto o_iter = n->OutputEdgesBegin(); o_iter != n->OutputEdgesEnd(); ++o_iter) {
        const auto& p = *o_iter;
        const Node& out_node = p.GetNode();

        // preprocess for the case 1
        out_node.ForEachWithIndex(
            out_node.InputDefs(),
            [&input_names_from_the_output_node](const onnxruntime::NodeArg& in_def, size_t) {
              input_names_from_the_output_node.insert(in_def.Name());
              return Status::OK();
            });
        // handle the case 2
        if (node_indices.count(out_node.Index()) == 0) {
          const NodeArg* def = n->OutputDefs()[p.GetSrcArgIndex()];
          InsertOutputToSubgraph(def);
        }
      }

      // handle case 1 and 3
      n->ForEachWithIndex(
          n->OutputDefs(),
          [&](const onnxruntime::NodeArg& def, size_t) {
            if (input_names_from_the_output_node.count(def.Name()) == 0 ||
                real_output_names.count(def.Name()) > 0) {
              InsertOutputToSubgraph(&def);
            }

            return Status::OK();
          });
    }

    // Handle immediate nested subgraphs
    // Note we put all info from immediate nested subgraphs in the end
    for (auto& n_idx : meta.nodes) {
      const Node* n = graph_viewer.GetNode(n_idx);
      auto immediate_nested_subgraph = GetSubgraph(*n);
      if (nullptr != immediate_nested_subgraph) {
        for (auto& nn : immediate_nested_subgraph->Nodes()) {
          nn.ForEachWithIndex(
              nn.InputDefs(),
              [&subgraph, &initializers](const onnxruntime::NodeArg& def, size_t) {
                auto ini_iter = initializers.find(def.Name());
                if (ini_iter != initializers.end()) {
                  subgraph.initializers.emplace(ini_iter->first, ini_iter->second);

                  // an intializer is an input
                  subgraph.inputs.push_back(&def);
                }
                return Status::OK();
              });
        }
      }
    }

    // push back
    results.push_back(subgraph);

    if (codegen::CodeGenSettings::Instance().HasOption(nuphar_codegen::kNupharDumpFusedNodes)) {
      std::ostringstream stream;
      stream << "[NUPHAR_DUMP_FUSED_NODES]" << std::endl;
      stream << "NupharSubgraphUnit of size " << results.back().nodes.size() << " [";
      for (const auto& n : results.back().nodes) {
        stream << "(" << n->Name() << ", " << n->OpType() << ") ";
      }
      stream << "]";

      LOGS_DEFAULT(CODEGEN_SETTINGS_LOG_LEVEL) << stream.str();
    }
  }

  return Status::OK();
}

}  // namespace nuphar
}  // namespace onnxruntime
