// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/partition/graph_partitioner.h"

#include "core/codegen/common/common.h"
#include "core/common/logging/logging.h"
#include "core/framework/tensorprotoutils.h"
#include "core/providers/nuphar/common/analysis/subgraph_partition_stats.h"
#include "core/providers/nuphar/common/nuphar_settings.h"
#include "core/providers/nuphar/common/utils.h"

namespace onnxruntime {
namespace nuphar {

bool GraphPartitioner::IsNodeSupported(const Node& node) const {
  const auto* subgraph = GetSubgraph(node);
  if (nullptr != subgraph) {
    // for control flow ops, only support the ones registered
    if (node.NodeType() != Node::Type::Fused && !is_op_type_supported_func_(node))
      return false;

    if (subgraph->NumberOfNodes() == 1) {
      // In Ort, subgraph is processed before main graph
      // We only need to detect whether a subgraph is already fused to One node.
      // And the first node in that fused node is also supported.
      const Node& fused_node = *subgraph->Nodes().begin();
      if (fused_node.NodeType() == Node::Type::Fused) {
        const Node& first_node = *fused_node.GetFunctionBody()->Body().Nodes().begin();
        return is_op_type_supported_func_(first_node);
      }
    }
    return false;
  }

  // check single node
  if (is_op_type_supported_func_(node)) {
    // currently, our tvm runtime has some issue for inferring the output shape
    // that's computed from input dimensions. Mark those nodes are not supported
    auto get_symbolic_dimensions = [](const Node& node, bool check_input) {
      std::unordered_set<std::string> symbolic_dimensions;
      node.ForEachDef([&](const NodeArg& def, bool is_input) {
        if (is_input == check_input && def.Shape() != nullptr) {
          for (const auto& dim : def.Shape()->dim()) {
            if (utils::HasDimParam(dim))
              symbolic_dimensions.insert(dim.dim_param());
            else
              ORT_ENFORCE(utils::HasDimValue(dim) && dim.dim_value() > 0);
          }
        }
      });
      return symbolic_dimensions;
    };
    // if there are any output symbols not in input symbols, fallback to CPU
    auto input_sym = get_symbolic_dimensions(node, true);
    auto output_sym = get_symbolic_dimensions(node, false);
    if (std::count_if(output_sym.begin(),
                      output_sym.end(),
                      [&input_sym](const std::string& name) {
                        return input_sym.count(name) == 0;
                      })) {
      return false;
    }
    return true;
  }

  return false;
}

void GraphPartitioner::HandleSubgraph(const onnxruntime::GraphViewer& graph) {
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

void GraphPartitioner::CreateNewPartition(
    const Node& node,
    const std::vector<NodeIndex>& immedidate_rejected_partitions) {
  Partitioner::CreateNewPartition(node, immedidate_rejected_partitions);
  const NodeIndex node_idx = node.Index();
  PartitionMeta& part_meta = partitions_[node_idx];
  // also add input
  for (const NodeArg* input_def : node.OutputDefs()) {
    if (input_def->Exists()) {
      part_meta.frontier_node_args.insert(input_def->Name());
    }
  }
}

// FORCE_ONE_SUBGRAPH is a marco to generate a single subgraph partition
// It is mainly for debug and reproducing older version
#ifdef FORCE_ONE_SUBGRAPH
bool GraphPartitioner::ForcePartition(
    const onnxruntime::GraphViewer& /*graph*/,
    const Node& node, const std::vector<NodeIndex>& candiates,
    const std::vector<NodeIndex>& immedidate_rejected_partitions) {
  const NodeIndex node_idx = node.Index();
  if (IsRecurrentNode(node)) {
    // a new partition
    partitions_.insert(std::make_pair(node_idx, PartitionMeta(node_idx, topology_idx)));
    PartitionMeta& part_meta = partitions_[node_idx];
    // update cost
    part_meta.cost = Cost(node, candiates);
    // update frontier_nodes and rejected_frontier_nodes
    UpdateFrontiers(part_meta, node);

    // update rejected predomiate partitions, all candidates become its dominators
    for (const auto& id : candiates) {
      part_meta.predecessor_partitions.insert(id);
    }

    // update predomiate partitions, all rejected partitions become its dominators
    for (const auto& id : immedidate_rejected_partitions) {
      UpdatePredecessors(part_meta, id);
    }

    // all children of node become current partition's rejected_frontier_nodes
    // to avoid any child be merged with current partition
    for (auto it = node.OutputEdgesBegin(); it != node.OutputEdgesEnd(); ++it) {
      const Node& dst_node = it->GetNode();
      if (part_meta.rejected_frontier_nodes.count(dst_node.Index()) == 0) {
        part_meta.rejected_frontier_nodes.insert(dst_node.Index());
      }
    }

    return true;
  }
  return false;
}
#endif

// Partition the graph (fusing ops) based on the dependency and whether ops are supported:
Status GraphPartitioner::Partition(const onnxruntime::GraphViewer& graph,
                                   int& fused_count,
                                   std::vector<std::unique_ptr<ComputeCapability>>& result) {
  // call partition
  ORT_RETURN_IF_ERROR(Evaluate(graph, /*distinguish_subgraph*/ true));

  std::vector<NodeIndex> erase_partitions;

  // remove single alias node (aka isolated alias op)
  // TODO: change this logic to removing a partition with only all alias ops
  for (const auto& iter : partitions_) {
    if (iter.second.nodes.size() == 1 &&
        IsAliasNode(*graph.GetNode(iter.second.nodes.front()))) {
      erase_partitions.push_back(iter.first);
    }
  }

  for (const auto& n_idx : erase_partitions) {
    partitions_.erase(n_idx);
  }

  // create results
  for (const auto& iter : partitions_) {
    std::unique_ptr<IndexedSubGraph> partition = std::make_unique<IndexedSubGraph>();

    for (auto& n : iter.second.nodes) {
      partition->nodes.push_back(n);
    }

    if (codegen::CodeGenSettings::Instance().HasOption(kNupharDumpPartition)) {
      std::ostringstream stream;
      if (graph.IsSubgraph()) {
        stream << "[NUPHAR_DUMP_PARTITION] ## Subgraph ## Fused graph ID " << fused_count << std::endl;
      } else {
        stream << "[NUPHAR_DUMP_PARTITION] ## Fused graph ID " << fused_count << std::endl;
      }
      stream << "Partition of size " << iter.second.nodes.size() << " [";
      for (const auto& node_index : partition->nodes) {
        const Node* node = graph.GetNode(node_index);
        stream << "(" << node->Name() << ", " << node->OpType() << ", " << node->Index() << ") ";
      }
      stream << "]";
      LOGS_DEFAULT(CODEGEN_SETTINGS_LOG_LEVEL) << stream.str();
    }

    result.emplace_back(
        ToCapacity(
            graph,
            fused_count++,
            partition));
  }

  return Status::OK();
}

}  // namespace nuphar
}  // namespace onnxruntime
