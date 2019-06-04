// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/partition/graph_partitioner.h"

#include "core/codegen/common/common.h"
#include "core/common/logging/logging.h"
#include "core/providers/nuphar/common/analysis/subgraph_partition_stats.h"
#include "core/providers/nuphar/common/nuphar_settings.h"
#include "core/providers/nuphar/common/utils.h"

namespace onnxruntime {
namespace nuphar {

bool GraphPartitioner::IsNodeSupported(const Node& node) {
  auto subgraph = GetSubgraph(node);
  if (nullptr != subgraph) {
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

  // currently, our tvm runtime has some issue for inferring the output shape
  // of Concat if its input and output shapes are symbolic or unknown
  if (node.OpType() == "Concat") {
    const onnxruntime::NodeAttributes& attrs = node.GetAttributes();
    auto it = attrs.find("axis");
    ORT_ENFORCE(it != attrs.end());
    int64_t axis = it->second.i();

    if (nuphar_codegen::HasUnknownShapeOnAxis(node.InputDefs(), axis) &&
        nuphar_codegen::HasUnknownShapeOnAxis(node.OutputDefs(), axis)) {
      for (auto iter = node.OutputNodesBegin(); iter != node.OutputNodesEnd(); ++iter) {
        unsupported_nodes_.insert(GetKey(*iter));
      }
      return false;
    }
  }

  if (unsupported_nodes_.count(GetKey(node)) > 0) {
    return false;
  }
  // check single node
  return is_op_type_supported_func_(node);
}

// FORCE_ONE_SUBGRAPH is a marco to generate a single subgraph partition
// It is mainly for debug and reproducing older version
#ifdef FORCE_ONE_SUBGRAPH
bool GraphPartitioner::ForcePartition(
    const NodeIndex& node_idx, const int topology_idx,
    const Node& node, const std::vector<NodeIndex>& candiates,
    const std::vector<NodeIndex>& rejected_partitions) {
  if (IsRecurrentNode(node)) {
    // a new partition
    partitions_.insert(std::make_pair(node_idx, PartitionMeta(node_idx, topology_idx)));
    PartitionMeta& part_meta = partitions_[node_idx];
    // update cost
    part_meta.cost = Cost(node, candiates);
    // update frontier_nodes and rejected_nodes
    UpdateNodesInPartitionMeta(part_meta, node);

    // update rejected predomiate partitions, all candidates become its dominators
    for (auto& id : candiates) {
      part_meta.predominate_partitions.insert(id);
    }

    // update predomiate partitions, all rejected partitions become its dominators
    for (auto& id : rejected_partitions) {
      part_meta.predominate_partitions.insert(id);
    }

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
#endif

// Partition the graph (fusing ops) based on the dependency and whether ops are supported:
Status GraphPartitioner::Partition(const onnxruntime::GraphViewer& graph,
                                   std::vector<std::unique_ptr<ComputeCapability>>& result) {
  // call partition
  ORT_RETURN_IF_ERROR(Evaluate(graph));

  // remove single alias node (aka isolated alias op)
  std::vector<NodeIndex> erase_partitions;
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

    if (codegen::CodeGenSettings::Instance().HasOption(nuphar_codegen::kNupharDumpPartition)) {
      std::ostringstream stream;
      if (graph.IsSubgraph()) {
        stream << "[NUPHAR_DUMP_PARTITION] ## Subgraph ## " << std::endl;
      } else {
        stream << "[NUPHAR_DUMP_PARTITION]" << std::endl;
      }
      stream << "Partition of size " << iter.second.nodes.size() << " [";
      for (const auto& node_index : partition->nodes) {
        const Node* node = graph.GetNode(node_index);
        stream << "(" << node->Name() << ", " << node->OpType() << ") ";
      }
      stream << "]";
      LOGS_DEFAULT(CODEGEN_SETTINGS_LOG_LEVEL) << stream.str();
    }

    result.emplace_back(
        ToCapacity(
            graph,
            partition));
  }

  return Status::OK();
}

}  // namespace nuphar
}  // namespace onnxruntime
