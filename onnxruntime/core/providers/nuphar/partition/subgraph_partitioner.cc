// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/partition/subgraph_partitioner.h"

#include "core/codegen/common/common.h"
#include "core/common/logging/logging.h"
#include "core/providers/nuphar/common/analysis/subgraph_partition_stats.h"
#include "core/providers/nuphar/common/nuphar_settings.h"

#include "core/framework/op_kernel.h"
#include "core/framework/tensorprotoutils.h"

#include <algorithm>

namespace onnxruntime {
namespace nuphar {

// A paramter for COST_UPPER_BOUND.
// This is for mimicing the criteria of the old greedy algorithm.
// TODO remove this after rewriting Cost function and ForcePartition
constexpr int COST_UPPER_BOUND = 180000;

// Here, we implement a partitioner using the new merge algorithm mimicing the criteria of the old greedy algorithm.
// It will generate the SAME subgraph partition as the old greedy algorithm did.
// This might change soon after the interface becomes stable.
// TODO: change the Cost and ForcePartition to a more complex form.

// Here we use NodeUseCount as Cost to meet the criteria of the old greedy algorithm.
// Note Cost function can use function, E.g. weight size or L2 pressure.
// TODO replace NodeUseCount approximation
int SubgraphPartitioner::Cost(const Node& node) const {
  return Promote<SubgraphPartitionStats>(graph_stats_)->NodeUseCount(&node);
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
bool SubgraphPartitioner::IsNodeSupported(const Node&) const {
  return true;
}

void SubgraphPartitioner::SetSpecifiedNodeNames(const std::vector<std::string>& specified_names) {
  for (const auto& name : specified_names) {
    specified_names_.insert(name);
  }
}

bool SubgraphPartitioner::SpecifiedNodePartition(const Node& node,
                                                 const std::vector<NodeIndex>& candidates,
                                                 const std::vector<NodeIndex>& rejected_partitions) {
  if (specified_names_.count(node.Name()) > 0) {
    // Here the old algorithm-equalivent. Merge two partitions
    MergePartitions(node, candidates, rejected_partitions);

    PartitionMeta& part_meta = partitions_[candidates[0]];

    // all children of node become current partition's rejected_nodes
    // to avoid any child be merged with current partition

    for (const NodeArg* output_arg : node.OutputDefs()) {
      if (output_arg->Exists()) {
        part_meta.rejected_frontiner_node_args.insert(output_arg->Name());
      }
    }

    return true;
  }

  return false;
}

static void RecordScanStates(
    const Node& node,
    const NodeIndex node_idx,
    std::map<std::string, NodeIndex>& no_merged_args_to_nodes) {
  size_t num_variadic_inputs = GetSubgraph(node)->GetInputs().size();
  int64_t num_scan_inputs;
  ProtoHelperNodeContext ctx(node);
  OpNodeProtoHelper<ProtoHelperNodeContext> attrs(&ctx);
  ORT_ENFORCE(attrs.GetAttr<int64_t>("num_scan_inputs", &num_scan_inputs).IsOK());
  size_t num_state_variables = num_variadic_inputs - gsl::narrow_cast<size_t>(num_scan_inputs);

  for (size_t idx = 0; idx < num_state_variables; ++idx) {
    const NodeArg* def = node.InputDefs()[idx];
    no_merged_args_to_nodes.insert(std::make_pair(def->Name(), node_idx));
  }
}

// ForcePartition implements the logic equalivent to the old greedy algorithm using a merging algorithm.
// It first check whether it is a Scan node. If so, make it a single partition.
// If not, check whether estimated cost is larger than an upperbound And current UseCount >= 2.
// If so, merge candidates with the node, and then force a partition for the merged partitions.
// If not, go the default process.
bool SubgraphPartitioner::ForcePartition(
    const onnxruntime::GraphViewer& graph,
    const Node& node,
    const std::vector<NodeIndex>& candidates,
    const std::vector<NodeIndex>& immedidate_rejected_partitions) {
  const NodeIndex node_idx = node.Index();

  if (IsRecurrentNode(node) ||
      node.OpType() == "Concat") {
    // a new partition
    CreateNewPartition(node, immedidate_rejected_partitions);
    PartitionMeta& part_meta = partitions_[node_idx];

    // update candidate's predecessor partitions, all candidates become its dominators
    for (auto& id : candidates) {
      UpdatePredecessors(part_meta, id);
    }

    // all children of node become current partition's rejected_frontier_nodes
    // to avoid any child be merged with the current partition in the future
    for (const NodeArg* output_arg : node.OutputDefs()) {
      if (output_arg->Exists()) {
        part_meta.rejected_frontiner_node_args.insert(output_arg->Name());
      }
    }

    // record Scan's states
    if (node.OpType() == "Scan") {
      RecordScanStates(node, node_idx, no_merged_args_to_nodes_);
    }

    return true;
  }

  // add specified node support
  if (SpecifiedNodePartition(node, candidates, immedidate_rejected_partitions)) {
    return true;
  }

  if (candidates.empty()) {
    return false;
  }

  // estimated the cost < COST_UPPER_BOUND
  if (Cost(node, candidates) < COST_UPPER_BOUND) {
    return false;
  }

  int use_cnt = Promote<SubgraphPartitionStats>(graph_stats_)->NodeUseCount(&node);

  if (use_cnt >= 2) {
    // Here the old algorithm-equalivent. Merge two partitions
    MergePartitions(node, candidates, immedidate_rejected_partitions);

    PartitionMeta& part_meta = partitions_[candidates[0]];

    // all children of node become current partition's rejected_frontier_nodes
    // to avoid any child be merged with current partition
    for (const NodeArg* output_arg : node.OutputDefs()) {
      if (output_arg->Exists()) {
        part_meta.rejected_frontiner_node_args.insert(output_arg->Name());
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
    FindInitializerFunc find_initializer_func) {
  const Graph* onnx_subgraph = GetSubgraph(node);

  // Handle single node
  if (nullptr == onnx_subgraph) {
    NupharSubgraphUnit subgraph;
    // set node
    subgraph.nodes.push_back(&node);

    node.ForEachWithIndex(
        node.InputDefs(),
        [&](const NodeArg& def, size_t i) {
          const Tensor* t = find_initializer_func(def.Name());
          bool unused_initializer = false;
          if (t != nullptr) {
            // note for Reshape and Tile, shape/repeats as initializer is not used at runtime
            // neither for any scalar
            unused_initializer = ((node.OpType() == "Reshape" || node.OpType() == "Tile") && i == 1) ||
                                 t->Shape().Size() == 1;

            if (!unused_initializer) {
              subgraph.initializers.emplace(def.Name(), t);
            }
          }
          // set real inputs
          if (!unused_initializer) {
            subgraph.inputs.push_back(&def);
          }
          return Status::OK();
        });

    // set real outputs
    for (const auto def : node.OutputDefs()) {
      subgraph.outputs.push_back(def);
    }

    // push back
    results.push_back(subgraph);
    return Status::OK();
  }

  ///////////////////////////////////
  // The rest code handles a subgraph
  ///////////////////////////////////
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
  graph_stats_ = std::make_unique<SubgraphPartitionStats>();
  Promote<SubgraphPartitionStats>(graph_stats_)->SetShapeInference(whole_partition_shape_infer);
  graph_stats_->Evaluate(graph_viewer);

  // perform partition
  ORT_RETURN_IF_ERROR(Evaluate(graph_viewer, false));

  // A group topology sort using predecessor set
  bool sorted = true;
  while (sorted) {
    sorted = false;
    for (const auto& iter : partitions_) {
      if (std::find(sorted_partitions_.begin(), sorted_partitions_.end(), iter.first) != sorted_partitions_.end())
        continue;  // already sorted, skip

      const auto& predecessor = iter.second.predecessor_partitions;
      std::vector<int> result;
      auto count_predecessor_not_sorted =
          std::count_if(predecessor.begin(),
                        predecessor.end(),
                        [this](NodeIndex idx) {
                          return sorted_partitions_.end() ==
                                 std::find(sorted_partitions_.begin(), sorted_partitions_.end(), idx);
                        });
      if (0 == count_predecessor_not_sorted) {
        // all predecessors are sorted, add it to sorted
        sorted_partitions_.push_back(iter.first);
        sorted = true;
        break;
      }
    }
  }

  ORT_ENFORCE(sorted_partitions_.size() == partitions_.size());

  // create results
  for (const auto& partition : sorted_partitions_) {
    const PartitionMeta& meta = partitions_.at(partition);

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
          [&](const onnxruntime::NodeArg& def, size_t) {
            const onnxruntime::Node* input_node = GetInputNode(*n, &def);
            bool input_from_subgraph = (nullptr != input_node && node_indices.count(input_node->Index()) > 0);
            const Tensor* t = find_initializer_func(def.Name());

            if (!input_from_subgraph && t == nullptr) {
              // input is from weights or outside of graph
              subgraph.inputs.push_back(&def);
            }

            if (t != nullptr) {
              subgraph.initializers.emplace(def.Name(), t);

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
              [&](const onnxruntime::NodeArg& def, size_t) {
                const Tensor* t = find_initializer_func(def.Name());
                if (t != nullptr) {
                  subgraph.initializers.emplace(def.Name(), t);

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

    if (codegen::CodeGenSettings::Instance().HasOption(kNupharDumpFusedNodes)) {
      std::ostringstream stream;
      stream << "[NUPHAR_DUMP_FUSED_NODES] ID " << subgraph.UniqueId() << std::endl;
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
