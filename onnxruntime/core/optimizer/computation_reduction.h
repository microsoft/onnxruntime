// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"
#include "core/optimizer/utils.h"
namespace onnxruntime {

struct GatherInfo {
  GatherInfo() = default;

  GatherInfo(int gather_axis, int is_slice_scalar, Node* gather_node)
      : axis(gather_axis), is_slice_scalar(is_slice_scalar), gather_node(gather_node) {
    const NodeArg* input = gather_node->InputDefs()[0];
    const NodeArg* output = gather_node->OutputDefs()[0];
    input_dim_on_axis = input->Shape()->dim(axis);
    if (!is_slice_scalar) {
      output_dim_on_axis = output->Shape()->dim(axis);
    }

    input_rank = input->Shape()->dim_size();
    output_rank = output->Shape()->dim_size();
  }

  int axis;
  bool is_slice_scalar;
  Node* gather_node;

  int input_rank;
  int output_rank;

  ONNX_NAMESPACE::TensorShapeProto_Dimension input_dim_on_axis;
  ONNX_NAMESPACE::TensorShapeProto_Dimension output_dim_on_axis;  // Only useful when is_slice_scalar is false.
};

typedef std::function<bool(const Graph& graph, const Node& target_node, const GatherInfo& info,
                           std::unordered_map<int, int>& target_node_input_indices,
                           std::vector<int>& input_dices, bool& /*input_has_dim_1_for_axis*/,
                           const logging::Logger& logger)>
    PreCheckFunctionType;

typedef std::function<bool(Graph& graph, Node& target_node, const GatherInfo& info,
                           const std::unordered_map<int, GatherInfo>& new_gather_infos, int target_output_index,
                           bool input_has_dim_1_for_axis,
                           const logging::Logger& logger)>
    PostProcessFunctionType;

struct OpPassThroughConfig {
  OpPassThroughConfig() = default;

  // OpPassThroughConfig(const std::vector<int>& input_indices, PreCheckFunctionType pre_check_func)
  //     : input_indices(input_indices), pre_check_func(pre_check_func) {}

  OpPassThroughConfig(const std::vector<int>& input_indices, PreCheckFunctionType pre_check_func,
                      PostProcessFunctionType post_process_func)
      : input_indices(input_indices),
        pre_check_func(pre_check_func),
        post_process_func(post_process_func) {}

  std::vector<int> input_indices;
  PreCheckFunctionType pre_check_func;
  PostProcessFunctionType post_process_func;
};

struct ReorderHandle {
  ReorderHandle(const std::string& node_name) : entry_node_name(node_name) {
    RegisterOperators();
  }

  bool operator()(Graph& graph, Node& gathernd_node, Node& target_node, GatherInfo& info,
                  const logging::Logger& logger, std::deque<GatherInfo>& queue) {
    const std::string& op_type = target_node.OpType();
    if (AllowedPassThroughOps.count(op_type)) {
      std::vector<GatherInfo> duplicated_gather_nodes;
      auto ret = CommonHandle(graph, gathernd_node, target_node, info, logger, AllowedPassThroughOps[op_type], duplicated_gather_nodes);
      queue.insert(queue.end(), duplicated_gather_nodes.begin(), duplicated_gather_nodes.end());
      return ret;
    } else {
      LOGS(logger, WARNING) << "op_type not supported for " << target_node.Name() << "(" << target_node.OpType() << ")";
      return false;
    }
  }

 private:
  void RegisterOperators();

  bool CommonHandle(Graph& graph, Node& gathernd_node, Node& target_node, GatherInfo& info,
                    const logging::Logger& logger, OpPassThroughConfig& config,
                    std::vector<GatherInfo>& added_nodes) {
    LOGS(logger, WARNING) << "Enter CommonHandle for tartget node " << target_node.Name() << "("
                          << target_node.OpType() << ")"
                          << "config.pre_check_func " << !(config.pre_check_func)
                          << "config.post_process_func " << !(config.post_process_func);
    auto target_output_shape = gathernd_node.InputDefs()[0]->Shape();
    if (!target_output_shape) {
      LOGS(logger, WARNING) << "Gather input node arg " << gathernd_node.InputDefs()[0]->Name() << " doesn't have shape info";
      return false;
    }

    std::unordered_map<int, int> target_node_input_indices;
    bool input_has_dim_1_for_axis = false;
    if (!config.pre_check_func(graph, target_node, info,
                               target_node_input_indices, config.input_indices, input_has_dim_1_for_axis, logger)) {
      LOGS(logger, WARNING) << "Pre-check failed for " << target_node.Name() << "(" << target_node.OpType() << ")";
      return false;
    }

    if (target_node_input_indices.empty()) {
      LOGS(logger, WARNING) << "Skip handling target node " << target_node.Name() << "(" << target_node.OpType()
                            << ") because the requirement is not met.";
      return false;
    }

    added_nodes.clear();
    std::unordered_map<int, GatherInfo> new_gather_infos;
    for (auto pair : target_node_input_indices) {
      auto target_node_input_index = pair.first;
      int new_axis = pair.second;
      GatherInfo gather_info = DuplicatedGatherNodeForOneInput(graph, &gathernd_node, &target_node, target_node_input_index,
                                                               info, logger, new_axis);

      ORT_ENFORCE(gather_info.gather_node, "New added gather node should not be null.");
      added_nodes.push_back(gather_info);
      new_gather_infos[target_node_input_index] = gather_info;
    }

    int target_node_output_index = optimizer_utils::IndexOfNodeOutput(target_node, *gathernd_node.InputDefs()[0]);

    ORT_ENFORCE(RemoveGatherNodeAndUpdateTargetNode(graph, gathernd_node, target_node, logger, info).IsOK());

    if (config.post_process_func) {
      LOGS(logger, WARNING) << "CommonHandle PostProcessFunc stage for tartget node " << target_node.Name() << "(" << target_node.OpType() << ")";
      if (!config.post_process_func(graph, target_node, info, new_gather_infos, target_node_output_index, input_has_dim_1_for_axis, logger)) {
        ORT_THROW("Post-process failed for " + target_node.Name() + "(" + target_node.OpType() + ")");
      }
    }

    return true;
  }

  GatherInfo DuplicatedGatherNodeForOneInput(Graph& graph, Node* gather_node,
                                             Node* target_node,
                                             int target_node_input_index,
                                             GatherInfo& info,
                                             const logging::Logger& logger,
                                             int new_axis);

  Status RemoveGatherNodeAndUpdateTargetNode(Graph& graph, Node& gathernd_node, Node& target_node,
                                             const logging::Logger& logger, GatherInfo& info);

  std::string entry_node_name;

  std::unordered_map<std::string, OpPassThroughConfig> AllowedPassThroughOps;
};

class ComputationReductionTransformer : public GraphTransformer {
 public:
  ComputationReductionTransformer(const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("ComputationReductionTransformer", compatible_execution_providers) {}

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;

 private:
  std::optional<GatherInfo> IsSupportedGatherND(Graph& graph, Node& node, const logging::Logger& logger) const;
  std::optional<GatherInfo> IsSupportedGather(Graph& /*graph*/, Node& node, const logging::Logger& logger) const;
};

}  // namespace onnxruntime
