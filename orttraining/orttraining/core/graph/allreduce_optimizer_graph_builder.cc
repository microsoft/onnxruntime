// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/graph/allreduce_optimizer_graph_builder.h"

#include "orttraining/core/framework/distributed_run_context.h"

namespace onnxruntime {
namespace training {

static bool IsNcclAvailable() {
#ifdef ORT_USE_NCCL
  return true;
#else
  return false;
#endif
}

static Status AddNcclAllReduceForGradients(
    std::vector<ArgDef>& gradient_argdefs,
    std::vector<ArgDef>& input_gradient_argdef,
    GraphAugmenter::GraphDefs& graph_defs) {
  std::vector<ArgDef> allreduce_outputs(gradient_argdefs.size());
  for (size_t i = 0; i < gradient_argdefs.size(); i++) {
    TypeProto* allreduced_gradient_type_proto = graph_defs.CopyTypeProto(gradient_argdefs[i]);
    allreduced_gradient_type_proto->mutable_tensor_type()->set_elem_type(
        input_gradient_argdef[0].type_proto->tensor_type().elem_type());

    allreduce_outputs[i] = ArgDef(gradient_argdefs[i].name + "_AllReduce_Out", allreduced_gradient_type_proto);
  }

  // Add NCCL Allreduce node.
  graph_defs.AddNodeDefs({NodeDef(OpDef{"NcclAllReduce", kMSDomain, 1},
                                  input_gradient_argdef,
                                  allreduce_outputs,
                                  {ONNX_NAMESPACE::MakeAttribute("group_type",
                                                                 static_cast<int64_t>(WorkerGroupType::DataParallel))},
                                  "NcclAllReduce")});

  gradient_argdefs = allreduce_outputs;
  return Status::OK();
}

AllreduceOptimizerGraphBuilder::AllreduceOptimizerGraphBuilder(
    const OptimizerBuilderRegistry& opt_builder_registry,
    const OptimizerGraphConfig& opt_graph_config,
    const std::unordered_map<std::string, OptimizerNodeConfig>& weight_names_to_opt_configs,
    std::unordered_map<std::string, std::string>& updated_weight_names_map,
    std::unordered_map<std::string, TrainingSession::PartitionInfo>& weight_partition_info)
    : OptimizerGraphBuilder(opt_builder_registry,
                            opt_graph_config,
                            weight_names_to_opt_configs,
                            updated_weight_names_map,
                            weight_partition_info) {
  ORT_ENFORCE(opt_graph_config.data_parallel_group_size > 1,
              "Allreduce optimizer graph builder can only be used for distributed training.");
  if (opt_graph_config.use_nccl) {
    ORT_ENFORCE(IsNcclAvailable(), "Distributed training with NCCL is not supported, as NCCL is not enabled in this build.");
  } else if (!opt_graph_config.use_nccl && opt_graph_config.adasum_reduction_type == AdasumReductionType::None) {
    ORT_THROW("Performing Allreduce is only supported using NCCL.");
  }
}

Status AllreduceOptimizerGraphBuilder::BuildInternal(
    bool should_add_gradient_norm,
    bool should_add_gradient_finite_check,
    Graph& graph,
    GraphAugmenter::GraphDefs& graph_defs,
    std::vector<ArgDef>& weight_argdefs,
    std::vector<ArgDef>& gradient_argdefs,
    std::unordered_map<std::string, std::unordered_map<std::string, std::string>>& weight_to_opt_mapping,
    OptimizerOutputKeyMap<std::string>& optimizer_graph_outputs) {
  auto nodearg_name_generator = [&graph](const std::string& base_name) {
    return graph.GenerateNodeArgName(base_name);
  };

  // add gradient scaling
  std::vector<ArgDef> output_gradient_argdef;
  const auto total_num_accumulations =
      opt_graph_config_.gradient_accumulation_steps * opt_graph_config_.data_parallel_group_size;
  ORT_RETURN_IF_NOT(total_num_accumulations > 0, "total_num_accumulations <= 0");
  const float scale = 1.0f / total_num_accumulations;
  ORT_RETURN_IF_ERROR(AddGradientScalingNodes(nodearg_name_generator, scale, gradient_argdefs, output_gradient_argdef, graph_defs,
                                              opt_graph_config_.AllReduceDataType()));

  ORT_RETURN_IF_ERROR(AddNcclAllReduceForGradients(gradient_argdefs, output_gradient_argdef, graph_defs));

  // check if all gradients are finite
  ArgDef global_grad_norm_argdef;
  ArgDef global_grad_norm_finite_argdef;

  if (should_add_gradient_norm) {
    ORT_RETURN_IF_ERROR(AddGradientNorm(
        nodearg_name_generator, gradient_argdefs, graph_defs, global_grad_norm_argdef));
    optimizer_graph_outputs[OptimizerOutputKey::GlobalGradientNorm] = global_grad_norm_argdef.name;
  }

  if (should_add_gradient_finite_check) {
    ORT_RETURN_IF_ERROR(AddFiniteGradientCheck(
        nodearg_name_generator, {global_grad_norm_argdef}, graph_defs, global_grad_norm_finite_argdef));
    optimizer_graph_outputs[OptimizerOutputKey::GradientAllIsFinite] = global_grad_norm_finite_argdef.name;
  }

  // add weight update
  ORT_RETURN_IF_ERROR(AddDirectWeightUpdate(
      opt_builder_registry_, weight_argdefs, gradient_argdefs,
      &global_grad_norm_argdef,
      &global_grad_norm_finite_argdef,
      opt_configs_, graph_defs,
      weight_to_opt_mapping));

  return Status::OK();
}

}  // namespace training
}  // namespace onnxruntime
