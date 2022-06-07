// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/graph/adasum_optimizer_graph_builder.h"
#include "orttraining/core/framework/distributed_run_context.h"

namespace onnxruntime {
namespace training {

ArgDef AdasumOptimizerGraphBuilder::BuildWeightUpdateNode(
    const NodeArgNameGeneratorFn& nodearg_name_generator,
    const ArgDef& gradient,
    ArgDef& weight,
    const ArgDef& gradient_finite_argdef,
    GraphAugmenter::GraphDefs& graph_defs) {
  TypeProto* gradient_fp32_type_proto = graph_defs.CopyTypeProto(weight);
  ArgDef weight_update_output = ArgDef(nodearg_name_generator(weight.name + "_update_out"), gradient_fp32_type_proto);
  graph_defs.AddNodeDefs({NodeDef(OpDef{"InPlaceAccumulator", kMSDomain, 1},
                                  {weight, gradient, gradient_finite_argdef},
                                  {weight_update_output},
                                  NodeAttributes(),
                                  weight_update_output.name)});
  return weight_update_output;
}

Status AdasumOptimizerGraphBuilder::AddWeightUpdateNodes(const NodeArgNameGeneratorFn& nodearg_name_generator,
                                                         std::vector<ArgDef>& gradient_argdefs,
                                                         std::vector<ArgDef>& weight_argdefs,
                                                         const ArgDef& adasum_gradient_finite_argdef,
                                                         GraphAugmenter::GraphDefs& graph_defs,
                                                         std::vector<ArgDef>& output_weight_argdefs) {
  output_weight_argdefs.clear();
  for (size_t i = 0; i < gradient_argdefs.size(); ++i) {
    output_weight_argdefs.push_back(BuildWeightUpdateNode(nodearg_name_generator,
                                                          gradient_argdefs[i],
                                                          weight_argdefs[i],
                                                          adasum_gradient_finite_argdef,
                                                          graph_defs));
  }
  weight_argdefs = std::move(output_weight_argdefs);
  return Status::OK();
}

AdasumOptimizerGraphBuilder::AdasumOptimizerGraphBuilder(
    const OptimizerBuilderRegistry& opt_builder_registry,
    const OptimizerGraphConfig& opt_graph_config,
    const std::unordered_map<std::string, OptimizerNodeConfig>& weight_names_to_opt_configs,
    std::unordered_map<std::string, std::string>& updated_weight_names_map,
    std::unordered_map<std::string, TrainingSession::PartitionInfo>& weight_partition_info)
    : AllreduceOptimizerGraphBuilder(opt_builder_registry,
                                     opt_graph_config,
                                     weight_names_to_opt_configs,
                                     updated_weight_names_map,
                                     weight_partition_info) {
  ORT_ENFORCE(opt_graph_config.data_parallel_group_size > 1,
              "Adasum optimizer graph builder can only be used for distributed training.");
}

Status AdasumOptimizerGraphBuilder::BuildOptimizerNode(
    const std::unique_ptr<OptimizerBuilder>& opt_builder,
    const std::vector<ArgDef>& weight_argdefs,
    const std::vector<ArgDef>& gradient_argdefs,
    const ArgDef* global_gradient_norm_argdef,
    const ArgDef* global_gradient_norm_finite_argdef,
    const std::vector<OptimizerNodeConfig>& opt_configs,
    GraphAugmenter::GraphDefs& graph_defs,
    std::vector<TensorProto>& new_initializers,
    std::unordered_map<std::string, std::unordered_map<std::string, std::string>>& weight_to_opt_mapping,
    std::vector<ArgDef>& output_weight_argdefs,
    std::vector<ArgDef>& output_gradient_argdefs) {
  OptimizerBuilderConfig config;
  config.weight_argdefs = weight_argdefs;
  config.gradient_argdefs = gradient_argdefs;
  if (global_gradient_norm_argdef != nullptr) {
    config.gradient_norm_argdef = *global_gradient_norm_argdef;
  }
  if (global_gradient_norm_finite_argdef != nullptr) {
    config.gradient_norm_finite_argdef = *global_gradient_norm_finite_argdef;
  }
  config.opt_configs = opt_configs;
  // Always enable grad clipping for Adasum
  config.enable_grad_clipping = true;
  config.shared_optimizer_states = opt_graph_config_.shared_optimizer_states;
  ORT_RETURN_IF_ERROR(opt_builder->Build(
      config, graph_defs,
      new_initializers,
      weight_to_opt_mapping,
      output_weight_argdefs, output_gradient_argdefs));

  return Status::OK();
}

#ifdef ORT_USE_NCCL
static Status AddNcclAllReduceForGradientsWithGroups(
    std::vector<ArgDef>& gradient_argdefs,
    ArgDef& fused_gradient_argdef,
    GraphAugmenter::GraphDefs& graph_defs,
    ArgDef& fused_allreduce_output,
    const WorkerGroupType group_type = WorkerGroupType::GlobalParallel) {
  fused_allreduce_output = ArgDef(fused_gradient_argdef.name + "AllReduce_Out", fused_gradient_argdef.type_proto);

  // Add NCCL Allreduce node.
  graph_defs.AddNodeDefs({NodeDef(OpDef{"NcclAllReduce", kMSDomain, 1},
                                  {fused_gradient_argdef},
                                  {fused_allreduce_output},
                                  {ONNX_NAMESPACE::MakeAttribute("group_type", static_cast<int64_t>(group_type))},
                                  "NcclAllReduce")});

  std::vector<ArgDef> view_inputs(gradient_argdefs.size() + 1);
  view_inputs[0] = fused_allreduce_output;

  for (size_t i = 0; i < gradient_argdefs.size(); i++) {
    ArgDef& gradient_shape = view_inputs[i + 1];
    gradient_shape = ArgDef(gradient_argdefs[i].name + "_Shape");

    graph_defs.AddNodeDefs({NodeDef("Shape",
                                    {gradient_argdefs[i]},
                                    {gradient_shape},
                                    NodeAttributes(),
                                    gradient_shape.name)});
  }

  std::vector<ArgDef> allreduce_outputs(gradient_argdefs.size());
  for (size_t i = 0; i < gradient_argdefs.size(); i++) {
    TypeProto* allreduced_gradient_type_proto = graph_defs.CopyTypeProto(gradient_argdefs[i]);
    allreduced_gradient_type_proto->mutable_tensor_type()->set_elem_type(
        fused_gradient_argdef.type_proto->tensor_type().elem_type());

    allreduce_outputs[i] = ArgDef(gradient_argdefs[i].name + "_AllReduce_Out", allreduced_gradient_type_proto);
  }
  graph_defs.AddNodeDefs({NodeDef(OpDef{"View", kMSDomain, 1},
                                  view_inputs,
                                  allreduce_outputs,
                                  NodeAttributes(),
                                  "AllReduceOutputView")});

  gradient_argdefs = allreduce_outputs;
  return Status::OK();
}
#endif

static Status AddAdasumAllReduceForGradients(
    std::vector<ArgDef>& gradient_argdefs,
    GraphAugmenter::GraphDefs& graph_defs,
    AdasumReductionType adasum_reduction_type) {
  std::vector<ArgDef> adasum_output_argdefs;
  for (size_t i = 0; i < gradient_argdefs.size(); i++) {
    adasum_output_argdefs.emplace_back(ArgDef(gradient_argdefs[i].name + "Adasum_Out", gradient_argdefs[i].type_proto));
  }

  // Add Adasum Allreduce node.
  graph_defs.AddNodeDefs({NodeDef(OpDef{"AdasumAllReduce", kMSDomain, 1},
                                  gradient_argdefs,
                                  adasum_output_argdefs,
                                  {ONNX_NAMESPACE::MakeAttribute("reduce_algo",
                                                                 static_cast<int64_t>(adasum_reduction_type))},
                                  "AdasumAllReduce")});
  gradient_argdefs = std::move(adasum_output_argdefs);
  return Status::OK();
}

Status AdasumOptimizerGraphBuilder::BuildInternal(
    bool should_add_gradient_norm,
    bool should_add_gradient_finite_check,
    Graph& graph,
    GraphAugmenter::GraphDefs& graph_defs,
    std::vector<ArgDef>& weight_argdefs,
    std::vector<ArgDef>& gradient_argdefs,
    std::unordered_map<std::string, std::unordered_map<std::string, std::string>>& weight_to_opt_mapping,
    OptimizerOutputKeyMap<std::string>& optimizer_graph_outputs) {
  // Set weight update to false for optimizer
  for (auto& opt_config : opt_configs_) {
    opt_config.update_weight = false;
  }

  auto nodearg_name_generator = [&graph](const std::string& base_name) {
    return graph.GenerateNodeArgName(base_name);
  };

  // add gradient scaling
  ArgDef fused_gradient_argdef;
  const auto total_num_accumulations = opt_graph_config_.gradient_accumulation_steps;
  ORT_RETURN_IF_NOT(total_num_accumulations > 0, "total_num_accumulations <= 0");

  auto scale_divisor = total_num_accumulations;
  //If Adasum GPU hierarchical reduce is used, then divide gradients by local size.
  if (opt_graph_config_.adasum_reduction_type == AdasumReductionType::GpuHierarchicalReduction) {
    scale_divisor *= opt_graph_config_.local_size;
  }

  const float scale = 1.0f / scale_divisor;
  // Only fuse if using hierarchical reduce.
  const bool fuse_scaling_outputs = opt_graph_config_.adasum_reduction_type == AdasumReductionType::GpuHierarchicalReduction ? true : false;
  ORT_RETURN_IF_ERROR(AddGradientScalingNodes(nodearg_name_generator, scale, gradient_argdefs, fused_gradient_argdef, graph_defs,
                                              opt_graph_config_.AllReduceDataType(), fuse_scaling_outputs));

  // add Allreduce for gradients
  ArgDef reduced_fused_gradient_argdef;
  if (opt_graph_config_.adasum_reduction_type == AdasumReductionType::GpuHierarchicalReduction) {
#ifdef ORT_USE_NCCL
    ORT_RETURN_IF_ERROR(AddNcclAllReduceForGradientsWithGroups(gradient_argdefs, fused_gradient_argdef, graph_defs,
                                                               reduced_fused_gradient_argdef, WorkerGroupType::NodeLocalDataParallel));
#else
    ORT_THROW("ORT is not built with NCCL.");
#endif
  }

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

  // Perform allreduce on deltas after step() for Adasum
  ORT_RETURN_IF_ERROR(AddAdasumAllReduceForGradients(gradient_argdefs,
                                                     graph_defs,
                                                     opt_graph_config_.adasum_reduction_type));

  //check if allreduced deltas are finite
  ArgDef adasum_global_grad_finite_argdef;
  if (opt_graph_config_.use_mixed_precision) {
    ORT_RETURN_IF_ERROR(AddFiniteGradientCheck(
        nodearg_name_generator, gradient_argdefs, graph_defs, adasum_global_grad_finite_argdef,
        "adasum_all_deltas_finite"));
    optimizer_graph_outputs[OptimizerOutputKey::DeltaAllIsFinite] = adasum_global_grad_finite_argdef.name;
  }

  // //Add weight update.
  std::vector<ArgDef> output_weight_args;
  ORT_RETURN_IF_ERROR(AddWeightUpdateNodes(nodearg_name_generator,
                                           gradient_argdefs,
                                           weight_argdefs,
                                           adasum_global_grad_finite_argdef,
                                           graph_defs,
                                           output_weight_args));

  return Status::OK();
}

}  // namespace training
}  // namespace onnxruntime
