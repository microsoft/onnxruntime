// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/graph/zero_optimizer_graph_builder.h"

#include "core/common/common.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph.h"
#include "orttraining/core/graph/graph_augmenter.h"

namespace onnxruntime {
namespace training {

static bool IsNcclAvailable() {
#ifdef USE_NCCL
  return true;
#else
  return false;
#endif
}

static Status AddNcclReduceScatterForGradients(
    std::vector<ArgDef>& gradient_argdefs,
    GraphAugmenter::GraphDefs& graph_defs) {
  std::vector<ArgDef> reducescatter_outputs(gradient_argdefs.size());
  for (size_t i = 0; i < gradient_argdefs.size(); i++) {
    reducescatter_outputs[i] = ArgDef(gradient_argdefs[i].name + "_ReduceScatter_Out",
                                      gradient_argdefs[i].type_proto);
  }

  // Add NCCL ReduceScatter node.
  graph_defs.AddNodeDefs({NodeDef(OpDef{"NcclReduceScatter", kMSDomain, 1},
                                  gradient_argdefs,
                                  reducescatter_outputs,
                                  NodeAttributes(),
                                  "NcclReduceScatter")});

  gradient_argdefs = std::move(reducescatter_outputs);
  return Status::OK();
}

static Status AddNcclAllGatherForWeights(
    std::vector<ArgDef>& weight_argdefs,
    GraphAugmenter::GraphDefs& graph_defs) {
  std::vector<ArgDef> allgather_outputs(weight_argdefs.size());
  for (size_t i = 0; i < weight_argdefs.size(); i++) {
    allgather_outputs[i] = ArgDef(weight_argdefs[i].name + "_AllGather_Out",
                                  weight_argdefs[i].type_proto);
  }

  // Add NCCL AllGather node.
  graph_defs.AddNodeDefs({NodeDef(OpDef{"NcclAllGather", kMSDomain, 1},
                                  weight_argdefs,
                                  allgather_outputs,
                                  NodeAttributes(),
                                  "NcclAllGather")});

  weight_argdefs = std::move(allgather_outputs);
  return Status::OK();
}

static Status AddL2NormNcclAllReduce(
    ArgDef& norm_argdef,
    GraphAugmenter::GraphDefs& graph_defs) {
  // Square the L2 norm.
  ArgDef exponent(norm_argdef.name + "_pow2",
                  graph_defs.CreateTypeProto({}, ONNX_NAMESPACE::TensorProto_DataType_FLOAT));
  graph_defs.AddInitializers({CreateTensorProto<float>(exponent.name, 2.0f, {})});
  ArgDef norm_squared(norm_argdef.name + "_squared", norm_argdef.type_proto);
  graph_defs.AddNodeDefs({NodeDef("Pow",
                                  {norm_argdef, exponent},
                                  {norm_squared},
                                  NodeAttributes(),
                                  norm_squared.name)});

  // AllReduce the squared L2 norms.
  ArgDef allreduce_output(norm_argdef.name + "_AllReduce_Out", norm_argdef.type_proto);
  graph_defs.AddNodeDefs({NodeDef(OpDef{"NcclAllReduce", kMSDomain, 1},
                                  {norm_squared},
                                  {allreduce_output},
                                  NodeAttributes(),
                                  allreduce_output.name)});

  // Sqrt the reduced L2 norm.
  ArgDef sqrt_output(norm_argdef.name + "_sqrt", norm_argdef.type_proto);
  graph_defs.AddNodeDefs({NodeDef("Sqrt",
                                  {allreduce_output},
                                  {sqrt_output},
                                  NodeAttributes(),
                                  sqrt_output.name)});

  norm_argdef = sqrt_output;
  return Status::OK();
}

static std::vector<ArgDef> AddViewForParameter(
    GraphAugmenter::GraphDefs& graph_defs,
    ArgDef argdef,
    const std::vector<TensorShape>& shapes) {
  std::vector<ArgDef> view_inputs = {argdef};
  std::vector<ArgDef> view_outputs;

  for (size_t i = 0; i < shapes.size(); i++) {
    const TensorShape& shape = shapes[i];
    const int64_t dims = shape.NumDimensions();

    ArgDef shape_argdef(argdef.name + "_view_shape_" + std::to_string(i),
                        graph_defs.CreateTypeProto({dims}, ONNX_NAMESPACE::TensorProto_DataType_INT64));
    graph_defs.AddInitializers({CreateTensorProto<int64_t>(shape_argdef.name, shape.GetDims(), {dims})});

    auto dtype = static_cast<ONNX_NAMESPACE::TensorProto_DataType>(argdef.type_proto->tensor_type().elem_type());
    ArgDef view_argdef(argdef.name + "_view_" + std::to_string(i),
                       graph_defs.CreateTypeProto(shape.GetDims(), dtype));

    view_inputs.push_back(shape_argdef);
    view_outputs.push_back(view_argdef);
  }

  graph_defs.AddNodeDefs({NodeDef(OpDef{"View", kMSDomain, 1},
                                  view_inputs,
                                  view_outputs,
                                  NodeAttributes(),
                                  argdef.name + "_view")});

  return view_outputs;
}

static Status AddViewForParameters(
    Graph& graph,
    GraphAugmenter::GraphDefs& graph_defs,
    ArgDef weight_argdef,
    ArgDef gradient_argdef,
    const OptimizerNodeConfig& opt_config,
    const std::vector<TensorShape>& view_shapes,
    const std::vector<bool>& enabled,
    std::vector<OptimizerNodeConfig>& opt_configs,
    std::vector<ArgDef>& weight_argdefs,
    std::vector<ArgDef>& gradient_argdefs) {
  // Add View for weight.
  std::vector<ArgDef> weight_views = AddViewForParameter(graph_defs, weight_argdef, view_shapes);
  weight_argdefs.insert(weight_argdefs.end(), weight_views.begin(), weight_views.end());

  // Add View for gradient.
  std::vector<ArgDef> gradient_views = AddViewForParameter(graph_defs, gradient_argdef, view_shapes);
  gradient_argdefs.insert(gradient_argdefs.end(), gradient_views.begin(), gradient_views.end());

  // (Optional) Add View for FP16 weight.
  std::vector<ArgDef> fp16_weight_views;
  if (opt_config.fp16_weight_arg != nullptr) {
    ArgDef fp16_weight_argdef(opt_config.fp16_weight_arg->Name(), opt_config.fp16_weight_arg->TypeAsProto());
    fp16_weight_views = AddViewForParameter(graph_defs, fp16_weight_argdef, view_shapes);
  }

  // Update Optimizer node configs.
  ORT_RETURN_IF_NOT(weight_views.size() == gradient_views.size());
  for (size_t i = 0; i < weight_views.size(); i++) {
    OptimizerNodeConfig new_config = opt_config;
    new_config.enabled = enabled[i];

    if (opt_config.fp16_weight_arg != nullptr) {
      new_config.fp16_weight_arg = &graph.GetOrCreateNodeArg(fp16_weight_views[i].name, fp16_weight_views[i].type_proto);
    }

    opt_configs.push_back(new_config);
  }

  return Status::OK();
}

static Status ModifyParametersForOptimizerPartitioning(
    Graph& graph,
    GraphAugmenter::GraphDefs& graph_defs,
    const OptimizerGraphConfig& opt_graph_config,
    std::vector<OptimizerNodeConfig>& opt_configs,
    std::vector<ArgDef>& weight_argdefs,
    std::vector<ArgDef>& gradient_argdefs) {
  ORT_ENFORCE(weight_argdefs.size() == gradient_argdefs.size());
  ORT_ENFORCE(weight_argdefs.size() == opt_configs.size());

  // Compute total element count to reduce.
  int64_t total_count = 0;
  for (size_t i = 0; i < weight_argdefs.size(); i++) {
    ArgDef weight_argdef = weight_argdefs[i];
    ORT_ENFORCE(weight_argdef.type_proto != nullptr);
    const auto& weight_shape_proto = weight_argdef.type_proto->tensor_type().shape();
    const TensorShape& weight_shape = utils::GetTensorShapeFromTensorShapeProto(weight_shape_proto);

    ArgDef gradient_argdef = gradient_argdefs[i];
    ORT_ENFORCE(gradient_argdef.type_proto != nullptr);
    const auto& gradient_shape_proto = gradient_argdef.type_proto->tensor_type().shape();
    const TensorShape& gradient_shape = utils::GetTensorShapeFromTensorShapeProto(gradient_shape_proto);

    ORT_ENFORCE(weight_shape == gradient_shape);
    total_count += weight_shape.Size();
  }

  // Compute split points for parameters.
  // Note: the alignment here needs to be kept in-sync with the alignment in nccl_kernels.cc
  const int data_parallel_group_rank = opt_graph_config.data_parallel_group_rank;
  const int data_parallel_group_size = opt_graph_config.data_parallel_group_size;
  const int64_t alignment = data_parallel_group_size * 32;
  const int64_t padded_count = total_count + alignment - (total_count % alignment);
  const int64_t rank_count = padded_count / data_parallel_group_size;
  const int64_t rank_start = data_parallel_group_rank * rank_count;
  const int64_t rank_end = rank_start + rank_count;

  std::vector<OptimizerNodeConfig> new_opt_configs;
  std::vector<ArgDef> new_weight_argdefs;
  std::vector<ArgDef> new_gradient_argdefs;

  int64_t offset = 0;
  for (size_t i = 0; i < weight_argdefs.size(); i++) {
    const OptimizerNodeConfig& opt_config = opt_configs[i];
    ArgDef weight_argdef = weight_argdefs[i];
    ArgDef gradient_argdef = gradient_argdefs[i];

    const auto& tensor_shape_proto = weight_argdef.type_proto->tensor_type().shape();
    const TensorShape& tensor_shape = utils::GetTensorShapeFromTensorShapeProto(tensor_shape_proto);
    const int64_t tensor_count = tensor_shape.Size();

    if (offset < rank_end && offset + tensor_count > rank_start) {
      // Parameter is handled by this rank.  There are 4 cases:
      // 1. parameter is fully handled by this rank
      // 2. parameter is split between previous rank and this rank
      // 3. parameter is split between this rank and next rank
      // 4. parameter is split between previous rank, this rank, and next rank
      if (offset >= rank_start && offset + tensor_count <= rank_end) {
        new_opt_configs.push_back(opt_config);
        new_weight_argdefs.push_back(weight_argdef);
        new_gradient_argdefs.push_back(gradient_argdef);
      } else if (offset < rank_start && offset + tensor_count <= rank_end) {
        int64_t size_for_previous_rank = rank_start - offset;
        int64_t size_for_current_rank = offset + tensor_count - rank_start;
        std::vector<TensorShape> view_shapes = {{size_for_previous_rank}, {size_for_current_rank}};
        std::vector<bool> enabled = {false, true};
        AddViewForParameters(graph, graph_defs, weight_argdef, gradient_argdef, opt_config, view_shapes, enabled,
                             new_opt_configs, new_weight_argdefs, new_gradient_argdefs);
      } else if (offset >= rank_start && offset + tensor_count > rank_end) {
        int64_t size_for_current_rank = rank_end - offset;
        int64_t size_for_next_rank = offset + tensor_count - rank_end;
        std::vector<TensorShape> view_shapes = {{size_for_current_rank}, {size_for_next_rank}};
        std::vector<bool> enabled = {true, false};
        AddViewForParameters(graph, graph_defs, weight_argdef, gradient_argdef, opt_config, view_shapes, enabled,
                             new_opt_configs, new_weight_argdefs, new_gradient_argdefs);
      } else {  // offset < rank_start && offset + tensor_count > rank_end
        int64_t size_for_previous_rank = rank_start - offset;
        int64_t size_for_current_rank = rank_end - rank_start;
        int64_t size_for_next_rank = offset + tensor_count - rank_end;
        std::vector<TensorShape> view_shapes = {{size_for_previous_rank}, {size_for_current_rank}, {size_for_next_rank}};
        std::vector<bool> enabled = {false, true, false};
        AddViewForParameters(graph, graph_defs, weight_argdef, gradient_argdef, opt_config, view_shapes, enabled,
                             new_opt_configs, new_weight_argdefs, new_gradient_argdefs);
      }
    } else {
      // Parameter is handled by a different rank.
      OptimizerNodeConfig new_config = opt_config;
      new_config.enabled = false;

      new_opt_configs.push_back(new_config);
      new_weight_argdefs.push_back(weight_argdef);
      new_gradient_argdefs.push_back(gradient_argdef);
    }

    offset += tensor_count;
  }

  // Update outputs.
  opt_configs = std::move(new_opt_configs);
  weight_argdefs = std::move(new_weight_argdefs);
  gradient_argdefs = std::move(new_gradient_argdefs);
  return Status::OK();
}

static std::vector<ArgDef> GetGradientNormInputs(
    const std::vector<ArgDef>& gradient_argdefs,
    const std::vector<OptimizerNodeConfig>& opt_configs) {
  std::vector<ArgDef> inputs;
  for (size_t i = 0; i < gradient_argdefs.size(); i++) {
    if (opt_configs[i].enabled) {
      inputs.push_back(gradient_argdefs[i]);
    }
  }

  return inputs;
}

ZeROOptimizerGraphBuilder::ZeROOptimizerGraphBuilder(
    const OptimizerBuilderRegistry& opt_builder_registry,
    const OptimizerGraphConfig& opt_graph_config,
    const std::unordered_map<std::string, OptimizerNodeConfig>& weight_names_to_opt_configs)
    : OptimizerGraphBuilder(opt_builder_registry,
                            opt_graph_config,
                            weight_names_to_opt_configs) {
  ORT_ENFORCE(opt_graph_config.data_parallel_group_size > 1, "ZeRO optimizer graph builder can only be used for distributed training.");
  ORT_ENFORCE(opt_graph_config.use_nccl, "Distributed training with ZeRO is only supported with NCCL.");
  ORT_ENFORCE(IsNcclAvailable(), "Distributed training with NCCL is not supported, as NCCL is not enabled in this build.");
}

Status ZeROOptimizerGraphBuilder::BuildInternal(
    Graph& graph,
    GraphAugmenter::GraphDefs& graph_defs,
    std::vector<ArgDef>& weight_argdefs,
    std::vector<ArgDef>& gradient_argdefs,
    std::unordered_set<std::string>& optimizer_state_initializer_names,
    OptimizerOutputKeyMap<std::string>& optimizer_graph_outputs) {
  auto nodearg_name_generator = [&graph](const std::string& base_name) {
    return graph.GenerateNodeArgName(base_name);
  };

  // handle optimizer partitioning
  ORT_RETURN_IF_ERROR(ModifyParametersForOptimizerPartitioning(
      graph, graph_defs, opt_graph_config_, opt_configs_, weight_argdefs, gradient_argdefs));

  // add gradient scaling
  ArgDef fused_gradient_argdef;
  const auto total_num_accumulations = opt_graph_config_.gradient_accumulation_steps * opt_graph_config_.data_parallel_group_size;
  ORT_RETURN_IF_NOT(total_num_accumulations > 0);
  const float scale = 1.0f / total_num_accumulations;
  ORT_RETURN_IF_ERROR(AddGradientScalingNodes(nodearg_name_generator, scale, gradient_argdefs, fused_gradient_argdef, graph_defs,
                                              opt_graph_config_.allreduce_in_fp16, false));

  // add Reducescatter for gradients
  ORT_RETURN_IF_ERROR(AddNcclReduceScatterForGradients(gradient_argdefs, graph_defs));

  // check if all gradients are finite
  ArgDef global_grad_norm_argdef;
  ArgDef global_grad_norm_finite_argdef;
  if (opt_graph_config_.use_mixed_precision) {
    auto gradient_norm_inputs = GetGradientNormInputs(gradient_argdefs, opt_configs_);
    ORT_RETURN_IF_ERROR(AddGradientNorm(
        nodearg_name_generator, gradient_norm_inputs, graph_defs, global_grad_norm_argdef));
    optimizer_graph_outputs[OptimizerOutputKey::GlobalGradientNorm] = global_grad_norm_argdef.name;

    ORT_RETURN_IF_ERROR(AddL2NormNcclAllReduce(global_grad_norm_argdef, graph_defs));

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
      optimizer_state_initializer_names));

  // add Allgather for weights
  ORT_RETURN_IF_ERROR(AddNcclAllGatherForWeights(weight_argdefs, graph_defs));

  return Status::OK();
}

}  // namespace training
}  // namespace onnxruntime
