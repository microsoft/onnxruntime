// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/graph/zero_optimizer_graph_builder.h"

#include "core/common/common.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph.h"
#include "core/optimizer/initializer.h"
#include "orttraining/core/graph/graph_augmenter.h"

namespace onnxruntime {
namespace training {

// Data structure to track weights/gradients/optimizer state owned by a rank.
struct ParameterPartition {
  std::vector<ArgDef> weights;
  std::vector<ArgDef> gradients;
  std::vector<OptimizerNodeConfig> opt_configs;
};

// Data structure to track weights, gradients, and optimizer state split between ranks.
struct OptimizerParameterPartitions {
  ParameterPartition previous_ranks; // weights/gradients belong to a smaller rank
  ParameterPartition current_rank;   // weights/gradients belong to this rank (process)
  ParameterPartition next_ranks;     // weights/gradients belong to a larger rank

  std::vector<ArgDef> Weights() {
    std::vector<ArgDef> weights;
    weights.insert(weights.end(), previous_ranks.weights.begin(), previous_ranks.weights.end());
    weights.insert(weights.end(), current_rank.weights.begin(), current_rank.weights.end());
    weights.insert(weights.end(), next_ranks.weights.begin(), next_ranks.weights.end());
    return weights;
  }

  void UpdateWeights(const std::vector<ArgDef>& weights) {
    ORT_ENFORCE(weights.size() == previous_ranks.weights.size() + current_rank.weights.size() + next_ranks.weights.size());
    size_t previous_ranks_start = 0;
    size_t previous_ranks_end = previous_ranks_start + previous_ranks.weights.size();
    size_t current_rank_start = previous_ranks_end;
    size_t current_rank_end = current_rank_start + current_rank.weights.size();
    size_t next_ranks_start = current_rank_end;
    size_t next_ranks_end = next_ranks_start + next_ranks.weights.size();

    previous_ranks.weights.assign(weights.begin() + previous_ranks_start, weights.begin() + previous_ranks_end);
    current_rank.weights.assign(weights.begin() + current_rank_start, weights.begin() + current_rank_end);
    next_ranks.weights.assign(weights.begin() + next_ranks_start, weights.begin() + next_ranks_end);
  }

  std::vector<ArgDef> Gradients() {
    std::vector<ArgDef> gradients;
    gradients.insert(gradients.end(), previous_ranks.gradients.begin(), previous_ranks.gradients.end());
    gradients.insert(gradients.end(), current_rank.gradients.begin(), current_rank.gradients.end());
    gradients.insert(gradients.end(), next_ranks.gradients.begin(), next_ranks.gradients.end());
    return gradients;
  }

  void UpdateGradients(const std::vector<ArgDef>& gradients) {
    ORT_ENFORCE(gradients.size() == previous_ranks.gradients.size() + current_rank.gradients.size() + next_ranks.gradients.size());
    size_t previous_ranks_start = 0;
    size_t previous_ranks_end = previous_ranks_start + previous_ranks.gradients.size();
    size_t current_rank_start = previous_ranks_end;
    size_t current_rank_end = current_rank_start + current_rank.gradients.size();
    size_t next_ranks_start = current_rank_end;
    size_t next_ranks_end = next_ranks_start + next_ranks.gradients.size();

    previous_ranks.gradients.assign(gradients.begin() + previous_ranks_start, gradients.begin() + previous_ranks_end);
    current_rank.gradients.assign(gradients.begin() + current_rank_start, gradients.begin() + current_rank_end);
    next_ranks.gradients.assign(gradients.begin() + next_ranks_start, gradients.begin() + next_ranks_end);
  }
};

static bool IsNcclAvailable() {
#ifdef USE_NCCL
  return true;
#else
  return false;
#endif
}

static Status AddNcclReduceScatterForGradients(
    OptimizerParameterPartitions& partitions,
    GraphAugmenter::GraphDefs& graph_defs) {
  std::vector<ArgDef> gradient_argdefs = partitions.Gradients();
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

  partitions.UpdateGradients(reducescatter_outputs);
  return Status::OK();
}

static Status AddNcclAllGatherForWeights(
    OptimizerParameterPartitions& partitions,
    GraphAugmenter::GraphDefs& graph_defs) {
  std::vector<ArgDef> weight_argdefs = partitions.Weights();
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

  partitions.UpdateWeights(allgather_outputs);
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

static ArgDef AddPartitionForParameter(
    Graph& graph,
    GraphAugmenter::GraphDefs& graph_defs,
    const std::string& initializer_name,
    int64_t partition_offset,
    int64_t partition_size) {
  const ONNX_NAMESPACE::TensorProto* tensor_proto;
  ORT_ENFORCE(graph.GetInitializedTensor(initializer_name, tensor_proto));
  auto dtype = static_cast<ONNX_NAMESPACE::TensorProto_DataType>(tensor_proto->data_type());
  ORT_ENFORCE(dtype == ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

  // Find the initializer partition to read out.
  auto initializer = onnxruntime::make_unique<Initializer>(*tensor_proto, graph.ModelPath());
  const float* initializer_data = initializer->data<float>();

  // Create new initializer tensor proto.
  ONNX_NAMESPACE::TensorProto initializer_partition;
  initializer_partition.set_name(initializer_name);
  initializer_partition.set_data_type(dtype);
  initializer_partition.add_dims(partition_size);
  initializer_partition.set_raw_data(initializer_data + partition_offset, partition_size * sizeof(float));

  // Clear the initializer shape, as we are replacing the full weight with a segment of the weight.
  NodeArg* initializer_arg = graph.GetNodeArg(initializer_name);
  initializer_arg->ClearShape();

  graph.RemoveInitializedTensor(initializer_name);
  graph.AddInitializedTensor(initializer_partition);

  return ArgDef(initializer_name, graph_defs.CreateTypeProto({partition_size}, dtype));
}

// Compute total number of elements in weights/gradients (must match).
static int64_t GetTotalParameterCount(
    const std::vector<ArgDef>& weight_argdefs,
    const std::vector<ArgDef>& gradient_argdefs) {
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

  return total_count;
}

static std::vector<TensorShape> GetViewShapes(
    int64_t size_for_previous_rank,
    int64_t size_for_current_rank,
    int64_t size_for_next_rank) {
  std::vector<TensorShape> view_shapes;
  if (size_for_previous_rank > 0)
    view_shapes.push_back({size_for_previous_rank});
  if (size_for_current_rank > 0)
    view_shapes.push_back({size_for_current_rank});
  if (size_for_next_rank > 0)
    view_shapes.push_back({size_for_next_rank});
  return view_shapes;
}

static Status AddParameterPartitionMixedPrecision(
    Graph& graph,
    GraphAugmenter::GraphDefs& graph_defs,
    OptimizerNodeConfig opt_config,
    ArgDef weight_argdef,
    ArgDef gradient_argdef,
    int64_t size_for_previous_rank,
    int64_t size_for_current_rank,
    int64_t size_for_next_rank,
    OptimizerParameterPartitions& partitions) {
  ORT_RETURN_IF_NOT(opt_config.fp16_weight_arg != nullptr);

  // Build View shapes.
  std::vector<TensorShape> view_shapes = GetViewShapes(size_for_previous_rank, size_for_current_rank, size_for_next_rank);
  ORT_RETURN_IF_NOT(view_shapes.size() > 1, "Parameter is not partitioned across 2 or more ranks.");

  // Partition the FP32 weight.
  ArgDef weight_for_current_rank = AddPartitionForParameter(graph, graph_defs, weight_argdef.name, size_for_previous_rank, size_for_current_rank);

  // Add View for FP16 weights.
  ArgDef fp16_weight_argdef(opt_config.fp16_weight_arg->Name(), opt_config.fp16_weight_arg->TypeAsProto());
  std::vector<ArgDef> fp16_weight_views = AddViewForParameter(graph_defs, fp16_weight_argdef, view_shapes);

  // Add View for gradient.
  std::vector<ArgDef> gradient_views = AddViewForParameter(graph_defs, gradient_argdef, view_shapes);

  // Update parameter partitions.
  size_t index = 0;
  if (size_for_previous_rank > 0) {
    partitions.previous_ranks.weights.push_back(fp16_weight_views[index]);
    partitions.previous_ranks.gradients.push_back(gradient_views[index]);
    index++;
  }
  if (size_for_current_rank > 0) {
    // Update optimizer config with FP16 weight view.
    opt_config.fp16_weight_arg = &graph.GetOrCreateNodeArg(fp16_weight_views[index].name, fp16_weight_views[index].type_proto);
    partitions.current_rank.weights.push_back(weight_for_current_rank);
    partitions.current_rank.gradients.push_back(gradient_views[index]);
    partitions.current_rank.opt_configs.push_back(opt_config);
    index++;
  }
  if (size_for_next_rank > 0) {
    partitions.next_ranks.weights.push_back(fp16_weight_views[index]);
    partitions.next_ranks.gradients.push_back(gradient_views[index]);
    index++;
  }

  return Status::OK();
}

static Status AddParameterPartition(
    Graph& graph,
    GraphAugmenter::GraphDefs& graph_defs,
    OptimizerNodeConfig opt_config,
    ArgDef weight_argdef,
    ArgDef gradient_argdef,
    int64_t size_for_previous_rank,
    int64_t size_for_current_rank,
    int64_t size_for_next_rank,
    OptimizerParameterPartitions& partitions) {
  // Handle mixed-precision mode.
  if (opt_config.fp16_weight_arg != nullptr) {
    return AddParameterPartitionMixedPrecision(
        graph, graph_defs, opt_config, weight_argdef, gradient_argdef,
        size_for_previous_rank, size_for_current_rank, size_for_next_rank, partitions);
  }

  // Build View shapes.
  std::vector<TensorShape> view_shapes = GetViewShapes(size_for_previous_rank, size_for_current_rank, size_for_next_rank);
  ORT_RETURN_IF_NOT(view_shapes.size() > 1, "Parameter is not partitioned across 2 or more ranks.");

  // Add View for weight.
  std::vector<ArgDef> weight_views = AddViewForParameter(graph_defs, weight_argdef, view_shapes);

  // Add View for gradient.
  std::vector<ArgDef> gradient_views = AddViewForParameter(graph_defs, gradient_argdef, view_shapes);

  // Update parameter partitions.
  size_t index = 0;
  if (size_for_previous_rank > 0) {
    partitions.previous_ranks.weights.push_back(weight_views[index]);
    partitions.previous_ranks.gradients.push_back(gradient_views[index]);
    index++;
  }
  if (size_for_current_rank > 0) {
    partitions.current_rank.weights.push_back(weight_views[index]);
    partitions.current_rank.gradients.push_back(gradient_views[index]);
    partitions.current_rank.opt_configs.push_back(opt_config);
    index++;
  }
  if (size_for_next_rank > 0) {
    partitions.next_ranks.weights.push_back(weight_views[index]);
    partitions.next_ranks.gradients.push_back(gradient_views[index]);
    index++;
  }

  return Status::OK();
}

static Status GetParameterPartitions(
    Graph& graph,
    GraphAugmenter::GraphDefs& graph_defs,
    const OptimizerGraphConfig& opt_graph_config,
    const std::vector<OptimizerNodeConfig>& opt_configs,
    const std::vector<ArgDef>& weight_argdefs,
    const std::vector<ArgDef>& gradient_argdefs,
    OptimizerParameterPartitions& partitions) {
  ORT_ENFORCE(weight_argdefs.size() == gradient_argdefs.size());
  ORT_ENFORCE(weight_argdefs.size() == opt_configs.size());

  // Compute total number of elements across all parameters.
  int64_t total_count = GetTotalParameterCount(weight_argdefs, gradient_argdefs);

  // Compute split points for parameters.
  // Note: the alignment here needs to be kept in-sync with the alignment in nccl_kernels.cc
  const int data_parallel_group_rank = opt_graph_config.data_parallel_group_rank;
  const int data_parallel_group_size = opt_graph_config.data_parallel_group_size;
  const int64_t alignment = data_parallel_group_size * 32;
  const int64_t padded_count = total_count + alignment - (total_count % alignment);
  const int64_t rank_count = padded_count / data_parallel_group_size;
  const int64_t rank_start = data_parallel_group_rank * rank_count;
  const int64_t rank_end = rank_start + rank_count;

  // Partition the weight/gradients into those belonging to the previous rank(s),
  // current rank, and next rank(s).
  int64_t offset = 0;
  for (size_t i = 0; i < weight_argdefs.size(); i++) {
    const OptimizerNodeConfig& opt_config = opt_configs[i];
    ArgDef weight_argdef = weight_argdefs[i];
    ArgDef gradient_argdef = gradient_argdefs[i];

    // In mixed-precision mode, we pass the FP32 weights to the optimizer builder and it returns the FP16 weights.
    // We only run the optimizer builder for weights belonging to this rank. For weights handled by other ranks,
    // immediately grab the FP16 weight (if it exists).
    ArgDef fp16_or_fp32_weight_argdef = opt_config.fp16_weight_arg != nullptr ? 
        ArgDef(opt_config.fp16_weight_arg->Name(), opt_config.fp16_weight_arg->TypeAsProto()) : weight_argdef;

    // Get the total number of elements in this parameter.
    const auto& tensor_shape_proto = weight_argdef.type_proto->tensor_type().shape();
    const TensorShape& tensor_shape = utils::GetTensorShapeFromTensorShapeProto(tensor_shape_proto);
    const int64_t tensor_count = tensor_shape.Size();

    if (offset + tensor_count <= rank_start) {
      // Parameter is fully handled by previous rank(s).
      partitions.previous_ranks.weights.push_back(fp16_or_fp32_weight_argdef);
      partitions.previous_ranks.gradients.push_back(gradient_argdef);
    } else if (offset >= rank_start && offset + tensor_count <= rank_end) {
      // Parameter is fully handled by this rank.
      partitions.current_rank.weights.push_back(weight_argdef);
      partitions.current_rank.gradients.push_back(gradient_argdef);
      partitions.current_rank.opt_configs.push_back(opt_config);
    } else if (offset >= rank_end) {
      // Parameter is fully handled by next rank(s).
      partitions.next_ranks.weights.push_back(fp16_or_fp32_weight_argdef);
      partitions.next_ranks.gradients.push_back(gradient_argdef);
    } else if (offset < rank_start && offset + tensor_count <= rank_end) {
      // Parameter is split between previous rank and this rank.
      int64_t size_for_previous_rank = rank_start - offset;
      int64_t size_for_current_rank = offset + tensor_count - rank_start;
      int64_t size_for_next_rank = 0;
      ORT_RETURN_IF_ERROR(AddParameterPartition(
          graph, graph_defs, opt_config, weight_argdef, gradient_argdef, 
          size_for_previous_rank, size_for_current_rank, size_for_next_rank, partitions));
    } else if (offset >= rank_start && offset + tensor_count > rank_end) {
      // Parameter is split between this rank and next rank.
      int64_t size_for_previous_rank = 0;
      int64_t size_for_current_rank = rank_end - offset;
      int64_t size_for_next_rank = offset + tensor_count - rank_end;
      ORT_RETURN_IF_ERROR(AddParameterPartition(
          graph, graph_defs, opt_config, weight_argdef, gradient_argdef, 
          size_for_previous_rank, size_for_current_rank, size_for_next_rank, partitions));
    } else if (offset < rank_start && offset + tensor_count > rank_end) {
      // Parameter is split between previous rank, this rank, and next rank.
      int64_t size_for_previous_rank = rank_start - offset;
      int64_t size_for_current_rank = rank_end - rank_start;
      int64_t size_for_next_rank = offset + tensor_count - rank_end;
      ORT_RETURN_IF_ERROR(AddParameterPartition(
          graph, graph_defs, opt_config, weight_argdef, gradient_argdef, 
          size_for_previous_rank, size_for_current_rank, size_for_next_rank, partitions));
    } else {
      ORT_THROW("Unhandled case in ZeRO optimizer parameter partitioning logic.");
    }

    offset += tensor_count;
  }

  return Status::OK();
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

  // add gradient scaling
  const auto total_num_accumulations = opt_graph_config_.gradient_accumulation_steps * opt_graph_config_.data_parallel_group_size;
  ORT_RETURN_IF_NOT(total_num_accumulations > 0);
  const float scale = 1.0f / total_num_accumulations;
  ORT_RETURN_IF_ERROR(AddGradientScalingNodes(nodearg_name_generator, scale, gradient_argdefs, graph_defs, opt_graph_config_.allreduce_in_fp16));

  // handle optimizer partitioning
  OptimizerParameterPartitions partitions;
  ORT_RETURN_IF_ERROR(GetParameterPartitions(graph, graph_defs, opt_graph_config_, opt_configs_, weight_argdefs, gradient_argdefs, partitions));

  // add Reducescatter for gradients
  ORT_RETURN_IF_ERROR(AddNcclReduceScatterForGradients(partitions, graph_defs));

  // check if all gradients are finite
  ArgDef global_grad_norm_argdef;
  ArgDef global_grad_norm_finite_argdef;
  if (opt_graph_config_.use_mixed_precision) {
    ORT_RETURN_IF_ERROR(AddGradientNorm(nodearg_name_generator, partitions.current_rank.gradients, graph_defs, global_grad_norm_argdef));
    optimizer_graph_outputs[OptimizerOutputKey::GlobalGradientNorm] = global_grad_norm_argdef.name;
    ORT_RETURN_IF_ERROR(AddL2NormNcclAllReduce(global_grad_norm_argdef, graph_defs));
    ORT_RETURN_IF_ERROR(AddFiniteGradientCheck(nodearg_name_generator, {global_grad_norm_argdef}, graph_defs, global_grad_norm_finite_argdef));
    optimizer_graph_outputs[OptimizerOutputKey::GradientAllIsFinite] = global_grad_norm_finite_argdef.name;
  }

  // add weight update
  ORT_RETURN_IF_ERROR(AddDirectWeightUpdate(
      opt_builder_registry_,
      partitions.current_rank.weights,
      partitions.current_rank.gradients,
      &global_grad_norm_argdef,
      &global_grad_norm_finite_argdef,
      partitions.current_rank.opt_configs,
      graph_defs,
      optimizer_state_initializer_names));

  // add Allgather for weights
  ORT_RETURN_IF_ERROR(AddNcclAllGatherForWeights(partitions, graph_defs));

  weight_argdefs = std::move(partitions.Weights());
  gradient_argdefs = std::move(partitions.Gradients());
  return Status::OK();
}

}  // namespace training
}  // namespace onnxruntime
