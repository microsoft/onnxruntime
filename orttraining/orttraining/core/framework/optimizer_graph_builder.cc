// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/framework/optimizer_graph_builder.h"

#include <cassert>
#include <algorithm>
#include <functional>
#include <iterator>

#include "core/common/common.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph.h"
#include "core/graph/model.h"
#include "orttraining/core/graph/gradient_builder_base.h"
#include "orttraining/core/graph/graph_augmenter.h"
#include "orttraining/core/graph/optimizer_builder.h"
#include "onnx/defs/attr_proto_util.h"

namespace onnxruntime {
namespace training {

Status GetArgDefsFromGraph(
    const Graph& graph, const std::vector<std::string>& node_arg_names,
    std::vector<ArgDef>& argdefs) {
  std::vector<ArgDef> result;
  result.reserve(node_arg_names.size());
  for (const auto& node_arg_name : node_arg_names) {
    const auto* node_arg = graph.GetNodeArg(node_arg_name);
    ORT_RETURN_IF_NOT(node_arg, "Failed to get NodeArg with name ", node_arg_name);
    result.emplace_back(node_arg_name, node_arg->TypeAsProto());
  }
  argdefs = std::move(result);
  return Status::OK();
}

ArgDef BuildGradientAccumulationNode(const NodeArgNameGeneratorFn& nodearg_name_generator,
                                     const ArgDef& gradient,
                                     ArgDef& gradient_accumulation_buffer,
                                     GraphAugmenter::GraphDefs& graph_defs,
                                     bool add_accumulate_buffer_as_initializers) {
  TypeProto* gradient_fp32_type_proto = graph_defs.CopyTypeProto(gradient);
  gradient_fp32_type_proto->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

  ArgDef gradient_accumulate_buffer(nodearg_name_generator(gradient.name + "_accumulate_buffer"),
                                    gradient_fp32_type_proto);
  ArgDef gradient_accumulator_output(nodearg_name_generator(gradient.name + "_accumulator_output"),
                                     gradient_fp32_type_proto);

  std::vector<int64_t> dims;
  ORT_ENFORCE(gradient.type_proto &&
              gradient.type_proto->has_tensor_type() &&
              gradient.type_proto->tensor_type().has_shape());
  for (const auto& dim : gradient.type_proto->tensor_type().shape().dim()) {
    dims.push_back(dim.dim_value());
  }
  if (add_accumulate_buffer_as_initializers)
    graph_defs.AddInitializers({CreateTensorProto<float>(gradient_accumulate_buffer.name, 0.f, dims)});
  graph_defs.AddNodeDefs({NodeDef("GradientAccumulator",
                                  {gradient_accumulate_buffer, gradient},
                                  {gradient_accumulator_output},
                                  NodeAttributes(),
                                  gradient_accumulator_output.name)});

  gradient_accumulation_buffer = gradient_accumulate_buffer;
  return gradient_accumulator_output;
}

namespace {

#ifdef USE_HOROVOD
const std::string global_barrier_name = "horovod/barrier";
const std::string global_barrier_ready = "horovod/barrier/ready";

// TODO generate NodeArg names instead of using hardcoded ones
NodeDef BuildGlobalHorovodBarrierNode(const std::vector<std::string>& ready_names, GraphAugmenter::GraphDefs& graph_defs) {
  std::string barrier_input_name = global_barrier_name + "/input";
  std::string barrier_output_name = global_barrier_name + "/output";

  // Global horovod barrier no-op input.
  TensorProto tensor_proto;
  tensor_proto.add_dims(0);
  tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  tensor_proto.set_name(barrier_input_name);
  graph_defs.AddInitializers({tensor_proto});

  std::vector<ArgDef> barrier_inputs{barrier_input_name};
  std::transform(ready_names.begin(), ready_names.end(), std::back_inserter(barrier_inputs), [](const std::string& name) { return ArgDef(name); });
  std::vector<ArgDef> barrier_outputs{barrier_output_name, global_barrier_ready};

  return NodeDef("HorovodBarrier", barrier_inputs, barrier_outputs, NodeAttributes(), global_barrier_name);
}

NodeDef& GetGlobalHorovodBarrierNode(GraphAugmenter::GraphDefs& graph_defs) {
  // Find the global horovod barrier node.
  auto& nodes = graph_defs.NodeDefs();
  auto barrier_iter = std::find_if(nodes.begin(), nodes.end(), [&](const NodeDef& def) { return def.name == global_barrier_name; });
  if (barrier_iter != nodes.end())
    return *barrier_iter;

  // Create the global horovod barrier.
  graph_defs.AddNodeDefs({BuildGlobalHorovodBarrierNode({}, graph_defs)});
  return *std::find_if(nodes.begin(), nodes.end(), [&](const NodeDef& def) { return def.name == global_barrier_name; });
}

ArgDef BuildHorovodAllReduceNode(const ArgDef& gradient_argdef, GraphAugmenter::GraphDefs& graph_defs) {
  const std::string& gradient_name = gradient_argdef.name;
  ArgDef reduce_output(gradient_name + "_AllReduce_Out", gradient_argdef.type_proto);
  ArgDef reduce_ready(gradient_name + "_AllReduce_Ready");
  ArgDef local_barrier_output(gradient_name + "_Barrier_Out", gradient_argdef.type_proto);
  ArgDef local_barrier_ready(gradient_name + "_Barrier_Ready");

  // Add horovod all reduce node.
  graph_defs.AddNodeDefs({NodeDef("HorovodAllReduce", {gradient_argdef}, {reduce_output, reduce_ready}, NodeAttributes(), gradient_name + "_AllReduce")});

  // Add ready check to global horovod barrier.
  NodeDef& global_barrier_node = GetGlobalHorovodBarrierNode(graph_defs);
  global_barrier_node.input_args.push_back(reduce_ready);

  // Add local horovod barrier node.
  graph_defs.AddNodeDefs({NodeDef("HorovodBarrier", {reduce_output, global_barrier_ready}, {local_barrier_output, local_barrier_ready}, NodeAttributes(), gradient_name + "_Barrier")});

  return local_barrier_output;
}
#else
ArgDef BuildHorovodAllReduceNode(const ArgDef& /*gradient*/, GraphAugmenter::GraphDefs& /*graph_defs*/) {
  ORT_NOT_IMPLEMENTED("Distributed training with Horovod is not supported, as Horovod is not enabled in this build.");
}
#endif

#ifdef USE_NCCL
Status AddNcclAllReduceForGradients(
    std::vector<ArgDef>& gradient_argdefs,
    ArgDef& fused_gradient_argdef,
    GraphAugmenter::GraphDefs& graph_defs,
    ArgDef& fused_allreduce_output) {
  fused_allreduce_output = ArgDef(fused_gradient_argdef.name + "AllReduce_Out", fused_gradient_argdef.type_proto);

  // Add NCCL Allreduce node.
  graph_defs.AddNodeDefs({NodeDef("NcclAllReduce",
                                  {fused_gradient_argdef},
                                  {fused_allreduce_output},
                                  NodeAttributes(),
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

  graph_defs.AddNodeDefs({NodeDef("View",
                                  view_inputs,
                                  allreduce_outputs,
                                  NodeAttributes(),
                                  "AllReduceOutputView")});

  gradient_argdefs = allreduce_outputs;
  return Status::OK();
}

Status AddNcclReduceScatterForGradients(
    std::vector<ArgDef>& gradient_argdefs,
    GraphAugmenter::GraphDefs& graph_defs) {
  std::vector<ArgDef> reducescatter_outputs(gradient_argdefs.size());
  for (size_t i = 0; i < gradient_argdefs.size(); i++) {
    reducescatter_outputs[i] = ArgDef(gradient_argdefs[i].name + "_ReduceScatter_Out",
                                      gradient_argdefs[i].type_proto);
  }

  // Add NCCL ReduceScatter node.
  graph_defs.AddNodeDefs({NodeDef("NcclReduceScatter",
                                  gradient_argdefs,
                                  reducescatter_outputs,
                                  NodeAttributes(),
                                  "NcclReduceScatter")});

  gradient_argdefs = std::move(reducescatter_outputs);
  return Status::OK();
}

Status AddNcclAllGatherForWeights(
    std::vector<ArgDef>& weight_argdefs,
    GraphAugmenter::GraphDefs& graph_defs) {
  std::vector<ArgDef> allgather_outputs(weight_argdefs.size());
  for (size_t i = 0; i < weight_argdefs.size(); i++) {
    allgather_outputs[i] = ArgDef(weight_argdefs[i].name + "_AllGather_Out",
                                  weight_argdefs[i].type_proto);
  }

  // Add NCCL AllGather node.
  graph_defs.AddNodeDefs({NodeDef("NcclAllGather",
                                  weight_argdefs,
                                  allgather_outputs,
                                  NodeAttributes(),
                                  "NcclAllGather")});

  weight_argdefs = std::move(allgather_outputs);
  return Status::OK();
}

Status AddL2NormNcclAllReduce(
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
  graph_defs.AddNodeDefs({NodeDef("NcclAllReduce",
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
#else
Status AddNcclAllReduceForGradients(
    std::vector<ArgDef>& /*gradient_argdefs*/,
    ArgDef& /*fused_gradient_argdef*/,
    GraphAugmenter::GraphDefs& /*graph_defs*/,
    ArgDef& /*fused_allreduce_output*/) {
  ORT_NOT_IMPLEMENTED("Distributed training with NCCL is not supported, as NCCL is not enabled in this build.");
}

Status AddNcclReduceScatterForGradients(
    std::vector<ArgDef>& /*gradient_argdefs*/,
    GraphAugmenter::GraphDefs& /*graph_defs*/) {
  ORT_NOT_IMPLEMENTED("Distributed training with NCCL is not supported, as NCCL is not enabled in this build.");
}

Status AddNcclAllGatherForWeights(
    std::vector<ArgDef>& /*weight_argdefs*/,
    GraphAugmenter::GraphDefs& /*graph_defs*/) {
  ORT_NOT_IMPLEMENTED("Distributed training with NCCL is not supported, as NCCL is not enabled in this build.");
}

Status AddL2NormNcclAllReduce(
    ArgDef& /*norm_argdef*/,
    GraphAugmenter::GraphDefs& /*graph_defs*/) {
  ORT_NOT_IMPLEMENTED("Distributed training with NCCL is not supported, as NCCL is not enabled in this build.");
}
#endif

ArgDef BuildGroupNode(const std::string& group_output_name,
                      const std::vector<ArgDef>& input_argdefs,
                      GraphAugmenter::GraphDefs& graph_defs) {
  ArgDef group_output(group_output_name,
                      graph_defs.CreateTypeProto({}, ONNX_NAMESPACE::TensorProto_DataType_BOOL));
  graph_defs.AddNodeDefs({NodeDef("Group",
                                  input_argdefs,
                                  {group_output},
                                  NodeAttributes(),
                                  group_output.name)});
  return group_output;
}

Status AddGradientScalingNodes(const NodeArgNameGeneratorFn& nodearg_name_generator,
                               const float scale,
                               std::vector<ArgDef>& gradient_argdefs,  // update argdefs in place
                               ArgDef& fused_gradient_argdef,          // update argdef in place
                               GraphAugmenter::GraphDefs& graph_defs,
                               const bool allreduce_in_fp16,
                               const bool fuse_scaling_outputs) {
  ArgDef pre_allreduce_scale(nodearg_name_generator("pre_allreduce_scale"),
                             graph_defs.CreateTypeProto({}, ONNX_NAMESPACE::TensorProto_DataType_FLOAT));
  graph_defs.AddInitializers({CreateTensorProto<float>(pre_allreduce_scale.name, scale, {})});

  auto target_type = allreduce_in_fp16 ? ONNX_NAMESPACE::TensorProto_DataType_FLOAT16
                                       : ONNX_NAMESPACE::TensorProto_DataType_FLOAT;

  if (fuse_scaling_outputs) {
    TypeProto* fused_gradient_type_proto = graph_defs.CreateTypeProto();
    fused_gradient_type_proto->mutable_tensor_type()->set_elem_type(target_type);
    fused_gradient_argdef = ArgDef("fused_gradient", fused_gradient_type_proto);

    std::vector<ArgDef> inputs;
    inputs.emplace_back(pre_allreduce_scale);
    for (size_t i = 0; i < gradient_argdefs.size(); ++i) {
      inputs.emplace_back(gradient_argdefs[i]);
    }
    graph_defs.AddNodeDefs({NodeDef("MixedPrecisionScale",
                                    inputs,
                                    {fused_gradient_argdef},
                                    std::vector<AttributeProto>({ONNX_NAMESPACE::MakeAttribute("to", static_cast<int64_t>(target_type)),
                                                                 ONNX_NAMESPACE::MakeAttribute("fuse_outputs", static_cast<int64_t>(true))}),
                                    pre_allreduce_scale.name)});
  } else {
    for (size_t i = 0; i < gradient_argdefs.size(); ++i) {
      ArgDef& gradient_argdef = gradient_argdefs[i];

      TypeProto* scaled_gradient_type_proto = graph_defs.CopyTypeProto(gradient_argdef);
      scaled_gradient_type_proto->mutable_tensor_type()->set_elem_type(target_type);

      ArgDef scaled_gradient_argdef = ArgDef(nodearg_name_generator(gradient_argdef.name + "_scaled"),
                                             scaled_gradient_type_proto);
      graph_defs.AddNodeDefs({NodeDef("MixedPrecisionScale",
                                      {pre_allreduce_scale, gradient_argdef},
                                      {scaled_gradient_argdef},
                                      {ONNX_NAMESPACE::MakeAttribute("to", static_cast<int64_t>(target_type))},
                                      scaled_gradient_argdef.name)});

      gradient_argdef = scaled_gradient_argdef;
    }
  }

  return Status::OK();
}

ArgDef AddGradientAccumulationNodes(const NodeArgNameGeneratorFn& nodearg_name_generator,
                                    std::vector<ArgDef>& gradient_argdefs,               // update argdefs in place
                                    std::vector<ArgDef>& gradient_accumulation_buffers,  // output
                                    GraphAugmenter::GraphDefs& graph_defs) {
  gradient_accumulation_buffers.resize(gradient_argdefs.size());
  for (size_t i = 0; i < gradient_argdefs.size(); ++i) {
    gradient_argdefs[i] = BuildGradientAccumulationNode(
        nodearg_name_generator, gradient_argdefs[i], gradient_accumulation_buffers[i], graph_defs);
  }

  ArgDef group_accumulate_gradient_output = BuildGroupNode(nodearg_name_generator("Group_Accumulated_Gradients"),
                                                           gradient_argdefs,
                                                           graph_defs);
  graph_defs.AddGraphOutputs({group_accumulate_gradient_output.name});
  return group_accumulate_gradient_output;
}

ArgDef BuildZeroGradientNode(const NodeArgNameGeneratorFn& nodearg_name_generator,
                             const ArgDef& control_signal,
                             const ArgDef& gradient,
                             GraphAugmenter::GraphDefs& graph_defs) {
  ArgDef gradient_zero_output(nodearg_name_generator(gradient.name + "_zero_out"), gradient.type_proto);
  graph_defs.AddNodeDefs({NodeDef("ZeroGradient",
                                  {gradient, control_signal},
                                  {gradient_zero_output},
                                  NodeAttributes(),
                                  gradient_zero_output.name)});
  return gradient_zero_output;
}

Status AddZeroGradientNodes(const NodeArgNameGeneratorFn& nodearg_name_generator,
                            const std::vector<ArgDef>& control_signals,
                            std::vector<ArgDef>& gradient_argdefs,  // update argdefs in place
                            GraphAugmenter::GraphDefs& graph_defs) {
  assert(gradient_argdefs.size() == control_signals.size());
  for (size_t i = 0; i < gradient_argdefs.size(); ++i) {
    gradient_argdefs[i] = BuildZeroGradientNode(nodearg_name_generator, control_signals[i], gradient_argdefs[i], graph_defs);
  }

  return Status::OK();
}

Status AddHorovodAllReduceForGradients(std::vector<ArgDef>& gradient_argdefs,  // update argdefs in place
                                       GraphAugmenter::GraphDefs& graph_defs) {
  for (size_t i = 0; i < gradient_argdefs.size(); ++i) {
    gradient_argdefs[i] = BuildHorovodAllReduceNode(gradient_argdefs[i], graph_defs);
  }
  return Status::OK();
}

Status AddDirectWeightUpdate(
    const OptimizerBuilderRegistry& opt_builder_registry,
    std::vector<ArgDef>& weight_argdefs,    // update argdefs in place
    std::vector<ArgDef>& gradient_argdefs,  // update argdefs in place
    const ArgDef* global_gradient_norm_argdef,
    const ArgDef* global_gradient_norm_finite_argdef,
    const std::vector<OptimizerNodeConfig>& opt_configs,
    GraphAugmenter::GraphDefs& graph_defs,
    std::unordered_set<std::string>& optimizer_state_initializer_names) {
  assert(weight_argdefs.size() == gradient_argdefs.size() &&
         weight_argdefs.size() == opt_configs.size());

  std::vector<TensorProto> new_initializers;
  std::vector<ArgDef> output_weight_argdefs;
  std::vector<ArgDef> output_gradient_argdefs;

  for (size_t i = 0; i < opt_configs.size(); ++i) {
    ORT_RETURN_IF_NOT(
        opt_configs[i].name == opt_configs[0].name,
        "All optimizers must be the same type, but the graph contains ",
        opt_configs[0].name, " and ", opt_configs[i].name);
  }

  auto opt_builder = opt_builder_registry.MakeUnique(opt_configs[0].name);
  ORT_RETURN_IF_NOT(
      opt_builder, "Failed to get Optimizer builder for ", opt_configs[0].name);

  ORT_RETURN_IF_ERROR(opt_builder->Build(
      weight_argdefs, gradient_argdefs,
      global_gradient_norm_argdef, global_gradient_norm_finite_argdef,
      opt_configs, graph_defs,
      new_initializers,
      output_weight_argdefs, output_gradient_argdefs));

  graph_defs.AddInitializers(new_initializers);

  weight_argdefs = std::move(output_weight_argdefs);
  gradient_argdefs = std::move(output_gradient_argdefs);

  std::unordered_set<std::string> all_new_initializer_names{};
  std::transform(
      new_initializers.begin(), new_initializers.end(),
      std::inserter(all_new_initializer_names, all_new_initializer_names.end()),
      [](const TensorProto& initializer) { return initializer.name(); });
  optimizer_state_initializer_names = std::move(all_new_initializer_names);

  return Status::OK();
}

std::vector<ArgDef> GetGradientNormInputs(
    const std::vector<ArgDef>& gradient_argdefs,
    const std::vector<OptimizerNodeConfig>& opt_configs,
    ArgDef fused_gradient_argdef) {
  if (!fused_gradient_argdef.name.empty()) {
    return {fused_gradient_argdef};
  }

  std::vector<ArgDef> inputs;
  for (size_t i = 0; i < gradient_argdefs.size(); i++) {
    if (opt_configs[i].enabled) {
      inputs.push_back(gradient_argdefs[i]);
    }
  }

  return inputs;
}

Status AddGradientNorm(
    const NodeArgNameGeneratorFn& nodearg_name_generator,
    const std::vector<ArgDef>& grad_argdefs,
    GraphAugmenter::GraphDefs& graph_defs,
    ArgDef& grad_norm_argdef) {
  ONNX_NAMESPACE::TensorProto_DataType grad_type =
      static_cast<ONNX_NAMESPACE::TensorProto_DataType>(grad_argdefs[0].type_proto->tensor_type().elem_type());
  if (grad_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT &&
      grad_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
    return Status(common::ONNXRUNTIME, common::FAIL,
                  "Unsupport gradient type: it has to be either float or MLFloat16.");
  }

  for (const auto argdef : grad_argdefs) {
    ONNX_NAMESPACE::TensorProto_DataType elem_type =
        static_cast<ONNX_NAMESPACE::TensorProto_DataType>(argdef.type_proto->tensor_type().elem_type());
    if (elem_type != grad_type) {
      return Status(common::ONNXRUNTIME, common::FAIL,
                    "All gradient tensors' types must be the same.");
    }
  }

  const TypeProto* const grad_norm_type = graph_defs.CreateTypeProto({}, ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  grad_norm_argdef = ArgDef{nodearg_name_generator("global_gradient_norm"), grad_norm_type};

  graph_defs.AddNodeDefs({NodeDef{"ReduceAllL2",
                                  grad_argdefs,
                                  {grad_norm_argdef},
                                  NodeAttributes(),
                                  grad_norm_argdef.name}});

  graph_defs.AddGraphOutputs({grad_norm_argdef.name});

  return Status::OK();
}

Status AddFiniteGradientCheck(
    const NodeArgNameGeneratorFn& nodearg_name_generator,
    const ArgDef& grad_norm_argdef,
    GraphAugmenter::GraphDefs& graph_defs,
    ArgDef& grad_norm_finite_argdef) {
  const TypeProto* const grad_norm_finite_type =
      graph_defs.CreateTypeProto({1}, ONNX_NAMESPACE::TensorProto_DataType_BOOL);
  grad_norm_finite_argdef =
      ArgDef{nodearg_name_generator("all_gradients_finite"), grad_norm_finite_type};

  graph_defs.AddNodeDefs({NodeDef{"IsAllFinite",
                                  {grad_norm_argdef},
                                  {grad_norm_finite_argdef},
                                  NodeAttributes(),
                                  grad_norm_finite_argdef.name}});

  graph_defs.AddGraphOutputs({grad_norm_finite_argdef.name});

  return Status::OK();
}

Status AddLearningRateGraphInputs(Graph& graph, const std::vector<OptimizerNodeConfig>& opt_configs) {
  auto graph_inputs = graph.GetInputsIncludingInitializers();
  std::vector<const NodeArg*> inputs_args_sets(graph_inputs.begin(), graph_inputs.end());
  std::unordered_set<std::string> added_feed_names;
  for (auto& cfg : opt_configs) {
    if (added_feed_names.find(cfg.lr_feed_name) == added_feed_names.end()) {
      TypeProto tensor_float;
      tensor_float.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
      tensor_float.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
      const auto& out_def = graph.GetOrCreateNodeArg(cfg.lr_feed_name, &tensor_float);
      inputs_args_sets.push_back(&out_def);
      added_feed_names.emplace(cfg.lr_feed_name);
    }
  }

  graph.SetInputs(inputs_args_sets);
  return Status::OK();
}

std::vector<ArgDef> AddViewForParameter(GraphAugmenter::GraphDefs& graph_defs, ArgDef argdef, const std::vector<TensorShape>& shapes) {
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

  graph_defs.AddNodeDefs({NodeDef("View",
                                  view_inputs,
                                  view_outputs,
                                  NodeAttributes(),
                                  argdef.name + "_view")});

  return view_outputs;
}

void AddViewForParameters(
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
  ORT_ENFORCE(weight_views.size() == gradient_views.size());
  for (size_t i = 0; i < weight_views.size(); i++) {
    OptimizerNodeConfig new_config = opt_config;
    new_config.enabled = enabled[i];

    if (opt_config.fp16_weight_arg != nullptr) {
      new_config.fp16_weight_arg = &graph.GetOrCreateNodeArg(fp16_weight_views[i].name, fp16_weight_views[i].type_proto);
    }

    opt_configs.push_back(new_config);
  }
}

Status ModifyParametersForOptimizerPartitioning(
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
  const int world_rank = opt_graph_config.world_rank;
  const int world_size = opt_graph_config.world_size;
  const int64_t alignment = world_size * 32;
  const int64_t padded_count = total_count + alignment - (total_count % alignment);
  const int64_t rank_count = padded_count / world_size;
  const int64_t rank_start = world_rank * rank_count;
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

}  // namespace

OptimizerGraphBuilder::OptimizerGraphBuilder(
    const OptimizerBuilderRegistry& opt_builder_registry,
    const OptimizerGraphConfig& opt_graph_config,
    const std::unordered_map<std::string, OptimizerNodeConfig>& weight_names_to_opt_configs)
    : opt_builder_registry_{opt_builder_registry},
      opt_graph_config_{opt_graph_config} {
  weight_names_.reserve(weight_names_to_opt_configs.size());
  std::transform(
      weight_names_to_opt_configs.begin(), weight_names_to_opt_configs.end(),
      std::back_inserter(weight_names_),
      [](const std::pair<std::string, OptimizerNodeConfig>& name_and_info) {
        return name_and_info.first;
      });

  // deterministic ordering for consistent generated nodearg names
  std::sort(weight_names_.begin(), weight_names_.end());

  opt_configs_.reserve(weight_names_.size());
  std::transform(
      weight_names_.begin(), weight_names_.end(), std::back_inserter(opt_configs_),
      [&weight_names_to_opt_configs](const std::string& weight_name) {
        return weight_names_to_opt_configs.at(weight_name);
      });
}

Status OptimizerGraphBuilder::Build(
    Graph& graph,
    std::unordered_set<std::string>& optimizer_state_initializer_names,
    std::unordered_map<std::string, std::string>& optimizer_graph_outputs) {
  if (weight_names_.empty()) {
    // nothing to do
    return Status::OK();
  }

  // from here, we assume there is at least one weight/gradient to process

  auto nodearg_name_generator = [&graph](const std::string& base_name) {
    return graph.GenerateNodeArgName(base_name);
  };

  GraphAugmenter::GraphDefs graph_defs{};
  std::vector<ArgDef> weight_argdefs{}, gradient_argdefs{};

  ORT_RETURN_IF_ERROR(GetArgDefsFromGraph(graph, weight_names_, weight_argdefs));
  {
    std::vector<std::string> gradient_names{};
    gradient_names.reserve(weight_names_.size());
    std::transform(
        weight_names_.begin(), weight_names_.end(), std::back_inserter(gradient_names),
        GradientBuilderBase::GradientName);
    ORT_RETURN_IF_ERROR(GetArgDefsFromGraph(graph, gradient_names, gradient_argdefs));
  }

  const bool is_gradient_accumulation_enabled = opt_graph_config_.gradient_accumulation_steps > 1;
  const bool is_distributed = opt_graph_config_.world_size > 1;
  const bool is_optimizer_partitioned = opt_graph_config_.partition_optimizer;
  const bool overlap_compute_distributed = is_distributed && opt_graph_config_.use_nccl;

  // add grad accumulation
  std::vector<ArgDef> gradient_accumulation_buffers;
  if (is_gradient_accumulation_enabled) {
    ArgDef group_accumulate_gradient_output =
        AddGradientAccumulationNodes(nodearg_name_generator, gradient_argdefs, gradient_accumulation_buffers, graph_defs);
    optimizer_graph_outputs[kGradientAccumulationOutputKey] = group_accumulate_gradient_output.name;
  }

  // handle optimizer partitioning
  if (is_distributed && is_optimizer_partitioned) {
    ORT_RETURN_IF_ERROR(ModifyParametersForOptimizerPartitioning(
        graph, graph_defs, opt_graph_config_, opt_configs_, weight_argdefs, gradient_argdefs));
  }

  // add gradient scaling
  ArgDef fused_gradient_argdef{};
  if (is_gradient_accumulation_enabled || is_distributed) {
    const auto total_num_accumulations = opt_graph_config_.gradient_accumulation_steps * opt_graph_config_.world_size;
    ORT_RETURN_IF_NOT(total_num_accumulations > 0);
    const float scale = 1.0f / total_num_accumulations;
    const bool fuse_scaling_outputs = is_distributed && overlap_compute_distributed && !is_optimizer_partitioned;
    ORT_RETURN_IF_ERROR(AddGradientScalingNodes(nodearg_name_generator, scale, gradient_argdefs, fused_gradient_argdef, graph_defs,
                                                opt_graph_config_.allreduce_in_fp16, fuse_scaling_outputs));
  }

  // add distributed ops
  ArgDef reduced_fused_gradient_argdef{};
  if (is_distributed) {
    if (opt_graph_config_.use_nccl) {
      if (is_optimizer_partitioned) {
        ORT_RETURN_IF_ERROR(AddNcclReduceScatterForGradients(gradient_argdefs, graph_defs));
      } else {
        ORT_RETURN_IF_ERROR(AddNcclAllReduceForGradients(gradient_argdefs, fused_gradient_argdef, graph_defs, reduced_fused_gradient_argdef));
      }
    } else {
      ORT_RETURN_IF_ERROR(AddHorovodAllReduceForGradients(gradient_argdefs, graph_defs));
    }
  }

  // check if all gradients are finite
  ArgDef global_grad_norm_argdef{};
  ArgDef global_grad_norm_finite_argdef{};
  if (opt_graph_config_.use_mixed_precision) {
    auto gradient_norm_inputs = GetGradientNormInputs(gradient_argdefs, opt_configs_, reduced_fused_gradient_argdef);
    ORT_RETURN_IF_ERROR(AddGradientNorm(
        nodearg_name_generator, gradient_norm_inputs, graph_defs, global_grad_norm_argdef));
    optimizer_graph_outputs[kGlobalGradientNormOutputKey] = global_grad_norm_argdef.name;

    if (is_distributed && is_optimizer_partitioned) {
      ORT_RETURN_IF_ERROR(AddL2NormNcclAllReduce(global_grad_norm_argdef, graph_defs));
    }

    ORT_RETURN_IF_ERROR(AddFiniteGradientCheck(
        nodearg_name_generator, global_grad_norm_argdef, graph_defs, global_grad_norm_finite_argdef));
    optimizer_graph_outputs[kGradientAllIsFiniteOutputKey] = global_grad_norm_finite_argdef.name;
  }

  // add weight update
  std::unordered_set<std::string> optimizer_state_initializer_names_result{};

  ORT_RETURN_IF_ERROR(AddDirectWeightUpdate(
      opt_builder_registry_, weight_argdefs, gradient_argdefs,
      &global_grad_norm_argdef,
      &global_grad_norm_finite_argdef,
      opt_configs_, graph_defs,
      optimizer_state_initializer_names_result));

  // add distributed ops
  if (is_distributed) {
    if (opt_graph_config_.use_nccl && is_optimizer_partitioned) {
      ORT_RETURN_IF_ERROR(AddNcclAllGatherForWeights(weight_argdefs, graph_defs));
    }
  }

  // add zero gradient
  if (is_gradient_accumulation_enabled) {
    ORT_RETURN_IF_ERROR(AddZeroGradientNodes(
        nodearg_name_generator, weight_argdefs, gradient_accumulation_buffers, graph_defs));
  }

  // add learning rate inputs
  ORT_RETURN_IF_ERROR(AddLearningRateGraphInputs(graph, opt_configs_));

  ORT_RETURN_IF_ERROR(GraphAugmenter::AugmentGraph(graph, graph_defs));

  optimizer_state_initializer_names = std::move(optimizer_state_initializer_names_result);

  return Status::OK();
}

}  // namespace training
}  // namespace onnxruntime
