// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/training/optimizer_graph_builder.h"

#include <cassert>
#include <algorithm>
#include <functional>
#include <iterator>

#include "core/common/common.h"
#include "core/graph/graph.h"
#include "core/graph/model.h"
#include "core/graph/training/gradient_builder_base.h"
#include "core/graph/training/graph_augmenter.h"
#include "core/graph/training/training_optimizer.h"

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
  ONNX_NAMESPACE::TensorProto tensor_proto;
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
#else
Status AddNcclAllReduceForGradients(
  std::vector<ArgDef>& /*gradient_argdefs*/,
  ArgDef& /*fused_gradient_argdef*/,
  GraphAugmenter::GraphDefs& /*graph_defs*/,
  ArgDef& /*fused_allreduce_output*/) {
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
                                    std::vector<AttributeProto>({MakeAttribute("to", static_cast<int64_t>(target_type)),
                                                                 MakeAttribute("fuse_outputs", static_cast<int64_t>(true))}),
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
                                      {MakeAttribute("to", static_cast<int64_t>(target_type))},
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
    const std::vector<ArgDef>& weight_argdefs,
    const std::vector<ArgDef>& gradient_argdefs,
    const ArgDef* global_gradient_norm_argdef,
    const ArgDef* global_gradient_norm_finite_argdef,
    const std::vector<OptimizerNodeConfig>& opt_configs,
    std::vector<ArgDef>& updated_weight_argdefs,
    GraphAugmenter::GraphDefs& graph_defs,
    std::unordered_set<std::string>& optimizer_state_initializer_names) {
  assert(weight_argdefs.size() == gradient_argdefs.size() &&
         weight_argdefs.size() == opt_configs.size());

  std::vector<ArgDef> inputs_including_initializers;
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
      inputs_including_initializers, new_initializers,
      output_weight_argdefs, output_gradient_argdefs));

  graph_defs.AddInitializers(new_initializers);

  updated_weight_argdefs = std::move(output_weight_argdefs);

  std::unordered_set<std::string> all_new_initializer_names{};
  std::transform(
      new_initializers.begin(), new_initializers.end(),
      std::inserter(all_new_initializer_names, all_new_initializer_names.end()),
      [](const TensorProto& initializer) { return initializer.name(); });
  optimizer_state_initializer_names = std::move(all_new_initializer_names);

  return Status::OK();
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

  for (const auto argdef: grad_argdefs) {
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

using GraphInitFn = std::function<Status(Graph&)>;

Status MakeGraphProto(GraphInitFn graph_init_fn, GraphProto& graph_proto) {
  Model model{"model", false, logging::LoggingManager::DefaultLogger()};
  Graph& graph = model.MainGraph();
  ORT_RETURN_IF_ERROR(graph_init_fn(graph));
  if (graph.GraphResolveNeeded()) {
    ORT_RETURN_IF_ERROR(graph.Resolve());
  }
  graph_proto = graph.ToGraphProto();
  return Status::OK();
}

Status AddConditionalWeightUpdate(
    const NodeArgNameGeneratorFn& nodearg_name_generator,
    const ArgDef& condition_argdef,
    const OptimizerBuilderRegistry& opt_builder_registry,
    const std::vector<ArgDef>& weight_argdefs,
    const std::vector<ArgDef>& gradient_argdefs,
    const std::vector<OptimizerNodeConfig>& opt_configs,
    ArgDef& conditional_node_output,
    GraphAugmenter::GraphDefs& graph_defs,
    std::unordered_set<std::string>& optimizer_state_initializer_names) {
  assert(weight_argdefs.size() == gradient_argdefs.size() &&
         weight_argdefs.size() == opt_configs.size());

  GraphProto then_subgraph_proto, else_subgraph_proto;

  // just use this same output ArgDef for parent graph and subgraphs
  const ArgDef conditional_output_argdef{
      nodearg_name_generator("conditional_output"),
      graph_defs.CreateTypeProto({}, ONNX_NAMESPACE::TensorProto_DataType_BOOL)};

  std::unordered_set<std::string> all_new_optimizer_external_initializer_names{};

  // condition == true
  ORT_RETURN_IF_ERROR(MakeGraphProto(
      [&opt_builder_registry, &weight_argdefs, &gradient_argdefs,
       &opt_configs, &graph_defs, &conditional_output_argdef,
       &all_new_optimizer_external_initializer_names](Graph& then_subgraph) {
        /* subgraph structure:
         * the idea is to minimize any copying incurred by subgraph outputs
         *
         * optimizer 1 ---|
         * optimizer 2 ---|---> group ---> (subgraph output)
         * ...            |
         * optimizer N ---|
         */

        GraphAugmenter::GraphDefs then_subgraph_defs{};
        std::vector<ArgDef> group_input_argdefs{};

        auto opt_builder = opt_builder_registry.MakeUnique(opt_configs[0].name);
        ORT_RETURN_IF_NOT(
            opt_builder, "Failed to get Optimizer builder for ", opt_configs[0].name);

        for (size_t i = 0; i < opt_configs.size(); ++i) {
          ORT_RETURN_IF_NOT(
              opt_configs[i].name == opt_configs[0].name,
              "One graph can only contains one optimizer but ", opt_configs[0].name, " and ", opt_configs[i].name, " are found.");
        }

        std::vector<ArgDef> external_inputs_including_initializers;
        std::vector<TensorProto> new_external_initializers;
        std::vector<ArgDef> output_weight_argdefs;
        std::vector<ArgDef> output_gradient_argdefs;

        ORT_RETURN_IF_ERROR(opt_builder->Build(
            weight_argdefs, gradient_argdefs,
            /*do_update_argdef*/ nullptr, /* global_gradient_norm */ nullptr,
            opt_configs, then_subgraph_defs,
            external_inputs_including_initializers, new_external_initializers,
            output_weight_argdefs, output_gradient_argdefs));

        for (auto& arg : output_weight_argdefs) {
          group_input_argdefs.emplace_back(arg);
        }

        for (const auto& external_input : external_inputs_including_initializers) {
          then_subgraph.AddOuterScopeNodeArg(external_input.name);
        }

        graph_defs.AddInitializers(new_external_initializers);

        std::transform(
            new_external_initializers.begin(), new_external_initializers.end(),
            std::inserter(
                all_new_optimizer_external_initializer_names,
                all_new_optimizer_external_initializer_names.end()),
            [](const TensorProto& initializer) { return initializer.name(); });

        then_subgraph_defs.AddNodeDefs({NodeDef{"Group", group_input_argdefs, {conditional_output_argdef}}});

        then_subgraph_defs.AddGraphOutputs({conditional_output_argdef.name});

        ORT_RETURN_IF_ERROR(GraphAugmenter::AugmentGraph(then_subgraph, then_subgraph_defs));

        return Status::OK();
      },
      then_subgraph_proto));

  // condition == false
  ORT_RETURN_IF_ERROR(MakeGraphProto(
      [&conditional_output_argdef](Graph& else_subgraph) {
        /* subgraph structure:
         * output needs to match that of then_branch subgraph
         *
         * (local initializer) ---> (subgraph output)
         */

        GraphAugmenter::GraphDefs else_subgraph_defs{};

        TensorProto local_initializer = CreateTensorProto(conditional_output_argdef.name, true, {});
        else_subgraph.AddInitializedTensor(local_initializer);

        else_subgraph_defs.AddGraphOutputs({conditional_output_argdef.name});

        ORT_RETURN_IF_ERROR(GraphAugmenter::AugmentGraph(else_subgraph, else_subgraph_defs));

        return Status::OK();
      },
      else_subgraph_proto));

  const std::vector<AttributeProto> conditional_attributes{
      MakeAttribute("then_branch", then_subgraph_proto),
      MakeAttribute("else_branch", else_subgraph_proto)};

  graph_defs.AddNodeDefs({NodeDef{"If", {condition_argdef}, {conditional_output_argdef}, conditional_attributes}});

  conditional_node_output = conditional_output_argdef;

  optimizer_state_initializer_names = std::move(all_new_optimizer_external_initializer_names);

  return Status::OK();
}

Status AddLearningRateGraphInputs(Graph& graph, const std::vector<OptimizerNodeConfig>& opt_configs) {
  auto graph_inputs = graph.GetInputsIncludingInitializers();
  std::vector<const NodeArg*> inputs_args_sets(graph_inputs.begin(), graph_inputs.end());
  std::unordered_set<std::string> added_feed_names;
  for (auto& cfg : opt_configs) {
    if (added_feed_names.find(cfg.lr_feed_name) == added_feed_names.end()) {
      TypeProto tensor_float;
      tensor_float.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
      tensor_float.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
      const auto& out_def = graph.GetOrCreateNodeArg(cfg.lr_feed_name, &tensor_float);
      inputs_args_sets.push_back(&out_def);
      added_feed_names.emplace(cfg.lr_feed_name);
    }
  }

  graph.SetInputs(inputs_args_sets);
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
  const bool is_all_reduce_enabled = opt_graph_config_.world_size > 1;
  const bool overlap_compute_allreduce = !opt_graph_config_.use_nccl;

  // add grad accumulation
  std::vector<ArgDef> gradient_accumulation_buffers;
  if (is_gradient_accumulation_enabled) {
    ArgDef group_accumulate_gradient_output =
        AddGradientAccumulationNodes(nodearg_name_generator, gradient_argdefs, gradient_accumulation_buffers, graph_defs);
    optimizer_graph_outputs[kGradientAccumulationOutputKey] = group_accumulate_gradient_output.name;
  }

  // add gradient scaling
  ArgDef fused_gradient_argdef{};
  if (is_gradient_accumulation_enabled || is_all_reduce_enabled) {
    const auto total_num_accumulations = opt_graph_config_.gradient_accumulation_steps * opt_graph_config_.world_size;
    ORT_RETURN_IF_NOT(total_num_accumulations > 0);
    const float scale = 1.0f / total_num_accumulations;
    const bool fuse_scaling_outputs = is_all_reduce_enabled && !overlap_compute_allreduce;
    ORT_RETURN_IF_ERROR(AddGradientScalingNodes(nodearg_name_generator, scale, gradient_argdefs, fused_gradient_argdef, graph_defs,
                                                opt_graph_config_.allreduce_in_fp16, fuse_scaling_outputs));
  }

  // add all-reduce
  ArgDef reduced_fused_gradient_argdef{};
  if (is_all_reduce_enabled) {
    if (overlap_compute_allreduce) {
      ORT_RETURN_IF_ERROR(AddHorovodAllReduceForGradients(gradient_argdefs, graph_defs));
    } else {
      ORT_RETURN_IF_ERROR(AddNcclAllReduceForGradients(gradient_argdefs, fused_gradient_argdef, graph_defs, reduced_fused_gradient_argdef));
    }
  }

  // check if all gradients are finite
  ArgDef global_grad_norm_argdef{};
  ArgDef global_grad_norm_finite_argdef{};
  if (opt_graph_config_.use_mixed_precision) {
    std::vector<ArgDef> is_finite_inputs{};
    if (!fused_gradient_argdef.name.empty()) {
      is_finite_inputs.emplace_back(reduced_fused_gradient_argdef);
    } else {
      is_finite_inputs = gradient_argdefs;
    }
    ORT_RETURN_IF_ERROR(AddGradientNorm(
        nodearg_name_generator, is_finite_inputs, graph_defs, global_grad_norm_argdef));
    optimizer_graph_outputs[kGlobalGradientNormOutputKey] = global_grad_norm_argdef.name;

    ORT_RETURN_IF_ERROR(AddFiniteGradientCheck(
        nodearg_name_generator, global_grad_norm_argdef, graph_defs, global_grad_norm_finite_argdef));
    optimizer_graph_outputs[kGradientAllIsFiniteOutputKey] = global_grad_norm_finite_argdef.name;
  }

  // add weight update
  std::vector<ArgDef> zero_gradients_control_signals;

  std::unordered_set<std::string> optimizer_state_initializer_names_result{};

  // TODO: enable conditional weight update for mixed precision when If op data copy issue is resolved
  if (false) {
    ArgDef conditional_weight_update_output{};
    ORT_RETURN_IF_ERROR(AddConditionalWeightUpdate(
        nodearg_name_generator, global_grad_norm_argdef, opt_builder_registry_,
        weight_argdefs, gradient_argdefs, opt_configs_,
        conditional_weight_update_output, graph_defs, optimizer_state_initializer_names_result));
    // TODO not ideal to pass N copies of the same argdef as control signals
    //      maybe update AddZeroGradientNodes() to accept 1 or N of them
    zero_gradients_control_signals.assign(weight_argdefs.size(), conditional_weight_update_output);
  } else {
    ORT_RETURN_IF_ERROR(AddDirectWeightUpdate(
        opt_builder_registry_, weight_argdefs, gradient_argdefs,
        &global_grad_norm_argdef,
        &global_grad_norm_finite_argdef,
        opt_configs_, zero_gradients_control_signals, graph_defs,
        optimizer_state_initializer_names_result));
  }

  // add zero gradient
  if (is_gradient_accumulation_enabled) {
    ORT_RETURN_IF_ERROR(AddZeroGradientNodes(
        nodearg_name_generator, zero_gradients_control_signals, gradient_accumulation_buffers, graph_defs));
  }

  // add learning rate inputs
  ORT_RETURN_IF_ERROR(AddLearningRateGraphInputs(graph, opt_configs_));

  ORT_RETURN_IF_ERROR(GraphAugmenter::AugmentGraph(graph, graph_defs));

  optimizer_state_initializer_names = std::move(optimizer_state_initializer_names_result);

  return Status::OK();
}

}  // namespace training
}  // namespace onnxruntime
