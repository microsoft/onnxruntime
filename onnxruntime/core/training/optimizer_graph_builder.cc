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
NodeDef BuildGlobalBarrierNode(const std::vector<std::string>& ready_names, GraphAugmenter::GraphDefs& graph_defs) {
  std::string barrier_input_name = global_barrier_name + "/input";
  std::string barrier_output_name = global_barrier_name + "/output";

  // Global barrier no-op input.
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

NodeDef& GetGlobalBarrierNode(GraphAugmenter::GraphDefs& graph_defs) {
  // Find the global barrier node.
  auto& nodes = graph_defs.NodeDefs();
  auto barrier_iter = std::find_if(nodes.begin(), nodes.end(), [&](const NodeDef& def) { return def.name == global_barrier_name; });
  if (barrier_iter != nodes.end())
    return *barrier_iter;

  // Create the global barrier.
  graph_defs.AddNodeDefs({BuildGlobalBarrierNode({}, graph_defs)});
  return *std::find_if(nodes.begin(), nodes.end(), [&](const NodeDef& def) { return def.name == global_barrier_name; });
}

ArgDef BuildAllReduceNode(const ArgDef& gradient_argdef, GraphAugmenter::GraphDefs& graph_defs) {
  const std::string& gradient_name = gradient_argdef.name;
  ArgDef reduce_output(gradient_name + "_AllReduce_Out", gradient_argdef.type_proto);
  ArgDef reduce_ready(gradient_name + "_AllReduce_Ready");
  ArgDef local_barrier_output(gradient_name + "_Barrier_Out", gradient_argdef.type_proto);
  ArgDef local_barrier_ready(gradient_name + "_Barrier_Ready");

  // Add horovod all reduce node.
  graph_defs.AddNodeDefs({NodeDef("HorovodAllReduce", {gradient_argdef}, {reduce_output, reduce_ready}, NodeAttributes(), gradient_name + "_AllReduce")});

  // Add ready check to global barrier.
  NodeDef& global_barrier_node = GetGlobalBarrierNode(graph_defs);
  global_barrier_node.input_args.push_back(reduce_ready);

  // Add local barrier node.
  graph_defs.AddNodeDefs({NodeDef("HorovodBarrier", {reduce_output, global_barrier_ready}, {local_barrier_output, local_barrier_ready}, NodeAttributes(), gradient_name + "_Barrier")});

  return local_barrier_output;
}
#else
ArgDef BuildAllReduceNode(const ArgDef& /*gradient*/, GraphAugmenter::GraphDefs& /*graph_defs*/) {
  ORT_NOT_IMPLEMENTED("Distributed training is not supported, as Horovod is not enabled in this build.");
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

ArgDef BuildGradientScalingNode(const NodeArgNameGeneratorFn& nodearg_name_generator,
                                const ArgDef& scale_argdef,
                                const ArgDef& gradient_argdef,
                                GraphAugmenter::GraphDefs& graph_defs,
                                const bool allreduce_in_fp16) {
  auto target_type = allreduce_in_fp16
                         ? ONNX_NAMESPACE::TensorProto_DataType_FLOAT16
                         : ONNX_NAMESPACE::TensorProto_DataType_FLOAT;

  TypeProto* scaled_gradient_type_proto = graph_defs.CopyTypeProto(gradient_argdef);
  scaled_gradient_type_proto->mutable_tensor_type()->set_elem_type(target_type);

  ArgDef scaled_gradient_argdef = ArgDef(nodearg_name_generator(gradient_argdef.name + "_scaled"),
                                         scaled_gradient_type_proto);

  graph_defs.AddNodeDefs({NodeDef("MixedPrecisionScale",
                                  {gradient_argdef, scale_argdef},
                                  {scaled_gradient_argdef},
                                  {MakeAttribute("to", static_cast<int64_t>(target_type))},
                                  scaled_gradient_argdef.name)});

  return scaled_gradient_argdef;
}

Status AddGradientScalingNodes(const NodeArgNameGeneratorFn& nodearg_name_generator,
                               const float scale,
                               std::vector<ArgDef>& gradient_argdefs,  // update argdefs in place
                               GraphAugmenter::GraphDefs& graph_defs,
                               const bool allreduce_in_fp16) {
  ArgDef pre_allreduce_scale(nodearg_name_generator("pre_allreduce_scale"),
                             graph_defs.CreateTypeProto({}, ONNX_NAMESPACE::TensorProto_DataType_FLOAT));
  graph_defs.AddInitializers({CreateTensorProto<float>(pre_allreduce_scale.name, scale, {})});

  for (size_t i = 0; i < gradient_argdefs.size(); ++i) {
    gradient_argdefs[i] = BuildGradientScalingNode(nodearg_name_generator, pre_allreduce_scale, gradient_argdefs[i], graph_defs, allreduce_in_fp16);
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

Status AddAllReduceForGradients(std::vector<ArgDef>& gradient_argdefs,  // update argdefs in place
                                GraphAugmenter::GraphDefs& graph_defs) {
  for (size_t i = 0; i < gradient_argdefs.size(); ++i) {
    gradient_argdefs[i] = BuildAllReduceNode(gradient_argdefs[i], graph_defs);
  }
  return Status::OK();
}

Status AddDirectWeightUpdate(
    const OptimizerBuilderRegistry& opt_builder_registry,
    const std::vector<ArgDef>& weight_argdefs,
    const std::vector<ArgDef>& gradient_argdefs,
    const ArgDef* all_gradients_finite_argdef,
    const std::vector<OptimizerNodeConfig>& opt_configs,
    std::vector<ArgDef>& updated_weight_argdefs,
    GraphAugmenter::GraphDefs& graph_defs) {
  assert(weight_argdefs.size() == gradient_argdefs.size() &&
         weight_argdefs.size() == opt_configs.size());

  const size_t num_weights = weight_argdefs.size();
  std::vector<ArgDef> output_weight_argdefs(num_weights);
  for (size_t i = 0; i < num_weights; ++i) {
    auto opt_builder = opt_builder_registry.MakeUnique(opt_configs[i].name);
    ORT_RETURN_IF_NOT(
        opt_builder, "Failed to get Optimizer builder for ", opt_configs[i].name);

    std::vector<ArgDef> inputs_including_initializers{};
    std::vector<TensorProto> new_initializers{};

    ORT_RETURN_IF_ERROR(opt_builder->Build(
        weight_argdefs[i], gradient_argdefs[i], all_gradients_finite_argdef, opt_configs[i],
        graph_defs, inputs_including_initializers, new_initializers, output_weight_argdefs[i]));

    graph_defs.AddInitializers(new_initializers);
  }

  updated_weight_argdefs = std::move(output_weight_argdefs);
  return Status::OK();
}

Status AddFiniteGradientChecks(
    const NodeArgNameGeneratorFn& nodearg_name_generator,
    const std::vector<ArgDef>& gradient_argdefs,
    GraphAugmenter::GraphDefs& graph_defs,
    ArgDef& all_gradients_finite_argdef) {
  std::vector<ArgDef> fp32_gradient_argdefs, fp16_gradient_argdefs;
  for (auto it = gradient_argdefs.begin(); it != gradient_argdefs.end(); ++it) {
    auto type = it->type_proto->tensor_type().elem_type();
    switch (type) {
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
        fp32_gradient_argdefs.push_back(*it);
        break;
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
        fp16_gradient_argdefs.push_back(*it);
        break;
      default:
        return Status(common::ONNXRUNTIME, common::FAIL,
                      "Unsupport gradient type: it has to be either float or MLFloat16.");
    }
  }

  const bool has_fp32_grad = !fp32_gradient_argdefs.empty();
  const bool has_fp16_grad = !fp16_gradient_argdefs.empty();

  const TypeProto* const is_all_finite_output_type = graph_defs.CreateTypeProto({1}, ONNX_NAMESPACE::TensorProto_DataType_BOOL);
  ArgDef all_fp32_gradients_finite_argdef;
  if (has_fp32_grad) {
    all_fp32_gradients_finite_argdef = ArgDef{nodearg_name_generator("all_fp32_gradients_finite"), is_all_finite_output_type};
    graph_defs.AddNodeDefs({NodeDef{"IsAllFinite",
                                    fp32_gradient_argdefs,
                                    {all_fp32_gradients_finite_argdef},
                                    NodeAttributes(),
                                    all_fp32_gradients_finite_argdef.name}});
  }

  ArgDef all_fp16_gradients_finite_argdef;
  if (has_fp16_grad) {
    all_fp16_gradients_finite_argdef = ArgDef{nodearg_name_generator("all_fp16_gradients_finite"), is_all_finite_output_type};
    graph_defs.AddNodeDefs({NodeDef{"IsAllFinite",
                                    fp16_gradient_argdefs,
                                    {all_fp16_gradients_finite_argdef},
                                    NodeAttributes(),
                                    all_fp16_gradients_finite_argdef.name}});
  }

  if (has_fp32_grad && !has_fp16_grad) {
    all_gradients_finite_argdef = all_fp32_gradients_finite_argdef;
  } else if (!has_fp32_grad && has_fp16_grad) {
    all_gradients_finite_argdef = all_fp16_gradients_finite_argdef;
  } else if (has_fp32_grad && has_fp16_grad) {
    all_gradients_finite_argdef = ArgDef{nodearg_name_generator("all_gradients_finite"), is_all_finite_output_type};
    graph_defs.AddNodeDefs({NodeDef{"And",
                                    {all_fp32_gradients_finite_argdef, all_fp16_gradients_finite_argdef},
                                    {all_gradients_finite_argdef},
                                    NodeAttributes(),
                                    all_gradients_finite_argdef.name}});
  }

  graph_defs.AddGraphOutputs({all_gradients_finite_argdef.name});

  return Status::OK();
}

using GraphInitFn = std::function<Status(Graph&)>;

Status MakeGraphProto(GraphInitFn graph_init_fn, GraphProto& graph_proto) {
  Model model{"model"};
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
    GraphAugmenter::GraphDefs& graph_defs) {
  assert(weight_argdefs.size() == gradient_argdefs.size() &&
         weight_argdefs.size() == opt_configs.size());

  GraphProto then_subgraph_proto, else_subgraph_proto;

  // just use this same output ArgDef for parent graph and subgraphs
  const ArgDef conditional_output_argdef{
      nodearg_name_generator("conditional_output"),
      graph_defs.CreateTypeProto({}, ONNX_NAMESPACE::TensorProto_DataType_BOOL)};

  // condition == true
  ORT_RETURN_IF_ERROR(MakeGraphProto(
      [&opt_builder_registry, &weight_argdefs, &gradient_argdefs,
       &opt_configs, &graph_defs, &conditional_output_argdef](Graph& then_subgraph) {
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

        const size_t num_weights = weight_argdefs.size();
        for (size_t i = 0; i < num_weights; ++i) {
          auto opt_builder = opt_builder_registry.MakeUnique(opt_configs[i].name);
          ORT_RETURN_IF_NOT(
              opt_builder, "Failed to get Optimizer builder for ", opt_configs[i].name);

          std::vector<ArgDef> external_inputs_including_initializers{};
          std::vector<TensorProto> new_external_initializers{};
          ArgDef output_weight_argdef{};

          ORT_RETURN_IF_ERROR(opt_builder->Build(
              weight_argdefs[i], gradient_argdefs[i], /*do_update_argdef*/ nullptr, opt_configs[i],
              then_subgraph_defs, external_inputs_including_initializers,
              new_external_initializers, output_weight_argdef));

          group_input_argdefs.emplace_back(output_weight_argdef);

          for (const auto& external_input : external_inputs_including_initializers) {
            then_subgraph.AddOuterScopeNodeArg(external_input.name);
          }

          graph_defs.AddInitializers(new_external_initializers);
        }

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

  // deterministic ordering for debugging
  std::sort(weight_names_.begin(), weight_names_.end());

  opt_configs_.reserve(weight_names_.size());
  std::transform(
      weight_names_.begin(), weight_names_.end(), std::back_inserter(opt_configs_),
      [&weight_names_to_opt_configs](const std::string& weight_name) {
        return weight_names_to_opt_configs.at(weight_name);
      });
}

Status OptimizerGraphBuilder::Build(Graph& graph,
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

  // add grad accumulation
  std::vector<ArgDef> gradient_accumulation_buffers;
  if (is_gradient_accumulation_enabled) {
    ArgDef group_accumulate_gradient_output =
        AddGradientAccumulationNodes(nodearg_name_generator, gradient_argdefs, gradient_accumulation_buffers, graph_defs);
    optimizer_graph_outputs[kGradientAccumulationOutputKey] = group_accumulate_gradient_output.name;
  }

  // add gradient scaling
  if (is_gradient_accumulation_enabled || is_all_reduce_enabled) {
    const auto total_num_accumulations = opt_graph_config_.gradient_accumulation_steps * opt_graph_config_.world_size;
    ORT_RETURN_IF_NOT(total_num_accumulations > 0);
    const float scale = 1.0f / total_num_accumulations;
    ORT_RETURN_IF_ERROR(AddGradientScalingNodes(nodearg_name_generator, scale, gradient_argdefs, graph_defs, opt_graph_config_.allreduce_in_fp16));
  }

  // add all-reduce
  if (is_all_reduce_enabled) {
    ORT_RETURN_IF_ERROR(AddAllReduceForGradients(gradient_argdefs, graph_defs));
  }

  // check if all gradients are finite
  ArgDef all_grads_finite_argdef{};
  if (opt_graph_config_.use_mixed_precision) {
    ORT_RETURN_IF_ERROR(AddFiniteGradientChecks(
        nodearg_name_generator, gradient_argdefs, graph_defs, all_grads_finite_argdef));
    optimizer_graph_outputs[kGradientAllIsFiniteOutputKey] = all_grads_finite_argdef.name;
  }

  // add weight update
  std::vector<ArgDef> zero_gradients_control_signals;

  // TODO: enable conditional weight update for mixed precision when If op data copy issue is resolved
  if (false) {
    ArgDef conditional_weight_update_output{};
    ORT_RETURN_IF_ERROR(AddConditionalWeightUpdate(
        nodearg_name_generator, all_grads_finite_argdef, opt_builder_registry_,
        weight_argdefs, gradient_argdefs, opt_configs_,
        conditional_weight_update_output, graph_defs));
    // TODO not ideal to pass N copies of the same argdef as control signals
    //      maybe update AddZeroGradientNodes() to accept 1 or N of them
    zero_gradients_control_signals.assign(weight_argdefs.size(), conditional_weight_update_output);
  } else {
    ArgDef* is_all_finite_argdef = (opt_graph_config_.use_mixed_precision && !opt_graph_config_.always_do_update)
                                       ? &all_grads_finite_argdef
                                       : nullptr;
    ORT_RETURN_IF_ERROR(AddDirectWeightUpdate(
        opt_builder_registry_, weight_argdefs, gradient_argdefs, is_all_finite_argdef,
        opt_configs_, zero_gradients_control_signals, graph_defs));
  }

  // add zero gradient
  if (is_gradient_accumulation_enabled) {
    ORT_RETURN_IF_ERROR(AddZeroGradientNodes(
        nodearg_name_generator, zero_gradients_control_signals, gradient_accumulation_buffers, graph_defs));
  }

  // add learning rate inputs
  ORT_RETURN_IF_ERROR(AddLearningRateGraphInputs(graph, opt_configs_));

  ORT_RETURN_IF_ERROR(GraphAugmenter::AugmentGraph(graph, graph_defs));

  return Status::OK();
}

}  // namespace training
}  // namespace onnxruntime
