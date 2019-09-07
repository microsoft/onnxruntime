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

namespace {
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

#ifdef USE_HOROVOD
const std::string global_barrier_name = "horovod/barrier";
const std::string global_barrier_ready = "horovod/barrier/ready";

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
  ArgDef reduce_output(gradient_name + "_AllReduce_Out");
  ArgDef reduce_ready(gradient_name + "_AllReduce_Ready");
  ArgDef local_barrier_output(gradient_name + "_Barrier_Out");
  ArgDef local_barrier_ready(gradient_name + "_Barrier_Ready");

  // Add horovod all reduce node.
  graph_defs.AddNodeDefs({NodeDef("HorovodAllReduce", {gradient_argdef}, {reduce_output, reduce_ready}, NodeAttributes(), gradient_name)});

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

Status AddAllReduceForGradients(
    std::vector<ArgDef>& gradient_argdefs,  // update argdefs in place
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
    const std::vector<OptimizerNodeConfig>& opt_configs,
    GraphAugmenter::GraphDefs& graph_defs) {
  assert(weight_argdefs.size() == gradient_argdefs.size() &&
         weight_argdefs.size() == opt_configs.size());

  const size_t num_weights = weight_argdefs.size();
  for (size_t i = 0; i < num_weights; ++i) {
    auto opt_builder = opt_builder_registry.MakeUnique(opt_configs[i].name);
    ORT_RETURN_IF_NOT(
        opt_builder, "Failed to get Optimizer builder for ", opt_configs[i].name);

    std::vector<ArgDef> inputs_including_initializers{};
    std::vector<TensorProto> new_initializers{};
    ArgDef output_weight_argdef{};

    ORT_RETURN_IF_ERROR(opt_builder->Build(
        weight_argdefs[i], gradient_argdefs[i], opt_configs[i],
        graph_defs, inputs_including_initializers, new_initializers, output_weight_argdef));

    graph_defs.AddInitializers(new_initializers);
  }

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
    const ArgDef& condition_argdef,
    const std::string& conditional_output_name,
    const OptimizerBuilderRegistry& opt_builder_registry,
    const std::vector<ArgDef>& weight_argdefs,
    const std::vector<ArgDef>& gradient_argdefs,
    const std::vector<OptimizerNodeConfig>& opt_configs,
    GraphAugmenter::GraphDefs& graph_defs) {
  assert(weight_argdefs.size() == gradient_argdefs.size() &&
         weight_argdefs.size() == opt_configs.size());

  GraphProto then_subgraph_proto, else_subgraph_proto;

  const TypeProto* const group_output_type_proto =
      graph_defs.CreateTypeProto({1}, ONNX_NAMESPACE::TensorProto_DataType_BOOL);
  // just use this same output ArgDef for parent graph and subgraphs
  const ArgDef conditional_output_argdef{conditional_output_name, group_output_type_proto};

  // condition == true
  ORT_RETURN_IF_ERROR(MakeGraphProto(
      [&condition_argdef, &opt_builder_registry, &weight_argdefs, &gradient_argdefs,
       &opt_configs, &graph_defs, &conditional_output_argdef](Graph& then_subgraph) {
        /* subgraph structure:
         * the idea is to minimize any copying incurred by subgraph outputs
         * condition is included as an input to group to ensure there is at least one
         *
         * (condition) ---|
         * optimizer 1 ---|
         * optimizer 2 ---|---> group ---> (subgraph output)
         * ...            |
         * optimizer N ---|
         */

        GraphAugmenter::GraphDefs then_subgraph_defs{};
        std::vector<ArgDef> group_input_argdefs{};

        group_input_argdefs.emplace_back(condition_argdef);
        then_subgraph.AddOuterScopeNodeArg(condition_argdef.name);

        const size_t num_weights = weight_argdefs.size();
        for (size_t i = 0; i < num_weights; ++i) {
          auto opt_builder = opt_builder_registry.MakeUnique(opt_configs[i].name);
          ORT_RETURN_IF_NOT(
              opt_builder, "Failed to get Optimizer builder for ", opt_configs[i].name);

          std::vector<ArgDef> external_inputs_including_initializers{};
          std::vector<TensorProto> new_external_initializers{};
          ArgDef output_weight_argdef{};

          ORT_RETURN_IF_ERROR(opt_builder->Build(
              weight_argdefs[i], gradient_argdefs[i], opt_configs[i],
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
      [&condition_argdef, &conditional_output_argdef](Graph& else_subgraph) {
        /* subgraph structure:
         *
         * (condition) ---> group ---> (subgraph output)
         */

        GraphAugmenter::GraphDefs else_subgraph_defs{};

        else_subgraph_defs.AddNodeDefs({NodeDef{"Group", {condition_argdef}, {conditional_output_argdef}}});

        else_subgraph_defs.AddGraphOutputs({conditional_output_argdef.name});

        else_subgraph.AddOuterScopeNodeArg(condition_argdef.name);

        ORT_RETURN_IF_ERROR(GraphAugmenter::AugmentGraph(else_subgraph, else_subgraph_defs));

        return Status::OK();
      },
      else_subgraph_proto));

  const std::vector<AttributeProto> conditional_attributes{
      MakeAttribute("then_branch", then_subgraph_proto),
      MakeAttribute("else_branch", else_subgraph_proto)};

  graph_defs.AddNodeDefs({NodeDef{"If", {condition_argdef}, {conditional_output_argdef}, conditional_attributes}});

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

Status OptimizerGraphBuilder::Build(Graph& graph) {
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

  // TODO grad accumulation

  if (opt_graph_config_.world_size > 1) {
    // all-reduce
    ORT_RETURN_IF_ERROR(AddAllReduceForGradients(gradient_argdefs, graph_defs));
  }

  // add weight update
  if (opt_graph_config_.use_mixed_precision) {
    // TODO check overflow

    // TODO hardcoded condition for now...
    ArgDef all_grads_finite_argdef{graph.GenerateNodeArgName("all_grads_finite")};
    graph_defs.AddInitializers({CreateTensorProto(all_grads_finite_argdef.name, true)});

    // TODO loss scale (maybe in opt_configs_?)

    ORT_RETURN_IF_ERROR(AddConditionalWeightUpdate(
        all_grads_finite_argdef, graph.GenerateNodeArgName("opt_conditional_out"),
        opt_builder_registry_, weight_argdefs, gradient_argdefs, opt_configs_,
        graph_defs));
  } else {
    ORT_RETURN_IF_ERROR(AddDirectWeightUpdate(
        opt_builder_registry_, weight_argdefs, gradient_argdefs, opt_configs_,
        graph_defs));
  }

  // TODO zero gradient

  ORT_RETURN_IF_ERROR(GraphAugmenter::AugmentGraph(graph, graph_defs));

  return Status::OK();
}

}  // namespace training
}  // namespace onnxruntime
