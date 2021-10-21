// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/graph/tensorboard_transformer.h"
#include "orttraining/core/graph/graph_augmenter.h"
#include "onnx/defs/attr_proto_util.h"

using namespace onnxruntime::common;

namespace onnxruntime {
namespace training {

Status TransformGraphForTensorboard(Graph& graph,
                                    const std::string& summary_name,
                                    const std::vector<std::string>& scalar_nodes,
                                    const std::vector<std::string>& histogram_nodes,
                                    const std::vector<std::string>& norm_nodes,
                                    const bool dump_convergence_metrics) {
  std::vector<ArgDef> summary_args;
  std::vector<NodeDef> new_nodes;

  // SummaryScalar nodes.
  for (const std::string& scalar_input : scalar_nodes) {
    std::string scalar_output = summary_name + "/scalar/" + scalar_input;
    summary_args.push_back(ArgDef(scalar_output));
    new_nodes.emplace_back(NodeDef(OpDef{"SummaryScalar", kMSDomain, 1},
                                   {ArgDef(scalar_input)},
                                   {ArgDef(scalar_output)},
                                   {ONNX_NAMESPACE::MakeAttribute("tags", std::vector<std::string>{scalar_output})},
                                   graph.GenerateNodeName(scalar_output)));
  }

  // SummaryHistogram nodes.
  for (const std::string& histogram_input : histogram_nodes) {
    std::string histogram_output = summary_name + "/histogram/" + histogram_input;
    summary_args.push_back(ArgDef(histogram_output));
    new_nodes.emplace_back(NodeDef(OpDef("SummaryHistogram", kMSDomain, 1),
                                   {ArgDef(histogram_input)},
                                   {ArgDef(histogram_output)},
                                   {ONNX_NAMESPACE::MakeAttribute("tag", histogram_output)},
                                   graph.GenerateNodeName(histogram_output)));
  }

  // SummaryScalar nodes for norms.
  for (const std::string& norm_input : norm_nodes) {
    const auto &node_arg = graph.GetNodeArg(norm_input);
    if (node_arg == nullptr) {
      continue;
    }

    std::vector<int64_t> axes;
    for (int i = 0; i < node_arg->Shape()->dim_size(); ++i) {
      axes.push_back(i);
    }

    std::vector<ONNX_NAMESPACE::AttributeProto> attribute_protos;
    attribute_protos.push_back(ONNX_NAMESPACE::MakeAttribute("keepdims", int64_t(0)));
    attribute_protos.push_back(ONNX_NAMESPACE::MakeAttribute("axes", axes));

    std::string norm_output = graph.GenerateNodeArgName(summary_name + "/L2-norm/" + norm_input);
    new_nodes.emplace_back(NodeDef("ReduceL2",
                                   {ArgDef(norm_input)},
                                   {ArgDef(norm_output)},
                                   attribute_protos,
                                   graph.GenerateNodeName(norm_output)));

    std::string scalar_output = graph.GenerateNodeArgName(summary_name + "/scalar/L2-norm/" + norm_input);
    summary_args.push_back(ArgDef(scalar_output));
    new_nodes.emplace_back(NodeDef(OpDef("SummaryScalar", kMSDomain, 1),
                                   {ArgDef(norm_output)},
                                   {ArgDef(scalar_output)},
                                   {ONNX_NAMESPACE::MakeAttribute("tags", std::vector<std::string>{scalar_output})},
                                   graph.GenerateNodeName(scalar_output)));
  }

  // If user wants to output gradient norm.
  if (dump_convergence_metrics) {
    auto initializer_set = graph.GetAllInitializedTensors();
    std::vector<ArgDef> squared_grad_sum_arg_defs;
    for (const auto& pair: initializer_set) {
      auto name = pair.first;
      auto grad_node = graph.GetNodeArg(name + "_grad");

      // Gradient node doesn't exist.
      if (grad_node == nullptr) {
        continue;
      }

      std::vector<int64_t> axes;
      for (int i = 0; i < grad_node->Shape()->dim_size(); ++i) {
        axes.push_back(i);
      }
      std::vector<ONNX_NAMESPACE::AttributeProto> attribute_protos;
      attribute_protos.push_back(ONNX_NAMESPACE::MakeAttribute("keepdims", int64_t(0)));
      attribute_protos.push_back(ONNX_NAMESPACE::MakeAttribute("axes", axes));
      std::string squared_sum_name = graph.GenerateNodeArgName(summary_name + grad_node->Name() + "/squared_sum");
      std::string squared_sum_node_name = graph.GenerateNodeName(squared_sum_name);
      squared_grad_sum_arg_defs.push_back(ArgDef(squared_sum_name)),
      new_nodes.emplace_back(NodeDef("ReduceSumSquare",
                                    {ArgDef(grad_node->Name())},
                                    {ArgDef(squared_sum_name)},
                                    attribute_protos,
                                    squared_sum_node_name));
    }

    std::string total_squared_gradient_sum = graph.GenerateNodeArgName(summary_name + "/total_squared_gradient_sum");
    std::string sum_node_name = graph.GenerateNodeName(total_squared_gradient_sum);
    new_nodes.emplace_back(NodeDef("Sum",
                                  squared_grad_sum_arg_defs,
                                  {ArgDef(total_squared_gradient_sum)},
                                  NodeAttributes(),
                                  sum_node_name));

    std::string total_gradient_norm = graph.GenerateNodeArgName(summary_name + "/total_gradient_norm");
    std::string sqrt_node_name = graph.GenerateNodeName(total_gradient_norm);
    new_nodes.emplace_back(NodeDef("Sqrt",
                                  {ArgDef(total_squared_gradient_sum)},
                                  {ArgDef(total_gradient_norm)},
                                  NodeAttributes(),
                                  sqrt_node_name));

    std::string total_gradient_norm_scalar_output = graph.GenerateNodeArgName(summary_name + "/tb_total_gradient_norm");
    std::string summary_scalar_node_name = graph.GenerateNodeName(total_gradient_norm_scalar_output);
    summary_args.push_back(ArgDef(total_gradient_norm_scalar_output));
    new_nodes.emplace_back(NodeDef(OpDef("SummaryScalar", kMSDomain, 1),
                                   {ArgDef(total_gradient_norm)},
                                   {ArgDef(total_gradient_norm_scalar_output)},
                                   {ONNX_NAMESPACE::MakeAttribute("tags", std::vector<std::string>{total_gradient_norm_scalar_output})},
                                   summary_scalar_node_name));
  }

  // SummaryMerge (if any tensorboard nodes exist).
  if (summary_args.size() > 0) {
    new_nodes.emplace_back(NodeDef(OpDef("SummaryMerge", kMSDomain, 1),
                                    summary_args,             // Inputs
                                    {ArgDef(summary_name)}, // Outputs
                                    NodeAttributes(),
                                    summary_name));

    // Modify graph.
    GraphAugmenter::GraphDefs graph_defs;
    graph_defs.AddNodeDefs(new_nodes);
    graph_defs.AddGraphOutputs({summary_name});
    return GraphAugmenter::AugmentGraph(graph, graph_defs);
  }

  return Status::OK();
}

}  // namespace training
}  // namespace onnxruntime
