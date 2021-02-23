// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/compiler/traverse_shape_infer.h"

#include "core/codegen/common/common.h"
#include "core/common/common.h"
#include "core/framework/tensorprotoutils.h"

// TODO retire this file

namespace onnxruntime {
namespace nuphar {

// local shape infernece function for input
static bool CreateInput(const NodeArg* def,
                        const GraphViewer& graph,
                        ShapeExpr& input,
                        bool initializer_only) {
  if (initializer_only && graph.GetAllInitializedTensors().count(def->Name()) == 0)
    return false;

  auto def_shape = def->Shape();
  if (!def_shape)
    return false;

  int rank = def_shape->dim_size();
  input = ShapeExpr(rank);
  for (int i = 0; i < rank; ++i) {
    const auto& dim = def_shape->dim()[i];
    if (utils::HasDimValue(dim))
      input[i] = DimExpr(dim.dim_value());
    else if (utils::HasDimParam(dim))
      input[i] = DimExpr(dim.dim_param());
    else {
      input[i] = DimExpr(NormalizeNodeArgName(def) + "_dim" + std::to_string(i));
    }
  }
  return true;
}

// local shape infernece function for output
static Status CreateOutputs(
    const Node* node,
    const std::vector<const ShapeExpr*>& inputs,
    std::vector<ShapeExpr>& outputs) {
  outputs.resize(node->OutputDefs().size());
  node->ForEachWithIndex(
      node->OutputDefs(),
      [&](const NodeArg& def, size_t index) {
        auto shape_proto = def.Shape();
        if (shape_proto) {
          TensorShape shape{utils::GetTensorShapeFromTensorShapeProto(*shape_proto)};
          ShapeExpr output_shape(shape.NumDimensions());
          for (int d = 0; d < gsl::narrow<int>(shape.NumDimensions()); ++d) {
            if (shape[d] > 0) {
              output_shape[d] = DimExpr(shape[d]);
            } else {
              ORT_RETURN_IF_NOT(shape_proto->dim_size() > d && utils::HasDimParam(shape_proto->dim(d)),
                                "shape_proto->dim_size() > d && utils::HasDimParam(shape_proto->dim(d) was false");
              output_shape[d] = DimExpr(shape_proto->dim(d).dim_param());
            }
          }
          outputs[index] = output_shape;
        }
        return Status::OK();
      });
  return Status::OK();
}

// The main function for shape infernece
Status ShapeInference(
    const GraphViewer& graph,
    ShapeExprContext& context) {
  // build graph inputs
  const auto& graph_inputs = graph.GetInputs();
  for (size_t i = 0; i < graph_inputs.size(); ++i) {
    ShapeExpr value;
    if (CreateInput(graph_inputs[i], graph, value, /*initializer_only*/ false)) {
      context.inputs.emplace(graph_inputs[i]->Name(), std::move(value));
    }
  }

  // perform shape inference using the topological order from ORT
  for (const NodeIndex& node_index : graph.GetNodesInTopologicalOrder()) {
    const Node* p_node = graph.GetNode(node_index);
    if(p_node == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "invalid node index");
    const Node& node = *p_node;
    // initializers
    node.ForEachWithIndex(
        node.InputDefs(),
        [&graph, &context](const NodeArg& def, size_t) {
          ShapeExpr value;
          if (CreateInput(&def, graph, value, /*initializer_only*/ true)) {
            context.inputs.emplace(def.Name(), std::move(value));
          }
          return Status::OK();
        });

    // handle subgraph
    const Graph* subgraph = GetSubgraph(node);
    if (nullptr != subgraph) {
      GraphViewer subgraph_viewer(*subgraph);
      ShapeInference(subgraph_viewer, context);
    }

    // collect inputs before creating outputs
    std::vector<const ShapeExpr*> inputs;
    for (const NodeArg* def : node.InputDefs()) {
      inputs.push_back(def->Exists() ? context.Lookup(def) : nullptr);
    }

    // create outputs
    std::vector<ShapeExpr> op_outputs;
    ORT_RETURN_IF_ERROR(CreateOutputs(&node, inputs, op_outputs));
    context.ops.emplace(&node, std::move(op_outputs));

    // recall input_from_
    node.ForEachWithIndex(
        node.OutputDefs(),
        [&node, &context](const NodeArg& def, size_t index) {
          context.input_from.emplace(def.Name(), std::make_pair(&node, index));
          return Status::OK();
        });
  }

  return Status::OK();
}

}  // namespace nuphar
}  // namespace onnxruntime
