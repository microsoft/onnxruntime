// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <core/common/safeint.h>

#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/coreml/builders/model_builder.h"
#include "core/providers/coreml/builders/op_builder_factory.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace coreml {

class ArgMaxOpBuilder : public BaseOpBuilder {
  // Add operator related
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node, const GraphViewer& graph_viewer,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

  // Operator support related
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node, const GraphViewer& graph_viewer,
                         const logging::Logger& logger) const override;
};

// Add operator related

Status ArgMaxOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                              const Node& node,
                                              const GraphViewer& graph_viewer,
                                              const logging::Logger& /*logger*/) const {
  std::unique_ptr<COREML_SPEC::NeuralNetworkLayer> layer = CreateNNLayer(node);

  NodeAttrHelper helper(node);
  const auto axis = helper.Get("axis", 0);
  const auto keep_dims = helper.Get("keep_dims", 1);
  bool removedim = false;
  if (keep_dims != 1) {
    removedim = true;
  }

  auto* coreml_argmax = layer->mutable_argmax();
  coreml_argmax->set_axis(axis);
  coreml_argmax->set_removedim(removedim);

  // Get ArgMax's next node(Cast)'s outputdefs
  const auto& node_indices = graph_viewer.GetNodesInTopologicalOrder();
  auto it = node.InputEdgesBegin();
  const auto* succ_node(graph_viewer.GetNode(node_indices[it->GetNode().Index()]));

  *layer->mutable_input()->Add() = node.InputDefs()[0]->Name();
  *layer->mutable_input()->Add() = succ_node->OutputDefs()[0]->Name();

  model_builder.AddLayer(std::move(layer));
  return Status::OK();
}

// Operator support related

bool ArgMaxOpBuilder::IsOpSupportedImpl(const InitializedTensorSet& /* initializers */, const Node& node,
                                        const GraphViewer& graph_viewer, const logging::Logger& logger) const {
  // Check if Argmax's output is the graph output
  const auto& graph_output_list = graph_viewer.GetOutputs();
  std::unordered_set<const NodeArg*> graph_outputs(graph_output_list.cbegin(), graph_output_list.cend());
  const auto& output_defs = node.OutputDefs();
  for (const auto* output_def : output_defs) {
    if (graph_outputs.count(output_def) != 0) {
      LOGS(logger, VERBOSE) << "Case: ArgMax's output is the graph output: Not supported.";
      return false;
    }
  }

  const auto& node_indices = graph_viewer.GetNodesInTopologicalOrder();
  std::vector<size_t> succ_node_indices;
  for (size_t i = 0; i < node_indices.size(); i++) {
    const auto* curr_node(graph_viewer.GetNode(node_indices[i]));
    if (curr_node->OpType() == "ArgMax") {
      for (auto it = curr_node->OutputEdgesBegin(), end = curr_node->OutputEdgesEnd(); it != end; ++it) {
        succ_node_indices.push_back(it->GetNode().Index());
      }
    }
  }
   //Case where argmax has multiple successive nodes is not supported
  if (succ_node_indices.size() > 1) {
    LOGS(logger, VERBOSE) << "Case - [ArgMax] has multiple sucessive nodes: Not supported.";
    return false;
  }
  
  if (succ_node_indices.empty()) {
    LOGS(logger, VERBOSE) << "Case - [ArgMax] has no sucessive nodes: Not supported.";
    return false;
  }

  const auto* succ_node(graph_viewer.GetNode(node_indices[succ_node_indices[0]]));

  // Case where argmax's successive node is not "cast" is not supported
  if (succ_node->OpType() != "Cast") {
    LOGS(logger, VERBOSE) << "Case - [ArgMax]'s next node is not [Cast]: Not supported. "
                          << "Current next node: [" << succ_node->OpType()
                          << "]";
    return false;
  }

  // Check if the output type of cast node is int32
  const auto& succ_node_output = *succ_node->OutputDefs()[0];
  int32_t succ_node_output_type;
  if (!GetType(succ_node_output, succ_node_output_type, logger)) {
    return false;
  }
  if (succ_node_output_type != ONNX_NAMESPACE::TensorProto_DataType_INT32) {
    LOGS(logger, VERBOSE) << "[" << succ_node->OpType()
                          << "] Output type: [" << succ_node_output_type
                          << "] is not supported for now";
    return false;
  }

  //Attribute `select_last_index` of ArgMax op is not supported
  NodeAttrHelper helper(node);
  const auto select_last_index = helper.Get("select_last_index", 0);
  if (select_last_index != 0) {
    LOGS(logger, VERBOSE) << "selected_last_index for ArgMax is not supported";
    return false;
  }

  return true;
}

void CreateArgMaxOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<ArgMaxOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace coreml
}  // namespace onnxruntime
