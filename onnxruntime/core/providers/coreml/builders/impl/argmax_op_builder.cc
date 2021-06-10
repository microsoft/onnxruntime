// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared/utils/utils.h"
#include "core/providers/coreml/builders/model_builder.h"
#include "core/providers/coreml/builders/op_builder_factory.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace coreml {

class ArgMaxOpBuilder : public BaseOpBuilder {
  // Add operator related
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

  // Operator support related
 private:
  bool IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                         const logging::Logger& logger) const override;
};

// Add operator related

Status ArgMaxOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                              const Node& node,
                                              const logging::Logger& /* logger */) const {
  std::unique_ptr<COREML_SPEC::NeuralNetworkLayer> layer = CreateNNLayer(node);
  const auto& graph_viewer = model_builder.GetGraphViewer();

  NodeAttrHelper helper(node);
  const auto axis = helper.Get("axis", 0);
  const auto keep_dims = helper.Get("keep_dims", 1);
  const bool removedim = keep_dims != 1;

  auto* coreml_argmax = layer->mutable_argmax();
  coreml_argmax->set_axis(axis);
  coreml_argmax->set_removedim(removedim);

  // Get ArgMax's next node(Cast)'s outputdefs
  auto it = node.OutputEdgesBegin();
  const auto* succ_node(graph_viewer.GetNode(it->GetNode().Index()));

  // Skip the cast's input/argmax's output
  *layer->mutable_input()->Add() = node.InputDefs()[0]->Name();
  *layer->mutable_output()->Add() = succ_node->OutputDefs()[0]->Name();

  model_builder.AddLayer(std::move(layer));
  return Status::OK();
}

// Operator support related

bool ArgMaxOpBuilder::IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                                        const logging::Logger& logger) const {
  // Check if Argmax's output is the graph output
  const auto& graph_output_list = input_params.graph_viewer.GetOutputs();
  std::unordered_set<const NodeArg*> graph_outputs(graph_output_list.cbegin(), graph_output_list.cend());
  const auto& output_defs = node.OutputDefs();
  for (const auto* output_def : output_defs) {
    if (graph_outputs.count(output_def) != 0) {
      LOGS(logger, VERBOSE) << "ArgMax not supported when it produces a graph output";
      return false;
    }
  }

  // Case where argmax has multiple succeeding nodes is not supported
  if (node.GetOutputEdgesCount() > 1) {
    LOGS(logger, VERBOSE) << "Multiple nodes consuming ArgMax's output";
    return false;
  }

  const auto& succ_node = node.OutputEdgesBegin()->GetNode();

  /*We're only handling the case: an ArgMax op followed by a Cast to int32 type right now so as to fuse
  the int64 output (not a supported output type by CoreML model) of ArgMax.*/
  if (succ_node.OpType() != "Cast") {
    LOGS(logger, VERBOSE) << "ArgMax not supported when next node is not [Cast]"
                          << "Current next node: [" << succ_node.OpType()
                          << "]";
    return false;
  }

  // Attribute `select_last_index` of ArgMax op is not supported
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
