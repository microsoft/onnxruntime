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

  // TODO: 1. Special Case 2. Otherwise
  if (node.GetOutputEdgesCount() == 1) {
    auto it = node.OutputEdgesBegin();
    const auto* succ_node(graph_viewer.GetNode(it->GetNode().Index()));
    if (succ_node->OpType() == "Cast") {
      // Skip the cast's input/argmax's output
      *layer->mutable_input()->Add() = node.InputDefs()[0]->Name();
      *layer->mutable_output()->Add() = succ_node->OutputDefs()[0]->Name();
      model_builder.AddLayer(std::move(layer));
      return Status::OK();
    } 
  }

  *layer->mutable_input()->Add() = node.InputDefs()[0]->Name();
  *layer->mutable_output()->Add() = node.OutputDefs()[0]->Name();

  model_builder.AddLayer(std::move(layer));
  return Status::OK();
}

// Operator support related

bool ArgMaxOpBuilder::IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& /*input_params*/,
                                        const logging::Logger& logger) const {
  
  // Attribute `select_last_index` of ArgMax op is not supported
  NodeAttrHelper helper(node);
  const auto select_last_index = helper.Get("select_last_index", 0);
  if (select_last_index != 0) {
    LOGS(logger, VERBOSE) << "selected_last_index for ArgMax is not supported";
    return false;
  }

  // Case where argmax has multiple succeeding nodes(cast node among them) is not supported
  if (node.GetOutputEdgesCount() > 1) {
    // TODO: Check if the succeeding nodes contains cast
    // If Yes: Then not supported
    // Otherwise: supported
    for (auto it = node.OutputEdgesBegin(), end = node.OutputEdgesEnd(); it != end; ++it) {
      const auto& op_type = it->GetNode().OpType();
      if (op_type == "Cast") {
        LOGS(logger, VERBOSE) << "ArgMax has multiple output nodes including Cast";
        return false;
      }
    }
  }

  return true;
}

void CreateArgMaxOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<ArgMaxOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace coreml
}  // namespace onnxruntime
