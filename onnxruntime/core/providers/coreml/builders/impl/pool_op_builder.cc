// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared/utils/utils.h"
#include "core/providers/coreml/builders/helper.h"
#include "core/providers/coreml/builders/model_builder.h"
#include "core/providers/coreml/builders/op_builder_factory.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace coreml {

class PoolOpBuilder : public BaseOpBuilder {
  // Add operator related
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

  // Operator support related
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                         const logging::Logger& logger) const override;
};

// Add operator related

Status PoolOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                            const Node& node,
                                            const logging::Logger& /* logger */) const {
  std::unique_ptr<COREML_SPEC::NeuralNetworkLayer> layer = CreateNNLayer(node);

  auto* coreml_pool = layer->mutable_pooling();
  const auto& op_type = node.OpType();

  // We only support global pool now
  coreml_pool->set_globalpooling(true);
  coreml_pool->mutable_valid();

  if (op_type == "GlobalAveragePool") {
    coreml_pool->set_type(COREML_SPEC::PoolingLayerParams_PoolingType_AVERAGE);
  } else if (op_type == "GlobalMaxPool") {
    coreml_pool->set_type(COREML_SPEC::PoolingLayerParams_PoolingType_MAX);
  }

  *layer->mutable_input()->Add() = node.InputDefs()[0]->Name();
  *layer->mutable_output()->Add() = node.OutputDefs()[0]->Name();

  model_builder.AddLayer(layer.release());
  return Status::OK();
}

// Operator support related
bool PoolOpBuilder::IsOpSupportedImpl(const InitializedTensorSet& /* initializers */, const Node& node,
                                      const logging::Logger& logger) const {
  const auto& op_type = node.OpType();
  std::vector<int64_t> input_shape;
  if (!GetShape(*node.InputDefs()[0], input_shape, logger))
    return false;

  const auto input_size = input_shape.size();
  if (input_size != 4) {
    LOGS(logger, VERBOSE)
        << op_type << " only supports rank-4 tensor, input ["
        << node.InputDefs()[0]->Name() << "] has actual dim count " << input_size;
    return false;
  }

  return true;
}

void CreatePoolOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  if (op_registrations.op_builder_map.find(op_type) != op_registrations.op_builder_map.cend())
    return;

  static std::vector<std::string> op_types =
      {
          "GlobalAveragePool",
          "GlobalMaxPool",
      };

  op_registrations.builders.push_back(onnxruntime::make_unique<PoolOpBuilder>());
  for (const auto& op_type : op_types) {
    op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
  }
}

}  // namespace coreml
}  // namespace onnxruntime
