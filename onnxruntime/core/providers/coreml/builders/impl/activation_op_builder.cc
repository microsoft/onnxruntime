// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef __APPLE__
#include "core/providers/coreml/builders/impl/builder_utils.h"
#include "core/providers/coreml/builders/model_builder.h"
#endif
#include "core/providers/coreml/builders/helper.h"
#include "core/providers/coreml/builders/impl/base_op_builder.h"
#include "core/providers/coreml/builders/op_builder_factory.h"

namespace onnxruntime {
namespace coreml {

class ActivationOpBuilder : public BaseOpBuilder {
  // Add operator related
#ifdef __APPLE__
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;
#endif

  // Operator support related
 private:
  bool IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                         const logging::Logger& logger) const override;
  int GetMinSupportedOpSet(const Node& node) const override;
};

// Add operator related

#ifdef __APPLE__
void ActivationOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const {
  const auto& op_type = node.OpType();
  const auto& input_defs = node.InputDefs();
  if (op_type == "PRelu") {
    // skip slope as it's already embedded as a weight in the coreml layer
    model_builder.AddInitializerToSkip(input_defs[1]->Name());
  }
}

Status ActivationOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                                  const Node& node,
                                                  const logging::Logger& /* logger */) const {
  std::unique_ptr<COREML_SPEC::NeuralNetworkLayer> layer = CreateNNLayer(model_builder, node);

  const auto& op_type(node.OpType());
  if (op_type == "Sigmoid") {
    layer->mutable_activation()->mutable_sigmoid();
  } else if (op_type == "Tanh") {
    layer->mutable_activation()->mutable_tanh();
  } else if (op_type == "Relu") {
    layer->mutable_activation()->mutable_relu();
  } else if (op_type == "PRelu") {
    auto* prelu = layer->mutable_activation()->mutable_prelu();
    // add slope initializer as alpha weight
    const auto& slope_tensor = *model_builder.GetInitializerTensors().at(node.InputDefs()[1]->Name());
    ORT_RETURN_IF_ERROR(CreateCoreMLWeight(*prelu->mutable_alpha(), slope_tensor));
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "ActivationOpBuilder::AddToModelBuilderImpl, unknown op: ", op_type);
  }

  *layer->mutable_input()->Add() = node.InputDefs()[0]->Name();
  *layer->mutable_output()->Add() = node.OutputDefs()[0]->Name();

  model_builder.AddLayer(std::move(layer));
  return Status::OK();
}
#endif

// Operator support related

namespace {
// assumes that node.OpType() == "PRelu"
bool IsPReluOpSupported(const Node& node, const OpBuilderInputParams& input_params,
                        const logging::Logger& logger) {
  const auto& input_defs = node.InputDefs();

  // X input rank must be at least 3
  {
    std::vector<int64_t> x_shape;
    if (!GetShape(*input_defs[0], x_shape, logger)) {
      return false;
    }
    if (x_shape.size() < 3) {
      LOGS(logger, VERBOSE) << "PRelu 'X' input must have at least 3 dimensions";
      return false;
    }
  }

  // slope input must be an initializer
  {
    const auto& initializers = input_params.graph_viewer.GetAllInitializedTensors();
    const auto initializer_it = initializers.find(input_defs[1]->Name());
    if (initializer_it == initializers.end()) {
      LOGS(logger, VERBOSE) << "PRelu 'slope' input must be an initializer tensor";
      return false;
    }
  }

  // slope must either:
  // - have shape [C, 1, 1]
  // - have 1 element
  // TODO: CoreML crashes with single element slope, support it later when fixed
  // https://github.com/apple/coremltools/issues/1488
  {
    std::vector<int64_t> slope_shape;
    if (!GetShape(*input_defs[1], slope_shape, logger)) {
      return false;
    }
    const bool has_supported_slope_shape =
        (slope_shape.size() == 3 && std::all_of(slope_shape.begin() + 1, slope_shape.end(),
                                                [](int64_t dim) { return dim == 1; }))
        /* || Product(slope_shape) == 1 */;
    if (!has_supported_slope_shape) {
      LOGS(logger, VERBOSE) << "PRelu 'slope' input must have shape [C, 1, 1]" /*" or have a single value"*/;
      return false;
    }
  }

  return true;
}
}  // namespace

bool ActivationOpBuilder::IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                                            const logging::Logger& logger) const {
  const auto& op_type = node.OpType();
  if (op_type == "PRelu") {
    return IsPReluOpSupported(node, input_params, logger);
  }
  return true;
}

int ActivationOpBuilder::GetMinSupportedOpSet(const Node& /* node */) const {
  // All ops opset 5- uses consumed_inputs attribute which is not supported for now
  return 6;
}

void CreateActivationOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  if (op_registrations.op_builder_map.find(op_type) != op_registrations.op_builder_map.cend())
    return;

  static std::vector<std::string> op_types =
      {
          "Sigmoid",
          "Tanh",
          "Relu",
          "PRelu",
      };

  op_registrations.builders.push_back(std::make_unique<ActivationOpBuilder>());
  for (const auto& type : op_types) {
    op_registrations.op_builder_map.emplace(type, op_registrations.builders.back().get());
  }
}

}  // namespace coreml
}  // namespace onnxruntime
