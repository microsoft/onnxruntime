// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef __APPLE__
#include "core/framework/tensorprotoutils.h"
#include "core/providers/coreml/builders/impl/builder_utils.h"
#include "core/providers/coreml/builders/model_builder.h"
#endif
#include "core/common/narrow.h"
#include "core/providers/common.h"
#include "core/providers/coreml/builders/helper.h"
#include "core/providers/coreml/builders/impl/base_op_builder.h"
#include "core/providers/coreml/builders/op_builder_factory.h"
#include "core/providers/shared/utils/utils.h"
#include "core/optimizer/initializer.h"

namespace onnxruntime {
namespace coreml {

class ActivationOpBuilder : public BaseOpBuilder {
  // Add operator related
#ifdef __APPLE__
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const override;

 private:
  [[nodiscard]] Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                             const logging::Logger& logger) const override;
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

namespace {
Status AddPReluWeight(ModelBuilder& model_builder, const Node& node,
                      const logging::Logger& logger,
                      COREML_SPEC::ActivationPReLU& prelu) {
  // add slope initializer as alpha weight
  const auto& slope_tensor = *model_builder.GetInitializerTensors().at(node.InputDefs()[1]->Name());
  const auto slope_tensor_num_elements = narrow<size_t>(Product(slope_tensor.dims()));
  if (slope_tensor_num_elements != 1) {
    ORT_RETURN_IF_ERROR(CreateCoreMLWeight(*prelu.mutable_alpha(), slope_tensor));
  } else {
    // TODO: CoreML crashes with single element slope, hence this special case. Remove when fixed.
    // https://github.com/apple/coremltools/issues/1488

    // "broadcast" single value by creating a CoreML weight with num_channels copies of it
    ORT_RETURN_IF_NOT(slope_tensor.data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
                      "slope initializer has unsupported data type: ", slope_tensor.data_type());

    std::vector<int64_t> x_shape;
    ORT_RETURN_IF_NOT(GetShape(*node.InputDefs()[0], x_shape, logger), "Failed to get shape of X.");

    // assume X has 3 or 4 dimensions, that was checked in IsPReluOpSupported()
    const auto num_channels = x_shape[x_shape.size() - 3];

    Initializer unpacked_tensor(slope_tensor);
    float value = unpacked_tensor.DataAsSpan<float>()[0];

    auto& weight_values = *prelu.mutable_alpha()->mutable_floatvalue();
    weight_values.Clear();
    weight_values.Resize(narrow<int>(num_channels), value);
  }
  return Status::OK();
}
}  // namespace

Status ActivationOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                                  const Node& node,
                                                  const logging::Logger& logger) const {
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
    ORT_RETURN_IF_ERROR(AddPReluWeight(model_builder, node, logger, *prelu));
  } else if (op_type == "LeakyRelu") {
    NodeAttrHelper helper(node);
    const auto alpha = helper.Get("alpha", 0.01f);

    auto* leaky_relu = layer->mutable_activation()->mutable_leakyrelu();
    leaky_relu->set_alpha(alpha);
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

  // X input rank must be 3 or 4
  std::vector<int64_t> x_shape;
  if (!GetShape(*input_defs[0], x_shape, logger)) {
    return false;
  }

  const auto x_rank = x_shape.size();
  if (x_rank != 3 && x_rank != 4) {
    LOGS(logger, VERBOSE) << "PRelu 'X' input must have 3 or 4 dimensions, it has " << x_rank << " dimensions";
    return false;
  }

  // slope input must be a constant initializer
  if (!input_params.graph_viewer.IsConstantInitializer(input_defs[1]->Name(), true)) {
    LOGS(logger, VERBOSE) << "PRelu 'slope' input must be a constant initializer tensor";
    return false;
  }

  // slope must either:
  // - have shape [C, 1, 1]
  // - have 1 element
  {
    std::vector<int64_t> slope_shape;
    if (!GetShape(*input_defs[1], slope_shape, logger)) {
      return false;
    }
    const bool has_per_channel_slopes =
        (slope_shape.size() == 3 && std::all_of(slope_shape.begin() + 1, slope_shape.end(),
                                                [](int64_t dim) { return dim == 1; }));
    const bool has_single_slope = Product(slope_shape) == 1;
    if (!has_per_channel_slopes && !has_single_slope) {
      LOGS(logger, VERBOSE) << "PRelu 'slope' input must either have shape [C, 1, 1] or have a single value";
      return false;
    }

    if (has_single_slope && x_shape[x_rank - 3] == 1) {
      // TODO: CoreML crashes with single element slope, hence this special case. Remove when fixed.
      // https://github.com/apple/coremltools/issues/1488
      LOGS(logger, VERBOSE) << "PRelu single 'slope' value in CoreML weight is not supported";
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
          "LeakyRelu",
      };

  op_registrations.builders.push_back(std::make_unique<ActivationOpBuilder>());
  for (const auto& type : op_types) {
    op_registrations.op_builder_map.emplace(type, op_registrations.builders.back().get());
  }
}

}  // namespace coreml
}  // namespace onnxruntime
