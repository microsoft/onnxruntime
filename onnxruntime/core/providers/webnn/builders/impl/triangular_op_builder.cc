// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/webnn/builders/helper.h"
#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace webnn {

class TriangularOpBuilder : public BaseOpBuilder {
  // Add operator related.
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

  // Operator support related.
 private:
  bool IsOpSupportedImpl(const GraphViewer& graph_viewer, const Node& node,
                         const WebnnDeviceType /* device_type */, const logging::Logger& logger) const override;
};

// Add operator related.

void TriangularOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const {
  // Skip diagonal initializer if present.
  if (node.InputDefs().size() > 1) {
    model_builder.AddInitializerToSkip(node.InputDefs()[1]->Name());
  }
}

Status TriangularOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                                  const Node& node,
                                                  const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const auto& initializers = model_builder.GetInitializerTensors();
  emscripten::val input = model_builder.GetOperand(input_defs[0]->Name());
  emscripten::val output = emscripten::val::object();
  NodeAttrHelper helper(node);
  emscripten::val options = emscripten::val::object();
  options.set("label", node.Name());

  const bool upper = helper.Get("upper", 1);
  options.set("upper", upper);

  const std::string diagonal_name = GetTensorName(input_defs, 1);
  if (!diagonal_name.empty()) {
    // Optional input diagonal is provided, use diagonal initializer data.
    const auto diagonal_tensor = *initializers.at(diagonal_name);

    std::vector<uint8_t> unpacked_tensor;
    ORT_RETURN_IF_ERROR(onnxruntime::utils::UnpackInitializerData(diagonal_tensor, unpacked_tensor));
    const auto diagonal = *reinterpret_cast<int64_t*>(unpacked_tensor.data());
    options.set("diagonal", SafeInt<int32_t>(diagonal).Ref());
  }

  output = model_builder.GetBuilder().call<emscripten::val>("triangular", input, options);

  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
  return Status::OK();
}

// Operator support related.
bool TriangularOpBuilder::IsOpSupportedImpl(const GraphViewer& graph_viewer,
                                            const Node& node,
                                            const WebnnDeviceType /* device_type */,
                                            const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  std::vector<int64_t> input_shape;
  if (!GetShape(*input_defs[0], input_shape, logger))
    return false;
  const auto input_size = input_shape.size();
  if (input_size < 2) {
    LOGS(logger, VERBOSE) << "Triangular only supports input size >= 2D shape, input is "
                          << input_size << "d shape";
    return false;
  }

  const std::string diagonal_name = GetTensorName(input_defs, 1);
  // Inputs contain optional 'diagonal' input.
  if (!diagonal_name.empty()) {
    const auto* init = graph_viewer.GetConstantInitializer(diagonal_name);
    if (!init) {
      LOGS(logger, VERBOSE) << "The diagonal must be a constant initializer.";
      return false;
    }
  }
  return true;
}

void CreateTriangularOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<TriangularOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime
