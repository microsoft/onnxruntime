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
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

  // Operator support related.
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                         const WebnnDeviceType /* device_type */, const logging::Logger& logger) const override;
};


Status TriangularOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                               const Node& node,
                                               const logging::Logger& logger) const {
    const auto& input_defs = node.InputDefs();
    emscripten::val input = model_builder.GetOperand(node.InputDefs()[0]->Name());
    emscripten::val output = emscripten::val::object();
    const auto& initializers = model_builder.GetInitializerTensors();
    NodeAttrHelper helper(node);


    emscripten::val options = emscripten::val::object();

    if (!GetTensorName(input_defs, 1).empty()) {
      // Optional input axes is provided, use axes initializer data.
      const auto diagonal_tensor = *initializers.at(input_defs[1]->Name());
      emscripten::val diagonal = emscripten::val::object();
      ORT_RETURN_IF_NOT(ReadScalarTensorData(diagonal_tensor, diagonal, logger), "Cannot read diagonal value");
      options.set("diagonal", diagonal);
    }
    else {
      options.set("diagonal", 0);
    }

    int32_t upper = helper.Get("upper", 1);
    options.set("upper", upper);

    output = model_builder.GetBuilder().call<emscripten::val>("triangular", input, options);

    model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
    return Status::OK();
}

// Operator support related.
bool TriangularOpBuilder::IsOpSupportedImpl(const InitializedTensorSet& /* initializers */,
                                         const Node& node,
                                         const WebnnDeviceType /* device_type */,
                                         const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  std::vector<int64_t> input_shape;
  if (!GetShape(*input_defs[0], input_shape, logger))
    return false;
  const auto input_size = input_shape.size();
  if (input_size < 2) {
    LOGS(logger, VERBOSE) << "Triangular only support input size >= 2d shape, input is "
                          << input_size << "d shape";
    return false;
  }

  return true;
}

void CreateTriangularOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<TriangularOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime
