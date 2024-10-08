// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/safeint.h"
#include "core/framework/tensorprotoutils.h"
#include "core/optimizer/initializer.h"
#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/webnn/builders/helper.h"
#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace webnn {

class TileOpBuilder : public BaseOpBuilder {
  // Add operator related.
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

  // Operator support related.
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                         const WebnnDeviceType /* device_type */, const logging::Logger& logger) const override;
};

// Add operator related.

void TileOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const {
  model_builder.AddInitializerToSkip(node.InputDefs()[1]->Name());
}

Status TileOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                            const Node& node,
                                            const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const auto& initializers(model_builder.GetInitializerTensors());
  const auto& repetitions_initializer = *initializers.at(input_defs[1]->Name());
  const int64_t* raw_repetitions_data = repetitions_initializer.int64_data().empty()
                                            ? reinterpret_cast<const int64_t*>(repetitions_initializer.raw_data().data())
                                            : repetitions_initializer.int64_data().data();
  const auto size = repetitions_initializer.dims()[0];
  TensorShapeVector repetitions_data{raw_repetitions_data, raw_repetitions_data + size};
  emscripten::val input = model_builder.GetOperand(input_defs[0]->Name());
  std::vector<uint32_t> repetitions;
  std::transform(repetitions_data.cbegin(), repetitions_data.cend(),
                 std::back_inserter(repetitions),
                 [](int64_t repetition) -> uint32_t { return SafeInt<uint32_t>(repetition); });

  emscripten::val options = emscripten::val::object();
  options.set("label", node.Name());
  emscripten::val output = model_builder.GetBuilder().call<emscripten::val>("tile",
                                                                            input,
                                                                            emscripten::val::array(repetitions),
                                                                            options);
  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
  return Status::OK();
}

// Operator support related.

bool TileOpBuilder::IsOpSupportedImpl(const InitializedTensorSet& initializers,
                                      const Node& node,
                                      const WebnnDeviceType /* device_type */,
                                      const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const auto& repetitions_name = input_defs[1]->Name();
  if (!Contains(initializers, repetitions_name)) {
    LOGS(logger, VERBOSE) << "Repetitions of tile must be a constant initializer";
    return false;
  }

  std::vector<int64_t> input_shape;
  if (!GetShape(*input_defs[0], input_shape, logger))
    return false;

  if (input_shape.empty()) {
    LOGS(logger, VERBOSE) << "Tile does not support empty input shape";
    return false;
  }

  return true;
}

void CreateTileOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<TileOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime
