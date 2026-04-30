// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/initializer.h"
#include "core/providers/coreml/builders/helper.h"
#include "core/providers/coreml/builders/impl/base_op_builder.h"
#include "core/providers/coreml/builders/impl/builder_utils.h"
#include "core/providers/coreml/builders/model_builder.h"
#include "core/providers/coreml/builders/op_builder_factory.h"
#include "core/providers/coreml/shape_utils.h"
#include "core/providers/shared/utils/utils.h"

namespace onnxruntime {
namespace coreml {

class TileOpBuilder : public BaseOpBuilder {
  void AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const override;

  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override;

  bool HasSupportedInputsImpl(const Node& node, const OpBuilderInputParams& input_params,
                              const logging::Logger& logger) const override;

  bool IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                         const logging::Logger& logger) const override;

  bool SupportsMLProgram() const override { return true; }
};

void TileOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const {
  // If 'repeats' is a constant initializer we bake it into the MIL constant
  // and don't need the original to land in the model. If it's a runtime
  // tensor the dynamic-shape MIL path consumes it directly.
  if (model_builder.GetConstantInitializer(node.InputDefs()[1]->Name())) {
    model_builder.AddInitializerToSkip(node.InputDefs()[1]->Name());
  }
}

Status TileOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                            const logging::Logger& /*logger*/) const {
  const auto& input_defs = node.InputDefs();
  const auto& output_def = *node.OutputDefs()[0];
  const auto* repeats_init = model_builder.GetConstantInitializer(input_defs[1]->Name());

  if (model_builder.CreateMLProgram()) {
    using namespace CoreML::Specification::MILSpec;
    auto op = model_builder.CreateOperation(node, "tile");
    AddOperationInput(*op, "x", input_defs[0]->Name());
    if (repeats_init) {
      Initializer unpacked(model_builder.GetGraphViewer().GetGraph(), *repeats_init);
      auto repeats = unpacked.DataAsSpan<int64_t>();
      AddOperationInput(*op, "reps", model_builder.AddConstant(op->type(), "reps", repeats));
    } else {
      // Runtime 'reps' (e.g. emitted by a Loop). Pass the tensor through.
      AddOperationInput(*op, "reps", input_defs[1]->Name());
    }
    AddOperationOutput(*op, output_def);
    model_builder.AddOperation(std::move(op));
  } else {
    if (!repeats_init) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "TileOpBuilder NeuralNetwork path requires constant 'repeats'");
    }
    Initializer unpacked(model_builder.GetGraphViewer().GetGraph(), *repeats_init);
    auto repeats = unpacked.DataAsSpan<int64_t>();
    auto layer = model_builder.CreateNNLayer(node);
    auto* tile_params = layer->mutable_tile();
    for (int64_t r : repeats) {
      tile_params->add_reps(r);
    }
    *layer->mutable_input()->Add() = input_defs[0]->Name();
    *layer->mutable_output()->Add() = output_def.Name();
    model_builder.AddLayer(std::move(layer));
  }
  return Status::OK();
}

bool TileOpBuilder::HasSupportedInputsImpl(const Node& node, const OpBuilderInputParams& input_params,
                                           const logging::Logger& logger) const {
  // Tile is shape-only data movement, so it can carry any element type CoreML
  // can represent. ONNX Tile is commonly used in graph post-processing on
  // INT32 grid-index tensors (e.g. YOLO anchor expansion), which the default
  // base check (float-only) would reject.
  int32_t input_type;
  if (!GetType(*node.InputDefs()[0], input_type, logger)) {
    return false;
  }
  switch (input_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:
    case ONNX_NAMESPACE::TensorProto_DataType_BOOL:
      return true;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      if (input_params.create_mlprogram && input_params.coreml_version >= 6) {
        return true;
      }
      [[fallthrough]];
    default:
      LOGS(logger, VERBOSE) << "[Tile] input type " << input_type << " is not supported";
      return false;
  }
}

bool TileOpBuilder::IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                                      const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();

  // The NeuralNetwork emitter only supports constant 'repeats'; the MLProgram
  // path also accepts a runtime 'reps' tensor.
  const auto& repeats_name = input_defs[1]->Name();
  const auto* repeats_tensor = input_params.graph_viewer.GetConstantInitializer(repeats_name);
  if (!input_params.create_mlprogram && !repeats_tensor) {
    LOGS(logger, VERBOSE) << "Tile NeuralNetwork path requires 'repeats' to be a constant initializer";
    return false;
  }

  std::vector<int64_t> input_shape;
  if (!GetShape(*input_defs[0], input_shape, logger)) {
    return false;
  }

  if (input_shape.size() > 5) {
    LOGS(logger, VERBOSE) << "Tile does not support input rank greater than 5. Input rank: " << input_shape.size();
    return false;
  }

  if (repeats_tensor) {
    Initializer unpacked(input_params.graph_viewer.GetGraph(), *repeats_tensor);
    auto repeats = unpacked.DataAsSpan<int64_t>();
    if (repeats.size() != input_shape.size()) {
      LOGS(logger, VERBOSE) << "Tile 'repeats' length (" << repeats.size()
                            << ") must match input rank (" << input_shape.size() << ")";
      return false;
    }
    for (int64_t r : repeats) {
      if (r < 1) {
        LOGS(logger, VERBOSE) << "Tile 'repeats' values must be positive; got " << r;
        return false;
      }
    }
  }

  return true;
}

void CreateTileOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<TileOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace coreml
}  // namespace onnxruntime
