// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/initializer.h"
#include "core/providers/common.h"
#include "core/providers/coreml/builders/helper.h"
#include "core/providers/coreml/builders/impl/base_op_builder.h"
#include "core/providers/coreml/builders/impl/builder_utils.h"
#include "core/providers/coreml/builders/model_builder.h"
#include "core/providers/coreml/builders/op_builder_factory.h"
#include "core/providers/coreml/shape_utils.h"
#include "core/providers/shared/utils/utils.h"

namespace onnxruntime {
namespace coreml {

class SplitOpBuilder : public BaseOpBuilder {
  void AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const override;

  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override;

  bool IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                         const logging::Logger& logger) const override;

  // Split opset 13- uses "split" as attribute. Currently it's not supported.
  int GetMinSupportedOpSet(const Node& /* node */) const override { return 13; }

  bool SupportsMLProgram() const override { return true; }
};

void SplitOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const {
  const auto& input_defs = node.InputDefs();

  if (input_defs.size() > 1 && input_defs[1]->Exists()) {  // optional second input "split"
    model_builder.AddInitializerToSkip(input_defs[1]->Name());
  }
}

Status SplitOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                             const Node& node,
                                             const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();

  std::vector<int64_t> data_shape;
  ORT_RETURN_IF_NOT(GetShape(*node.InputDefs()[0], data_shape, logger), "Failed to get input shape.");

  NodeAttrHelper helper(node);
  int64_t axis = helper.Get("axis", 0);

  auto calculate_remainder_and_chunk_size = [&](int32_t num_outputs) {
    // note: checked in IsOpSupportedImpl that ensures the dim value at splitting axis exists
    auto split_dim_size = data_shape[HandleNegativeAxis(axis, data_shape.size())];
    int64_t chunk_size = (split_dim_size + num_outputs - 1) / num_outputs;
    int64_t remainder = split_dim_size % chunk_size;
    return std::make_tuple(remainder, chunk_size);
  };

#if defined(COREML_ENABLE_MLPROGRAM)
  if (model_builder.CreateMLProgram()) {
    using namespace CoreML::Specification::MILSpec;
    std::unique_ptr<Operation> split_op = model_builder.CreateOperation(node, "split");
    AddOperationInput(*split_op, "axis", model_builder.AddScalarConstant(split_op->type(), "axis", axis));

    if (input_defs.size() > 1) {
      // if "split" is explicitly provided as an input
      Initializer unpacked_tensor(*model_builder.GetConstantInitializer(input_defs[1]->Name()));
      auto split_span = unpacked_tensor.DataAsSpan<int64_t>();
      AddOperationInput(*split_op, "split_sizes",
                        model_builder.AddConstant(split_op->type(), "split_sizes", split_span));
    } else if (node.SinceVersion() < 18) {
      int64_t num_outputs = narrow<int64_t>(node.OutputDefs().size());
      AddOperationInput(*split_op, "num_splits",
                        model_builder.AddScalarConstant(split_op->type(), "num_splits", num_outputs));
    } else {
      // note: for opset 18+ 'num_outputs' is a required attribute
      int64_t num_outputs = helper.GetInt64("num_outputs").value();
      auto [remainder, chunk_size] = calculate_remainder_and_chunk_size(static_cast<int32_t>(num_outputs));
      if (remainder) {
        // uneven
        std::vector<int64_t> split_sizes(num_outputs, chunk_size);
        split_sizes.back() = remainder;
        AddOperationInput(*split_op, "split_sizes",
                          model_builder.AddConstant(split_op->type(), "split_sizes", split_sizes));
      } else {
        // even
        AddOperationInput(*split_op, "num_splits",
                          model_builder.AddScalarConstant(split_op->type(), "num_splits", num_outputs));
      }
    }

    AddOperationInput(*split_op, "x", input_defs[0]->Name());
    for (const auto& output_def : node.OutputDefs()) {
      AddOperationOutput(*split_op, *output_def);
    }
    model_builder.AddOperation(std::move(split_op));

  } else
#endif
  {
    std::unique_ptr<COREML_SPEC::NeuralNetworkLayer> layer = model_builder.CreateNNLayer(node);
    auto* coreml_splitnd = layer->mutable_splitnd();
    coreml_splitnd->set_axis(axis);

    if (input_defs.size() > 1) {
      // if "split" is explicitly provided as an input
      // const auto& split_tensor = *model_builder.GetInitializerTensors().at(input_defs[1]->Name());
      Initializer unpacked_tensor(*model_builder.GetConstantInitializer(input_defs[1]->Name()));
      auto split_span = unpacked_tensor.DataAsSpan<int64_t>();
      for (const auto& split_size : split_span) {
        coreml_splitnd->add_splitsizes(split_size);
      }
    } else if (node.SinceVersion() < 18) {
      int64_t num_outputs = narrow<int64_t>(node.OutputDefs().size());
      coreml_splitnd->set_numsplits(num_outputs);
    } else {
      // note: for opset 18+ 'num_outputs' is a required attribute
      int64_t num_outputs = narrow<int64_t>(helper.GetInt64("num_outputs").value());
      auto [remainder, chunk_size] = calculate_remainder_and_chunk_size(static_cast<int32_t>(num_outputs));
      if (remainder) {
        // uneven
        auto split_sizes = InlinedVector<int64_t>(num_outputs, chunk_size);
        split_sizes.back() = remainder;
        for (size_t i = 0; i < split_sizes.size(); i++) {
          coreml_splitnd->add_splitsizes(split_sizes[i]);
        }
      } else {
        // even
        coreml_splitnd->set_numsplits(num_outputs);
      }
    }

    *layer->mutable_input()->Add() = node.InputDefs()[0]->Name();
    // variadic number of outputs. Calculated based on the length of the given splitSizes if provided.
    // Otherwise, uses attribute value 'num_outputs'.
    for (const auto& output_def : node.OutputDefs()) {
      *layer->mutable_output()->Add() = output_def->Name();
    }
    model_builder.AddLayer(std::move(layer));
  }

  return Status::OK();
}

bool SplitOpBuilder::IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                                       const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();

  NodeAttrHelper helper(node);
  const auto axis = helper.Get("axis", 0);

  std::vector<int64_t> input_shape;
  if (!GetShape(*input_defs[0], input_shape, logger))
    return false;

  const auto split_dims_at_axis = input_shape[HandleNegativeAxis(axis, input_shape.size())];
  if (input_defs.size() > 1 && input_defs[1]->Exists()) {
    const auto* splits_tensor = input_params.graph_viewer.GetConstantInitializer(input_defs[1]->Name());
    if (!splits_tensor) {
      LOGS(logger, VERBOSE) << "CoreML 'splits' input must be a constant initializer.";
      return false;
    }

    const auto split_shape = *input_defs[1]->Shape();
    if (split_shape.dim(0).dim_value() < 2) {
      LOGS(logger, VERBOSE) << "CoreML Split must produce at least 2 outputs.";
      return false;
    }

    Initializer unpacked_tensor(*splits_tensor);
    auto splits_span = unpacked_tensor.DataAsSpan<int64_t>();
    int64_t sum_of_splits = std::accumulate(splits_span.begin(), splits_span.end(), int64_t{0});
    if (sum_of_splits != split_dims_at_axis) {
      LOGS(logger, VERBOSE) << "Mismatch between the sum of 'split'. Expected: "
                            << split_dims_at_axis
                            << "Actual: "
                            << sum_of_splits;
      return false;
    }
    auto it = std::find(splits_span.begin(), splits_span.end(), 0);
    if (it != splits_span.end()) {
      LOGS(logger, VERBOSE) << "Invalid value in 'splits' input.";
      return false;
    }
    if (split_dims_at_axis == -1) {
      LOGS(logger, VERBOSE) << "Dim at the splitting axis is not allowed to be dynamic.";
      return false;
    }
  } else {
    if (node.SinceVersion() >= 18) {
      const auto num_outputs = helper.GetInt64("num_outputs");
      if (!num_outputs.has_value()) {
        LOGS(logger, VERBOSE) << "No 'num_outputs' provided. For split 18+, num_outputs is a required attribute.";
        return false;
      }
      if (num_outputs.value() < 2) {
        LOGS(logger, VERBOSE) << "Invalid num_outputs. The value cannot be lower than 2.\n"
                              << "CoreML SplitND requires at least 2 outputs. num_outputs: " << num_outputs.value();
        return false;
      }
      if (num_outputs.value() != static_cast<int32_t>(node.OutputDefs().size()) ||
          num_outputs.value() > split_dims_at_axis) {
        LOGS(logger, VERBOSE) << "Invalid num_outputs provided.\n. The value should be smaller or equal to the size "
                                 "of dimension being split. num_outputs: "
                              << num_outputs.value();
        return false;
      }
    }
  }
  return true;
}

void CreateSplitOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<SplitOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace coreml
}  // namespace onnxruntime
