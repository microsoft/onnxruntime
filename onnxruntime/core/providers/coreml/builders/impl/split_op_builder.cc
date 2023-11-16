// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/coreml/builders/impl/base_op_builder.h"

#include "core/optimizer/initializer.h"
#include "core/providers/common.h"
#include "core/providers/coreml/builders/helper.h"
#include "core/providers/coreml/builders/op_builder_factory.h"
#include "core/providers/coreml/shape_utils.h"
#include "core/providers/shared/utils/utils.h"

#if defined(__APPLE__)
#include "core/providers/coreml/builders/model_builder.h"
#endif

namespace onnxruntime {
namespace coreml {

class SplitOpBuilder : public BaseOpBuilder {
  // Add operator related
#ifdef __APPLE__
 private:
  void AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override;
#endif

  // Operator support related
 private:
  bool IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                         const logging::Logger& logger) const override;

  // Split opset 13- uses "split" as attribute. Currently it's not supported.
  int GetMinSupportedOpSet(const Node& /* node */) const override { return 13; }
};

// Add operator related

#ifdef __APPLE__

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
  ORT_RETURN_IF_NOT(GetStaticShape(*node.InputDefs()[0], data_shape, logger), "Failed to get input shape.");

  NodeAttrHelper helper(node);
  const auto axis = helper.Get("axis", 0);

  // attribute introduced since opset 18
  uint64_t num_outputs = 2;

  std::unique_ptr<COREML_SPEC::NeuralNetworkLayer> layer = CreateNNLayer(model_builder, node);
  auto* coreml_splitnd = layer->mutable_splitnd();
  coreml_splitnd->set_axis(axis);

  if (input_defs.size() > 1) {
    // if "split" is explicitly provided as an input
    const auto& split_tensor = *model_builder.GetInitializerTensors().at(input_defs[1]->Name());
    Initializer unpacked_tensor(split_tensor);
    auto split_span = unpacked_tensor.DataAsSpan<int64_t>();
    auto split_sizes = split_span.size();
    num_outputs = SafeInt<uint64_t>(split_sizes);
    for (size_t i = 0; i < split_sizes; i++) {
      coreml_splitnd->add_splitsizes(SafeInt<uint64_t>(split_span[i]));
    }
  } else if (node.SinceVersion() < 18) {
    num_outputs = node.OutputDefs().size();
    coreml_splitnd->set_numsplits(num_outputs);
  } else {
    num_outputs = SafeInt<uint64_t>(helper.Get("num_outputs", -1));
    auto split_dim_size = data_shape[HandleNegativeAxis(axis, data_shape.size())];
    uint64_t chunk_size = narrow<uint64_t>(std::ceil(float(split_dim_size) / num_outputs));
    uint64_t remainder = split_dim_size % chunk_size;
    if (remainder) {
      // uneven
      auto split_sizes = std::vector<uint64_t>(num_outputs, chunk_size);
      split_sizes.back() = remainder;
      for (size_t i = 0; i < split_sizes.size(); i++) {
        coreml_splitnd->add_splitsizes(SafeInt<uint64_t>(split_sizes[i]));
      }
    } else {
      // even
      num_outputs = node.OutputDefs().size();
      coreml_splitnd->set_numsplits(num_outputs);
    }
  }

  *layer->mutable_input()->Add() = node.InputDefs()[0]->Name();
  // variadic number of outputs. Calculated based on the length of the given splitSizes if provided.
  // Otherwise, uses attribute value 'num_outputs'.
  for (uint64_t i = 0; i < num_outputs; i++) {
    *layer->mutable_output()->Add() = node.OutputDefs()[i]->Name();
  }
  model_builder.AddLayer(std::move(layer));

  return Status::OK();
}

#endif

// Operator support related

bool SplitOpBuilder::IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                                       const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const auto& initializers = input_params.graph_viewer.GetAllInitializedTensors();

  NodeAttrHelper helper(node);
  const auto axis = helper.Get("axis", 0);
  const auto num_outputs = helper.Get("num_outputs", -1);

  std::vector<int64_t> input_shape;
  if (!GetStaticShape(*input_defs[0], input_shape, logger))
    return false;

  const auto split_dims_at_axis = input_shape[HandleNegativeAxis(axis, input_shape.size())];
  if (input_defs.size() > 1 && input_defs[1]->Exists()) {
    if (!CheckIsConstantInitializer(*input_defs[1], input_params.graph_viewer, logger, "'split'")) {
      return false;
    }
    const auto split_shape = *input_defs[1]->Shape();
    if (split_shape.dim_size() < 2) {
      LOGS(logger, VERBOSE) << "CoreML SplitND requires to produce at least 2 outputs.";
      return false;
    }
    const auto& splits_tensor = *initializers.at(input_defs[1]->Name());
    Initializer unpacked_tensor(splits_tensor);
    auto splits_span = unpacked_tensor.DataAsSpan<int64_t>();
    int sum_of_splits = std::accumulate(splits_span.begin(), splits_span.end(), 0);
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
  } else {
    if (node.SinceVersion() >= 18) {
      if (num_outputs < 2) {
        LOGS(logger, VERBOSE) << "Invalid num_outputs. The value can not be lower than 1.\n"
                              << "CoreML SplitND requires at least 2 outputs. num_outputs: " << num_outputs;
        return false;
      }
      if (num_outputs != static_cast<int32_t>(node.OutputDefs().size()) || num_outputs > split_dims_at_axis) {
        LOGS(logger, VERBOSE) << "Invalid num_outputs provided.\n."
                              << "The value should be smaller or equal to the size of dimension being split. num_outputs: "
                              << num_outputs;
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
