// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/safeint.h"
#include "core/framework/tensorprotoutils.h"
#include "core/providers/common.h"
#include "core/providers/coreml/builders/impl/base_op_builder.h"
#include "core/providers/coreml/builders/model_builder.h"
#include "core/providers/coreml/builders/op_builder_factory.h"
#include "core/providers/shared/utils/utils.h"
#include "core/optimizer/initializer.h"

namespace onnxruntime {
namespace coreml {

class SqueezeOpBuilder : public BaseOpBuilder {
  void AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const override;

  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override;

  bool IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                         const logging::Logger& logger) const override;
};

namespace {
Status GetAxes(ModelBuilder& model_builder, const Node& node, std::vector<int64_t>& axes) {
  // Squeeze opset 13 use input as axes
  if (node.SinceVersion() > 12) {
    // If axes is not provided, return an empty axes as default to squeeze all
    if (node.InputDefs().size() > 1) {
      const auto& initializers(model_builder.GetInitializerTensors());
      const auto& axes_tensor = *initializers.at(node.InputDefs()[1]->Name());
      Initializer unpacked_tensor(axes_tensor);
      auto raw_axes = unpacked_tensor.DataAsSpan<int64_t>();
      const auto size = SafeInt<size_t>(axes_tensor.dims()[0]);
      axes.reserve(size);
      axes.insert(axes.end(), raw_axes.begin(), raw_axes.end());
    }
  } else {
    NodeAttrHelper helper(node);
    axes = helper.Get("axes", std::vector<int64_t>());
  }

  return Status::OK();
}
}  // namespace

void SqueezeOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const {
  if (node.SinceVersion() > 12 && node.InputDefs().size() > 1) {
    model_builder.AddInitializerToSkip(node.InputDefs()[1]->Name());
  }
}

Status SqueezeOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                               const Node& node,
                                               const logging::Logger& /* logger */) const {
  std::unique_ptr<COREML_SPEC::NeuralNetworkLayer> layer = model_builder.CreateNNLayer(node);

  auto* coreml_squeeze = layer->mutable_squeeze();
  std::vector<int64_t> axes;
  ORT_RETURN_IF_ERROR(GetAxes(model_builder, node, axes));
  if (axes.empty()) {
    coreml_squeeze->set_squeezeall(true);
  } else {
    *coreml_squeeze->mutable_axes() = {axes.cbegin(), axes.cend()};
    coreml_squeeze->set_squeezeall(false);
  }

  *layer->mutable_input()->Add() = node.InputDefs()[0]->Name();
  *layer->mutable_output()->Add() = node.OutputDefs()[0]->Name();

  model_builder.AddLayer(std::move(layer));
  return Status::OK();
}

bool SqueezeOpBuilder::IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                                         const logging::Logger& /*logger*/) const {
  // Squeeze opset 13 uses input 1 as axes, if we have input 1 then it needs to be an initializer
  const auto& initializers = input_params.graph_viewer.GetAllInitializedTensors();
  if (node.SinceVersion() > 12 && node.InputDefs().size() > 1) {
    const auto& axes_name = node.InputDefs()[1]->Name();
    if (!Contains(initializers, axes_name)) {
      LOGS_DEFAULT(VERBOSE) << "Input axes of Squeeze must be known";
      return false;
    }
  }

  return true;
}

void CreateSqueezeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<SqueezeOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace coreml
}  // namespace onnxruntime
