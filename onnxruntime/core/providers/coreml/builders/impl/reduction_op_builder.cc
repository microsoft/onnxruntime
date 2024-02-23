// Copyright (c) Shukant Pal.
// Licensed under the MIT License.

#include "core/optimizer/initializer.h"
#include "core/providers/common.h"
#include "core/providers/coreml/builders/helper.h"
#include "core/providers/coreml/builders/impl/base_op_builder.h"
#include "core/providers/coreml/builders/model_builder.h"
#include "core/providers/coreml/builders/op_builder_factory.h"
#include "core/providers/shared/utils/utils.h"

namespace onnxruntime {
namespace coreml {

class ReductionOpBuilder : public BaseOpBuilder {
  void AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const override;

  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override;

  bool IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                         const logging::Logger& logger) const override;
};

namespace {
template <typename T>
void AddReductionParams(T* params, const std::vector<int64_t>& axes, bool keepdims, bool noop_with_empty_axes) {
  params->set_keepdims(keepdims);

  for (auto& axis : axes)
    params->add_axes(axis);

  if (axes.size() == 0 && !noop_with_empty_axes)
    params->set_reduceall(true);
}
}  // namespace

void ReductionOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const {
  const auto& input_defs(node.InputDefs());

  // We have already embedded the axes into the CoreML layer.
  // No need to copy them later to reduce memory consumption.
  if (input_defs.size() > 1)
    model_builder.AddInitializerToSkip(input_defs[1]->Name());
}

Status ReductionOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                                 const logging::Logger& /* logger */) const {
  const auto& op_type(node.OpType());
  const auto& input_defs(node.InputDefs());
  const auto& initializers(model_builder.GetInitializerTensors());

  std::vector<int64_t> axes;

  NodeAttrHelper helper(node);
  if (input_defs.size() > 1 && input_defs[1]->Exists()) {
    auto& axes_tensor = *initializers.at(input_defs[1]->Name());
    Initializer axes_initializer(axes_tensor);
    int64_t* data = axes_initializer.data<int64_t>();
    int64_t size = axes_initializer.size();

    axes = std::vector<int64_t>(data, data + size);
  } else if (helper.HasAttr("axes")) {
    axes = helper.Get("axes", std::vector<int64_t>{});
  }

  const bool keepdims = helper.Get("keepdims", 1) != 0;
  const bool noop_with_empty_axes = helper.Get("noop_with_empty_axes", 0) != 0;

  std::unique_ptr<COREML_SPEC::NeuralNetworkLayer> layer = model_builder.CreateNNLayer(node);

  if (op_type == "ReduceSum") {
    AddReductionParams(layer->mutable_reducesum(), axes, keepdims, noop_with_empty_axes);
  } else if (op_type == "ReduceMean") {
    AddReductionParams(layer->mutable_reducemean(), axes, keepdims, noop_with_empty_axes);
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "ReductionOpBuilder::AddToModelBuilderImpl, unknown op: ", op_type);
  }

  *layer->mutable_input()->Add() = node.InputDefs()[0]->Name();
  *layer->mutable_output()->Add() = node.OutputDefs()[0]->Name();

  model_builder.AddLayer(std::move(layer));
  return Status::OK();
}

bool ReductionOpBuilder::IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                                           const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();

  if (input_defs.size() > 1 && input_defs[1]->Exists()) {
    const auto& axes_name = input_defs[1]->Name();
    const auto& initializers = input_params.graph_viewer.GetAllInitializedTensors();
    if (!Contains(initializers, axes_name)) {
      LOGS(logger, VERBOSE) << "Axes of reduction must be a constant initializer";
      return false;
    }

    NodeAttrHelper helper(node);

    if (initializers.at(axes_name)->int64_data_size() == 0 && helper.Get("noop_with_empty_axes", 0) != 0) {
      LOGS(logger, VERBOSE) << "CoreML doesn't support noop on empty axes for reduction layers" << std::endl;
      return false;
    }
  }

  return true;
}

void CreateReductionOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<ReductionOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace coreml
}  // namespace onnxruntime
