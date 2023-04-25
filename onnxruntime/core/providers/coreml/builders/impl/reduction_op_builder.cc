// Copyright (c) Shukant Pal.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"

#ifdef __APPLE__
#include "core/providers/coreml/builders/model_builder.h"
#endif
#include "core/providers/coreml/builders/helper.h"
#include "core/providers/coreml/builders/op_builder_factory.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace coreml {

class ReductionOpBuilder : public BaseOpBuilder {
 private:
#ifdef __APPLE__
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override;
#endif

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
} // namespace

#ifdef __APPLE__
Status ReductionOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                                 const logging::Logger& /* logger */) const {
  const auto& op_type(node.OpType());
  const auto& input_defs(node.InputDefs());
  const auto& initializers(model_builder.GetInitializerTensors());

  std::vector<int64_t> axes;

  NodeAttrHelper helper(node);
  if (input_defs.size() > 1) {
    auto& axes_tensor = *initializers.at(input_defs[1]->Name());
    const int64_t* raw_axes = axes_tensor.int64_data().empty()
      ? reinterpret_cast<const int64_t*>(axes_tensor.raw_data().data())
      : axes_tensor.int64_data().data();
    const auto size = axes_tensor.dims()[0];
    axes = std::vector<int64_t>(raw_axes, raw_axes + size);
  } else {
    axes = helper.Get("axes", std::vector<int64_t>{});
  }
  auto keepdims = helper.Get("keepdims", false);
  auto noop_with_empty_axes = helper.Get("noop_with_empty_axes", 0);

  std::unique_ptr<COREML_SPEC::NeuralNetworkLayer> layer = CreateNNLayer(model_builder, node);

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
#endif

bool ReductionOpBuilder::IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                                           const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();

  if (input_defs.size() == 1) {
    NodeAttrHelper helper(node);

    if (!helper.HasAttr("axes")) {
      LOGS(logger, VERBOSE) << "Axes of reduction must be an attribute if not present in inputs";
      return false;
    }

    return true;
  }

  const auto& axes_name = input_defs[1]->Name();
  const auto& initializers = input_params.graph_viewer.GetAllInitializedTensors();
  if (!Contains(initializers, axes_name)) {
    LOGS(logger, VERBOSE) << "Axes of reduction must be a constant initializer";
    return false;
  }

  return true;
}

void CreateReductionOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<ReductionOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

} // coreml
} // onnxruntime