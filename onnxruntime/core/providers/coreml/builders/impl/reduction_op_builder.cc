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
};

namespace {
template <typename T>
void AddReductionParams(T* params, std::vector<int64_t>& axes, bool keepdims) {
  params->set_keepdims(keepdims);
  for (auto& axis : axes) {
    params->add_axes(axis);
  }
}
} // namespace

#ifdef __APPLE__
Status ReductionOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                                 const logging::Logger& /* logger */) const {
  const auto& op_type(node.OpType());
  const auto& input_defs(node.InputDefs());

  NodeAttrHelper helper(node);
  auto axes = helper.Get("axes", std::vector<int64_t>{});
  auto keepdims = helper.Get("keepdims", false);

  std::unique_ptr<COREML_SPEC::NeuralNetworkLayer> layer = CreateNNLayer(model_builder, node);

  if (op_type == "ReduceSum") {
    AddReductionParams(layer->mutable_reducesum(), axes, keepdims);
  } else if (op_type == "ReduceMean") {
    AddReductionParams(layer->mutable_reducemean(), axes, keepdims);
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

void CreateReductionOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<ReductionOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

} // coreml
} // onnxruntime