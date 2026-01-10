// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/coreml/builders/helper.h"
#include "core/providers/coreml/builders/impl/base_op_builder.h"
#include "core/providers/coreml/builders/model_builder.h"
#include "core/providers/coreml/builders/op_builder_factory.h"
#include "core/providers/coreml/shape_utils.h"
#include "core/providers/shared/utils/utils.h"

namespace onnxruntime {
namespace coreml {

class LRNOpBuilder : public BaseOpBuilder {
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override;

  bool IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                         const logging::Logger& logger) const override;
};

Status LRNOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                           const Node& node,
                                           const logging::Logger& /*logger*/) const {
  std::unique_ptr<COREML_SPEC::NeuralNetworkLayer> layer = model_builder.CreateNNLayer(node);

  auto* coreml_lrn = layer->mutable_lrn();

  NodeAttrHelper helper(node);
  const auto alpha = helper.Get("alpha", 0.0001f);
  const auto beta = helper.Get("beta", 0.75f);
  const auto bias = helper.Get("bias", 1.0f);  // k
  const auto size = helper.Get("size", 1);     // localSize

  coreml_lrn->set_alpha(alpha);
  coreml_lrn->set_beta(beta);
  coreml_lrn->set_localsize(size);
  coreml_lrn->set_k(bias);

  *layer->mutable_input()->Add() = node.InputDefs()[0]->Name();
  *layer->mutable_output()->Add() = node.OutputDefs()[0]->Name();

  model_builder.AddLayer(std::move(layer));
  return Status::OK();
}

bool LRNOpBuilder::IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& /*input_params*/,
                                     const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();

  std::vector<int64_t> input_shape;
  if (!GetShape(*input_defs[0], input_shape, logger))
    return false;

  if (input_shape.empty()) {
    LOGS(logger, VERBOSE) << "LRN does not support empty input shape";
    return false;
  }

  // Note: For higher ranks ( > 3), CoreML LRN treats all leading dimensions as the batch,
  // which differs from ONNX LRN. Only support the case - input rank equals 3 or 4 here.
  // CoreML Spec:https://apple.github.io/coremltools/mlmodel/Format/NeuralNetwork.html#lrnlayerparams
  const auto input_rank = input_shape.size();
  if (input_rank != 3 && input_rank != 4) {
    LOGS(logger, VERBOSE) << "LRN only supports input rank equals to 3 or 4, input rank is "
                          << input_rank;
    return false;
  }

  return true;
}

void CreateLRNOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<LRNOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace coreml
}  // namespace onnxruntime
