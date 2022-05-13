// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <core/common/safeint.h>

#include "core/providers/shared/utils/utils.h"
#include "core/providers/coreml/builders/helper.h"
#ifdef __APPLE__
#include "core/providers/coreml/builders/model_builder.h"
#endif
#include "core/providers/coreml/builders/op_builder_factory.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace coreml {

class DepthToSpaceOpBuilder : public BaseOpBuilder {
  // Add operator related
 private:
#ifdef __APPLE__
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;
#endif

  // Operator support related
  bool IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                         const logging::Logger& logger) const override;
};

// Add operator related

#ifdef __APPLE__
Status DepthToSpaceOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                                    const Node& node,
                                                    const logging::Logger& /* logger */) const {
  std::unique_ptr<COREML_SPEC::NeuralNetworkLayer> layer = CreateNNLayer(model_builder, node);

  const auto& input_defs = node.InputDefs();
  const auto& output_defs = node.OutputDefs();
  const auto& input_name = input_defs[0]->Name();
  const auto& output_name = output_defs[0]->Name();

  uint64_t blocksize = SafeInt<uint64_t>(node.GetAttributes().at("blocksize").i());

  auto* coreml_depthtospace = layer->mutable_reorganizedata();
  coreml_depthtospace->set_blocksize(blocksize);
  coreml_depthtospace->set_mode(CoreML::Specification::ReorganizeDataLayerParams_ReorganizationType::
                                    ReorganizeDataLayerParams_ReorganizationType_DEPTH_TO_SPACE);

  *layer->mutable_input()->Add() = input_name;
  *layer->mutable_output()->Add() = output_name;

  model_builder.AddLayer(std::move(layer));
  return Status::OK();
}
#endif

// Operator support related

bool DepthToSpaceOpBuilder::IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& /*input_params*/,
                                              const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();

  std::vector<int64_t> input_shape;
  if (!GetShape(*input_defs[0], input_shape, logger)) {
    return false;
  }

  const auto input_size = input_shape.size();
  if (input_size != 4) {
    LOGS(logger, VERBOSE) << "DepthToSpace only supports 4d shape, input is " << input_size << "d shape.";
  }

  // CoreML spec ReorganizeDataLayer DEPTH_TO_SPACE mode only accepts input with one batch ([C, H, W]).
  if (input_shape[0] != 1) {
    LOGS(logger, VERBOSE) << "The batch size of DepthToSpace [" << input_shape[0] << "] is not supported.";
    return false;
  }

  NodeAttrHelper helper(node);
  if (node.SinceVersion() >= 11) {
    // For now, only DCR mode DepthToSpace is supported
    const auto mode = helper.Get("mode", "DCR");
    if (mode != "DCR") {
      LOGS(logger, VERBOSE) << "The mode: " << mode << "of DepthToSpace is not supported in CoreML EP for now.";
      return false;
    }
  }

  return true;
}

void CreateDepthToSpaceOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<DepthToSpaceOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace coreml
}  // namespace onnxruntime
