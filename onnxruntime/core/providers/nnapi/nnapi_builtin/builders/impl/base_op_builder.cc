// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nnapi/nnapi_builtin/builders/impl/base_op_builder.h"

namespace onnxruntime {
namespace nnapi {

// Add operator related

Status BaseOpBuilder::AddToModelBuilder(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  ORT_RETURN_IF_NOT(IsOpSupported(model_builder, node_unit), "Unsupported operator ", node_unit.OpType());
  ORT_RETURN_IF_ERROR(AddToModelBuilderImpl(model_builder, node_unit));
  LOGS_DEFAULT(VERBOSE) << "Operator name: [" << node_unit.Name()
                        << "] type: [" << node_unit.OpType() << "] was added";
  return Status::OK();
}

/* static */ bool BaseOpBuilder::IsOpSupported(const ModelBuilder& model_builder, const NodeUnit& node_unit) {
  OpSupportCheckParams params{
      model_builder.GetNNAPIFeatureLevel(),
      model_builder.UseNCHW(),
  };

  return IsNodeSupported(node_unit, model_builder.GetGraphViewer(), params);
}

}  // namespace nnapi
}  // namespace onnxruntime
