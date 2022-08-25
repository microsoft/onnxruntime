// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"

#include "core/providers/nnapi/nnapi_builtin/builders/model_builder.h"
#include "core/providers/nnapi/nnapi_builtin/builders/op_builder.h"
#include "core/providers/nnapi/nnapi_builtin/builders/op_support_checker.h"

namespace onnxruntime {

class Node;
class NodeUnit;

namespace common {
class Status;
}

namespace nnapi {

class ModelBuilder;

class BaseOpBuilder : public IOpBuilder {
 public:
  virtual ~BaseOpBuilder() = default;
  virtual void AddInitializersToSkip(ModelBuilder& /* model_builder */, const NodeUnit& /* node_unit */) const override {}
  Status AddToModelBuilder(ModelBuilder& model_builder, const NodeUnit& node_unit) const override final;

 protected:
  virtual Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const = 0;
  static bool IsOpSupported(const ModelBuilder& model_builder, const NodeUnit& node_unit) ORT_MUST_USE_RESULT;
  virtual bool IsQuantizedOp(const NodeUnit& /* node_unit */) const { return false; }
};

/* static */ bool BaseOpBuilder::IsOpSupported(const ModelBuilder& model_builder, const NodeUnit& node_unit) {
  OpSupportCheckParams params{
      model_builder.GetNNAPIFeatureLevel(),
      model_builder.UseNCHW(),
  };

  return IsNodeSupported(node_unit, model_builder.GetGraphViewer(), params);
}

Status BaseOpBuilder::AddToModelBuilder(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  ORT_RETURN_IF_NOT(IsOpSupported(model_builder, node_unit), "Unsupported operator ", node_unit.OpType());
  ORT_RETURN_IF_ERROR(AddToModelBuilderImpl(model_builder, node_unit));
  LOGS_DEFAULT(VERBOSE) << "Operator name: [" << node_unit.Name()
                        << "] type: [" << node_unit.OpType() << "] was added";
  return Status::OK();
}

}  // namespace nnapi
}  // namespace onnxruntime
