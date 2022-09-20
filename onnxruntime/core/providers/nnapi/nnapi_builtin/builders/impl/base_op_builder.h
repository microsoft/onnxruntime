// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/providers/nnapi/nnapi_builtin/builders/model_builder.h"
#include "core/providers/nnapi/nnapi_builtin/builders/op_builder.h"
#include "core/providers/nnapi/nnapi_builtin/builders/op_support_checker.h"
#include "core/providers/shared/node_unit/node_unit.h"

namespace onnxruntime {

namespace common {
class Status;
}

namespace nnapi {

class ModelBuilder;

class BaseOpBuilder : public IOpBuilder {
 public:
  virtual ~BaseOpBuilder() = default;

  // Add operator related
 public:
  virtual void AddInitializersToSkip(ModelBuilder& /* model_builder */,
                                     const NodeUnit& /* node_unit */) const override {}

  Status AddToModelBuilder(ModelBuilder& model_builder, const NodeUnit& node_unit) const override final;

 protected:
  virtual Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const = 0;

  static bool IsOpSupported(const ModelBuilder& model_builder, const NodeUnit& node_unit);

  virtual bool IsQuantizedOp(const NodeUnit& /* node_unit */) const { return false; }
};

}  // namespace nnapi
}  // namespace onnxruntime
