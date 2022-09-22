// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/providers/nnapi/nnapi_builtin/builders/model_builder.h"
#include "core/providers/nnapi/nnapi_builtin/builders/op_builder.h"
#include "core/providers/nnapi/nnapi_builtin/builders/op_support_checker.h"
#include "core/providers/nnapi/nnapi_builtin/builders/op_builder_factory.h"
#include "core/providers/shared/node_unit/node_unit.h"

namespace onnxruntime {

namespace common {
class Status;
}

namespace nnapi {

class ModelBuilder;

template <class T>
void CreateSharedOpBuilderImpl(const std::string& op_type,
                               OpBuilderRegistrations& op_registrations,
                               const std::vector<std::string>& op_types) {
  // The shared OpSupportChecker is already in the OpSupportCheckerRegistrations
  if (op_registrations.op_builder_map.find(op_type) != op_registrations.op_builder_map.cend())
    return;

  op_registrations.builders.push_back(std::make_unique<T>());
  for (const auto& op : op_types) {
    op_registrations.op_builder_map.emplace(op, op_registrations.builders.back().get());
  }
}

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
