// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/shared/node_unit/node_unit.h"
#include "core/providers/nnapi/nnapi_builtin/builders/model_builder.h"
#include "core/providers/nnapi/nnapi_builtin/builders/op_builder.h"
#include "core/providers/nnapi/nnapi_builtin/builders/op_builder_factory.h"

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
  static bool IsOpSupported(const ModelBuilder& model_builder, const NodeUnit& node_unit);

 protected:
  virtual Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const = 0;

  virtual bool IsQuantizedOp(const NodeUnit& /* node_unit */) const { return false; }

  // Operator support related
 public:
  bool IsOpSupported(const GraphViewer& graph_viewer, const NodeUnit& node_unit,
                     const OpSupportCheckParams& params) const override;

 protected:
  virtual bool IsOpSupportedImpl(const GraphViewer& /* graph_viewer */, const NodeUnit& /* node_unit */,
                                 const OpSupportCheckParams& /* params */) const {
    return true;
  }

  virtual int32_t GetMinSupportedNNAPIFeatureLevel(const NodeUnit& /* node_unit */,
                                                   const OpSupportCheckParams& /* params */) const {
    // ANEURALNETWORKS_FEATURE_LEVEL_1 is the baseline version of NNAPI,
    // There is no NNAPI support for Android API level 26-
    return ANEURALNETWORKS_FEATURE_LEVEL_1;
  }

  virtual bool HasSupportedInputOutputsImpl(const GraphViewer& graph_viewer, const NodeUnit& node_unit,
                                            const OpSupportCheckParams& params) const;

  virtual int GetMinSupportedOpSet(const NodeUnit& /* node_unit */) const { return 1; }
  virtual int GetMaxSupportedOpSet(const NodeUnit& /* node_unit */) const { return 19; }

  // Check if this node_unit's type is supported
  // SingleNode type NodeUnit is supported
  // QDQGroup type NodeUnit is by default unsupported, and this can be individually overwritten by inherited classes
  virtual bool IsNodeUnitTypeSupported(const NodeUnit& node_unit) const;

 private:
  bool HasSupportedOpSet(const NodeUnit& node_unit) const;
  bool HasSupportedInputOutputs(const GraphViewer& graph_viewer, const NodeUnit& node_unit,
                                const OpSupportCheckParams& params) const;
};

}  // namespace nnapi
}  // namespace onnxruntime
