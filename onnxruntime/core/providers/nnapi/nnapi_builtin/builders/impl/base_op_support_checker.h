// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/nnapi/nnapi_builtin/builders/op_support_checker.h"
#include "core/providers/nnapi/nnapi_builtin/builders/op_support_checker_factory.h"

namespace onnxruntime {

class NodeUnit;

namespace nnapi {

template <class T>
void CreateSharedOpSupportCheckerImpl(const std::string& op_type,
                                      OpSupportCheckerRegistrations& op_registrations,
                                      const std::vector<std::string>& op_types) {
  // The shared OpSupportChecker is already in the OpSupportCheckerRegistrations
  if (op_registrations.op_support_checker_map.find(op_type) != op_registrations.op_support_checker_map.cend())
    return;

  op_registrations.support_checkers.push_back(std::make_unique<T>());
  for (const auto& op : op_types) {
    op_registrations.op_support_checker_map.emplace(op, op_registrations.support_checkers.back().get());
  }
}

class BaseOpSupportChecker : public IOpSupportChecker {
 public:
  virtual ~BaseOpSupportChecker() = default;

  // Operator support related

  bool IsOpSupported(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                     const OpSupportCheckParams& params) const override;

 protected:
  virtual bool IsOpSupportedImpl(const InitializedTensorSet& /* initializers */, const NodeUnit& /* node_unit */,
                                 const OpSupportCheckParams& /* params */) const {
    return true;
  }

  virtual bool IsQuantizedOp(const NodeUnit& /* node_unit */) const { return false; }

  virtual int32_t GetMinSupportedNNAPIFeatureLevel(const NodeUnit& /* node_unit */,
                                                   const OpSupportCheckParams& /* params */) const {
    // ANEURALNETWORKS_FEATURE_LEVEL_1 is the baseline version of NNAPI,
    // There is no NNAPI support for Android API level 26-
    return ANEURALNETWORKS_FEATURE_LEVEL_1;
  }

  virtual bool HasSupportedInputOutputsImpl(
      const InitializedTensorSet& initializers, const NodeUnit& node_unit,
      const OpSupportCheckParams& params) const;

  virtual int GetMinSupportedOpSet(const NodeUnit& /* node_unit */) const { return 1; }
  virtual int GetMaxSupportedOpSet(const NodeUnit& /* node_unit */) const { return 15; }

  // Check if this node_unit's type is supported
  // SingleNode type NodeUnit is supported
  // QDQGroup type NodeUnit is by default unsupported, and this can be individually overwritten by inherited classes
  virtual bool IsNodeUnitTypeSupported(const NodeUnit& node_unit) const;

 private:
  bool HasSupportedOpSet(const NodeUnit& node_unit) const;
  bool HasSupportedInputOutputs(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                                const OpSupportCheckParams& params) const;
};

}  // namespace nnapi
}  // namespace onnxruntime
