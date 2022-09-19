// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/nnapi/nnapi_builtin/builders/op_support_checker.h"

namespace onnxruntime {

class NodeUnit;

namespace nnapi {

struct OpSupportCheckerRegistrations {
  std::vector<std::unique_ptr<IOpSupportChecker>> support_checkers;
  std::unordered_map<std::string, const IOpSupportChecker*> op_support_checker_map;
};

class BaseOpSupportChecker : public IOpSupportChecker {
 public:
  virtual ~BaseOpSupportChecker() = default;

  // Operator support related

  bool IsOpSupported(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                     const OpSupportCheckParams& params) const override;

  // This is for ops which are by default supported and do not have their own impl of OpSupportChecker
  // for those ops (Relu, Identity) we use BaseOpSupportChecker
  static void CreateSharedOpSupportChecker(
      const std::string& op_type, OpSupportCheckerRegistrations& op_registrations);

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
