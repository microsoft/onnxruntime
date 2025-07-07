// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

class QnnModelWrapper;

class IOpBuilder {
 public:
  virtual ~IOpBuilder() = default;

  // Check whether the operator is supported or not
  virtual Status IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                               const NodeUnit& node,
                               const logging::Logger& logger) const ORT_MUST_USE_RESULT = 0;

#if !BUILD_QNN_EP_STATIC_LIB
  // ABI-compatible version of IsOpSupported for shared library builds
  virtual bool IsOpSupportedForABI(const OrtNode* ort_node,
                                   const OrtApi& ort_api,
                                   const OrtGraph* graph,
                                   const OrtLogger* logger,
                                   const OrtEpGraphSupportInfo* graph_support_info) const = 0;
#endif

  // Add the operator to QNN model
  virtual Status AddToModelBuilder(QnnModelWrapper& qnn_model_wrapper,
                                   const NodeUnit& node,
                                   const logging::Logger& logger,
                                   bool do_op_validation = false) const ORT_MUST_USE_RESULT = 0;

  virtual std::string GetOpBuilderType() const = 0;
};

}  // namespace qnn
}  // namespace onnxruntime
