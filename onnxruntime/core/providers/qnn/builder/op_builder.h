// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/graph_viewer.h"
#include "core/framework/node_unit.h"
#include "core/providers/shared/utils/utils.h"

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

  // Add the operator to QNN model
  virtual Status AddToModelBuilder(QnnModelWrapper& qnn_model_wrapper,
                                   const NodeUnit& node,
                                   const logging::Logger& logger,
                                   bool do_op_validation = false) const ORT_MUST_USE_RESULT = 0;

  virtual std::string GetOpBuilderType() const = 0;
};

}  // namespace qnn
}  // namespace onnxruntime
