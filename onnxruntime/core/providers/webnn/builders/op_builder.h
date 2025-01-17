// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webnn/builders/helper.h"

#pragma once

namespace onnxruntime {
namespace webnn {

class ModelBuilder;

class IOpBuilder {
 public:
  virtual ~IOpBuilder() = default;

  // Add operator related.
 public:
  // Check if the initializers of this operator need preprocess,
  // which will not be copied.
  virtual void AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const = 0;

  // Add the operator to WebNN model.
  virtual Status AddToModelBuilder(ModelBuilder& model_builder, const Node& node,
                                   const logging::Logger& logger) const ORT_MUST_USE_RESULT = 0;

  // Operator support related.
 public:
  // Check if an operator is supported.
  virtual bool IsOpSupported(const InitializedTensorSet& initializers, const Node& node,
                             const WebnnDeviceType device_type, const emscripten::val& wnn_limits,
                             const logging::Logger& logger) const = 0;
};

}  // namespace webnn
}  // namespace onnxruntime
