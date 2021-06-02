// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/graph_viewer.h"
namespace onnxruntime {
namespace coreml {

class ModelBuilder;

struct OpBuilderInputParams {
  OpBuilderInputParams(const GraphViewer& graph_viewer)
      : graph_viewer(graph_viewer) {}

  const GraphViewer& graph_viewer;
};

class IOpBuilder {
 public:
  virtual ~IOpBuilder() = default;

  // Add operator related
 public:
  // Check if the initializers of this operator need preprocess
  // which will not be copied
  virtual void AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const = 0;

  // Add the operator to CoreML model
  virtual Status AddToModelBuilder(ModelBuilder& model_builder, const Node& node,
                                   const logging::Logger& logger) const ORT_MUST_USE_RESULT = 0;

  // Operator support related
 public:
  // Check if an operator is supported
  virtual bool IsOpSupported(const Node& node, OpBuilderInputParams& input_params,
                             const logging::Logger& logger) const = 0;
};

}  // namespace coreml
}  // namespace onnxruntime