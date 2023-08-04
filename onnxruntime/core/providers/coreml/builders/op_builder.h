// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/graph_viewer.h"

namespace onnxruntime {
namespace coreml {

class ModelBuilder;

struct OpBuilderInputParams {
  OpBuilderInputParams(const GraphViewer& graph_viewer, bool only_allow_static_input_shapes)
      : graph_viewer(graph_viewer),
        only_allow_static_input_shapes(only_allow_static_input_shapes) {}

  const GraphViewer& graph_viewer;
  const bool only_allow_static_input_shapes;
};

class IOpBuilder {
 public:
  virtual ~IOpBuilder() = default;

  // Add operator related
#ifdef __APPLE__
 public:
  // Check if the initializers of this operator need preprocess
  // which will not be copied
  virtual void AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const = 0;

  // Add the operator to CoreML model
  virtual Status AddToModelBuilder(ModelBuilder& model_builder, const Node& node,
                                   const OpBuilderInputParams& input_params,
                                   const logging::Logger& logger) const = 0;
#endif

  // Operator support related
 public:
  // Check if an operator is supported
  virtual bool IsOpSupported(const Node& node, const OpBuilderInputParams& input_params,
                             const logging::Logger& logger) const = 0;
};

}  // namespace coreml
}  // namespace onnxruntime
