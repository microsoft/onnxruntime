// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/graph_viewer.h"

namespace onnxruntime {
namespace coreml {

class ModelBuilder;

struct OpBuilderInputParams {
  OpBuilderInputParams(const GraphViewer& graph_viewer,
                       int32_t coreml_version,
                       bool only_allow_static_input_shapes,
                       bool create_mlprogram)
      : graph_viewer(graph_viewer),
        coreml_version(coreml_version),
        only_allow_static_input_shapes(only_allow_static_input_shapes),
        create_mlprogram(create_mlprogram) {}

  const GraphViewer& graph_viewer;
  const int32_t coreml_version;  // required to determine which version of an operation can be used.
  const bool only_allow_static_input_shapes;
  const bool create_mlprogram;  // whether to create ML Program (Core ML 5+) or NeuralNetwork (Core ML 3+)
};

class IOpBuilder {
 public:
  virtual ~IOpBuilder() = default;

  // Check if the initializers of this operator need preprocess
  // which will not be copied
  virtual void AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const = 0;

  // Add the operator to CoreML model
  virtual Status AddToModelBuilder(ModelBuilder& model_builder, const Node& node,
                                   const logging::Logger& logger) const = 0;

  // Check if an operator is supported
  virtual bool IsOpSupported(const Node& node, const OpBuilderInputParams& input_params,
                             const logging::Logger& logger) const = 0;

  // Does the builder implementation support creating an ML Program?
  virtual bool SupportsMLProgram() const = 0;
};

}  // namespace coreml
}  // namespace onnxruntime
