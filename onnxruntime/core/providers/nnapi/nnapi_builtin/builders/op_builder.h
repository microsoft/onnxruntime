// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <unordered_map>

#include "core/graph/basic_types.h"
#include "core/providers/nnapi/nnapi_builtin/builders/helper.h"

namespace onnxruntime {

class Node;
class NodeUnit;

namespace common {
class Status;
}

namespace nnapi {

class ModelBuilder;

struct OpSupportCheckParams {
  OpSupportCheckParams(int32_t android_feature_level, bool use_nchw)
      : android_feature_level(android_feature_level),
        use_nchw(use_nchw) {
  }

  int32_t android_feature_level = 0;
  bool use_nchw = false;
};

class IOpBuilder {
 public:
  virtual ~IOpBuilder() = default;

 public:
  // Add operator related

  // Check if the initializers of this operator need preprocess
  // which will not be copied
  virtual void AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const = 0;

  // Add the operator to NNAPI model
  virtual common::Status AddToModelBuilder(ModelBuilder& model_builder, const NodeUnit& node_unit) const = 0;

  // Get the lookup table with IOpBuilder delegates for different onnx operators
  // Note, the lookup table should have same number of entries as the result of CreateOpBuilders() in op_builder.h
  const std::unordered_map<std::string, const IOpBuilder*>& GetOpBuilders();

  // Transpose the NHWC input to NCHW output
  common::Status TransposeNHWCToNCHW(ModelBuilder& model_builder, const std::string& input, const std::string& output);

  // Operator support check related

  // Check if an operator is supported
  virtual bool IsOpSupported(const GraphViewer& graph_viewer, const NodeUnit& node_unit,
                             const OpSupportCheckParams& params) const = 0;
};

}  // namespace nnapi
}  // namespace onnxruntime
